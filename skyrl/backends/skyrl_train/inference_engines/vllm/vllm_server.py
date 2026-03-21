import asyncio
import os
import signal
import inspect
import uvloop
from vllm import AsyncLLMEngine
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils import FlexibleArgumentParser
try:
    from vllm.utils.system_utils import set_ulimit
except ImportError:
    from vllm.utils import set_ulimit
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    create_server_socket,
    build_app,
    init_app_state,
)
import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from fastapi import HTTPException, Request


# TODO(tgriggs): Handle errors and use best practices for vLLM server
# TODO(tgriggs): Return correct status codes.
class VllmServer:
    def __init__(self, args):
        self.server_args = args

    async def run_server(self, **uvicorn_kwargs) -> None:
        sock_addr = (self.server_args.host or "", self.server_args.port)
        sock = create_server_socket(sock_addr)

        set_ulimit()

        def signal_handler(*_) -> None:
            # Interrupt server on sigterm while initializing
            raise KeyboardInterrupt("terminated")

        signal.signal(signal.SIGTERM, signal_handler)

        # TODO(tgriggs): Move this elsewhere, make configurable.
        os.environ["VLLM_USE_V1"] = "1"
        engine_args = AsyncEngineArgs.from_cli_args(self.server_args)
        engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )

        sock_addr = (self.server_args.host or "", self.server_args.port)
        sock = create_server_socket(sock_addr)
        app = build_app(self.server_args)
        external_server_idx = int(os.environ.get("SKYRL_EXTERNAL_SERVER_IDX", "0"))
        paused_event = asyncio.Event()

        def _get_unfinished_request_ids(output_processor) -> list:
            if hasattr(output_processor, "external_req_ids"):
                return list(output_processor.external_req_ids.keys())
            return list(output_processor.request_states.keys())

        async def _abort_generation_requests() -> int:
            output_processor = getattr(engine, "output_processor", None)
            if output_processor is None:
                return 0
            unfinished_request_ids = _get_unfinished_request_ids(output_processor)
            if unfinished_request_ids:
                await engine.abort(unfinished_request_ids)
            await engine.reset_prefix_cache()
            return len(unfinished_request_ids)

        @app.middleware("http")
        async def _block_generation_when_paused(request: Request, call_next):
            if paused_event.is_set() and request.method == "POST" and request.url.path in (
                "/v1/chat/completions",
                "/v1/completions",
                "/inference/v1/generate",
            ):
                while paused_event.is_set():
                    await asyncio.sleep(0.05)
            return await call_next(request)

        async def _init_weight_receiver(request: Request, *, adjust_for_engine: bool) -> dict[str, str]:
            import pickle
            from skyrl.backends.skyrl_train.weight_sync import BroadcastInitInfo, CudaIpcInitInfo

            data = await request.json()
            try:
                init_info = BroadcastInitInfo(**data)
            except Exception:
                try:
                    init_info = CudaIpcInitInfo(**data)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail="Received invalid init info") from exc

            if adjust_for_engine:
                init_info = init_info.for_engine(
                    engine_index=external_server_idx,
                    tp_size=self.server_args.tensor_parallel_size,
                    pp_size=self.server_args.pipeline_parallel_size,
                )

            pickled_init_info = pickle.dumps(init_info)
            await engine.collective_rpc(
                "init_weight_update_communicator",
                args=(pickled_init_info,),
            )
            return {"status": "ok"}

        @app.post("/init_weight_update_communicator")
        async def _init_weight_update_communicator(request: Request):
            return await _init_weight_receiver(request, adjust_for_engine=False)

        @app.post("/init_weight_transfer")
        async def _init_weight_transfer(request: Request):
            return await _init_weight_receiver(request, adjust_for_engine=True)

        @app.post("/sleep")
        async def _sleep(request: Request):
            data = await request.json()
            level = data.get("level")

            # TODO(team): remove once vllm fixes this
            # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
            await engine.reset_prefix_cache()

            await engine.sleep(level)
            return {"status": "ok"}

        @app.post("/wake_up")
        async def _wake_up(request: Request):
            data = await request.json()
            tags = data.get("tags")
            await engine.wake_up(tags)
            return {"status": "ok"}

        @app.post("/reset_prefix_cache")
        async def _reset_prefix_cache(request: Request):
            await engine.reset_prefix_cache()
            return {"status": "ok"}

        @app.post("/pause")
        async def _pause(request: Request):
            data = await request.json()
            wait_for_inflight_request = data.get("wait_for_inflight_request", False)
            paused_event.set()
            if wait_for_inflight_request:
                return {"status": "ok", "aborted_requests": 0}
            aborted_requests = await _abort_generation_requests()
            return {"status": "ok", "aborted_requests": aborted_requests}

        @app.post("/resume")
        async def _resume(request: Request):
            paused_event.clear()
            return {"status": "ok"}

        # NOTE (sumanthrh): We use the _skyrl suffix to differentiate this from the native /update_weights endpoint
        # introduced in vLLM 0.16.0: https://github.com/vllm-project/vllm/pull/31943
        @app.post("/update_weights_skyrl")
        async def _update_weights(request: Request):
            import pickle
            from skyrl.backends.skyrl_train.weight_sync import BroadcastWeightUpdateRequest

            # Convert the HTTP request to a BroadcastWeightUpdateRequest
            # TODO(haochen): only the broadcast strategy is currently supported
            # for the remote inference engine path.
            # To support other strategies, we'll need to add a "strategy=xxx"
            # parameter in the HTTP request.
            data = await request.json()
            weight_request = BroadcastWeightUpdateRequest(**data)

            # Pickle to preserve type through collective_rpc
            pickled_request = pickle.dumps(weight_request)

            await engine.collective_rpc(
                "load_weights",
                args=(pickled_request,),
            )
            return {"status": "ok"}

        @app.post("/destroy_weights_update_group")
        async def _destroy_weights_update_group(request: Request):
            data = await request.json()  # noqa: F841
            await engine.collective_rpc(
                "teardown_weight_receiver",
                args=(),
            )
            return {"status": "ok"}

        @app.post("/finalize_weight_update")
        async def _finalize_weight_update(request: Request):
            data = await request.json()  # noqa: F841
            return {"status": "ok"}

        @app.get("/get_server_info")
        async def _get_server_info():
            """Return minimal server parallelism info for SkyRL remote clients."""
            return {
                "ip": self.server_args.host,
                "port": self.server_args.port,
                "url": f"http://{self.server_args.host}:{self.server_args.port}",
                "world_size": self.server_args.tensor_parallel_size * self.server_args.pipeline_parallel_size,
            }

        if len(inspect.signature(init_app_state).parameters) == 4:
            if hasattr(engine, "get_vllm_config"):
                vllm_config = await engine.get_vllm_config()
            else:
                vllm_config = engine_args.create_engine_config(
                    usage_context=UsageContext.OPENAI_API_SERVER,
                )
            await init_app_state(engine, vllm_config, app.state, args)
        else:
            await init_app_state(engine, app.state, args)

        shutdown_task = await serve_http(
            app,
            sock,
            host=self.server_args.host,
            port=self.server_args.port,
            log_level=self.server_args.uvicorn_log_level,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=self.server_args.ssl_keyfile,
            ssl_certfile=self.server_args.ssl_certfile,
            ssl_ca_certs=self.server_args.ssl_ca_certs,
            ssl_cert_reqs=self.server_args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

        await shutdown_task

        sock.close()

    def run_server_uvloop(self, **uvicorn_kwargs) -> None:
        uvloop.run(self.run_server(**uvicorn_kwargs))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    vllm_server = VllmServer(args)
    vllm_server.run_server_uvloop()
