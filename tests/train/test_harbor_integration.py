from copy import deepcopy
from types import SimpleNamespace

from examples.train_integrations.harbor.dataset import HarborTaskDataset
from examples.train_integrations.harbor.harbor_generator import (
    HarborAgentOutput,
    HarborGenerator,
)
from harbor.llms.base import LLMResponse, OutputLengthExceededError
from harbor.llms.chat import Chat
from skyrl.train.generators.base import TrajectoryID


def test_harbor_sampling_params_are_applied_to_trial_config():
    config = {
        "agent": {
            "kwargs": {
                "temperature": 0.3,
                "llm_call_kwargs": {
                    "extra_body": {
                        "include_reasoning": False,
                    }
                },
            }
        }
    }
    sampling_params = {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.1,
        "repetition_penalty": 1.2,
        "max_generate_length": 64,
        "stop": ["DONE"],
        "logprobs": 1,
    }

    updated = HarborGenerator._apply_sampling_params_to_trial_config(deepcopy(config), sampling_params)

    agent_kwargs = updated["agent"]["kwargs"]
    assert agent_kwargs["temperature"] == 0.0
    assert agent_kwargs["collect_rollout_details"] is True
    assert agent_kwargs["llm_call_kwargs"]["top_p"] == 0.95
    assert agent_kwargs["llm_call_kwargs"]["stop"] == ["DONE"]
    assert agent_kwargs["llm_call_kwargs"]["max_tokens"] == 64
    assert agent_kwargs["llm_call_kwargs"]["extra_body"]["include_reasoning"] is False
    assert agent_kwargs["llm_call_kwargs"]["extra_body"]["top_k"] == 40
    assert agent_kwargs["llm_call_kwargs"]["extra_body"]["min_p"] == 0.1
    assert agent_kwargs["llm_call_kwargs"]["extra_body"]["repetition_penalty"] == 1.2


def test_harbor_trial_routing_ids_attach_program_id():
    config = {
        "agent": {
            "kwargs": {
                "llm_call_kwargs": {
                    "extra_body": {
                        "include_reasoning": False,
                    }
                }
            }
        }
    }

    updated = HarborGenerator._attach_trial_routing_ids(deepcopy(config), session_id="trial-123")

    assert updated["agent"]["kwargs"]["session_id"] == "trial-123"
    assert updated["agent"]["kwargs"]["llm_call_kwargs"]["extra_body"]["program_id"] == "trial-123"
    assert updated["agent"]["kwargs"]["llm_call_kwargs"]["extra_body"]["include_reasoning"] is False


def test_mask_failed_instances_zeroes_rollout_logprobs():
    outputs = [
        HarborAgentOutput(
            response_ids=[1, 2],
            reward=1.0,
            stop_reason="error",
            loss_mask=[1, 1],
            prompt_ids=[10],
            rollout_logprobs=[-0.1, -0.2],
            trajectory_id=TrajectoryID(instance_id="task-a", repetition_id=0),
        ),
        HarborAgentOutput(
            response_ids=[3, 4],
            reward=2.0,
            stop_reason="complete",
            loss_mask=[1, 1],
            prompt_ids=[11],
            rollout_logprobs=[-0.3, -0.4],
            trajectory_id=TrajectoryID(instance_id="task-a", repetition_id=1),
        ),
    ]

    masked_outputs, metrics = HarborGenerator._mask_failed_instances_and_compute_metrics(outputs)

    assert masked_outputs[0].rollout_logprobs == [0.0]
    assert masked_outputs[1].rollout_logprobs == [0.0]
    assert metrics["generate/num_masked_instances"] == 1


def test_extract_assistant_rollout_fields_from_rollout_details():
    agent_result = {
        "rollout_details": [
            {
                "prompt_token_ids": [[1, 2], [3, 4]],
                "completion_token_ids": [[5, 6], [7]],
                "logprobs": [[-0.1, -0.2], [-0.3]],
            }
        ]
    }

    assistant_logprobs = HarborGenerator._extract_assistant_rollout_field(agent_result, "logprobs")
    assistant_completion_token_ids = HarborGenerator._extract_assistant_rollout_field(
        agent_result, "completion_token_ids"
    )

    assert assistant_logprobs == [[-0.1, -0.2], [-0.3]]
    assert assistant_completion_token_ids == [[5, 6], [7]]


def test_harbor_dataset_uses_stable_sorted_task_uids(tmp_path):
    task_b = tmp_path / "task_b"
    task_b.mkdir()
    (task_b / "instruction.md").write_text("b", encoding="utf-8")

    task_a = tmp_path / "task_a"
    task_a.mkdir()
    (task_a / "instruction.md").write_text("a", encoding="utf-8")

    dataset = HarborTaskDataset(data_files=[str(tmp_path)])

    assert [dataset[i]["prompt"] for i in range(len(dataset))] == [str(task_a.resolve()), str(task_b.resolve())]
    assert [dataset[i]["uid"] for i in range(len(dataset))] == [str(task_a.resolve()), str(task_b.resolve())]


def test_output_length_exceeded_error_carries_rollout_details():
    err = OutputLengthExceededError(
        "truncated",
        truncated_response="partial",
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5],
        logprobs=[-0.1, -0.2],
    )

    assert err.truncated_response == "partial"
    assert err.prompt_token_ids == [1, 2, 3]
    assert err.completion_token_ids == [4, 5]
    assert err.logprobs == [-0.1, -0.2]


def test_harbor_chat_append_external_turn_preserves_rollout_details():
    class DummyModel:
        pass

    chat = Chat(DummyModel())
    llm_response = LLMResponse(
        content="partial",
        prompt_token_ids=[11, 12],
        completion_token_ids=[21, 22],
        logprobs=[-0.3, -0.4],
    )

    chat.append_external_turn("original prompt", llm_response)

    assert chat.messages == [
        {"role": "user", "content": "original prompt"},
        {"role": "assistant", "content": "partial"},
    ]
    assert chat.rollout_details == [
        {
            "prompt_token_ids": [[11, 12]],
            "completion_token_ids": [[21, 22]],
            "logprobs": [[-0.3, -0.4]],
        }
    ]


def test_harbor_generator_uses_inference_proxy_url_for_api_base():
    generator_cfg = SimpleNamespace(
        inference_engine=SimpleNamespace(
            http_endpoint_host="127.0.0.1",
            http_endpoint_port=8000,
            served_model_name="Qwen3-8B",
            engine_init_kwargs={},
        )
    )
    inference_engine_client = SimpleNamespace(proxy_url="http://127.0.0.1:8081")

    generator = HarborGenerator(
        generator_cfg=generator_cfg,
        harbor_cfg={},
        inference_engine_client=inference_engine_client,
        tokenizer=None,
        max_seq_len=1024,
    )

    assert generator.base_url == "http://127.0.0.1:8081"
    assert generator._harbor_trial_config_template["agent"]["kwargs"]["api_base"] == "http://127.0.0.1:8081/v1"
