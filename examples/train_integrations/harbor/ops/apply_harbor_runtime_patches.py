#!/usr/bin/env python3
"""Apply or verify the Harbor site-packages patches needed by the runtime-full workflow."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import apply_harbor_rootless_patch as rootless_patch


ASSET_ROOT = Path(__file__).resolve().parent / "harbor_site_packages"
BASE_ASSET = ASSET_ROOT / "agents" / "installed" / "base.py"
TEMPLATE_ASSET = (
    ASSET_ROOT / "agents" / "installed" / "install-mini-swe-agent.sh.j2"
)
COMPOSE_ASSET = (
    ASSET_ROOT
    / "environments"
    / "docker"
    / "docker-compose-base.yaml"
)
COMPOSE_DEFAULT_BRIDGE_ASSET = (
    ASSET_ROOT
    / "environments"
    / "docker"
    / "docker-compose-default-bridge.yaml"
)
COMPOSE_SHARED_NETWORK_ASSET = (
    ASSET_ROOT
    / "environments"
    / "docker"
    / "docker-compose-shared-network.yaml"
)
VERIFIER_ASSET = ASSET_ROOT / "verifier" / "verifier.py"
MINI_SWE_API_BASE_MARKER = 'self._api_base = api_base or kwargs.pop("api_base", None)'
MINI_SWE_HOSTED_VLLM_MARKER = 'env["HOSTED_VLLM_API_BASE"] = self._api_base'
MINI_SWE_METADATA_MARKER = 'context.metadata = self._build_harbor_metadata_from_messages(messages)'
MINI_SWE_CONFIG_MARKER = 'mini -c {self._mini_swe_agent_config_path} -m {self.model_name} -t {escaped_instruction} -y '
MINI_SWE_HOSTED_VLLM_KEY_MARKER = 'elif self.model_name.startswith("hosted_vllm/"):'
MINI_SWE_LITELLM_REGISTRY_MARKER = 'env["LITELLM_MODEL_REGISTRY_PATH"] = str('
MINI_SWE_MAX_TURNS_MARKER = 'self._max_turns = kwargs.pop("max_turns", None)'
MINI_SWE_STEP_LIMIT_MARKER = 'agent_cfg["step_limit"] = max(int(self._max_turns), 0)'
DOCKER_TASK_INPUT_FIELDS_MARKER = "host_task_workspace_path: str"
DOCKER_TASK_INPUT_ENV_MARKER = 'env_task_workspace_path=str(Path("/harbor-mounted/workspace"))'
DOCKER_NON_TTY_MARKER = 'exec_command = ["exec", "-T"]'
DOCKER_DEFAULT_BRIDGE_MARKER = '_DOCKER_COMPOSE_DEFAULT_BRIDGE_PATH = ('
DOCKER_SHARED_NETWORK_MARKER = '_DOCKER_COMPOSE_SHARED_NETWORK_PATH = ('
DOCKER_SHARED_NETWORK_ENV_MARKER = 'shared_network_name = os.environ.get("HARBOR_DOCKER_SHARED_NETWORK_NAME", "").strip()'
DOCKER_START_CLEANUP_MARKER = '["down", "--remove-orphans", "--volumes"], check=False'
TRIAL_SETUP_TIMEOUT_MARKER = "_AGENT_SETUP_TIMEOUT_SEC = 600"
TRIAL_VERIFIER_TIMEOUT_ENV_MARKER = "def _verifier_timeout_max_attempts() -> int:"
TRIAL_VERIFIER_TIMEOUT_STOP_MARKER = (
    "stop=stop_after_attempt(_verifier_timeout_max_attempts()),\n"
    "        wait=wait_exponential(multiplier=1, min=1, max=10),\n"
    "        retry=retry_if_exception_type(VerifierTimeoutError),"
)
TRIAL_ENV_START_WRONG_MARKER = (
    "stop=stop_after_attempt(_verifier_timeout_max_attempts()),\n"
    "        wait=wait_exponential(multiplier=1, min=1, max=10),\n"
    "        retry=retry_if_exception_type(EnvironmentStartTimeoutError),"
)


def _patch_mini_swe_agent_text(text: str) -> str:
    if (
        MINI_SWE_API_BASE_MARKER in text
        and MINI_SWE_HOSTED_VLLM_MARKER in text
        and MINI_SWE_METADATA_MARKER in text
        and MINI_SWE_CONFIG_MARKER in text
        and MINI_SWE_HOSTED_VLLM_KEY_MARKER in text
        and MINI_SWE_LITELLM_REGISTRY_MARKER in text
        and MINI_SWE_MAX_TURNS_MARKER in text
        and MINI_SWE_STEP_LIMIT_MARKER in text
    ):
        return text

    if "import yaml" not in text:
        import_anchor = "from typing import Any\n"
        if import_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py import anchor")
        text = text.replace(import_anchor, import_anchor + "\nimport yaml\n", 1)

    if "from harbor.models.agent.rollout_detail import RolloutDetail" not in text:
        rollout_import_anchor = "from harbor.models.agent.name import AgentName\n"
        if rollout_import_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py rollout import anchor")
        text = text.replace(
            rollout_import_anchor,
            rollout_import_anchor + "from harbor.models.agent.rollout_detail import RolloutDetail\n",
            1,
        )

    old_key_block = """        if \"MSWEA_API_KEY\" in os.environ:\n            env[\"MSWEA_API_KEY\"] = os.environ[\"MSWEA_API_KEY\"]\n        else:\n"""
    new_key_block = """        if \"MSWEA_API_KEY\" in os.environ:\n            env[\"MSWEA_API_KEY\"] = os.environ[\"MSWEA_API_KEY\"]\n        elif self.model_name.startswith(\"hosted_vllm/\"):\n            env[\"HOSTED_VLLM_API_KEY\"] = (\n                os.environ.get(\"HOSTED_VLLM_API_KEY\")\n                or os.environ.get(\"OPENAI_API_KEY\")\n                or \"fake-api-key\"\n            )\n        else:\n"""
    if MINI_SWE_HOSTED_VLLM_KEY_MARKER not in text:
        if old_key_block not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py key block")
        text = text.replace(old_key_block, new_key_block, 1)

    if MINI_SWE_API_BASE_MARKER not in text:
        old_init_anchor = """    SUPPORTS_ATIF: bool = True\n\n    @staticmethod\n"""
        new_init_anchor = """    SUPPORTS_ATIF: bool = True\n\n    def __init__(\n        self,\n        api_base: str | None = None,\n        *args,\n        **kwargs,\n    ):\n        self._api_base = api_base or kwargs.pop(\"api_base\", None)\n        self._collect_rollout_details = bool(kwargs.pop(\"collect_rollout_details\", False))\n        self._llm_call_kwargs = dict(kwargs.pop(\"llm_call_kwargs\", {}) or {})\n        self._temperature = kwargs.get(\"temperature\")\n        self._max_turns = kwargs.pop(\"max_turns\", None)\n        super().__init__(*args, **kwargs)\n\n    @staticmethod\n"""
        if old_init_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py init anchor")
        text = text.replace(old_init_anchor, new_init_anchor, 1)
    elif "self._collect_rollout_details = bool(kwargs.pop(\"collect_rollout_details\", False))" not in text:
        text = text.replace(
            MINI_SWE_API_BASE_MARKER,
            MINI_SWE_API_BASE_MARKER
            + "\n        self._collect_rollout_details = bool(kwargs.pop(\"collect_rollout_details\", False))"
            + "\n        self._llm_call_kwargs = dict(kwargs.pop(\"llm_call_kwargs\", {}) or {})"
            + "\n        self._temperature = kwargs.get(\"temperature\")",
            1,
        )
    if MINI_SWE_MAX_TURNS_MARKER not in text:
        init_tail_anchor = '        self._temperature = kwargs.get("temperature")'
        if init_tail_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py max_turns init anchor")
        text = text.replace(
            init_tail_anchor,
            init_tail_anchor + '\n        self._max_turns = kwargs.pop("max_turns", None)',
            1,
        )

    helper_anchor = """    def _atif_trajectory_path(self) -> Path:\n        \"\"\"Path where we write the ATIF-formatted trajectory.\"\"\"\n        return EnvironmentPaths.agent_dir / \"trajectory.json\"\n"""
    helper_block = """    def _atif_trajectory_path(self) -> Path:\n        \"\"\"Path where we write the ATIF-formatted trajectory.\"\"\"\n        return EnvironmentPaths.agent_dir / \"trajectory.json\"\n\n    @property\n    def _mini_swe_agent_config_path(self) -> Path:\n        \"\"\"Path where we write the mini-swe-agent config override.\"\"\"\n        return EnvironmentPaths.agent_dir / \"mini-swe-agent.config.yaml\"\n\n    @property\n    def _mini_swe_agent_config_host_path(self) -> Path:\n        return self.logs_dir / \"mini-swe-agent.config.yaml\"\n\n    @staticmethod\n    def _build_harbor_metadata_from_messages(\n        messages: list[dict[str, Any]],\n    ) -> dict[str, Any]:\n        harbor_messages = [dict(message) for message in messages if message.get(\"role\") != \"system\"]\n        return {\n            \"all_messages\": harbor_messages,\n            \"summarization_count\": 0,\n            \"n_episodes\": sum(1 for message in harbor_messages if message.get(\"role\") == \"assistant\"),\n        }\n\n    @staticmethod\n    def _build_rollout_details_from_messages(\n        messages: list[dict[str, Any]],\n    ) -> list[RolloutDetail] | None:\n        prompt_token_ids: list[list[int]] = []\n        completion_token_ids: list[list[int]] = []\n        logprobs: list[list[float]] = []\n\n        for message in messages:\n            if message.get(\"role\") != \"assistant\":\n                continue\n\n            response = ((message.get(\"extra\") or {}).get(\"response\") or {})\n            choices = response.get(\"choices\") or []\n            if not choices:\n                return None\n\n            choice = choices[0] or {}\n            provider_specific_fields = choice.get(\"provider_specific_fields\") or {}\n            raw_prompt_token_ids = response.get(\"prompt_token_ids\")\n            raw_completion_token_ids = provider_specific_fields.get(\"token_ids\")\n            raw_logprobs = ((choice.get(\"logprobs\") or {}).get(\"content\") or [])\n\n            if raw_prompt_token_ids is None or raw_completion_token_ids is None:\n                return None\n\n            cur_logprobs = []\n            for entry in raw_logprobs:\n                if not isinstance(entry, dict) or entry.get(\"logprob\") is None:\n                    return None\n                cur_logprobs.append(float(entry[\"logprob\"]))\n\n            cur_prompt_token_ids = [int(token_id) for token_id in raw_prompt_token_ids]\n            cur_completion_token_ids = [int(token_id) for token_id in raw_completion_token_ids]\n            if len(cur_logprobs) != len(cur_completion_token_ids):\n                return None\n\n            prompt_token_ids.append(cur_prompt_token_ids)\n            completion_token_ids.append(cur_completion_token_ids)\n            logprobs.append(cur_logprobs)\n\n        if not completion_token_ids:\n            return None\n\n        return [\n            {\n                \"prompt_token_ids\": prompt_token_ids,\n                \"completion_token_ids\": completion_token_ids,\n                \"logprobs\": logprobs,\n            }\n        ]\n\n    def _load_base_config(self) -> dict[str, Any]:\n        shared_tool_home = Path(\n            os.environ.get(\n                \"HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME\",\n                f\"/scratch/triton_cache/{os.environ.get('USER', 'hkang')}/harbor-mini-swe-home\",\n            )\n        )\n        config_path = (\n            shared_tool_home\n            / \".local/share/uv/tools/mini-swe-agent/lib/python3.10/site-packages/minisweagent/config/mini.yaml\"\n        )\n        if not config_path.exists():\n            return {}\n        loaded = yaml.safe_load(config_path.read_text())\n        return loaded if isinstance(loaded, dict) else {}\n\n    def _build_model_kwargs(self) -> dict[str, Any]:\n        model_kwargs = dict(self._llm_call_kwargs)\n        extra_body = dict(model_kwargs.get(\"extra_body\") or {})\n\n        if self._temperature is not None and \"temperature\" not in model_kwargs:\n            model_kwargs[\"temperature\"] = self._temperature\n        if self._api_base:\n            model_kwargs[\"api_base\"] = self._api_base\n        if self._collect_rollout_details:\n            model_kwargs[\"logprobs\"] = True\n            if self.model_name and self.model_name.startswith(\"hosted_vllm/\"):\n                extra_body[\"return_token_ids\"] = True\n        if extra_body:\n            model_kwargs[\"extra_body\"] = extra_body\n\n        return model_kwargs\n\n    def _write_runtime_config(self) -> None:\n        config = self._load_base_config()\n        agent_cfg = config.setdefault(\"agent\", {})\n        if self._max_turns is not None:\n            agent_cfg[\"step_limit\"] = max(int(self._max_turns), 0)\n        model_cfg = config.setdefault(\"model\", {})\n        base_model_kwargs = model_cfg.get(\"model_kwargs\") or {}\n        model_cfg[\"model_kwargs\"] = {**base_model_kwargs, **self._build_model_kwargs()}\n        self._mini_swe_agent_config_host_path.write_text(\n            yaml.safe_dump(config, sort_keys=False)\n        )\n"""
    if "def _mini_swe_agent_config_path" not in text:
        if helper_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py helper anchor")
        text = text.replace(helper_anchor, helper_block, 1)
    if MINI_SWE_STEP_LIMIT_MARKER not in text:
        write_anchor = "        config = self._load_base_config()\n        model_cfg = config.setdefault(\"model\", {})\n"
        if write_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py step_limit anchor")
        text = text.replace(
            write_anchor,
            "        config = self._load_base_config()\n        agent_cfg = config.setdefault(\"agent\", {})\n        if self._max_turns is not None:\n            agent_cfg[\"step_limit\"] = max(int(self._max_turns), 0)\n        model_cfg = config.setdefault(\"model\", {})\n",
            1,
        )

    registry_path_anchor = """    @property\n    def _mini_swe_agent_config_host_path(self) -> Path:\n        return self.logs_dir / \"mini-swe-agent.config.yaml\"\n"""
    registry_path_block = """    @property\n    def _mini_swe_agent_config_host_path(self) -> Path:\n        return self.logs_dir / \"mini-swe-agent.config.yaml\"\n\n    @property\n    def _mini_swe_agent_model_registry_path(self) -> Path:\n        \"\"\"Path where we write the LiteLLM model registry override.\"\"\"\n        return EnvironmentPaths.agent_dir / \"mini-swe-agent.litellm_model_registry.json\"\n\n    @property\n    def _mini_swe_agent_model_registry_host_path(self) -> Path:\n        return self.logs_dir / \"mini-swe-agent.litellm_model_registry.json\"\n"""
    if "_mini_swe_agent_model_registry_path" not in text:
        if registry_path_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py registry path anchor")
        text = text.replace(registry_path_anchor, registry_path_block, 1)

    registry_method_anchor = """    def _write_runtime_config(self) -> None:\n"""
    registry_method_block = """    def _build_litellm_model_registry(self) -> dict[str, Any]:\n        if not self.model_name or not self.model_name.startswith(\"hosted_vllm/\"):\n            return {}\n\n        base_model_name = self.model_name.split(\"/\", 1)[1]\n        entry = {\n            \"max_tokens\": 32768,\n            \"input_cost_per_token\": 0.0,\n            \"output_cost_per_token\": 0.0,\n            \"litellm_provider\": \"hosted_vllm\",\n            \"mode\": \"chat\",\n        }\n        return {\n            self.model_name: dict(entry),\n            base_model_name: dict(entry),\n        }\n\n    def _write_runtime_config(self) -> None:\n"""
    if "def _build_litellm_model_registry" not in text:
        if registry_method_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py registry method anchor")
        text = text.replace(registry_method_anchor, registry_method_block, 1)

    registry_write_anchor = """        self._mini_swe_agent_config_host_path.write_text(\n            yaml.safe_dump(config, sort_keys=False)\n        )\n"""
    registry_write_block = """        self._mini_swe_agent_config_host_path.write_text(\n            yaml.safe_dump(config, sort_keys=False)\n        )\n        model_registry = self._build_litellm_model_registry()\n        if model_registry:\n            self._mini_swe_agent_model_registry_host_path.write_text(\n                json.dumps(model_registry, indent=2, sort_keys=True)\n            )\n"""
    if "model_registry = self._build_litellm_model_registry()" not in text:
        if registry_write_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py registry write anchor")
        text = text.replace(registry_write_anchor, registry_write_block, 1)

    old_message_loop = "        # Extract token usage from mini-swe-agent format\n        n_input_tokens = 0\n"
    new_message_loop = "        # Extract token usage from mini-swe-agent format\n        messages = mini_trajectory.get(\"messages\") or []\n        n_input_tokens = 0\n"
    if old_message_loop in text:
        text = text.replace(old_message_loop, new_message_loop, 1)

    old_for_loop = "        for message in mini_trajectory.get(\"messages\") or []:\n"
    if old_for_loop in text:
        text = text.replace(old_for_loop, "        for message in messages:\n", 1)

    if MINI_SWE_METADATA_MARKER not in text:
        metadata_anchor = """        context.n_input_tokens = n_input_tokens\n        context.n_output_tokens = n_output_tokens\n        context.n_cache_tokens = n_cache_tokens\n        context.cost_usd = total_cost\n"""
        metadata_block = """        context.n_input_tokens = n_input_tokens\n        context.n_output_tokens = n_output_tokens\n        context.n_cache_tokens = n_cache_tokens\n        context.cost_usd = total_cost\n        context.metadata = self._build_harbor_metadata_from_messages(messages)\n        rollout_details = self._build_rollout_details_from_messages(messages)\n        if rollout_details is not None:\n            context.rollout_details = rollout_details\n"""
        if metadata_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py metadata anchor")
        text = text.replace(metadata_anchor, metadata_block, 1)

    old_env_block = """        if self._api_base:\n            env[\"OPENAI_API_BASE\"] = self._api_base\n            env[\"OPENAI_BASE_URL\"] = self._api_base\n        else:\n            if \"OPENAI_API_BASE\" in os.environ:\n                env[\"OPENAI_API_BASE\"] = os.environ[\"OPENAI_API_BASE\"]\n            if \"OPENAI_BASE_URL\" in os.environ:\n                env[\"OPENAI_BASE_URL\"] = os.environ[\"OPENAI_BASE_URL\"]\n"""
    new_env_block = """        if self._api_base:\n            env[\"OPENAI_API_BASE\"] = self._api_base\n            env[\"OPENAI_BASE_URL\"] = self._api_base\n            env[\"HOSTED_VLLM_API_BASE\"] = self._api_base\n        else:\n            if \"OPENAI_API_BASE\" in os.environ:\n                env[\"OPENAI_API_BASE\"] = os.environ[\"OPENAI_API_BASE\"]\n            if \"OPENAI_BASE_URL\" in os.environ:\n                env[\"OPENAI_BASE_URL\"] = os.environ[\"OPENAI_BASE_URL\"]\n            if \"HOSTED_VLLM_API_BASE\" in os.environ:\n                env[\"HOSTED_VLLM_API_BASE\"] = os.environ[\"HOSTED_VLLM_API_BASE\"]\n\n        if \"HOSTED_VLLM_API_KEY\" in os.environ:\n            env[\"HOSTED_VLLM_API_KEY\"] = os.environ[\"HOSTED_VLLM_API_KEY\"]\n        elif \"OPENAI_API_KEY\" in os.environ:\n            env[\"HOSTED_VLLM_API_KEY\"] = os.environ[\"OPENAI_API_KEY\"]\n\n        self._write_runtime_config()\n"""
    if MINI_SWE_HOSTED_VLLM_MARKER not in text:
        if old_env_block not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py env block")
        text = text.replace(old_env_block, new_env_block, 1)

    registry_env_anchor = """        self._write_runtime_config()\n"""
    registry_env_block = """        self._write_runtime_config()\n        if self._mini_swe_agent_model_registry_host_path.exists():\n            env[\"LITELLM_MODEL_REGISTRY_PATH\"] = str(\n                self._mini_swe_agent_model_registry_path\n            )\n"""
    if MINI_SWE_LITELLM_REGISTRY_MARKER not in text:
        if registry_env_anchor not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py registry env anchor")
        text = text.replace(registry_env_anchor, registry_env_block, 1)

    old_command = """                    f\"mini -m {self.model_name} -t {escaped_instruction} -y \"\n"""
    new_command = """                    f\"mini -c {self._mini_swe_agent_config_path} -m {self.model_name} -t {escaped_instruction} -y \"\n"""
    if MINI_SWE_CONFIG_MARKER not in text:
        if old_command not in text:
            raise SystemExit("PATCH_MISSING mini_swe_agent.py command anchor")
        text = text.replace(old_command, new_command, 1)

    return text


def _patch_docker_text(text: str) -> str:
    if not rootless_patch.is_patch_present(text):
        text = rootless_patch.patch_text(text)

    if (
        DOCKER_TASK_INPUT_FIELDS_MARKER in text
        and DOCKER_TASK_INPUT_ENV_MARKER in text
        and DOCKER_NON_TTY_MARKER in text
        and DOCKER_DEFAULT_BRIDGE_MARKER in text
        and DOCKER_SHARED_NETWORK_MARKER in text
        and DOCKER_SHARED_NETWORK_ENV_MARKER in text
        and DOCKER_START_CLEANUP_MARKER in text
    ):
        return text

    compose_paths_anchor = """    _DOCKER_COMPOSE_PREBUILT_PATH = (
        Path(__file__).parent / "docker-compose-prebuilt.yaml"
    )
    _DOCKER_COMPOSE_NO_NETWORK_PATH = (
        Path(__file__).parent / "docker-compose-no-network.yaml"
    )
"""
    compose_paths_block = """    _DOCKER_COMPOSE_PREBUILT_PATH = (
        Path(__file__).parent / "docker-compose-prebuilt.yaml"
    )
    _DOCKER_COMPOSE_DEFAULT_BRIDGE_PATH = (
        Path(__file__).parent / "docker-compose-default-bridge.yaml"
    )
    _DOCKER_COMPOSE_SHARED_NETWORK_PATH = (
        Path(__file__).parent / "docker-compose-shared-network.yaml"
    )
    _DOCKER_COMPOSE_NO_NETWORK_PATH = (
        Path(__file__).parent / "docker-compose-no-network.yaml"
    )
"""
    if DOCKER_DEFAULT_BRIDGE_MARKER not in text or DOCKER_SHARED_NETWORK_MARKER not in text:
        if compose_paths_anchor not in text:
            raise SystemExit("PATCH_MISSING docker.py compose paths anchor")
        text = text.replace(compose_paths_anchor, compose_paths_block, 1)

    if DOCKER_TASK_INPUT_FIELDS_MARKER not in text or DOCKER_TASK_INPUT_ENV_MARKER not in text:
        envvars_anchor = """    host_artifacts_path: str\n    env_verifier_logs_path: str\n    env_agent_logs_path: str\n    env_artifacts_path: str\n"""
        envvars_block = """    host_artifacts_path: str\n    host_task_workspace_path: str\n    host_task_tests_path: str\n    env_verifier_logs_path: str\n    env_agent_logs_path: str\n    env_artifacts_path: str\n    env_task_workspace_path: str\n    env_task_tests_path: str\n"""
        if envvars_anchor not in text:
            raise SystemExit("PATCH_MISSING docker.py env vars anchor")
        text = text.replace(envvars_anchor, envvars_block, 1)

        init_anchor = """        self._keep_containers = keep_containers\n\n        self._env_vars = DockerEnvironmentEnvVars(\n"""
        init_block = """        self._keep_containers = keep_containers\n\n        task_root = self.environment_dir.parent\n        host_task_tests_path = task_root / \"tests\"\n        host_task_workspace_path = self.environment_dir / \"workspace\"\n        if not host_task_workspace_path.is_dir():\n            host_task_workspace_path = trial_paths.trial_dir / \"_mounted-empty-workspace\"\n            host_task_workspace_path.mkdir(parents=True, exist_ok=True)\n\n        self._env_vars = DockerEnvironmentEnvVars(\n"""
        if init_anchor not in text:
            raise SystemExit("PATCH_MISSING docker.py init anchor")
        text = text.replace(init_anchor, init_block, 1)

        env_assign_anchor = """            host_artifacts_path=str(trial_paths.artifacts_dir.resolve().absolute()),\n            env_verifier_logs_path=str(EnvironmentPaths.verifier_dir),\n            env_agent_logs_path=str(EnvironmentPaths.agent_dir),\n            env_artifacts_path=str(EnvironmentPaths.artifacts_dir),\n"""
        env_assign_block = """            host_artifacts_path=str(trial_paths.artifacts_dir.resolve().absolute()),\n            host_task_workspace_path=str(host_task_workspace_path.resolve().absolute()),\n            host_task_tests_path=str(host_task_tests_path.resolve().absolute()),\n            env_verifier_logs_path=str(EnvironmentPaths.verifier_dir),\n            env_agent_logs_path=str(EnvironmentPaths.agent_dir),\n            env_artifacts_path=str(EnvironmentPaths.artifacts_dir),\n            env_task_workspace_path=str(Path(\"/harbor-mounted/workspace\")),\n            env_task_tests_path=str(Path(\"/harbor-mounted/tests\")),\n"""
        if env_assign_anchor not in text:
            raise SystemExit("PATCH_MISSING docker.py env assignment anchor")
        text = text.replace(env_assign_anchor, env_assign_block, 1)

    exec_tty_anchor = '        exec_command = ["exec", "-it"]\n'
    exec_tty_block = '        exec_command = ["exec", "-T"]\n'
    if DOCKER_NON_TTY_MARKER not in text:
        if exec_tty_anchor not in text:
            raise SystemExit("PATCH_MISSING docker.py exec tty anchor")
        text = text.replace(exec_tty_anchor, exec_tty_block, 1)

    network_selection_anchor = """        if not self.task_env_config.allow_internet:
            paths.append(self._DOCKER_COMPOSE_NO_NETWORK_PATH)

        return paths
"""
    network_selection_block = """        shared_network_name = os.environ.get("HARBOR_DOCKER_SHARED_NETWORK_NAME", "").strip()
        if not self.task_env_config.allow_internet:
            paths.append(self._DOCKER_COMPOSE_NO_NETWORK_PATH)
        elif shared_network_name:
            paths.append(self._DOCKER_COMPOSE_SHARED_NETWORK_PATH)
        else:
            paths.append(self._DOCKER_COMPOSE_DEFAULT_BRIDGE_PATH)

        return paths
"""
    if DOCKER_SHARED_NETWORK_ENV_MARKER not in text:
        if network_selection_anchor not in text:
            raise SystemExit("PATCH_MISSING docker.py network selection anchor")
        text = text.replace(network_selection_anchor, network_selection_block, 1)

    start_block = """        try:
            await self._run_docker_compose_command(["up", "-d"])
        except Exception:
            await self._run_docker_compose_command(
                ["down", "--remove-orphans", "--volumes"], check=False
            )
            raise
"""
    broken_start_block = """        try:
            try:
            await self._run_docker_compose_command(["up", "-d"])
        except Exception:
            await self._run_docker_compose_command(
                ["down", "--remove-orphans", "--volumes"], check=False
            )
            raise
        except Exception:
            await self._run_docker_compose_command(
                ["down", "--remove-orphans", "--volumes"], check=False
            )
            raise
"""
    if broken_start_block in text:
        text = text.replace(broken_start_block, start_block, 1)
    elif DOCKER_START_CLEANUP_MARKER not in text:
        start_anchor = """        await self._run_docker_compose_command(["up", "-d"])
"""
        if start_anchor not in text:
            raise SystemExit("PATCH_MISSING docker.py start cleanup anchor")
        text = text.replace(start_anchor, start_block, 1)

    return text


def _patch_trial_text(text: str) -> str:
    if (
        TRIAL_SETUP_TIMEOUT_MARKER in text
        and TRIAL_VERIFIER_TIMEOUT_ENV_MARKER in text
        and TRIAL_VERIFIER_TIMEOUT_STOP_MARKER in text
        and TRIAL_ENV_START_WRONG_MARKER not in text
    ):
        return text

    if "import os\n" not in text:
        import_anchor = "import logging\n"
        if import_anchor not in text:
            raise SystemExit("PATCH_MISSING trial.py import anchor")
        text = text.replace(import_anchor, import_anchor + "import os\n", 1)

    timeout_anchor = "    _AGENT_SETUP_TIMEOUT_SEC = 360\n"
    timeout_block = "    _AGENT_SETUP_TIMEOUT_SEC = 600\n"
    if timeout_anchor in text:
        text = text.replace(timeout_anchor, timeout_block, 1)
    elif TRIAL_SETUP_TIMEOUT_MARKER not in text:
        raise SystemExit("PATCH_MISSING trial.py setup timeout anchor")

    if TRIAL_VERIFIER_TIMEOUT_ENV_MARKER not in text:
        helper_anchor = "\n\nclass AgentSetupTimeoutError(asyncio.TimeoutError):\n"
        helper_block = """

def _verifier_timeout_max_attempts() -> int:
    value = os.environ.get("HARBOR_VERIFIER_TIMEOUT_MAX_ATTEMPTS", "1")
    try:
        return max(1, int(value))
    except ValueError:
        return 1


class AgentSetupTimeoutError(asyncio.TimeoutError):
"""
        if helper_anchor not in text:
            raise SystemExit("PATCH_MISSING trial.py verifier timeout helper anchor")
        text = text.replace(helper_anchor, helper_block, 1)

    stop_anchor = """    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(VerifierTimeoutError),
    )
"""
    stop_block = """    @retry(
        reraise=True,
        stop=stop_after_attempt(_verifier_timeout_max_attempts()),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(VerifierTimeoutError),
    )
"""
    if TRIAL_VERIFIER_TIMEOUT_STOP_MARKER not in text:
        if stop_anchor not in text:
            raise SystemExit("PATCH_MISSING trial.py verifier timeout stop anchor")
        text = text.replace(stop_anchor, stop_block, 1)

    env_start_wrong_block = """    @retry(
        reraise=True,
        stop=stop_after_attempt(_verifier_timeout_max_attempts()),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(EnvironmentStartTimeoutError),
    )
"""
    env_start_right_block = """    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(EnvironmentStartTimeoutError),
    )
"""
    if env_start_wrong_block in text:
        text = text.replace(env_start_wrong_block, env_start_right_block, 1)

    return text


def _resolve_targets() -> dict[str, Path]:
    base_mod = importlib.import_module("harbor.agents.installed.base")
    mini_swe_mod = importlib.import_module("harbor.agents.installed.mini_swe_agent")
    docker_mod = importlib.import_module("harbor.environments.docker.docker")
    trial_mod = importlib.import_module("harbor.trial.trial")
    verifier_mod = importlib.import_module("harbor.verifier.verifier")

    base_dir = Path(base_mod.__file__).resolve().parent
    docker_dir = Path(docker_mod.__file__).resolve().parent

    return {
        "base.py": Path(base_mod.__file__).resolve(),
        "docker.py": Path(docker_mod.__file__).resolve(),
        "trial.py": Path(trial_mod.__file__).resolve(),
        "mini_swe_agent.py": Path(mini_swe_mod.__file__).resolve(),
        "verifier.py": Path(verifier_mod.__file__).resolve(),
        "install-mini-swe-agent.sh.j2": base_dir / "install-mini-swe-agent.sh.j2",
        "docker-compose-base.yaml": docker_dir / "docker-compose-base.yaml",
        "docker-compose-default-bridge.yaml": docker_dir / "docker-compose-default-bridge.yaml",
        "docker-compose-shared-network.yaml": docker_dir / "docker-compose-shared-network.yaml",
    }


def _asset_map() -> dict[str, Path]:
    return {
        "base.py": BASE_ASSET,
        "verifier.py": VERIFIER_ASSET,
        "install-mini-swe-agent.sh.j2": TEMPLATE_ASSET,
        "docker-compose-base.yaml": COMPOSE_ASSET,
        "docker-compose-default-bridge.yaml": COMPOSE_DEFAULT_BRIDGE_ASSET,
        "docker-compose-shared-network.yaml": COMPOSE_SHARED_NETWORK_ASSET,
    }


def _ensure_backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(path.read_text())
        print(f"BACKUP_WRITTEN {backup_path}")


def _sync_text_asset(*, target: Path, asset: Path, check: bool, backup: bool) -> None:
    expected = asset.read_text()
    if not target.exists():
        if check:
            raise SystemExit(f"PATCH_MISSING {target}")
        target.write_text(expected)
        print(f"PATCH_APPLIED {target}")
        return

    current = target.read_text()
    if current == expected:
        print(f"PATCH_PRESENT {target}")
        return

    if check:
        raise SystemExit(f"PATCH_MISSING {target}")

    if backup:
        _ensure_backup(target)
    target.write_text(expected)
    print(f"PATCH_APPLIED {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only verify whether all patches are present")
    parser.add_argument("--backup", action="store_true", help="Write .bak files before patching")
    args = parser.parse_args()

    targets = _resolve_targets()

    docker_target = targets["docker.py"]
    docker_text = docker_target.read_text()
    docker_patched = (
        rootless_patch.is_patch_present(docker_text)
        and DOCKER_TASK_INPUT_FIELDS_MARKER in docker_text
        and DOCKER_TASK_INPUT_ENV_MARKER in docker_text
        and DOCKER_NON_TTY_MARKER in docker_text
        and DOCKER_DEFAULT_BRIDGE_MARKER in docker_text
        and DOCKER_SHARED_NETWORK_MARKER in docker_text
        and DOCKER_SHARED_NETWORK_ENV_MARKER in docker_text
        and DOCKER_START_CLEANUP_MARKER in docker_text
    )
    if args.check:
        if not docker_patched:
            raise SystemExit(f"PATCH_MISSING {docker_target}")
        print(f"PATCH_PRESENT {docker_target}")
    elif docker_patched:
        print(f"PATCH_PRESENT {docker_target}")
    else:
        new_text = _patch_docker_text(docker_text)
        if args.backup:
            _ensure_backup(docker_target)
        docker_target.write_text(new_text)
        print(f"PATCH_APPLIED {docker_target}")

    trial_target = targets["trial.py"]
    trial_text = trial_target.read_text()
    trial_patched = (
        TRIAL_SETUP_TIMEOUT_MARKER in trial_text
        and TRIAL_VERIFIER_TIMEOUT_ENV_MARKER in trial_text
        and TRIAL_VERIFIER_TIMEOUT_STOP_MARKER in trial_text
        and TRIAL_ENV_START_WRONG_MARKER not in trial_text
    )
    if args.check:
        if not trial_patched:
            raise SystemExit(f"PATCH_MISSING {trial_target}")
        print(f"PATCH_PRESENT {trial_target}")
    elif trial_patched:
        print(f"PATCH_PRESENT {trial_target}")
    else:
        new_text = _patch_trial_text(trial_text)
        if args.backup:
            _ensure_backup(trial_target)
        trial_target.write_text(new_text)
        print(f"PATCH_APPLIED {trial_target}")

    mini_swe_target = targets["mini_swe_agent.py"]
    mini_swe_text = mini_swe_target.read_text()
    mini_swe_patched = (
        MINI_SWE_API_BASE_MARKER in mini_swe_text
        and MINI_SWE_HOSTED_VLLM_MARKER in mini_swe_text
        and MINI_SWE_METADATA_MARKER in mini_swe_text
        and MINI_SWE_CONFIG_MARKER in mini_swe_text
        and MINI_SWE_HOSTED_VLLM_KEY_MARKER in mini_swe_text
        and MINI_SWE_LITELLM_REGISTRY_MARKER in mini_swe_text
        and MINI_SWE_MAX_TURNS_MARKER in mini_swe_text
        and MINI_SWE_STEP_LIMIT_MARKER in mini_swe_text
    )
    if args.check:
        if not mini_swe_patched:
            raise SystemExit(f"PATCH_MISSING {mini_swe_target}")
        print(f"PATCH_PRESENT {mini_swe_target}")
    elif mini_swe_patched:
        print(f"PATCH_PRESENT {mini_swe_target}")
    else:
        new_text = _patch_mini_swe_agent_text(mini_swe_text)
        if args.backup:
            _ensure_backup(mini_swe_target)
        mini_swe_target.write_text(new_text)
        print(f"PATCH_APPLIED {mini_swe_target}")

    for name, asset in _asset_map().items():
        _sync_text_asset(
            target=targets[name],
            asset=asset,
            check=args.check,
            backup=args.backup,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
