"""LIMA's Terminal-Bench launcher with official Terminus 2 / Claude Code agents."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatchcase
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from harbor.agents.installed.claude_code import ClaudeCodeOptions
from harbor.agents.terminus_2.terminus_2 import Terminus2Options
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.job.config import JobConfig

DATASET = "terminal-bench/terminal-bench"
VERSION = "4.0.0"
HARBOR = Path(sys.executable).parent / "harbor"
DOCKER_IMPORT = "src.eval.terminalbench.run:LocalDockerEnvironment"
GPU_TASKS = ("fp8-rmsnorm-gemm", "math-eval-grader", "jax-speedrun-gpu")
LOOPBACK = {"localhost", "127.0.0.1", "::1"}
PRIVATE_DOCKER = "unix:///run/lima-docker/docker.sock"
RELAY_HOST = "10.255.254.1"
KEY_TEMPLATE = "${LIMA_TB_ANTHROPIC_API_KEY}"
CLAUDE_BINARY_VERSION = "2.1.80"
CLAUDE_BINARY_CACHE = ".cache/terminalbench-claude-verify-20261026/claude-2.1.80-linux-x64"
CLAUDE_BINARY_TARGET = "/usr/local/bin/claude"
CLAUDE_BINARY_SHA256 = "49fa3cf7aaab9d54066e85eaa11911b7d25c629a82af323a76b22db2029d4fa2"
CLAUDE_BINARY_ID = "LIMA_TB_CLAUDE_BINARY_SHA256"


class ModelConfigurationError(ValueError):
    """A local validation message, safe to report without provider response data."""


class LocalDockerEnvironment(DockerEnvironment):
    """Pin the task AND its separate verifier to the explicitly selected GPU.

    Harbor clears agent-only Compose overlays for separate verifiers, but keeps
    environment kwargs/import_path. Use those hooks rather than editing tasks.
    Jobs using this single-GPU adapter must run serially to avoid contention.
    """

    def __init__(self, *args, gpu_device=None, **kwargs):
        self.gpu_device = gpu_device
        super().__init__(*args, **kwargs)
        if self._effective_gpus > 0:
            if self._effective_gpus != 1 or not self.gpu_device:
                raise ValueError("Local GPU tasks require exactly one TB_GPU_DEVICE")

    @classmethod
    def _engine_cmd(cls, *args):
        # Keep Harbor's kernel probe and result cache; only its temporary
        # container needs to avoid the broken default docker0 bridge.
        if args[:4] == ("container", "run", "--rm", cls._EGRESS_CONTROL_KERNEL_PROBE_IMAGE):
            args = (*args[:3], "--network", "none", *args[3:])
        return super()._engine_cmd(*args)

    async def _write_resources_compose_file(self):
        path = await super()._write_resources_compose_file()
        if self._effective_gpus > 0:
            compose = json.loads(path.read_text())
            main = compose["services"]["main"]
            # The explicit runtime works after a daemon config reload even when
            # Docker's GPU DeviceRequest handler has not discovered the toolkit.
            reservations = main.get("deploy", {}).get("resources", {}).get("reservations", {})
            reservations.pop("devices", None)
            main["runtime"] = "nvidia"
            main["environment"] = {
                "NVIDIA_VISIBLE_DEVICES": self.gpu_device,
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            }
            path.write_text(json.dumps(compose, indent=2))
        return path


def needs_gpu(config):
    for dataset in config["datasets"]:
        included = dataset.get("task_names") or ["*"]
        excluded = dataset.get("exclude_task_names") or []
        for name in GPU_TASKS:
            matches = lambda pattern: any(fnmatchcase(n, pattern) for n in (name, "terminal-bench/" + name))
            if any(map(matches, included)) and not any(map(matches, excluded)):
                return True
    return False


def integer(name, default=None, minimum=1):
    raw = os.environ.get(name) or default
    if raw is None:
        return None
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def number(name, default):
    value = float(os.environ.get(name) or default)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def component(value):
    if not re.fullmatch(r"[\w.-]+", value) or value in {".", ".."}:
        raise ValueError(f"Invalid output name: {value!r}; use letters, numbers, _, -, .")
    return value


def task_names(name):
    return [s.strip() for s in os.environ.get(name, "").split(",") if s.strip()] or None


def api_url(base):
    url = urlsplit(base)
    if url.scheme not in {"http", "https"} or not url.hostname or url.port == 0:
        raise ValueError("API/proxy endpoints must be HTTP(S) URLs with a valid host/port")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("Keep credentials, queries and fragments out of API/proxy URLs; use AGENT_API_KEY")
    return url


def private_docker():
    return not os.environ.get("DOCKER_CONTEXT") and os.environ.get("DOCKER_HOST") == PRIVATE_DOCKER


def configure_claude_preinstall(config, root):
    """Choose the verified local binary only for eligible new jobs."""
    agent = config["agents"][0]
    if agent["name"] != "claude-code":
        return
    enabled = os.environ.get("TB_CLAUDE_PREINSTALL", "1")
    if enabled not in {"0", "1"}:
        raise ValueError("TB_CLAUDE_PREINSTALL must be 0 or 1")
    if (enabled == "0" or config["environment"]["type"] != "docker"
            or not private_docker() or agent["kwargs"]["version"] != CLAUDE_BINARY_VERSION
            or sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}):
        return
    config["environment"]["mounts"] = [{
        "type": "bind", "source": str((root / CLAUDE_BINARY_CACHE).resolve()),
        "target": CLAUDE_BINARY_TARGET, "read_only": True,
        "bind": {"create_host_path": False},
    }]
    agent["env"][CLAUDE_BINARY_ID] = CLAUDE_BINARY_SHA256


def validate_claude_preinstall(config):
    """Check saved identity/mounts too; never add mounts or migrate a resume."""
    agent, environment = config["agents"][0], config["environment"]
    identity = agent["env"].get(CLAUDE_BINARY_ID)
    mounts = [m for m in environment.get("mounts") or []
              if Path(m["target"]) == Path(CLAUDE_BINARY_TARGET)]
    if identity is None and not mounts:
        return
    if (identity != CLAUDE_BINARY_SHA256 or agent["kwargs"]["version"] != CLAUDE_BINARY_VERSION
            or environment["type"] != "docker" or not private_docker()):
        raise ValueError("Saved Claude preinstall requires the pinned 2.1.80 identity and the private local Docker daemon; resume cannot change its installation protocol")
    if len(mounts) != 1:
        raise ValueError("Claude preinstall requires exactly one saved binary mount")
    mount = mounts[0]
    if (mount["type"] != "bind" or mount["target"] != CLAUDE_BINARY_TARGET
            or mount.get("read_only") is not True
            or mount.get("bind", {}).get("create_host_path") is not False):
        raise ValueError("Claude preinstall requires a read-only file bind with create_host_path=false")
    source = Path(mount["source"])
    if not source.is_absolute() or not source.is_file() or not os.access(source, os.X_OK):
        raise ValueError(f"Claude preinstall binary is missing or not executable: {source}; restore it, or set TB_CLAUDE_PREINSTALL=0 for a new job")
    with source.open("rb") as binary:
        header = binary.read(20)
        if header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\x3e\x00":
            raise ValueError("Claude preinstall binary must be ELF64 little-endian x86_64")
        binary.seek(0)
        digest = hashlib.file_digest(binary, "sha256").hexdigest()
    if digest != identity:
        raise ValueError("Claude preinstall binary SHA256 differs from the pinned official release; restore the verified file")


def host_api_base(agent):
    if agent["name"] == "claude-code":
        return agent["env"]["LIMA_TB_HOST_API_BASE"]
    return agent["kwargs"]["api_base"]


def claude_agent(base, model):
    url = api_url(base)
    version = os.environ.get("TB_CLAUDE_VERSION") or "2.1.80"
    context = integer("AGENT_CONTEXT_LENGTH", 40960)
    output = integer("MAX_OUTPUT_TOKENS", 4096)
    origin = base.rstrip("/").removesuffix("/v1")
    env = {
        "LIMA_TB_HOST_API_BASE": base.rstrip("/"),
        "ANTHROPIC_API_KEY": KEY_TEMPLATE,
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(context),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(output),
        "DISABLE_AUTOUPDATER": "1",
    }
    container_base = os.environ.get("TB_CLAUDE_BASE_URL")
    is_private = (os.environ.get("TB_ENVIRONMENT") or "docker") == "docker" and private_docker()
    if not container_base and url.hostname in LOOPBACK:
        if not is_private or url.scheme != "http":
            raise ValueError("Claude Code runs inside the task container. Set TB_CLAUDE_BASE_URL to its reachable Anthropic API origin, or use the private Docker daemon with a local HTTP API")
        port = integer("TB_CLAUDE_RELAY_PORT", 18004)
        container_base = urlsplit(origin)._replace(netloc=f"{RELAY_HOST}:{port}").geturl()
        env["LIMA_TB_RELAY_UPSTREAM"] = origin
    env["ANTHROPIC_BASE_URL"] = (container_base or origin).rstrip("/")
    proxy = os.environ.get("TB_CONTAINER_PROXY", f"http://{RELAY_HOST}:17890" if is_private else "")
    if proxy:
        api_url(proxy)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[name] = proxy
    bypass = os.environ.get("TB_CONTAINER_NO_PROXY", f"localhost,127.0.0.1,::1,{RELAY_HOST},deb.debian.org" if is_private else "localhost,127.0.0.1,::1")
    # The local model relay must not go through the download proxy.
    if "LIMA_TB_RELAY_UPSTREAM" in env:
        bypass = ",".join(filter(None, [bypass, RELAY_HOST]))
    env["NO_PROXY"] = env["no_proxy"] = ",".join(dict.fromkeys(bypass.split(",")))
    kwargs = {"version": version}
    if (steps := integer("MAX_STEPS")) is not None:
        kwargs["max_turns"] = steps
    return {"name": "claude-code", "model_name": model, "kwargs": kwargs, "env": env}


def validate(config):
    JobConfig.model_validate(config)
    environment = config["environment"]
    if environment["type"] not in {"docker", "modal"}:
        raise ValueError("TB_ENVIRONMENT must be docker or modal")
    if environment["type"] == "docker":
        device = environment.get("kwargs", {}).get("gpu_device")
        if device is not None and not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", device):
            raise ValueError("TB_GPU_DEVICE must be one GPU index or UUID, such as 4")
        if device is not None and config["n_concurrent_trials"] != 1:
            raise ValueError("A pinned local GPU requires MAX_CONCURRENCY=1; run CPU-only tasks separately for higher concurrency")
    if len(config["agents"]) != 1:
        raise ValueError("This launcher requires exactly one agent")
    agent = config["agents"][0]
    if agent["name"] == "terminus-2":
        Terminus2Options.model_validate(agent.get("kwargs", {}))
        if not agent["model_name"].startswith("openai/"):
            raise ValueError("Terminus 2 requires an openai/ model ID with this launcher")
    elif agent["name"] == "claude-code":
        ClaudeCodeOptions.model_validate(agent.get("kwargs", {}))
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", agent["kwargs"].get("version", "")):
            raise ValueError("TB_CLAUDE_VERSION must pin an exact version, e.g. 2.1.80")
        env = agent["env"]
        if env.get("ANTHROPIC_API_KEY") != KEY_TEMPLATE:
            raise ValueError("Claude credentials must use the launcher's runtime key template")
        context, output = (int(env[name]) for name in ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_CODE_MAX_OUTPUT_TOKENS"))
        if output <= 0 or output >= context or context <= min(output, 20000) + 13000:
            raise ValueError("Claude output must be positive and below the context window, with room for its 13000-token compaction buffer")
        endpoint = api_url(env["ANTHROPIC_BASE_URL"])
        if endpoint.hostname in LOOPBACK or endpoint.hostname in {"0.0.0.0", "::"}:
            raise ValueError("TB_CLAUDE_BASE_URL must be reachable from the task container, not its loopback; unset it to use the automatic private Docker relay")
        if endpoint.path.rstrip("/").endswith("/v1"):
            raise ValueError("TB_CLAUDE_BASE_URL is the Anthropic API origin, without the final /v1")
        if upstream := env.get("LIMA_TB_RELAY_UPSTREAM"):
            source = api_url(upstream)
            if environment["type"] != "docker" or not private_docker():
                raise ValueError(f"The saved local relay requires DOCKER_HOST={PRIVATE_DOCKER} and unset DOCKER_CONTEXT")
            if source.scheme != "http" or source.hostname not in LOOPBACK or endpoint.scheme != "http" or endpoint.hostname != RELAY_HOST or endpoint.port is None:
                raise ValueError("Invalid private Docker relay endpoints")
            if source.path.rstrip("/") != endpoint.path.rstrip("/"):
                raise ValueError("The transparent relay cannot rewrite URL paths")
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            if env.get(name):
                api_url(env[name])
        validate_claude_preinstall(config)
    else:
        raise ValueError("TB_AGENT must be terminus-2 or claude-code")
    api_url(host_api_base(agent))


def make_config(root):
    agent_name = os.environ.get("TB_AGENT") or "terminus-2"
    if agent_name not in {"terminus-2", "claude-code"}:
        raise ValueError("TB_AGENT must be terminus-2 or claude-code")
    model = Path(os.environ.get("MODEL") or root / "models/envfactory_baseline/EnvFactory-1.7B")
    ckpt = component(model.name)
    experiment = component(os.environ.get("EXPERIMENT") or model.parent.name)
    tag = component(os.environ.get("RUN_TAG") or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S"))
    if agent_name == "claude-code" and not os.environ.get("RUN_TAG"):
        tag = component(f"claude-code-{os.environ.get('TB_CLAUDE_VERSION') or '2.1.80'}_{tag}")
    base = os.environ.get("AGENT_BASE_URL") or f"http://localhost:{os.environ.get('SGLANG_PORT') or '8000'}/v1"
    agent_model = os.environ.get("AGENT_MODEL") or ckpt
    if agent_name == "claude-code":
        agent = claude_agent(base, agent_model)
    else:
        agent = terminus_agent(base, model, agent_model)
    config = {
        "job_name": tag,
        "jobs_dir": str(Path(os.environ.get("TB_JOBS_DIR") or root / "results" / experiment / "terminalbench" / ckpt).resolve()),
        "n_attempts": integer("NUM_TRIALS", 5),
        "n_concurrent_trials": integer("MAX_CONCURRENCY", 1),
        "timeout_multiplier": 1.0,
        "retry": {"max_retries": integer("MAX_RETRIES", 0, minimum=0)},
        "environment": {"type": os.environ.get("TB_ENVIRONMENT") or "docker", "delete": True},
        "agents": [agent],
        "datasets": [{
            "name": DATASET, "ref": VERSION,
            "download_dir": str(Path(os.environ.get("TB_DATA_DIR") or root / "data/eval/terminalbench/4.0.0").resolve()),
            "task_names": task_names("TASK_NAMES"),
            "exclude_task_names": task_names("EXCLUDE_TASK_NAMES"),
            "n_tasks": integer("NUM_TASKS"),
        }],
    }
    if config["environment"]["type"] == "docker":
        config["environment"].update(import_path=DOCKER_IMPORT, kwargs={})
        if device := os.environ.get("TB_GPU_DEVICE"):
            config["environment"]["kwargs"]["gpu_device"] = device
    configure_claude_preinstall(config, root)
    log = Path(os.environ.get("TB_LOG") or root / "logs" / experiment / f"terminalbench_{ckpt}_{tag}.log")
    return config, log


def terminus_agent(base, model, agent_model):
    local = urlsplit(base).hostname in {"localhost", "127.0.0.1", "::1"}
    think = os.environ.get("ENABLE_THINK") or ("true" if local else "auto")
    if think not in {"true", "false", "auto"}:
        raise ValueError("ENABLE_THINK must be true, false or auto")
    native_context = 40960
    if (model / "config.json").is_file():
        native_context = json.loads((model / "config.json").read_text()).get("max_position_embeddings", native_context)
    context = integer("AGENT_CONTEXT_LENGTH", native_context)
    output = integer("MAX_OUTPUT_TOKENS")
    if output is not None and output >= context:
        raise ValueError("MAX_OUTPUT_TOKENS must be smaller than AGENT_CONTEXT_LENGTH")
    kwargs = {
        "api_base": base.rstrip("/"),
        "model_info": {
            "max_input_tokens": context,
            "input_cost_per_token": number("AGENT_INPUT_PRICE", 0) / 1e6,
            "output_cost_per_token": number("AGENT_OUTPUT_PRICE", 0) / 1e6,
            "litellm_provider": "openai",
            "mode": "chat",
        },
        "proactive_summarization_threshold": integer("AGENT_SUMMARIZATION_THRESHOLD", 8000, minimum=0),
        "llm_call_kwargs": {},
    }
    # Leave the output budget to the model server unless explicitly overridden.
    if output is not None:
        kwargs["model_info"]["max_output_tokens"] = output
        kwargs["llm_call_kwargs"]["max_tokens"] = output
    if think != "auto":
        kwargs["llm_call_kwargs"]["extra_body"] = {"chat_template_kwargs": {"enable_thinking": think == "true"}}
        kwargs["temperature"] = number("AGENT_TEMPERATURE", 0.7 if think == "true" else 0)
    elif os.environ.get("AGENT_TEMPERATURE"):
        kwargs["temperature"] = number("AGENT_TEMPERATURE", 0)
    if (steps := integer("MAX_STEPS")) is not None:
        kwargs["max_turns"] = steps
    if not agent_model.startswith("openai/"):
        agent_model = "openai/" + agent_model
    return {"name": "terminus-2", "model_name": agent_model, "kwargs": kwargs}


def credentials(config):
    agent = config["agents"][0]
    local = api_url(host_api_base(agent)).hostname in LOOPBACK
    if agent["name"] == "claude-code":
        key = os.environ.get("AGENT_API_KEY")
        if not local:
            key = key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key and not local:
            raise ValueError("Set AGENT_API_KEY or ANTHROPIC_API_KEY for the remote model API")
        # Freeze the protocol in config.json, including on resume. Do not inherit
        # shell auth/provider/thinking/compaction overrides into official Harbor.
        for name in list(os.environ):
            if name.startswith(("CLAUDE_", "ANTHROPIC_")) or name in {
                "CLAUDECODE", "MAX_THINKING_TOKENS", "DISABLE_COMPACT",
                "DISABLE_AUTO_COMPACT", "AWS_BEARER_TOKEN_BEDROCK",
            }:
                del os.environ[name]
        key = key or "lima-local-sglang-no-auth-credential-placeholder"
        os.environ["LIMA_TB_ANTHROPIC_API_KEY"] = key
    else:
        key = os.environ.get("AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key and not local:
        raise ValueError("Set AGENT_API_KEY or OPENAI_API_KEY for the remote model API")
    if agent["name"] == "terminus-2":
        os.environ["OPENAI_API_KEY"] = key or "placeholder"
    if local:
        for name in ("NO_PROXY", "no_proxy"):
            os.environ[name] = ",".join(filter(None, [os.environ.get(name), "localhost,127.0.0.1,::1"]))
    return key or "placeholder"


def preflight(config, discover_model, key):
    errors = []
    agent = config["agents"][0]
    base = host_api_base(agent)
    try:
        request = Request(base + "/models", headers={"Authorization": "Bearer " + key})
        with urlopen(request, timeout=15) as response:
            models = json.load(response)
        ids = [m["id"] for m in models["data"] if isinstance(m.get("id"), str)]
        prefix = "openai/" if agent["name"] == "terminus-2" else ""
        if discover_model:
            if len(ids) != 1:
                raise ModelConfigurationError("Set AGENT_MODEL explicitly when /models does not list exactly one model")
            agent["model_name"] = prefix + ids[0]
        elif (agent["model_name"].removeprefix(prefix) if prefix else agent["model_name"]) not in ids:
            raise ModelConfigurationError("AGENT_MODEL is absent from /models; check the served model ID")
        if agent["name"] == "claude-code":
            metadata = next(m for m in models["data"] if m["id"] == agent["model_name"])
            limit = metadata.get("max_model_len")
            if limit is not None and int(agent["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]) > int(limit):
                raise ModelConfigurationError("Configured Claude compaction window exceeds the served max_model_len")
            print(f"Claude protocol: version={agent['kwargs']['version']}; compaction window={agent['env']['CLAUDE_CODE_AUTO_COMPACT_WINDOW']}; output cap={agent['env']['CLAUDE_CODE_MAX_OUTPUT_TOKENS']}; served context={limit or 'not reported'}", flush=True)
        print(f"Model API OK: {base}; model={agent['model_name']}", flush=True)
    except ModelConfigurationError as exc:
        errors.append(f"Model check failed: {exc}")
    except Exception as exc:
        # HTTP errors can contain provider details; avoid printing response bodies/keys.
        errors.append(f"Model check failed ({type(exc).__name__}); check endpoint, model ID and credentials")
    if config["environment"]["type"] == "docker":
        try:
            result = subprocess.run(["docker", "info", "--format", "{{json .}}"], capture_output=True, text=True, timeout=15)
            if result.returncode:
                raise ValueError("Docker daemon is inaccessible; check docker info and group membership (newgrp docker or a new login)")
            info = json.loads(result.stdout)
            if agent.get("env", {}).get(CLAUDE_BINARY_ID):
                if info.get("OSType") != "linux" or info.get("Architecture") not in {"x86_64", "amd64"}:
                    raise ValueError("The saved Claude linux-x64 binary requires a Linux x86_64 Docker daemon")
                print(f"Claude preinstall OK: official {CLAUDE_BINARY_VERSION}, SHA256={agent['env'][CLAUDE_BINARY_ID]}; agent-only read-only mount; task-image ABI is not checked", flush=True)
            result = subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=15)
            if result.returncode:
                raise ValueError("Docker Compose is unavailable")
            print(f"Docker OK: server={info['ServerVersion']}; data-root={info['DockerRootDir']}", flush=True)
            if needs_gpu(config):
                device = config["environment"].get("kwargs", {}).get("gpu_device")
                if not device:
                    raise ValueError("GPU tasks are selected; set TB_GPU_DEVICE to a dedicated GPU and MAX_CONCURRENCY=1")
                if "nvidia" not in info.get("Runtimes", {}):
                    raise ValueError("Docker has no NVIDIA runtime; install/configure NVIDIA Container Toolkit before running GPU tasks")
                result = subprocess.run(["nvidia-smi", "-i", device, "--query-gpu=uuid,name,memory.used", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
                if result.returncode:
                    raise ValueError("TB_GPU_DEVICE is not accessible to nvidia-smi")
                print(f"Task GPU (agent and verifier): {result.stdout.strip()}; hardware may differ from task gpu_types", flush=True)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            errors.append(f"Docker check failed: {exc}")
        if errors:
            raise ValueError("\n".join(errors))
        return
    try:
        from modal.config import Config
        modal_config = Config()
        if not modal_config.get("token_id") or not modal_config.get("token_secret"):
            raise ValueError("missing credentials")
        result = subprocess.run([str(HARBOR.parent / "modal"), "app", "list", "--json"], capture_output=True, timeout=30)
        if result.returncode:
            raise ValueError("authentication or connectivity failed")
        print("Modal authentication OK (no sandbox started)", flush=True)
    except Exception as exc:
        errors.append(f"Modal check failed ({type(exc).__name__}); use modal token new or export MODAL_TOKEN_ID and MODAL_TOKEN_SECRET")
    if errors:
        raise ValueError("\n".join(errors))


@contextmanager
def model_relay(config, key):
    agent = config["agents"][0]
    env = agent.get("env") or {}
    upstream = env.get("LIMA_TB_RELAY_UPSTREAM")
    if not upstream:
        yield
        return
    from src.eval.terminalbench.local_api_relay import local_api_relay
    source = api_url(upstream)
    target = api_url(env["ANTHROPIC_BASE_URL"])
    host = "127.0.0.1" if source.hostname == "localhost" else source.hostname
    with local_api_relay(target.hostname, target.port, host, source.port or 80,
                         ("10.255.254.2", RELAY_HOST, "127.0.0.1")):
        # Verify forwarding without generating tokens or starting a container.
        # Bypass the host's outbound proxy for this private, local listener.
        from urllib.request import ProxyHandler, build_opener
        request = Request(env["ANTHROPIC_BASE_URL"] + "/v1/models", headers={"Authorization": "Bearer " + key})
        with build_opener(ProxyHandler({})).open(request, timeout=15) as response:
            ids = [m["id"] for m in json.load(response)["data"]]
        if agent["model_name"] not in ids:
            raise ValueError("Private Docker relay returned a different model list")
        print(f"Temporary model relay OK: {env['ANTHROPIC_BASE_URL']} -> {upstream}", flush=True)
        yield


def execute(command, config, log):
    job = Path(config["jobs_dir"]) / config["job_name"]
    job.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        def report(line):
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()
        report(f"Terminal-Bench 4.0 | {config['agents'][0]['model_name']} | {config['environment']['type']} | attempts={config['n_attempts']} | concurrency={config['n_concurrent_trials']}")
        agent = config["agents"][0]
        report(f"Agent: {agent['name']} | version={agent.get('kwargs', {}).get('version', 'Harbor default')}")
        if agent["name"] == "claude-code":
            report(f"Claude compaction window={agent['env']['CLAUDE_CODE_AUTO_COMPACT_WINDOW']}; max output tokens={agent['env']['CLAUDE_CODE_MAX_OUTPUT_TOKENS']}; native compaction policy; no setup-timeout override.")
            if identity := agent["env"].get(CLAUDE_BINARY_ID):
                report(f"Claude preinstalled official binary SHA256={identity}; read-only agent mount. Harbor's version match skips installation and dependency provisioning; default setup timeout remains 360 seconds.")
            report("Claude cost estimates do not represent local inference charges.")
        report(f"Results: {job}\nLog: {log}")
        report("Task resources/timeouts inherited.")
        if agent["name"] == "terminus-2":
            report("Zero token prices mean cost is unpriced.")
        if config["environment"]["type"] == "docker":
            device = config["environment"].get("kwargs", {}).get("gpu_device")
            report(f"Local Docker task GPU: {device or 'none'} (agent and verifier); task GPU types are not enforced by Docker.")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        previous = signal.signal(signal.SIGTERM, lambda signum, frame: process.send_signal(signum))
        try:
            for line in process.stdout:
                report(line.rstrip("\n"))
            return process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait()
            return 130
        finally:
            signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--resume", type=Path, metavar="JOB_DIR")
    args = parser.parse_args()
    root = Path(os.environ["REPO_ROOT"])
    if args.resume:
        job = args.resume.resolve()
        config = json.loads((job / "config.json").read_text())
        if os.environ.get("TB_AGENT") and os.environ["TB_AGENT"] != config["agents"][0]["name"]:
            raise ValueError("--resume cannot switch agent harness; unset TB_AGENT or start a new job")
        requested = os.environ.get("TB_ENVIRONMENT") or "docker"
        if config["environment"]["type"] != requested:
            raise ValueError(f"Saved job uses {config['environment']['type']}; --resume cannot switch backends. Start a new Docker run, or explicitly set TB_ENVIRONMENT={config['environment']['type']} to resume it")
        config["jobs_dir"], config["job_name"] = str(job.parent), job.name
        log = Path(os.environ.get("TB_LOG") or root / "logs/terminalbench" / f"resume_{job.name}.log")
    else:
        config, log = make_config(root)
    validate(config)
    if config["agents"][0]["name"] == "claude-code":
        ignored = [name for name in ("ENABLE_THINK", "AGENT_TEMPERATURE", "AGENT_SUMMARIZATION_THRESHOLD", "AGENT_INPUT_PRICE", "AGENT_OUTPUT_PRICE") if os.environ.get(name)]
        if ignored:
            print("Claude Code uses its native protocol; these Terminus-only settings do not apply: " + ", ".join(ignored), file=sys.stderr)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    key = credentials(config)
    preflight(config, discover_model=not args.resume and not os.environ.get("AGENT_MODEL"), key=key)
    with model_relay(config, key):
        if args.check:
            print("Preflight passed without model generation or task containers. Installation, task execution and native compaction still need a smoke run.")
            return 0
        if args.resume:
            return execute([str(HARBOR), "job", "resume", "--job-path", str(job)], config, log)
        job = Path(config["jobs_dir"]) / config["job_name"]
        if job.exists():
            raise ValueError(f"Job already exists: {job}; choose RUN_TAG or use --resume")
        # Harbor persists this non-secret input as the job's config.json.
        with tempfile.TemporaryDirectory(prefix="lima-terminalbench-") as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config, indent=2))
            return execute([str(HARBOR), "run", "--config", str(path)], config, log)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
