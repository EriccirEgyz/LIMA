"""LIMA's host-side Terminus 2 launcher; task execution stays in Modal."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from harbor.agents.terminus_2.terminus_2 import Terminus2Options
from harbor.models.job.config import JobConfig

DATASET = "terminal-bench/terminal-bench"
VERSION = "4.0.0"
HARBOR = Path(sys.executable).parent / "harbor"


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


def validate(config):
    JobConfig.model_validate(config)
    Terminus2Options.model_validate(config["agents"][0].get("kwargs", {}))
    if config["environment"]["type"] != "modal":
        raise ValueError("This launcher requires a Modal job")
    agent = config["agents"][0]
    if len(config["agents"]) != 1 or agent["name"] != "terminus-2":
        raise ValueError("This launcher supports one Terminus 2 agent")
    if not agent["model_name"].startswith("openai/"):
        raise ValueError("Use an OpenAI-compatible model with this launcher")
    base = agent["kwargs"]["api_base"]
    url = urlsplit(base)
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise ValueError("AGENT_BASE_URL must be an HTTP(S) API base URL")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("Keep credentials out of AGENT_BASE_URL; use AGENT_API_KEY")


def make_config(root):
    model = Path(os.environ.get("MODEL") or root / "models/envfactory_baseline/EnvFactory-1.7B")
    ckpt = component(model.name)
    experiment = component(os.environ.get("EXPERIMENT") or model.parent.name)
    tag = component(os.environ.get("RUN_TAG") or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S"))
    base = os.environ.get("AGENT_BASE_URL") or f"http://localhost:{os.environ.get('SGLANG_PORT') or '8000'}/v1"
    local = urlsplit(base).hostname in {"localhost", "127.0.0.1", "::1"}
    think = os.environ.get("ENABLE_THINK") or ("true" if local else "auto")
    if think not in {"true", "false", "auto"}:
        raise ValueError("ENABLE_THINK must be true, false or auto")
    native_context = 40960
    if (model / "config.json").is_file():
        native_context = json.loads((model / "config.json").read_text()).get("max_position_embeddings", native_context)
    context = integer("AGENT_CONTEXT_LENGTH", native_context)
    output = integer("MAX_OUTPUT_TOKENS", 8192)
    if output >= context:
        raise ValueError("MAX_OUTPUT_TOKENS must be smaller than AGENT_CONTEXT_LENGTH")
    kwargs = {
        "api_base": base.rstrip("/"),
        "model_info": {
            "max_input_tokens": context,
            "max_output_tokens": output,
            "input_cost_per_token": number("AGENT_INPUT_PRICE", 0) / 1e6,
            "output_cost_per_token": number("AGENT_OUTPUT_PRICE", 0) / 1e6,
            "litellm_provider": "openai",
            "mode": "chat",
        },
        "proactive_summarization_threshold": output,
        "llm_call_kwargs": {"max_tokens": output},
    }
    if think != "auto":
        kwargs["llm_call_kwargs"]["extra_body"] = {"chat_template_kwargs": {"enable_thinking": think == "true"}}
        kwargs["temperature"] = number("AGENT_TEMPERATURE", 0.7 if think == "true" else 0)
    elif os.environ.get("AGENT_TEMPERATURE"):
        kwargs["temperature"] = number("AGENT_TEMPERATURE", 0)
    if (steps := integer("MAX_STEPS")) is not None:
        kwargs["max_turns"] = steps
    agent_model = os.environ.get("AGENT_MODEL") or ckpt
    if not agent_model.startswith("openai/"):
        agent_model = "openai/" + agent_model
    config = {
        "job_name": tag,
        "jobs_dir": str(Path(os.environ.get("TB_JOBS_DIR") or root / "results" / experiment / "terminalbench" / ckpt).resolve()),
        "n_attempts": integer("NUM_TRIALS", 1),
        "n_concurrent_trials": integer("MAX_CONCURRENCY", 2),
        "timeout_multiplier": 1.0,
        "retry": {"max_retries": integer("MAX_RETRIES", 0, minimum=0)},
        "environment": {"type": "modal", "delete": True},
        "agents": [{"name": "terminus-2", "model_name": agent_model, "kwargs": kwargs}],
        "datasets": [{
            "name": DATASET, "ref": VERSION,
            "download_dir": str(Path(os.environ.get("TB_DATA_DIR") or root / "data/eval/terminalbench/4.0.0").resolve()),
            "task_names": task_names("TASK_NAMES"),
            "exclude_task_names": task_names("EXCLUDE_TASK_NAMES"),
            "n_tasks": integer("NUM_TASKS"),
        }],
    }
    log = Path(os.environ.get("TB_LOG") or root / "logs" / experiment / f"terminalbench_{ckpt}_{tag}.log")
    return config, log


def credentials(config):
    host = urlsplit(config["agents"][0]["kwargs"]["api_base"]).hostname
    local = host in {"localhost", "127.0.0.1", "::1"}
    key = os.environ.get("AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key and not local:
        raise ValueError("Set AGENT_API_KEY or OPENAI_API_KEY for the remote model API")
    os.environ["OPENAI_API_KEY"] = key or "placeholder"
    if local:
        for name in ("NO_PROXY", "no_proxy"):
            os.environ[name] = ",".join(filter(None, [os.environ.get(name), "localhost,127.0.0.1,::1"]))


def preflight(config, discover_model):
    errors = []
    agent = config["agents"][0]
    base = agent["kwargs"]["api_base"]
    try:
        request = Request(base + "/models", headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
        with urlopen(request, timeout=15) as response:
            models = json.load(response)
        ids = [m["id"] for m in models["data"] if isinstance(m.get("id"), str)]
        if discover_model:
            if len(ids) != 1:
                raise ValueError("Set AGENT_MODEL explicitly when /models does not list exactly one model")
            agent["model_name"] = "openai/" + ids[0]
        elif agent["model_name"].removeprefix("openai/") not in ids:
            raise ValueError("AGENT_MODEL is absent from /models; check the served model ID")
        print(f"Model API OK: {base}; model={agent['model_name']}", flush=True)
    except Exception as exc:
        # HTTP errors can contain provider details; avoid printing response bodies/keys.
        errors.append(f"Model check failed ({type(exc).__name__}); check endpoint, model ID and credentials")
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


def execute(command, config, log):
    job = Path(config["jobs_dir"]) / config["job_name"]
    job.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        def report(line):
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()
        report(f"Terminal-Bench 4.0 | {config['agents'][0]['model_name']} | Modal | attempts={config['n_attempts']} | concurrency={config['n_concurrent_trials']}")
        report(f"Results: {job}\nLog: {log}")
        report("Task resources/timeouts inherited. Zero token prices mean cost is unpriced.")
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
        config["jobs_dir"], config["job_name"] = str(job.parent), job.name
        log = Path(os.environ.get("TB_LOG") or root / "logs/terminalbench" / f"resume_{job.name}.log")
    else:
        config, log = make_config(root)
    validate(config)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    credentials(config)
    preflight(config, discover_model=not args.resume and not os.environ.get("AGENT_MODEL"))
    if args.check:
        print("Preflight passed. This checks connectivity/authentication; task execution still needs a smoke run.")
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
