#!/usr/bin/env bash
# Local OpenAI-compatible model + upstream MCPMark Verified orchestration/verifiers.
set -euo pipefail
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"
export MCPMARK_DIR="$EVAL_DIR/mcpmark"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash src/eval/run_mcpmark.sh [--dry-run | --list-tasks | --check | --check-services]
Default: filesystem, standard (MCPMark Verified), all tasks, k=4, serial execution.
--dry-run        Print settings; no network, dependency imports or output files.
--list-tasks     List exact selected category/task IDs; no model/task execution.
--check-services Check selected services, Docker images, credentials and MCP/browser
                 startup without creating accounts, databases or task containers.
--check          Also query /models and verify one structured model tool call.
No flag          Check prerequisites, then run official agent and verifiers.

Model (start SGLang separately, with --tool-call-parser hermes):
 MODEL              Default models/Qwen3-8B; checkpoint/experiment naming.
 SGLANG_PORT        Default 8000. serve_h200-236.log previously used 8002.
 AGENT_BASE_URL     Default http://127.0.0.1:$SGLANG_PORT/v1.
 AGENT_MODEL        Exact request ID, otherwise discover a single ID via /models.
                    Mismatch with MODEL basename requires explicit AGENT_MODEL.
 AGENT_API_KEY      Fallback OPENAI_API_KEY; local endpoints allow placeholder.
 ENABLE_THINK       true/false/auto; default auto (use the model/server default).
 AGENT_TEMPERATURE  Optional override; unset means omit temperature from requests.
 MAX_OUTPUT_TOKENS  Optional positive override including reasoning; unset means
                    omit the output limit and use the endpoint's default behavior.
                    Model-card recommendations are not inferred automatically.
 AGENT_CONTEXT_LENGTH  Optional served limit; checked against /models when known.

Evaluation:
 MCPS               filesystem (default), a comma-separated list, or all.
                    all = filesystem,postgres,playwright,playwright_webarena,
                    github,notion (127 standard tasks at the pinned revision).
                    insforge and supabase may be explicitly selected as extensions.
 TASK_SUITE         standard (default) or easy; MCPS=all selects 127 or 50 tasks.
 TASKS              all (default), exact category, or category/task.
 <SERVICE>_TASKS    Per-service override, e.g. POSTGRES_TASKS=lego.
 NUM_TRIALS         Default 4 (upstream default); set 1 for a smoke run.
 TASK_TIMEOUT       Default 3600 seconds; MAX_STEPS default 100 (upstream defaults).
 COMPACTION_TOKEN   Default 999999999, disabled (upstream default).
 EXPERIMENT/RUN_TAG  Result naming; reuse both to resume exactly the same config.
 MCPMARK_RUN_ROOT / MCPMARK_LOG  Optional result/log path overrides.
 MCPMARK_ENV_FILE   Optional dotenv file (default mcpmark/.mcp_env).

Docker/services:
 GitHub: GITHUB_TOKENS, GITHUB_EVAL_ORG and the pinned Docker MCP image.
 Notion: SOURCE_NOTION_API_KEY, EVAL_NOTION_API_KEY, EVAL_PARENT_PAGE_TITLE,
         Source/Eval Hub grants and mcpmark/notion_state.json (setup --notion-login).
         NOTION_PROXY selects the browser proxy, as in setup --notion-login.
         Empty NOTION_PROXY uses a direct connection; otherwise HTTPS_PROXY /
         HTTP_PROXY is used as fallback when NOTION_PROXY is not set.
 Postgres: export POSTGRES_PASSWORD, POSTGRES_HOST (default localhost),
           POSTGRES_PORT (default 15432), POSTGRES_USERNAME (default postgres).
           Dedicated disposable evaluation DB only. Setup --start-postgres,
           then --init-postgres to restore all five official template databases.
 WebArena: setup --prepare-docker webarena; each task recreates its own container.
           Container names are prefixed lima-mcpmark-<uid>; default ports are
           17770 (shopping), 17780 (admin), 19999 (reddit). Optional overrides:
           WEBARENA_SHOPPING_PORT / WEBARENA_SHOPPING_ADMIN_PORT / WEBARENA_REDDIT_PORT.
 See setup_mcpmark.sh --help for browser/image installation.

Examples:
 SGLANG_PORT=8002 bash src/eval/run_mcpmark.sh --check
 SGLANG_PORT=8002 TASKS=file_property/size_classification bash src/eval/run_mcpmark.sh
 MCPS=all SGLANG_PORT=8002 NUM_TRIALS=4 bash src/eval/run_mcpmark.sh --check
 MCPS=all SGLANG_PORT=8002 NUM_TRIALS=4 bash src/eval/run_mcpmark.sh
Results: results/<experiment>/mcpmark/<checkpoint>/<run_tag>/<service>/...
Logs: logs/<experiment>/mcpmark_<checkpoint>_<run_tag>.log
Resume refuses changed config/dependency locks. Scores use upstream verifiers;
unset generation settings are recorded as null/auto (endpoint defaults), not as
inferred numeric values. Explicit overrides must accompany reported results.
The --check connectivity probe has a separate 256-token budget, outside evaluation.
HELP
    exit 0
fi
PYTHON="$MCPMARK_DIR/.venv/bin/python"
if [[ "${1:-}" == --dry-run || "${1:-}" == --list-tasks ]]; then
    PYTHON=python3
else
    [[ -x "$PYTHON" ]] || { echo 'ERROR: Run setup_mcpmark.sh first.' >&2; exit 1; }
    # shellcheck source=cache_env.sh
    source "$EVAL_DIR/cache_env.sh"
    export npm_config_cache="${npm_config_cache:-$CACHE_ROOT/npm}"
    export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CACHE_ROOT/ms-playwright}"
    export TMPDIR="${MCPMARK_TMPDIR:-$CACHE_ROOT/mcpmark-tmp}"
    mkdir -p "$TMPDIR"
fi
export PATH="$MCPMARK_DIR/.venv/bin:$MCPMARK_DIR/.lima-bin:$MCPMARK_DIR/.lima-node/node_modules/.bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LITELLM_LOCAL_MODEL_COST_MAP=True
exec "$PYTHON" - "$@" <<'PY'
import argparse
import asyncio
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit, quote, unquote
from urllib.request import Request, urlopen

PIN = 'cd45b7f57923b9b3985467f5139927575f83141c'
STANDARD = ['filesystem', 'postgres', 'playwright', 'playwright_webarena', 'github', 'notion']
ROOT = Path(os.environ['REPO_ROOT'])
UPSTREAM = Path(os.environ['MCPMARK_DIR'])
PG_IMAGE = 'pgvector/pgvector:0.8.0-pg17-bookworm'
GH_IMAGE = 'ghcr.io/github/github-mcp-server:v0.15.0'
WEB = {'shopping': ('shopping_final_0712', 17770),
       'shopping_admin': ('shopping_admin_final_0719', 17780),
       'reddit': ('postmill-populated-exposed-withimg', 19999)}


def positive(name, default=None):
    raw = os.environ.get(name) or default
    if raw is None:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def component(value):
    if value in {'.', '..'} or not re.fullmatch(r'[\w.-]+', value):
        raise ValueError(f'Invalid output name: {value!r}')
    return value


def process(args):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=30).stdout.strip()


def tasks_for(service, suite, selector):
    items = []
    for path in sorted((UPSTREAM / 'tasks' / service / suite).glob('*/*/meta.json')):
        name = '/'.join(path.parts[-3:-1])
        if selector == 'all' or selector == name or selector == path.parent.parent.name:
            items.append(name)
    if not items:
        raise ValueError(f'{service}/{suite}: no tasks match {selector!r}')
    return items


def config():
    model = Path(os.environ.get('MODEL') or ROOT / 'models/Qwen3-8B')
    ckpt = component(model.name)
    experiment = component(os.environ.get('EXPERIMENT') or model.parent.name or 'models')
    tag = component(os.environ.get('RUN_TAG') or datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    services = os.environ.get('MCPS') or 'filesystem'
    all_services = services == 'all'
    services = STANDARD.copy() if all_services else [s.strip() for s in services.split(',')]
    if len(set(services)) != len(services) or any(s not in STANDARD + ['insforge', 'supabase'] for s in services):
        raise ValueError('MCPS contains unknown or duplicate services')
    suite = os.environ.get('TASK_SUITE') or 'standard'
    if suite not in {'standard', 'easy'}:
        raise ValueError('TASK_SUITE must be standard or easy')
    if all_services and suite == 'easy':
        services.remove('playwright')  # This service has no easy split; WebArena supplies 10.
    thinking = os.environ.get('ENABLE_THINK') or 'auto'
    if thinking not in {'true', 'false', 'auto'}:
        raise ValueError('ENABLE_THINK must be true, false or auto')
    temperature = float(os.environ['AGENT_TEMPERATURE']) if os.environ.get('AGENT_TEMPERATURE') else None
    if temperature is not None and (not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError('AGENT_TEMPERATURE must be finite and between 0 and 2')
    base = (os.environ.get('AGENT_BASE_URL') or f"http://127.0.0.1:{os.environ.get('SGLANG_PORT') or '8000'}/v1").rstrip('/')
    url = urlsplit(base)
    if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('AGENT_BASE_URL must be an HTTP(S) base URL without embedded credentials/query')
    selectors = {s: os.environ.get(s.upper() + '_TASKS') or os.environ.get('TASKS') or 'all' for s in services}
    return dict(commit=PIN, model_weights=str(model), checkpoint=ckpt, experiment=experiment,
                run_tag=tag, services=services, task_suite=suite, selectors=selectors,
                api_base=base, api_model=os.environ.get('AGENT_MODEL'), enable_think=thinking,
                temperature=temperature,
                max_output_tokens=positive('MAX_OUTPUT_TOKENS'),
                generation_policy='endpoint defaults unless explicitly overridden',
                trials=positive('NUM_TRIALS', 4), timeout=positive('TASK_TIMEOUT', 3600),
                max_steps=positive('MAX_STEPS', 100), compaction_token=positive('COMPACTION_TOKEN', 999999999))


def notion_browser_proxy():
    value = os.environ.get('NOTION_PROXY')
    if value is None:
        value = next((os.environ[n] for n in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy')
                      if os.environ.get(n)), '')
    if not value:
        return None
    parsed = urlsplit(value)
    if (parsed.scheme not in {'http', 'https', 'socks5'} or not parsed.hostname
            or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
        raise ValueError('NOTION_PROXY must be an HTTP(S) or SOCKS5 proxy URL.')
    proxy = {'server': parsed.scheme + '://' + parsed.netloc.rsplit('@', 1)[-1]}
    if parsed.username is not None:
        proxy.update(username=unquote(parsed.username), password=unquote(parsed.password or ''))
    return proxy


def model_request_kwargs(c, kwargs):
    kwargs = dict(kwargs)
    # The pinned agent always supplies max_tokens=32768 and temperature=1.0.
    # Remove these too: merely removing this wrapper's overrides would still
    # impose universal values instead of allowing each endpoint's defaults.
    for name in ('enforcer_mode', 'think_mode', 'max_tokens', 'max_completion_tokens', 'temperature'):
        kwargs.pop(name, None)
    if c['max_output_tokens'] is not None:
        kwargs['max_tokens'] = c['max_output_tokens']
    if c['temperature'] is not None:
        kwargs['temperature'] = c['temperature']
    if c['enable_think'] != 'auto':
        extra = dict(kwargs.get('extra_body') or {})
        template = dict(extra.get('chat_template_kwargs') or {})
        template['enable_thinking'] = c['enable_think'] == 'true'
        extra['chat_template_kwargs'] = template
        kwargs['extra_body'] = extra
    return kwargs


def install_adapters(c):
    import litellm
    from src.agents import AGENT_REGISTRY
    from src.agents.mcpmark_agent import MCPMarkAgent
    from src.agents.mcp import MCPStdioServer
    from src.model_config import ModelConfig
    ModelConfig.MODEL_CONFIGS[c['checkpoint']] = {
        'api_key_var': 'LIMA_MCPMARK_KEY', 'base_url_var': 'LIMA_MCPMARK_BASE',
        'litellm_input_model_name': 'openai/' + c['api_model'], 'provider': 'openai'}
    original = litellm.acompletion

    async def completion(**kwargs):
        return await original(**model_request_kwargs(c, kwargs))

    litellm.acompletion = completion

    class LocalAgent(MCPMarkAgent):
        MAX_TURNS = c['max_steps']

        def _create_stdio_server(self):
            server = super()._create_stdio_server()
            params = server.params
            bins = UPSTREAM / '.lima-node/node_modules/.bin'
            if self.mcp_service == 'filesystem':
                return MCPStdioServer(str(bins / 'mcp-server-filesystem'), [str(self.service_config['test_directory'])])
            if self.mcp_service in {'playwright', 'playwright_webarena'}:
                params.command = str(bins / 'playwright-mcp')
                params.args = params.args[2:]
            elif self.mcp_service == 'notion':
                params.command = str(bins / 'notion-mcp-server')
                params.args = []
            elif self.mcp_service == 'postgres':
                params.command = str(UPSTREAM / '.venv/bin/postgres-mcp')
                params.args = ['--access-mode=unrestricted']
                cfg=self.service_config
                database=cfg.get('current_database') or cfg.get('database')
                params.env['DATABASE_URI']=f"postgresql://{quote(cfg['username'],safe='')}:{quote(cfg['password'],safe='')}@{cfg.get('host','localhost')}:{cfg.get('port',5432)}/{quote(database,safe='')}"
            elif self.mcp_service == 'github':
                params.args[1:1] = ['--network', 'host']
            return server

    AGENT_REGISTRY['mcpmark'] = LocalAgent
    # Isolate shared-host container names. Keep upstream's Magento configuration
    # branches working by calling them after the category-specific setup.
    from src.mcp_services.playwright_webarena.playwright_state_manager import PlaywrightStateManager
    class LocalWebArena(PlaywrightStateManager):
        CATEGORY_CONFIGS = {k: {**v, 'container_name': f'lima-mcpmark-{os.getuid()}-{k}',
                               'host_port': positive('WEBARENA_' + k.upper() + '_PORT', WEB[k][1])}
                            for k, v in PlaywrightStateManager.CATEGORY_CONFIGS.items()}

        def __init__(self, **kwargs):
            super().__init__(docker_container_name=f'lima-mcpmark-{os.getuid()}-shopping_admin', **kwargs)

        def _run_cmd(self, args, **kwargs):
            if args[:2] == ['docker', 'run']:
                args = args[:2] + ['--network', f'lima-mcpmark-{os.getuid()}',
                        '--label', f'lima.mcpmark.owner={os.getuid()}'] + args[2:]
                index = args.index('-p') + 1
                args[index] = '127.0.0.1:' + args[index]
            return super()._run_cmd(args, **kwargs)

        def _wait_until_ready(self):
            if self.config.container_name.endswith('-shopping'):
                self._configure_shopping_post_start()
            elif self.config.container_name.endswith('-shopping_admin'):
                self._configure_shopping_admin_post_start()
            return super()._wait_until_ready()

        def _stop_and_remove_container(self, name):
            if not name.startswith(f'lima-mcpmark-{os.getuid()}-'):
                raise ValueError('Refusing to remove a container outside the MCPMark namespace')
            super()._stop_and_remove_container(name)

    from src.factory import ServiceRegistry
    ServiceRegistry.get_components('playwright_webarena').state_manager_class = LocalWebArena
    from src.mcp_services.notion.notion_state_manager import NotionStateManager
    class LocalNotion(NotionStateManager):
        def _ensure_browser(self):
            proxy = notion_browser_proxy()
            if not proxy:
                return super()._ensure_browser()
            from unittest.mock import patch
            from playwright.sync_api import BrowserType
            original_launch = BrowserType.launch

            def launch(browser_type, **kwargs):
                kwargs.setdefault('proxy', proxy)
                return original_launch(browser_type, **kwargs)

            # Retain upstream browser/session/duplication logic. The patch is
            # scoped to this synchronous launch and never changes upstream files.
            with patch.object(BrowserType, 'launch', launch):
                return super()._ensure_browser()

    ServiceRegistry.get_components('notion').state_manager_class = LocalNotion
    return LocalAgent


def require_image(image):
    try:
        process(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'])
    except subprocess.CalledProcessError:
        raise ValueError(f'Missing Docker image {image}; run setup_mcpmark.sh --prepare-docker for this service') from None


def check_services(c):
    from src.config.config_schema import ConfigRegistry
    errors = []
    docker_ok = False
    if any(s in c['services'] for s in ['github', 'postgres', 'playwright_webarena']):
        try:
            process(['docker', 'info', '--format', '{{.DockerRootDir}}'])
            docker_ok = True
        except Exception:
            errors.append('Docker inaccessible; use a terminal/session with docker group membership.')
    for service in c['services']:
        try:
            required={'github': ['GITHUB_TOKENS','GITHUB_EVAL_ORG'],
                      'notion': ['SOURCE_NOTION_API_KEY','EVAL_NOTION_API_KEY','EVAL_PARENT_PAGE_TITLE'],
                      'postgres': ['POSTGRES_PASSWORD']}.get(service,[])
            missing=[name for name in required if not os.environ.get(name,'').strip()]
            if missing:
                raise ValueError('Fill '+', '.join(missing)+' in mcpmark/.mcp_env')
            ConfigRegistry.get_config(service)
            if service == 'filesystem':
                async def handshake():
                    from src.agents.mcp import MCPStdioServer
                    with tempfile.TemporaryDirectory() as folder:
                        async with MCPStdioServer(str(UPSTREAM / '.lima-node/node_modules/.bin/mcp-server-filesystem'), [folder]) as server:
                            tools = await server.list_tools()
                            if not tools:
                                raise ValueError('Filesystem MCP returned no tools')
                            print(f'filesystem MCP: {len(tools)} tools')
                asyncio.run(handshake())
            if service == 'github':
                if not os.environ.get('GITHUB_EVAL_ORG'):
                    raise ValueError('Set GITHUB_EVAL_ORG to your dedicated evaluation user/org')
                if docker_ok:
                    require_image(GH_IMAGE)
            if service == 'postgres':
                if not os.environ.get('POSTGRES_PASSWORD'):
                    raise ValueError('Export POSTGRES_PASSWORD for the dedicated evaluation DB')
                if docker_ok:
                    require_image(PG_IMAGE)
                import psycopg2
                with psycopg2.connect(host=os.environ.get('POSTGRES_HOST', 'localhost'),
                        port=os.environ.get('POSTGRES_PORT', '15432'), user=os.environ.get('POSTGRES_USERNAME', 'postgres'),
                        password=os.environ['POSTGRES_PASSWORD'], dbname='postgres', connect_timeout=5) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute('SHOW server_version_num')
                        if int(cursor.fetchone()[0]) // 10000 != 17:
                            raise ValueError('PostgreSQL 17 is required by the pinned configuration')
                        cursor.execute("SELECT default_version FROM pg_available_extensions WHERE name='vector'")
                        vector=cursor.fetchone()
                        if not vector or vector[0]!='0.8.0':
                            raise ValueError('pgvector 0.8.0 is required')
                        cursor.execute("SELECT datname FROM pg_database WHERE datname IN ('employees','chinook','dvdrental','sports','lego')")
                        missing=set(['employees','chinook','dvdrental','sports','lego'])-{row[0] for row in cursor.fetchall()}
                        if missing:
                            raise ValueError('Missing template databases '+', '.join(sorted(missing))+'; setup_mcpmark.sh --init-postgres')
            if service == 'notion' and not (UPSTREAM / 'notion_state.json').is_file():
                raise ValueError('Notion browser login missing: setup_mcpmark.sh --notion-login')
            if service == 'playwright_webarena' and docker_ok:
                try:
                    process(['docker', 'network', 'inspect', f'lima-mcpmark-{os.getuid()}', '--format', '{{.Id}}'])
                except subprocess.CalledProcessError:
                    raise ValueError('WebArena Docker network missing; run setup_mcpmark.sh --prepare-docker webarena') from None
                categories = {x.split('/')[0] for x in c['selected_tasks'][service]}
                for category in sorted(categories):
                    require_image(WEB[category][0])
                    port = positive('WEBARENA_' + category.upper() + '_PORT', WEB[category][1])
                    with socket.socket() as sock:
                        sock.bind(('127.0.0.1', port))
            if service in {'notion', 'playwright', 'playwright_webarena'}:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    options = {'headless': True}
                    if service == 'notion' and notion_browser_proxy():
                        options['proxy'] = notion_browser_proxy()
                    browser = getattr(p, os.environ.get('PLAYWRIGHT_BROWSER', 'chromium')).launch(**options)
                    browser.close()
                if service != 'notion':
                    script = "const p=require(process.argv[1]); (async()=>{const b=await p.chromium.launch({headless:true,args:['--no-sandbox']});await b.close()})().catch(e=>{console.error(e.message);process.exit(1)});"
                    process(['node', '-e', script, str(UPSTREAM / '.lima-node/node_modules/playwright')])
            print(f'{service}: prerequisites OK')
        except Exception as exc:
            detail = str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else type(exc).__name__
            errors.append(f'{service}: {detail}; see setup_mcpmark.sh --help and upstream docs/mcp/{service}.md')
    return errors


def check_model(c, tool_call=False):
    key = os.environ['LIMA_MCPMARK_KEY']
    def request(path, body=None):
        req = Request(c['api_base'] + path, headers={'Authorization': 'Bearer ' + key,
            'Content-Type': 'application/json'}, data=json.dumps(body).encode() if body else None)
        with urlopen(req, timeout=60) as response:
            return json.load(response)
    models = request('/models')['data']
    ids = [m['id'] for m in models]
    if not c['api_model']:
        if len(ids) != 1:
            raise ValueError('Set AGENT_MODEL explicitly; /models must list one model for discovery')
        c['api_model'] = ids[0]
        if Path(c['api_model']).name != c['checkpoint']:
            raise ValueError(f"Served model {c['api_model']!r} differs from MODEL label {c['checkpoint']!r}; set MODEL and/or AGENT_MODEL explicitly")
    elif c['api_model'] not in ids:
        raise ValueError('AGENT_MODEL is absent from /models')
    model = next(m for m in models if m['id'] == c['api_model'])
    context = int(os.environ.get('AGENT_CONTEXT_LENGTH') or model.get('max_model_len') or 0)
    c['context_length'] = context or None
    if context and c['max_output_tokens'] is not None and c['max_output_tokens'] >= context:
        raise ValueError('MAX_OUTPUT_TOKENS must be smaller than the served context limit')
    if tool_call:
        body = {'model': c['api_model'], 'messages': [{'role': 'user', 'content': 'Call mcpmark_ping with value "ok".'}],
                'tools': [{'type': 'function', 'function': {'name': 'mcpmark_ping', 'description': 'Connectivity test.',
                           'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}, 'required': ['value']}}}],
                'tool_choice': 'required', 'temperature': 0, 'max_tokens': 256}
        if c['enable_think'] != 'auto':
            body['chat_template_kwargs'] = {'enable_thinking': False}
        msg = request('/chat/completions', body)['choices'][0]['message']
        calls = msg.get('tool_calls') or []
        if not calls or calls[0]['function']['name'] != 'mcpmark_ping':
            raise ValueError('Structured tool_calls missing; check SGLang --tool-call-parser hermes')
        if json.loads(calls[0]['function']['arguments']).get('value') != 'ok':
            raise ValueError('Invalid tool-call arguments')
    print(f"Model API OK: {c['api_base']} model={c['api_model']}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    mode = parser.add_mutually_exclusive_group()
    for option in ['dry-run', 'list-tasks', 'check', 'check-services']:
        mode.add_argument('--' + option, action='store_true')
    args = parser.parse_args()
    if not args.dry_run and not args.list_tasks:
        if not UPSTREAM.is_dir():
            raise ValueError('Run setup_mcpmark.sh first')
        os.chdir(UPSTREAM)
        sys.path.insert(0, str(UPSTREAM))
        from dotenv import load_dotenv
        env_file = Path(os.environ.get('MCPMARK_ENV_FILE') or UPSTREAM / '.mcp_env').resolve()
        if os.environ.get('MCPMARK_ENV_FILE') and not env_file.is_file():
            raise ValueError('MCPMARK_ENV_FILE does not exist')
        if env_file.exists() and env_file.stat().st_mode & 0o077:
            raise ValueError('Credential file permissions must be 600 (owner only)')
        load_dotenv(env_file, override=False)
    c = config()
    c['selected_tasks'] = {s: tasks_for(s, c['task_suite'], c['selectors'][s]) for s in c['services']}
    if process(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD']) != PIN:
        raise ValueError('MCPMark source revision differs from the setup pin')
    if args.dry_run:
        print(json.dumps(c, ensure_ascii=False, indent=2))
        return 0
    if args.list_tasks:
        for service, tasks in c['selected_tasks'].items():
            print(f'{service}: {len(tasks)} tasks')
            print('\n'.join(tasks))
        return 0
    os.environ.setdefault('PLAYWRIGHT_BROWSER', 'chromium')
    os.environ.setdefault('POSTGRES_PORT', '15432')
    os.environ.setdefault('FILESYSTEM_TEST_ROOT', str(UPSTREAM / 'test_environments'))
    local = urlsplit(c['api_base']).hostname in {'localhost', '127.0.0.1', '::1'}
    key = os.environ.get('AGENT_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not key and not local and not args.check_services:
        raise ValueError('Set AGENT_API_KEY for a remote endpoint')
    os.environ['LIMA_MCPMARK_KEY'] = key or 'placeholder'
    os.environ['LIMA_MCPMARK_BASE'] = c['api_base']
    for name in ('NO_PROXY', 'no_proxy'):
        os.environ[name] = ','.join(filter(None, [os.environ.get(name), 'localhost,127.0.0.1,::1']))
    errors = check_services(c)
    if not args.check_services:
        try:
            check_model(c, tool_call=True)
        except Exception as exc:
            errors.append('Model API: ' + (str(exc) if isinstance(exc, ValueError) else type(exc).__name__))
    if errors:
        for error in errors:
            print('ERROR: ' + error, file=sys.stderr)
        return 1
    if args.check or args.check_services:
        print('Preflight complete; no evaluation tasks executed.')
        return 0
    install_adapters(c)
    for name in ('.lima-requirements.txt', '.lima-node/package-lock.json'):
        path = UPSTREAM / name
        c[name + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = Path(os.environ.get('MCPMARK_RUN_ROOT') or ROOT / 'results' / c['experiment'] / 'mcpmark' / c['checkpoint'] / c['run_tag']).resolve()
    log = Path(os.environ.get('MCPMARK_LOG') or ROOT / 'logs' / c['experiment'] / f"mcpmark_{c['checkpoint']}_{c['run_tag']}.log").resolve()
    output.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        web_lock = None
        if 'playwright_webarena' in c['services']:
            web_lock = (UPSTREAM / '.lima-webarena.lock').open('w')
            fcntl.flock(web_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = output / 'config.json'
        if saved.exists() and json.loads(saved.read_text()) != c:
            raise ValueError('Resume config/dependency mismatch; choose a new RUN_TAG')
        saved.write_text(json.dumps(c, ensure_ascii=False, indent=2) + '\n')
        with log.open('a') as log_file:
            class Tee:
                def __init__(self, stream): self.stream = stream
                def write(self, text):
                    self.stream.write(text)
                    if not log_file.closed: log_file.write(text)
                    return len(text)
                def flush(self):
                    self.stream.flush()
                    if not log_file.closed: log_file.flush()
                def isatty(self): return False
            with redirect_stdout(Tee(sys.stdout)), redirect_stderr(Tee(sys.stderr)):
                # Configure upstream logging after redirecting the stream.
                import logging
                for logger in logging.Logger.manager.loggerDict.values():
                    if isinstance(logger, logging.Logger):
                        for handler in logger.handlers:
                            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                                handler.setStream(sys.stderr)
                from src.evaluator import MCPEvaluator
                print(json.dumps(c, ensure_ascii=False, indent=2))
                from src.errors import is_retryable_error
                failed = False
                outcomes = {}
                for service in c['services']:
                    # Keep ordinary Playwright and WebArena outputs separate: upstream
                    # otherwise puts both in the same playwright output directory.
                    for run in range(1, c['trials'] + 1):
                        evaluator = MCPEvaluator(mcp_service=service, model=c['checkpoint'], timeout=c['timeout'],
                            exp_name=f'run-{run}', output_dir=output / service, task_suite=c['task_suite'],
                            compaction_token=c['compaction_token'])
                        report = evaluator.run_evaluation(c['selectors'][service])
                        print(f'{service} run-{run}: {report.successful_tasks}/{report.total_tasks}')
                        if report.total_tasks != len(c['selected_tasks'][service]):
                            raise ValueError(f'{service}: evaluator task count differs from the selected manifest')
                        for result in report.task_results:
                            key = f'{service}/{result.category_id}/{result.task_id}'
                            outcomes.setdefault(key, []).append(bool(result.success))
                        failed |= any(is_retryable_error(r.error_message) for r in report.task_results)
                n = len(outcomes)
                if any(len(row) != c['trials'] for row in outcomes.values()):
                    raise ValueError('Incomplete task/trial matrix; cannot aggregate')
                rates = [sum(row[i] for row in outcomes.values()) / n for i in range(c['trials'])]
                mean = sum(rates) / len(rates)
                summary = {'task_suite': c['task_suite'], 'services': c['services'],
                           'scope': 'full-standard-127' if c['task_suite'] == 'standard' and set(c['services']) == set(STANDARD) and n == 127 else ('full-easy-50' if c['task_suite'] == 'easy' and set(c['services']) == set(STANDARD) - {'playwright'} and n == 50 else 'selected-tasks'),
                           'total_tasks': n, 'trials_per_task': c['trials'],
                           'pass@1': mean, 'pass@1_per_run': rates,
                           'pass@1_std': (sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5,
                           'has_retryable_errors': failed}
                if c['trials'] > 1:
                    summary[f"pass@{c['trials']}"] = sum(any(row) for row in outcomes.values()) / n
                    summary[f"pass^{c['trials']}"] = sum(all(row) for row in outcomes.values()) / n
                (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                print(json.dumps(summary, indent=2))
                print(f'Results: {output}\nLog: {log}')
    # Model/verifier FAILs are scores; infrastructure errors make the job fail.
    return 1 if failed else 0

try:
    sys.exit(main())
except KeyboardInterrupt:
    sys.exit(130)
except Exception as exc:
    print('ERROR: ' + str(exc), file=sys.stderr)
    sys.exit(1)
PY
