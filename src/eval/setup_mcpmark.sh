#!/usr/bin/env bash
# MCPMark Verified. Custom entry points are this file and run_mcpmark.sh only.
set -euo pipefail
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"
MCP_DIR="$EVAL_DIR/mcpmark"
PIN=cd45b7f57923b9b3985467f5139927575f83141c
PG_IMAGE=pgvector/pgvector:0.8.0-pg17-bookworm
GH_IMAGE=ghcr.io/github/github-mcp-server:v0.15.0
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash src/eval/setup_mcpmark.sh [ACTION]
No ACTION: clone pinned MCPMark Verified, create Python 3.12 .venv, resolve/cache
Python dependencies, install pinned Node MCP servers, and install Chromium.
Generated environments, dependency locks and helpers stay inside ignored mcpmark/.
  INSTALL_BROWSERS=false     Skip Python/Node Chromium downloads (default true).
  MCPMARK_PYTHON=/path       Python 3.12 interpreter (default uv-managed 3.12).
  MCPMARK_REPO_URL=...       Git mirror; default official eval-sys/mcpmark.

Optional actions (do not run evaluations or change the Docker daemon):
  --check-docker            Print Docker permissions and storage information.
  --prepare-docker github   Pull GitHub MCP v0.15.0.
  --prepare-docker postgres Pull PostgreSQL 17 + pgvector 0.8.0.
  --prepare-docker webarena Download/load the 3 official WebArena image archives.
  --start-postgres          Start/wait for dedicated PostgreSQL, localhost:15432.
                           Credentials load from mcpmark/.mcp_env (or exported env).
  --init-postgres           Restore/validate all five official template databases.
                           Data: data/eval/mcpmark/postgres; backups: postgres_state/.
  --notion-login            Interactive Notion login; saves mcpmark/notion_state.json.
                           Uses the official login helper with navigation retries.
                           NOTION_PROXY overrides HTTPS_PROXY/HTTP_PROXY; empty
                           NOTION_PROXY explicitly selects a direct connection.

WebArena image archive cache: data/eval/mcpmark/webarena (MCPMARK_WEBARENA_DIR).
An existing verified oci/ cache is preferred over downloading image archives.
OCI imports validate pinned manifests/configs and Docker rootfs layer digests;
image references are normalized for Docker's containerd image store.
Docker data-root and containerd storage are system-managed; check free space before
loading large WebArena images. MCPMARK_IMAGE_STORE can identify the actual image
store; otherwise it is detected from Docker/containerd. WebArena downloads try
Google Drive, CMU and Archive.org. Set MCPMARK_WEBARENA_SOURCE=gdrive|cmu|archive
or WEBARENA_SHOPPING_URL / WEBARENA_SHOPPING_ADMIN_URL / WEBARENA_REDDIT_URL.
Notion requires SOURCE_NOTION_API_KEY, EVAL_NOTION_API_KEY, EVAL_PARENT_PAGE_TITLE
and Source/Eval Hub grants. GitHub requires GITHUB_TOKENS and GITHUB_EVAL_ORG.
Export credentials, or store dotenv values in ignored mcpmark/.mcp_env.

After setup (start the model separately):
  SGLANG_PORT=8002 bash src/eval/run_mcpmark.sh --check
  MCPS=all SGLANG_PORT=8002 bash src/eval/run_mcpmark.sh --check
HELP
    exit 0
fi
# shellcheck source=cache_env.sh
source "$EVAL_DIR/cache_env.sh"
export TMPDIR="${MCPMARK_TMPDIR:-$CACHE_ROOT/mcpmark-tmp}"
export npm_config_cache="${npm_config_cache:-$CACHE_ROOT/npm}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CACHE_ROOT/ms-playwright}"
mkdir -p "$TMPDIR"
action="${1:-install}"
# Load dotenv through Python rather than treating the credentials file as shell code.
if [[ "$action" != install && "$action" != --check-docker && "${LIMA_MCPMARK_ENV_LOADED:-}" != 1 ]]; then
    [[ -x "$MCP_DIR/.venv/bin/python" ]] || { echo 'Run setup first.' >&2; exit 1; }
    exec "$MCP_DIR/.venv/bin/python" - "$EVAL_DIR/setup_mcpmark.sh" "$MCP_DIR" "$@" 3<&0 <<'ENVLOAD'
import os, sys
from pathlib import Path
from dotenv import dotenv_values
path=Path(os.environ.get('MCPMARK_ENV_FILE') or Path(sys.argv[2])/'.mcp_env').resolve()
if os.environ.get('MCPMARK_ENV_FILE') and not path.is_file():
    sys.exit('ERROR: MCPMARK_ENV_FILE does not exist')
for name,value in dotenv_values(path).items():
    if value and name not in os.environ:
        os.environ[name]=value
os.environ['LIMA_MCPMARK_ENV_LOADED']='1'
# Restore the caller's terminal/pipe after consuming this Python heredoc.
os.dup2(3, 0)
os.close(3)
os.execv('/bin/bash',['bash',sys.argv[1],*sys.argv[3:]])
ENVLOAD
fi
case "$action" in
    --check-docker)
        [[ $# == 1 ]] || exit 2
        docker version
        docker info --format 'Docker root: {{.DockerRootDir}}; storage driver: {{.Driver}}'
        df -h / /data
        exit 0 ;;
    --prepare-docker)
        [[ $# == 2 ]] || { echo 'Specify github, postgres or webarena.' >&2; exit 2; }
        docker info >/dev/null
        "$MCP_DIR/.venv/bin/python" - "$2" <<'STORAGE'
import json, os, shutil, subprocess, tomllib
from pathlib import Path
info=json.loads(subprocess.check_output(['docker','info','--format','{{json .}}']))
store=os.environ.get('MCPMARK_IMAGE_STORE')
if not store:
    store=info['DockerRootDir']
    if any('containerd.snapshotter' in str(row) for row in info.get('DriverStatus',[])):
        path=Path('/etc/containerd/config.toml')
        store=tomllib.loads(path.read_text()).get('root','/var/lib/containerd') if path.is_file() else '/var/lib/containerd'
mount=subprocess.check_output(['findmnt','-n','-T',store,'-o','TARGET'],text=True).strip()
free=shutil.disk_usage(mount).free
print(f'Image store: {store}; free: {free/1024**3:.1f} GiB')
minimum=(40 if __import__('sys').argv[1]=='webarena' else 2)*1024**3
if free<minimum:
    raise SystemExit('ERROR: Insufficient space in the actual Docker image store')
if shutil.disk_usage('/').free < 1024**3:
    raise SystemExit('ERROR: Leave >=1 GiB free for Docker system metadata')
STORAGE
        case "$2" in
            github) docker pull "$GH_IMAGE" ;;
            postgres) docker pull "$PG_IMAGE" ;;
            webarena)
                dest="${MCPMARK_WEBARENA_DIR:-$REPO_ROOT/data/eval/mcpmark/webarena}"
                mkdir -p "$dest"
                "$MCP_DIR/.venv/bin/python" - "$dest" <<'DOWNLOAD'
import hashlib, json, os, subprocess, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path
root=Path(sys.argv[1]).resolve()
items=[('shopping','shopping_final_0712','1gxXalk9O0p9eu1YkIJcmZta1nvvyAJpA','webarena-env-shopping-image'),
       ('shopping_admin','shopping_admin_final_0719','1See0ZhJRw0WTTL9y8hFlgaduwPZ_nGfd','webarena-env-shopping-admin-image'),
       ('reddit','postmill-populated-exposed-withimg','17Qpp1iu_mPqzgO_73Z9BnFjHrzmX9DGf','webarena-env-forum-image')]
source=os.environ.get('MCPMARK_WEBARENA_SOURCE','auto')
if source not in {'auto','gdrive','cmu','archive'}:
    sys.exit('ERROR: Invalid MCPMARK_WEBARENA_SOURCE')
# These manifests have the same configuration SHA-256 as the official archives.
# Do not use a mutable mirror tag as the identity of an evaluation environment.
pinned={
    'shopping_final_0712': ('8f9618d9194a46bc9f7bc96dc14deba0b968abce6c393d59cc7cd1dfec86ed6f',
                          '73e22b803149f8efa3f1171d2347fbdcc1063a2a20ddfbab4d5a20ea0fc7cc9c'),
    'shopping_admin_final_0719': ('428cdedfc10996d7f5378299cff541c93c187d1243643e31d439a66c0156147d',
                                'd0709482086f0e4931771795ae2b1e4b13fb9237c22328c8df0d074763465ad4'),
    'postmill-populated-exposed-withimg': ('e5b9d29b26ca911048fb14a8d4806a871b796db85c2991d514a13df2af0e427f',
                                        'bd8ae768c200fc4d005672af33fe03d49c9a058ae33997f3d50576e84cf51a1d'),
}
oci=root/'oci'
def inspect(reference):
    result=subprocess.run(['docker','image','inspect',reference],capture_output=True,text=True)
    return json.loads(result.stdout)[0] if result.returncode==0 else None

def blob(digest):
    algorithm,hex_digest=digest.split(':',1)
    if algorithm!='sha256' or len(hex_digest)!=64 or any(c not in '0123456789abcdef' for c in hex_digest):
        raise ValueError('Invalid OCI digest')
    data=(oci/'blobs/sha256'/hex_digest).read_bytes()
    if hashlib.sha256(data).hexdigest()!=hex_digest:
        raise ValueError('OCI manifest/config checksum mismatch')
    return json.loads(data)

if (oci/'index.json').is_file():
    index=json.loads((oci/'index.json').read_text())
    descriptors={x['digest'] for x in index['manifests']}
    expected={}
    for image,(manifest_hash,config_hash) in pinned.items():
        digest='sha256:'+manifest_hash
        if digest not in descriptors: raise SystemExit('ERROR: Incomplete pinned OCI index')
        manifest=blob(digest)
        if manifest['config']['digest']!='sha256:'+config_hash:
            raise SystemExit('ERROR: OCI config differs from the official archive')
        config=blob(manifest['config']['digest'])
        expected[image]=(digest,'sha256:'+config_hash,config['rootfs']['diff_ids'])
    if any(inspect(digest) is None and inspect(config_digest) is None
           for digest,config_digest,layers in expected.values()):
        print('Loading the pinned local OCI cache; no network download.',flush=True)
        # Stream the existing layout; avoid another ~87 GiB archive on disk.
        archive=subprocess.Popen(['tar','-C',str(oci),'-cf','-','oci-layout','index.json','blobs'],stdout=subprocess.PIPE)
        try:
            result=subprocess.run(['docker','image','load'],stdin=archive.stdout)
        finally:
            archive.stdout.close()
        if archive.wait()!=0 or result.returncode!=0:
            raise SystemExit('ERROR: Local OCI import failed')
    verified={}
    for image,(digest,config_digest,layers) in expected.items():
        loaded=inspect(digest) or inspect(config_digest)
        if not loaded or loaded['Id'] not in {digest,config_digest} or loaded['RootFS']['Layers']!=layers:
            raise SystemExit('ERROR: Imported image identity/rootfs mismatch: '+image)
        # OCI annotations may preserve an unqualified name that Docker lists but
        # cannot resolve. Creating the canonical tag fixes lookup without copying.
        reference='docker.io/library/'+image+':latest'
        existing=inspect(reference)
        if existing and existing['Id']!=loaded['Id']:
            raise SystemExit('ERROR: Preserving a different existing image: '+reference)
        if not existing:
            subprocess.run(['docker','image','tag',loaded['Id'],reference],check=True)
        resolved=inspect(image)
        if not resolved or resolved['Id']!=loaded['Id']:
            raise SystemExit('ERROR: Canonical Docker image lookup failed: '+image)
        verified[image]={'docker_id':loaded['Id'],'official_config_digest':config_digest,'rootfs_layers':len(layers)}
        print(image+': pinned identity, official config and rootfs verified; tag usable',flush=True)
    network=f'lima-mcpmark-{os.getuid()}'
    if subprocess.run(['docker','network','inspect',network],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        subprocess.run(['docker','network','create','--label',f'lima.mcpmark.owner={os.getuid()}',network],check=True)
    status_path=oci/'import-status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    status.pop('error',None)
    if status.get('log'):
        status['previous_attempt_log']=status.pop('log')
    status.update(pid=os.getpid(),status='complete',completed_at=datetime.now(timezone.utc).isoformat(),
                  network_download=False,images=verified,network=network)
    status_path.write_text(json.dumps(status,indent=2)+'\n')
    print('Local OCI import/verification complete; no images downloaded.',flush=True)
    sys.exit(0)
for category,image,drive_id,archive_item in items:
    if subprocess.run(['docker','image','inspect',image],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
        print(f'{image}: already loaded',flush=True)
        continue
    archive=root/(image+'.tar')
    if not archive.is_file():
        override=os.environ.get('WEBARENA_'+category.upper()+'_URL')
        mirrors=[('gdrive',drive_id),('cmu','https://metis.lti.cs.cmu.edu/webarena-images/'+archive.name),
                 ('archive','https://archive.org/download/'+archive_item+'/'+archive.name)]
        if override: mirrors=[('custom',override)]
        elif source!='auto': mirrors=[item for item in mirrors if item[0]==source]
        for kind,url in mirrors:
            part=root/(archive.name+'.'+kind+'.part')
            print(f'{image}: trying {kind}',flush=True)
            if kind=='gdrive':
                command=['uv','tool','run','--from','gdown==5.2.0','gdown','--no-cookies','--continue','-O',str(part),url]
            else:
                command=['wget','--continue','--timeout=20','--tries=2','--progress=dot:giga','-O',str(part),url]
            result=subprocess.run(command)
            if result.returncode==0:
                try:
                    with tarfile.open(part) as tar:
                        if tar.next() is None: raise ValueError('empty tar')
                except (tarfile.TarError,ValueError):
                    print(f'{kind}: response is not a valid image tar',flush=True)
                    continue
                part.replace(archive)
                break
        else:
            sys.exit(f'ERROR: All mirrors failed for {image}. Supply a local archive at {archive} or an official mirror URL override.')
    subprocess.run(['docker','load','--input',str(archive)],check=True)
    subprocess.run(['docker','image','inspect',image],check=True,stdout=subprocess.DEVNULL)
DOWNLOAD
                network="lima-mcpmark-$(id -u)"
                if ! docker network inspect "$network" >/dev/null 2>&1; then
                    docker network create --label "lima.mcpmark.owner=$(id -u)" "$network"
                fi ;;
            *) echo "ERROR: Unknown Docker service: $2" >&2; exit 2 ;;
        esac
        exit 0 ;;
    --start-postgres)
        [[ $# == 1 ]] || exit 2
        : "${POSTGRES_PASSWORD:?Export POSTGRES_PASSWORD for this dedicated evaluation DB.}"
        export POSTGRES_USER="${POSTGRES_USERNAME:-postgres}" POSTGRES_PASSWORD
        port="${POSTGRES_PORT:-15432}"
        [[ "$port" =~ ^[0-9]{1,5}$ ]] && (( 10#$port > 0 && 10#$port < 65536 )) || exit 2
        name="lima-mcpmark-$(id -u)-postgres"
        data="${MCPMARK_PG_DATA:-$REPO_ROOT/data/eval/mcpmark/postgres}"
        mkdir -p "$data"
        data="$(realpath -m -- "$data")"
        create=true
        if docker container inspect "$name" >/dev/null 2>&1; then
            owner=$(docker inspect --format '{{index .Config.Labels "lima.mcpmark.owner"}}' "$name")
            [[ "$owner" == "$(id -u)" ]] || { echo 'ERROR: Container owner mismatch.' >&2; exit 1; }
            running=$(docker inspect --format '{{.State.Running}}' "$name")
            entry=$(docker inspect --format '{{json .Config.Entrypoint}}' "$name")
            if [[ "$running" != true && "$entry" != '["/bin/bash"]' ]]; then
                # Preserve the bind-mounted DB; replace only our failed legacy container.
                docker rm "$name" >/dev/null
            else
                create=false
                if [[ "$running" != true ]]; then docker start "$name" >/dev/null; fi
            fi
        fi
        if [[ "$create" == true ]]; then
            # This host's snapshot roots inherit mode 750. Fix only this container's
            # writable root so the upstream entrypoint can drop to the postgres UID.
            docker run -d --name "$name" --label "lima.mcpmark.owner=$(id -u)" \
                --network host --memory 4g --cpus 2 -e POSTGRES_PASSWORD -e POSTGRES_USER \
                --entrypoint /bin/bash -v "$data:/var/lib/postgresql/data" "$PG_IMAGE" \
                -c 'chmod 755 / && exec /usr/local/bin/docker-entrypoint.sh "$@"' -- \
                postgres -p "$port" -c listen_addresses=127.0.0.1
        fi
        ready=false
        for attempt in {1..30}; do
            if docker exec "$name" pg_isready -U "$POSTGRES_USER" -p "$port" >/dev/null 2>&1; then ready=true; break; fi
            sleep 1
        done
        [[ "$ready" == true ]] || { echo 'ERROR: PostgreSQL did not become ready; inspect its container logs.' >&2; exit 1; }
        echo "PostgreSQL ready: $name; localhost:$port. Initialize templates with --init-postgres."
        exit 0 ;;
    --init-postgres)
        [[ $# == 1 ]] || exit 2
        export MCPMARK_UPSTREAM="$MCP_DIR"
        "$MCP_DIR/.venv/bin/python" - <<'PGINIT'
import hashlib, os, subprocess, urllib.request
from pathlib import Path
import psycopg2
from psycopg2 import sql
up=Path(os.environ['MCPMARK_UPSTREAM'])
backups=up/'postgres_state'
backups.mkdir(parents=True,exist_ok=True)
container=f'lima-mcpmark-{os.getuid()}-postgres'
owner=subprocess.check_output(['docker','inspect','--format','{{index .Config.Labels "lima.mcpmark.owner"}}',container],text=True).strip()
if owner!=str(os.getuid()): raise SystemExit('ERROR: Container ownership mismatch')
params=dict(host=os.environ.get('POSTGRES_HOST','127.0.0.1'),port=os.environ.get('POSTGRES_PORT','15432'),
            user=os.environ.get('POSTGRES_USERNAME','postgres'),password=os.environ['POSTGRES_PASSWORD'],connect_timeout=10)
conn=psycopg2.connect(**params,dbname='postgres')
conn.autocommit=True
for db in ('employees','chinook','dvdrental','sports','lego'):
    backup=backups/(db+'.backup')
    if not backup.is_file():
        print(f'Downloading official {db} backup',flush=True)
        part=backup.with_suffix('.backup.part')
        request=urllib.request.Request('https://storage.mcpmark.ai/postgres/'+backup.name,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(request,timeout=60) as response, part.open('wb') as output:
            __import__('shutil').copyfileobj(response,output)
        if part.open('rb').read(5)!=b'PGDMP': raise SystemExit(f'ERROR: Invalid backup: {db}')
        part.replace(backup)
    with conn.cursor() as cur:
        cur.execute('SELECT shobj_description(oid, %s) FROM pg_database WHERE datname=%s',('pg_database',db))
        existing=cur.fetchone()
        if existing and existing[0]=='lima-mcpmark-template-ready-v1':
            print(f'{db}: already initialized',flush=True)
            continue
        if existing:
            raise SystemExit(f'ERROR: {db} exists without our initialization marker; preserving it for inspection')
        cur.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(db)))
    print(f'Restoring {db}',flush=True)
    with backup.open('rb') as source:
        result=subprocess.run(['docker','exec','-i',container,'pg_restore','--exit-on-error','--single-transaction',
                               '-p',str(params['port']),'-U',params['user'],'-d',db],stdin=source,capture_output=True)
    if result.returncode:
        print(result.stderr.decode(errors='replace')[-2500:])
        raise SystemExit(f'ERROR: Restore failed for {db}; no readiness marker was written')
    with psycopg2.connect(**params,dbname=db) as sample, sample.cursor() as cur:
        cur.execute("SELECT count(*) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema')")
        tables=cur.fetchone()[0]
    if tables<1: raise SystemExit(f'ERROR: Restored {db} has no user tables')
    with conn.cursor() as cur:
        cur.execute(sql.SQL("COMMENT ON DATABASE {} IS 'lima-mcpmark-template-ready-v1'").format(sql.Identifier(db)))
    print(f'{db}: ready ({tables} user tables)',flush=True)
conn.close()
PGINIT
        exit 0 ;;
    --notion-login)
        [[ $# == 1 && -x "$MCP_DIR/.venv/bin/python" ]] || exit 2
        cd "$MCP_DIR"
        # Keep stdin available for the official helper's email/code prompts.
        # All adaptations are process-local; the pinned upstream stays unchanged.
        exec .venv/bin/python -c "$(cat <<'NOTION_LOGIN'
import os, re, sys, tempfile, time
from pathlib import Path
from urllib.parse import urlsplit, unquote
from playwright.sync_api import Browser, BrowserType, Page, Error as BrowserError
from src.mcp_services.notion.notion_login_helper import NotionLoginHelper

if not sys.stdin.isatty():
    sys.exit('ERROR: Run --notion-login in an interactive terminal for the email/code prompts.')

proxy_url = os.environ.get('NOTION_PROXY')
if proxy_url is None:
    proxy_url = next((os.environ[n] for n in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy')
                      if os.environ.get(n)), '')
proxy = None
if proxy_url:
    parsed = urlsplit(proxy_url)
    if (parsed.scheme not in {'http', 'https', 'socks5'} or not parsed.hostname
            or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
        sys.exit('ERROR: NOTION_PROXY must be an HTTP(S) or SOCKS5 proxy URL.')
    proxy = {'server': parsed.scheme + '://' + parsed.netloc.rsplit('@', 1)[-1]}
    if parsed.username is not None:
        proxy.update(username=unquote(parsed.username), password=unquote(parsed.password or ''))
print('Notion browser connection: ' + ('explicit proxy' if proxy else 'direct'), flush=True)

original_launch, original_context, original_goto = BrowserType.launch, Browser.new_context, Page.goto
def launch(browser_type, **kwargs):
    if proxy:
        kwargs.setdefault('proxy', proxy)
    return original_launch(browser_type, **kwargs)

def new_context(browser, **kwargs):
    kwargs.setdefault('locale', 'en-US')
    context = original_context(browser, **kwargs)
    context.set_default_navigation_timeout(90_000)
    return context

def login_url(url):
    parsed = urlsplit(url)
    return (parsed.hostname in {'notion.so', 'www.notion.so', 'notion.com', 'www.notion.com', 'app.notion.com'}
            and (parsed.path.rstrip('/') == '/login' or parsed.path.startswith('/login/')))

def goto(page, url, **kwargs):
    if not login_url(url):
        return original_goto(page, url, **kwargs)
    email = page.locator('input[type="email"]').first
    # The official helper navigates to /login twice. Reuse an already ready form.
    if login_url(page.url) and email.is_visible():
        return None
    kwargs.update(wait_until='domcontentloaded', timeout=90_000)
    for attempt in range(3):
        print(f'Opening Notion login form: attempt {attempt + 1}/3 (90s navigation timeout)', flush=True)
        try:
            response = original_goto(page, url, **kwargs)
            email.wait_for(state='visible', timeout=90_000)
            return response
        except BrowserError:
            if login_url(page.url) and email.is_visible():
                return None
            if attempt == 2:
                raise RuntimeError('Notion login form did not become ready. Check the proxy/network and retry.') from None
            page.wait_for_timeout(1_000)

BrowserType.launch, Browser.new_context, Page.goto = launch, new_context, goto

class CheckedNotionLoginHelper(NotionLoginHelper):
    def _handle_headless_login(self, context):
        # Reuse upstream's email submission, verification code and redirect flow.
        super()._handle_headless_login(context)
        print('Verification code submitted; waiting for the Notion browser session.', flush=True)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            # Login can redirect between notion.so and notion.com. Querying just
            # www.notion.so hides cookies scoped to the other Notion domains.
            cookies = [c for c in context.cookies() if
                       c['domain'].lstrip('.') in {'notion.so', 'notion.com'} or
                       c['domain'].endswith(('.notion.so', '.notion.com'))]
            page = context.pages[0]
            host = urlsplit(page.url).hostname or ''
            notion_page = host in {'notion.so', 'notion.com'} or host.endswith(('.notion.so', '.notion.com'))
            if (any(c['name'] == 'token_v2' and c.get('value') for c in cookies)
                    and notion_page and not login_url(page.url)):
                return
            page.wait_for_timeout(500)
        # Report only structure and known UI signals. Never print input values,
        # full URLs/query parameters, page text, or cookie values.
        parsed = urlsplit(page.url)
        print('Login diagnostic: page host = ' + (parsed.hostname or '(none)'), flush=True)
        print('Login diagnostic: still on login route = ' + str(login_url(page.url)), flush=True)
        print('Login diagnostic: Notion cookie names/domains = ' + str(
            sorted({(c['name'], c['domain']) for c in cookies})), flush=True)
        try:
            print('Login diagnostic: email input visible = ' + str(
                page.locator('input[type="email"]').first.is_visible()), flush=True)
            print('Login diagnostic: code input visible = ' + str(
                page.locator('input[placeholder="Enter code"]').first.is_visible()), flush=True)
            text = page.locator('body').inner_text(timeout=5_000)
            for label, pattern in (
                ('invalid_or_expired_code', r'(invalid|incorrect|expired|wrong)[^\n]{0,60}(code|link)|(code|link)[^\n]{0,60}(invalid|incorrect|expired)'),
                ('additional_verification', r'captcha|verify you are human|two.step verification|security key'),
                ('workspace_selection', r'choose a workspace|select a workspace'),
            ):
                if re.search(pattern, text, re.I):
                    print('Login diagnostic: page signal = ' + label, flush=True)
        except BrowserError:
            print('Login diagnostic: page details unavailable.', flush=True)
        raise RuntimeError('Notion login was not confirmed; see the diagnostic above. The previous login state has been preserved.')

os.umask(0o077)
state = Path('notion_state.json').resolve()
fd, name = tempfile.mkstemp(prefix='.notion_state.', suffix='.json', dir=state.parent)
os.close(fd)
pending = Path(name)
helper = CheckedNotionLoginHelper(browser=os.environ.get('PLAYWRIGHT_BROWSER', 'chromium'),
                                  headless=True, state_path=pending)
try:
    helper.login()
    pending.chmod(0o600)
    pending.replace(state)
    print(f'Confirmed Notion login saved to {state}', flush=True)
except KeyboardInterrupt:
    sys.exit('Login cancelled; the previous login state has been preserved.')
except Exception as exc:
    sys.exit(f'ERROR: {exc}')
finally:
    try:
        helper.close()
    except Exception:
        # Ctrl-C can terminate the browser driver before upstream close() runs.
        if helper._playwright:
            try:
                helper._playwright.stop()
            except Exception:
                pass
    finally:
        pending.unlink(missing_ok=True)
NOTION_LOGIN
)" ;;
    install) [[ $# == 0 ]] || exit 2 ;;
    *) echo "ERROR: Unknown action: $action (see --help)." >&2; exit 2 ;;
esac
for tool in git uv node npm wget unzip rg; do
    command -v "$tool" >/dev/null || { echo "ERROR: Missing $tool." >&2; exit 1; }
done
node -e 'if (+process.versions.node.split(".")[0] < 18) process.exit(1)'
INSTALL_BROWSERS="${INSTALL_BROWSERS:-true}"
[[ "$INSTALL_BROWSERS" == true || "$INSTALL_BROWSERS" == false ]] || exit 2
url="${MCPMARK_REPO_URL:-https://github.com/eval-sys/mcpmark.git}"
if [[ ! -e "$MCP_DIR" ]]; then git clone --filter=blob:none "$url" "$MCP_DIR"; fi
[[ -e "$MCP_DIR/.git" ]] || { echo "ERROR: $MCP_DIR is not a checkout." >&2; exit 1; }
[[ -z "$(git -C "$MCP_DIR" status --porcelain --untracked-files=normal)" ]] || {
    echo 'ERROR: Preserve local MCPMark source changes before setup.' >&2; exit 1;
}
if ! git -C "$MCP_DIR" cat-file -e "$PIN^{commit}" 2>/dev/null; then
    git -C "$MCP_DIR" fetch --no-tags "$url" "$PIN"
fi
git -C "$MCP_DIR" checkout --detach -q "$PIN"
cd "$MCP_DIR"
# Only generated files are ignored. Upstream source stays unmodified.
for entry in '.lima-*' '.venv/'; do
    if ! rg -q -F -x "$entry" .git/info/exclude; then echo "$entry" >> .git/info/exclude; fi
done
if [[ ! -d .venv ]]; then uv venv --python "${MCPMARK_PYTHON:-3.12}" .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
if [[ ! -f .lima-requirements.txt ]]; then
    printf '%s\n' '-e .' 'postgres-mcp==0.3.0' > .lima-requirements.in
    uv pip compile --quiet --python .venv/bin/python .lima-requirements.in -o .lima-requirements.txt
fi
uv pip sync --python .venv/bin/python .lima-requirements.txt
if [[ -f .lima-node/package-lock.json ]]; then
    npm ci --prefix .lima-node --no-audit --no-fund
else
    npm install --prefix .lima-node --save-exact --no-audit --no-fund \
        @modelcontextprotocol/server-filesystem@2025.12.18 \
        @notionhq/notion-mcp-server@1.9.1 @playwright/mcp@0.0.68
fi
if [[ "$INSTALL_BROWSERS" == true ]]; then
    .venv/bin/playwright install chromium
    .lima-node/node_modules/.bin/playwright install chromium
fi
# Match pg_restore client to the pinned PG17 server without a system apt install.
mkdir -p .lima-bin
cat > .lima-bin/pg_restore <<'PGRESTORE'
#!/usr/bin/env bash
set -euo pipefail
upstream="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec docker run --rm --network host -i -e PGPASSWORD \
    -v "$upstream/postgres_state:$upstream/postgres_state:ro" \
    pgvector/pgvector:0.8.0-pg17-bookworm pg_restore "$@"
PGRESTORE
chmod +x .lima-bin/pg_restore
.venv/bin/python -c 'from importlib.metadata import version; print("LiteLLM:", version("litellm"), "MCP SDK:", version("mcp"))'
echo "Setup complete: MCPMark Verified @ $PIN"
echo 'No evaluations or Docker service deployment were started.'
echo 'Next: bash src/eval/run_mcpmark.sh --help'
