#!/usr/bin/env bash
# Silt on one Ubuntu 22.04 server (Vultr). Run as root from the clone at /opt/silt:
#
#     git clone https://github.com/ipsitapp8/ai-data-analyst.git /opt/silt
#     bash /opt/silt/deploy/vultr/setup.sh
#
# Safe to run again (existing secrets and settings are kept). What it sets up -- see deploy/vultr/README.md:
#   Caddy :80 (later :443)  ->  Streamlit 127.0.0.1:8501  ->  FastAPI 127.0.0.1:8000  ->  Postgres 127.0.0.1:5433
#   AI-generated code runs in sandbox containers started through a ROOTLESS Docker daemon owned by the
#   unprivileged `silt` user. No container gets the Docker socket; `silt` is not in the `docker` group.
#
# Options (environment): SKIP_FIREWALL=1, PUBLIC_URL=http://<address> (default: this server's IPv4 address).
# The sandbox uses Docker's default runtime (runc) under the rootless daemon. gVisor is deliberately not set up:
# under rootless Docker it needs --ignore-cgroups, which would drop the sandbox's memory / CPU / process limits.
set -euo pipefail

SILT_DIR=/opt/silt
SRV=/srv/silt
ETC=/etc/silt
SKIP_FIREWALL=${SKIP_FIREWALL:-0}

step() { printf '\n== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root (sudo bash $0)"
cd /                                                # commands run as silt must not start in root's home
[ -f "$SILT_DIR/backend/app/main.py" ] || die "clone the repository to $SILT_DIR first (see the header of this script)"
. /etc/os-release
[ "${VERSION_ID:-}" = "22.04" ] || warn "tested for Ubuntu 22.04; this is ${PRETTY_NAME:-unknown}"
command -v docker >/dev/null || die "Docker Engine is required (https://docs.docker.com/engine/install/ubuntu/)"
docker compose version >/dev/null 2>&1 || die "the Docker Compose v2 plugin is required (docker compose ...)"
[ -f /sys/fs/cgroup/cgroup.controllers ] || die "cgroup v2 is required (Ubuntu 22.04 uses it by default)"
[ "$(sysctl -n kernel.unprivileged_userns_clone 2>/dev/null || echo 1)" = 1 ] \
  || die "unprivileged user namespaces are disabled (kernel.unprivileged_userns_clone=0); rootless Docker needs them"

# ------------------------------------------------------------------------------------------- packages --
step "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q software-properties-common git curl gnupg openssl ufw uidmap dbus-user-session \
  slirp4netns fuse-overlayfs build-essential
# dockerd-rootless-setuptool.sh comes with Docker's own apt repository (docker-ce-rootless-extras)
apt-get install -y -q docker-ce-rootless-extras \
  || die "docker-ce-rootless-extras not found: install Docker Engine from Docker's apt repository"
if ! command -v python3.11 >/dev/null; then
  add-apt-repository -y ppa:deadsnakes/ppa          # Ubuntu 22.04 ships Python 3.10; the pinned stack needs 3.11
  apt-get update -q
fi
apt-get install -y -q python3.11 python3.11-venv python3.11-dev
if ! command -v caddy >/dev/null; then
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q
  apt-get install -y -q caddy
fi

# ------------------------------------------------------------------------------------------- user --
step "Service user 'silt' (unprivileged, not in the docker group)"
install -d -m 0755 "$SRV"
if ! id silt >/dev/null 2>&1; then
  useradd --create-home --home-dir "$SRV/home" --shell /bin/bash --user-group silt
  passwd -l silt >/dev/null                         # no password login
fi
if id -nG silt | tr ' ' '\n' | grep -qx docker; then
  gpasswd -d silt docker                            # the rootful daemon is root-equivalent
fi
# subordinate ids for the rootless daemon's user namespace: a fresh 65536-id range after every range in use
next_free() { awk -F: 'BEGIN { m = 100000 } { e = $2 + $3; if (e > m) m = e } END { print m }' "$1" 2>/dev/null || echo 100000; }
if ! grep -q '^silt:' /etc/subuid; then s=$(next_free /etc/subuid); usermod --add-subuids "$s-$((s + 65535))" silt; fi
if ! grep -q '^silt:' /etc/subgid; then s=$(next_free /etc/subgid); usermod --add-subgids "$s-$((s + 65535))" silt; fi
SILT_UID=$(id -u silt)
RUNDIR=/run/user/$SILT_UID
# The sandbox's --memory / --cpus / --pids-limit need these cgroup controllers delegated to user services
# (Ubuntu 22.04 delegates only memory and pids by default; without cpu every sandbox run fails).
install -d /etc/systemd/system/user@.service.d
printf '[Service]\nDelegate=cpu cpuset io memory pids\n' > /etc/systemd/system/user@.service.d/delegate.conf
systemctl daemon-reload
loginctl enable-linger silt
# start (or restart, so the delegation applies) the user manager; with linger it runs without a login
systemctl restart "user@$SILT_UID.service"
for _ in $(seq 1 30); do [ -S "$RUNDIR/bus" ] && break; sleep 1; done
[ -S "$RUNDIR/bus" ] || die "the systemd user manager for silt did not start ($RUNDIR/bus missing)"
as_silt() { sudo -u silt -H env XDG_RUNTIME_DIR="$RUNDIR" DBUS_SESSION_BUS_ADDRESS="unix:path=$RUNDIR/bus" \
  DOCKER_HOST="unix://$RUNDIR/docker.sock" "$@"; }

# ------------------------------------------------------------------------------ rootless docker --
step "Rootless Docker daemon for the sandbox (owned by silt)"
if [ ! -S "$RUNDIR/docker.sock" ]; then
  as_silt dockerd-rootless-setuptool.sh install
fi
as_silt systemctl --user enable --now docker >/dev/null
for _ in $(seq 1 30); do as_silt docker info >/dev/null 2>&1 && break; sleep 1; done
as_silt docker info --format '{{.SecurityOptions}}' | grep -q rootless \
  || die "the sandbox daemon at $RUNDIR/docker.sock is not running rootless"

# ------------------------------------------------------------------------------------------- code --
step "Python environment"
if [ ! -x "$SILT_DIR/.venv/bin/python" ]; then
  python3.11 -m venv "$SILT_DIR/.venv"
fi
"$SILT_DIR/.venv/bin/python" -m pip install -q --upgrade pip
"$SILT_DIR/.venv/bin/python" -m pip install -q -r "$SILT_DIR/backend/requirements.txt" -r "$SILT_DIR/frontend/requirements.txt"
chmod -R a+rX "$SILT_DIR"

# ------------------------------------------------------------------------------------------- data --
step "Data directories"
install -d -o silt -g silt -m 0700 "$SRV/data" "$SRV/data/uploads"
# the sandbox container's (remapped) user must traverse into its own workspace; no listing of the others
install -d -o silt -g silt -m 0711 "$SRV/workspaces"
install -d -o root -g root -m 0700 "$SRV/backups"

# ---------------------------------------------------------------------------------------- secrets --
step "Environment files in $ETC (kept if they exist)"
install -d -o root -g silt -m 0750 "$ETC"
if [ -n "${PUBLIC_URL:-}" ]; then
  URL=$PUBLIC_URL
else
  IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
  URL="http://${IP:-127.0.0.1}"
fi
if [ ! -f "$ETC/compose.env" ]; then
  umask 077
  printf 'POSTGRES_PASSWORD=%s\n' "$(openssl rand -hex 24)" > "$ETC/compose.env"
  umask 022
fi
chmod 0600 "$ETC/compose.env"
PGPASS=$(sed -n 's/^POSTGRES_PASSWORD=//p' "$ETC/compose.env")
if [ ! -f "$ETC/api.env" ]; then
  cat > "$ETC/api.env" <<ENV
# Silt API. systemd format: KEY=value, comments only on their own line. Fill in the two API keys.
GEMINI_API_KEY=
LLAMA_API_KEY=
JWT_SECRET_KEY=$(openssl rand -hex 32)
DATABASE_URL=postgresql://silt:$PGPASS@127.0.0.1:5433/silt
APP_URL=$URL
CORS_ALLOWED_ORIGINS=$URL
DATA_DIR=$SRV/data
UPLOADS_DIR=$SRV/data/uploads
WORKSPACES_DIR=$SRV/workspaces
SANDBOX_BACKEND=docker
SANDBOX_ALLOW_UNSAFE_SUBPROCESS=false
SANDBOX_IMAGE=ai-data-analyst-sandbox:latest
SANDBOX_DOCKER_RUNTIME=
DOCKER_HOST=unix://$RUNDIR/docker.sock
XDG_RUNTIME_DIR=$RUNDIR
HOME=$SRV/home
JOB_WORKERS=2
ENV
fi
if [ ! -f "$ETC/ui.env" ]; then
  cat > "$ETC/ui.env" <<ENV
# Silt web UI. APP_PASSWORD is the shared access password in front of the whole site (gated signup).
BACKEND_BASE_URL=http://127.0.0.1:8000
APP_URL=$URL
APP_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | cut -c1-20)
APP_PASSWORD_MAX_FAILURES=8
APP_PASSWORD_WINDOW_SECONDS=600
ENV
fi
chown root:silt "$ETC/api.env" "$ETC/ui.env"
chmod 0640 "$ETC/api.env" "$ETC/ui.env"
grep -q '^SANDBOX_ALLOW_UNSAFE_SUBPROCESS=false$' "$ETC/api.env" || die "$ETC/api.env must keep SANDBOX_ALLOW_UNSAFE_SUBPROCESS=false"
grep -q '^SANDBOX_BACKEND=docker$' "$ETC/api.env" || die "$ETC/api.env must keep SANDBOX_BACKEND=docker"

# --------------------------------------------------------------------------------------- postgres --
step "Postgres (loopback only)"
docker compose -p silt -f "$SILT_DIR/docker-compose.yml" --env-file "$ETC/compose.env" up -d postgres
for _ in $(seq 1 60); do
  docker compose -p silt -f "$SILT_DIR/docker-compose.yml" --env-file "$ETC/compose.env" exec -T postgres \
    pg_isready -U silt -d silt >/dev/null 2>&1 && break
  sleep 2
done

# -------------------------------------------------------------------------------- sandbox image --
step "Sandbox image (built in the rootless daemon)"
as_silt docker build -q -t ai-data-analyst-sandbox:latest "$SILT_DIR/backend/app/sandbox" >/dev/null
# Self-test with the isolation, limits and mounts the API's runner uses (backend/app/sandbox/runner.py): the
# container's remapped user reads a 0444 input through the 0711 workspace root and writes to a 0777 output dir.
# shellcheck disable=SC2054  # the commas are --tmpfs mount options, not array separators
SBX_FLAGS=(--network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 128
  --memory 512m --memory-swap 512m --cpus 1 --ipc none --ulimit fsize=268435456:268435456
  --ulimit nofile=256:256 --ulimit core=0:0 --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m)
T="$SRV/workspaces/_selftest"
rm -rf "$T"
as_silt sh -c "umask 022 && mkdir -p '$T/data' '$T/out' && printf 'a,b\n1,2\n' > '$T/data/input.csv' \
  && chmod 0444 '$T/data/input.csv' && chmod 0777 '$T/out' \
  && printf 'import pandas as pd\nn = len(pd.read_csv(\"/workspace/data/input.csv\"))\nopen(\"/workspace/output/ok.txt\", \"w\").write(str(n))\n' > '$T/main.py'"
as_silt docker run --rm "${SBX_FLAGS[@]}" -v "$T/data:/workspace/data:ro" -v "$T/out:/workspace/output:rw" \
  -v "$T/main.py:/workspace/main.py:ro" -w /tmp ai-data-analyst-sandbox:latest -I -B /workspace/main.py \
  || die "the sandbox cannot run with its limits and mounts under the rootless daemon (cgroup delegation? permissions?)"
[ "$(as_silt cat "$T/out/ok.txt")" = 1 ] || die "the sandbox self-test wrote no result"
rm -rf "$T"
echo "sandbox self-test passed (rootless, no network, read-only, limits enforced, workspace mounts readable)"

# A runtime other than the default must exist in the sandbox daemon; setup never turns one on by itself.
RT=$(sed -n 's/^SANDBOX_DOCKER_RUNTIME=//p' "$ETC/api.env")
if [ -n "$RT" ]; then
  as_silt docker info --format '{{json .Runtimes}}' | grep -q "\"$RT\"" \
    || die "SANDBOX_DOCKER_RUNTIME=$RT in $ETC/api.env is not a runtime of the sandbox daemon; leave it empty"
fi


# ------------------------------------------------------------------------------------------ services --
step "systemd services"
for unit in silt-api.service silt-ui.service silt-backup.service silt-backup.timer; do
  sed "s/@SILT_UID@/$SILT_UID/g" "$SILT_DIR/deploy/vultr/$unit" > "/etc/systemd/system/$unit"
done
chmod +x "$SILT_DIR/deploy/vultr/backup.sh"
systemctl daemon-reload
systemctl enable --now silt-api.service silt-ui.service silt-backup.timer
systemctl restart silt-api.service silt-ui.service

step "Caddy (public entry point)"
[ -f /etc/caddy/Caddyfile.orig ] || cp -p /etc/caddy/Caddyfile /etc/caddy/Caddyfile.orig 2>/dev/null || true
install -m 0644 "$SILT_DIR/deploy/vultr/Caddyfile" /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null
systemctl enable --now caddy
systemctl reload caddy

# ------------------------------------------------------------------------------------------ firewall --
if [ "$SKIP_FIREWALL" != 1 ]; then
  step "Firewall (SSH, HTTP, HTTPS only)"
  SSH_PORTS=$(ss -tlnpH 2>/dev/null | awk '/sshd/ {n = split($4, a, ":"); print a[n]}' | sort -u)
  for p in ${SSH_PORTS:-22}; do ufw allow "$p/tcp" >/dev/null; done
  ufw allow 80/tcp >/dev/null
  ufw allow 443/tcp >/dev/null
  ufw --force enable >/dev/null
  # Postgres is published on 127.0.0.1 only (docker-compose.yml): Docker's own rules would bypass UFW otherwise.
fi

# ------------------------------------------------------------------------------------------- checks --
step "Checks"
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1 && break; sleep 2; done
curl -fsS http://127.0.0.1:8000/api/health | "$SILT_DIR/.venv/bin/python" -c '
import json, sys
h = json.load(sys.stdin)
sb = h.get("sandbox") or {}
print("database ok:     ", h.get("database_ok"))
print("sandbox ready:   ", sb.get("ready"), "|", sb.get("isolation"))
print("gemini key set:  ", h.get("gemini_key_configured"))
print("llama key set:   ", h.get("llama_key_configured"))' \
  || warn "the API did not answer its health check -- see: journalctl -u silt-api -n 100"
curl -fsS -o /dev/null -w 'public page:      HTTP %{http_code}\n' http://127.0.0.1/ \
  || warn "the public page did not load -- see: journalctl -u silt-ui -u caddy -n 100"

cat <<DONE

Silt is running at $URL  (plain HTTP until a domain is set up -- see deploy/vultr/README.md)

Next:
  1. Put your API keys in $ETC/api.env (GEMINI_API_KEY, LLAMA_API_KEY), then:  systemctl restart silt-api
  2. The site's access password is APP_PASSWORD in $ETC/ui.env -- share it only with people you invite.
  3. Copy $SRV/backups off this server regularly.
DONE
