# Silt on a Vultr server (Ubuntu 22.04)

One server runs everything. Only Caddy is public:

```
internet ──► Caddy :80 (later :443, HTTPS) ──► Streamlit 127.0.0.1:8501 ──► FastAPI 127.0.0.1:8000 ──► Postgres 127.0.0.1:5433
                                                                                │
                                                                                └─► rootless Docker (user silt) ──► sandbox containers
```

- **Streamlit and FastAPI** run on the host as systemd services under the unprivileged `silt` user.
- **AI-generated code** runs in throwaway sandbox containers with the isolation set in
  `backend/app/sandbox/runner.py`:
  - no network, a read-only root filesystem and no Linux capabilities;
  - limits on processes, memory and CPU;
  - read-only data, and a `/tmp` that can't run programs.

  They are started through a **rootless** Docker daemon owned by `silt`. No container gets the Docker socket,
  and `silt` is not in the `docker` group, which would be root-equivalent. A container escape gives the attacker
  the `silt` user, not root.
- **Postgres** runs in Docker Compose (`docker-compose.yml`), reachable only from the server itself.
- **The access gate (gated signup):** a shared password (`APP_PASSWORD`) protects the whole site, signup included.
  Wrong guesses are limited per visitor address.

## Install

As root, on a server with Docker Engine installed from Docker's apt repository:

```bash
git clone https://github.com/ipsitapp8/ai-data-analyst.git /opt/silt
bash /opt/silt/deploy/vultr/setup.sh
```

The script is safe to run again, and it keeps existing secrets. It:
- installs Python 3.11, Caddy and the rootless-Docker tools;
- creates the `silt` user and its rootless Docker daemon, and builds the sandbox image there;
- self-tests the sandbox with the runner's real limits;
- starts Postgres on loopback;
- writes `/etc/silt/{api,ui,compose}.env` with generated secrets;
- installs the services and the nightly backup timer;
- configures Caddy and the firewall: SSH, 80 and 443 only.

Then:
1. **Add the API keys.** Put `GEMINI_API_KEY` and `LLAMA_API_KEY` in `/etc/silt/api.env`, then run
   `systemctl restart silt-api`.
2. **Share the site password.** It's `APP_PASSWORD` in `/etc/silt/ui.env`; give it only to people you invite.
   To change it, edit the file and run `systemctl restart silt-ui`.
3. **Check the setup:** `curl -s http://127.0.0.1:8000/api/health` should show `"database_ok": true` and the
   sandbox `"ready": true`.

## No domain yet: plain HTTP

Until a domain points at the server, Silt is served on `http://<server-ip>`. Traffic, passwords included, is
**not encrypted**, and the login cookie can't be marked Secure. Use it only for trials.

To switch to HTTPS:
1. Point a DNS A record at the server.
2. In `/etc/caddy/Caddyfile`, replace `:80` with the domain name.
3. Set `APP_URL` and `CORS_ALLOWED_ORIGINS` to `https://<domain>` in `/etc/silt/api.env` and `/etc/silt/ui.env`.
4. Run `systemctl reload caddy && systemctl restart silt-api silt-ui`.

Caddy then gets and renews the certificate itself.

## Update

```bash
cd /opt/silt && git pull
/opt/silt/.venv/bin/python -m pip install -r backend/requirements.txt -r frontend/requirements.txt
systemctl restart silt-api silt-ui
```

Database migrations run when the API starts. If `backend/app/sandbox/Dockerfile` changed, rebuild the image:

```bash
sudo -u silt -H env XDG_RUNTIME_DIR=/run/user/$(id -u silt) DOCKER_HOST=unix:///run/user/$(id -u silt)/docker.sock \
  docker build -t ai-data-analyst-sandbox:latest /opt/silt/backend/app/sandbox
```

## Backups

`silt-backup.timer` runs `backup.sh` nightly, as root.
- It writes a Postgres dump (`db-<time>.dump`) and an archive of `/srv/silt/data` (`data-<time>.tar.gz`: uploads,
  Copilot memory, documents) to `/srv/silt/backups`, and keeps them for 14 days.
- It also removes sandbox workspaces older than 8 days. The app removes its own after 7 days, except folders a
  sandbox run created, which belong to the rootless daemon's remapped user.

**Copy the backups off the server**, for example `rsync` or `scp` from another machine. A backup on the same disk
is lost with the server.

Run a backup now with `systemctl start silt-backup.service`. List them with `ls -l /srv/silt/backups`.

## Restore

This restores from one backup pair; use the same `<time>` for both files.

```bash
systemctl stop silt-ui silt-api
cd /opt/silt
# 1. the database: replaces every table with the dump's contents
docker compose -p silt -f docker-compose.yml --env-file /etc/silt/compose.env exec -T postgres \
  pg_restore -U silt -d silt --clean --if-exists --no-owner < /srv/silt/backups/db-<time>.dump
# 2. the data directory: put the current one aside, unpack the backup, keep ownership and modes
mv /srv/silt/data /srv/silt/data.before-restore
tar -C /srv/silt -xzpf /srv/silt/backups/data-<time>.tar.gz
chown -R silt:silt /srv/silt/data && chmod 0700 /srv/silt/data
systemctl start silt-api silt-ui
curl -s http://127.0.0.1:8000/api/health
```

When the restored site checks out, delete `/srv/silt/data.before-restore`.

**Moving to a new server:**
1. Run `setup.sh` there first.
2. Copy `/etc/silt/*.env` from the old server. It holds the keys, `JWT_SECRET_KEY` and the database password,
   so carry it as a secret.
3. Run `systemctl restart silt-api silt-ui`, then restore as above.

**Test a restore once before relying on the backups.**

## Where things are

| What | Where |
|---|---|
| Code, Python environment | `/opt/silt`, `/opt/silt/.venv` |
| Uploads, Copilot memory, documents | `/srv/silt/data` (0700, `silt`) |
| Sandbox workspaces (kept 7 days) | `/srv/silt/workspaces` |
| Secrets and settings | `/etc/silt/api.env`, `/etc/silt/ui.env` (root:silt 0640), `/etc/silt/compose.env` (0600) |
| Logs | `journalctl -u silt-api -u silt-ui -u caddy` |

## Security notes

- **Never** set `SANDBOX_BACKEND=subprocess` or `SANDBOX_ALLOW_UNSAFE_SUBPROCESS=true` on this server; the setup
  script refuses to continue if they're changed. Never add `silt` to the `docker` group, and never mount the
  Docker socket into a container.
- **The sandbox shares the server's kernel.** Rootless Docker limits what an escape can reach.
  - The sandbox uses Docker's default runtime (`SANDBOX_DOCKER_RUNTIME` empty).
  - gVisor is **not** set up. It would add kernel isolation, but under rootless Docker it needs
    `--ignore-cgroups`, which drops the sandbox's memory, CPU and process limits. A runaway script could then
    exhaust the server.
  - Adding gVisor needs a separate design: a dedicated daemon that keeps cgroups.
- **Spending on LLM calls.** Everyone who passes the gate can run analyses on your Gemini and Llama keys.
  - Set a spending limit with each provider.
  - Optionally set `RUN_MAX_COST_USD` in `api.env`.
  - `JOB_WORKERS=2` suits 4 vCPUs: each sandbox run is capped at 512 MB and 1 CPU.
- **The local fine-tuned Copilot model** (MLX, Apple Silicon only) is not used here. The Copilot answers with
  its deterministic tools.
