#!/usr/bin/env bash
# Nightly job (silt-backup.timer), run as root:
#   1. a Postgres dump and an archive of the data directory (uploads, Copilot memory, documents), kept for
#      BACKUP_KEEP_DAYS days in /srv/silt/backups. Copy these off the server as well -- a backup on the same disk
#      does not survive losing the server. Restore: deploy/vultr/README.md, "Restore".
#   2. sandbox workspaces older than the app's retention (RETENTION_WORKSPACE_DAYS, 7) + 1 day. The app removes
#      them itself, except folders a sandbox run created: those belong to the rootless daemon's remapped user,
#      which the silt user cannot delete.
set -euo pipefail
umask 077
DEST=/srv/silt/backups
KEEP_DAYS=${BACKUP_KEEP_DAYS:-14}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$DEST"
docker compose -p silt -f /opt/silt/docker-compose.yml --env-file /etc/silt/compose.env \
  exec -T postgres pg_dump -U silt -d silt --format=custom > "$DEST/db-$STAMP.dump"
tar -C /srv/silt -czf "$DEST/data-$STAMP.tar.gz" data
find "$DEST" -type f \( -name 'db-*.dump' -o -name 'data-*.tar.gz' \) -mtime +"$KEEP_DAYS" -delete
echo "backup written: $DEST/db-$STAMP.dump, $DEST/data-$STAMP.tar.gz"
find /srv/silt/workspaces -mindepth 1 -maxdepth 1 -mtime +"${WORKSPACE_KEEP_DAYS:-8}" -exec rm -rf {} +
