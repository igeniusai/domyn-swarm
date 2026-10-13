#!/bin/bash
# Start munge and the Slurm daemon for this container's role.
set -euo pipefail

role=${1:?usage: e2e-entrypoint controller|compute}
key=/shared/munge/munge.key

mkdir -p /shared/logs /shared/munge /var/spool/slurmctld /var/spool/slurmd \
  /etc/munge /run/munge /var/log/munge /var/lib/munge

if [ "$role" = controller ]; then
  mkdir -p /shared/home /shared/hf /shared/images
  if [ ! -f "$key" ]; then
    dd if=/dev/urandom bs=1 count=1024 of="$key.tmp" 2>/dev/null
    mv "$key.tmp" "$key"
  fi
else
  # Slurm wants a device node per GPU; reuse the /dev/null numbers.
  for i in 0 1; do [ -e /dev/fakegpu$i ] || mknod -m 666 /dev/fakegpu$i c 1 3; done
  for _ in $(seq 1 60); do
    [ -f "$key" ] && break
    sleep 1
  done
fi

install -o munge -g munge -m 400 "$key" /etc/munge/munge.key
chown -R munge:munge /etc/munge /run/munge /var/log/munge /var/lib/munge
chmod 700 /etc/munge
su -s /bin/bash munge -c /usr/sbin/munged

case "$role" in
  controller) /usr/sbin/slurmctld ;;
  compute) /usr/sbin/slurmd ;;
  *) echo "unknown role: $role" >&2; exit 2 ;;
esac

exec sleep infinity
