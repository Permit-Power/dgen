#!/usr/bin/env bash
set -euo pipefail

# Confirm the pushed image actually CONTAINS the net-billing changes before submitting a
# national run. Without this, a stale :latest silently produces another net-metering run --
# the jobs succeed, the numbers look plausible, and the only tell is the schema tag.
#
# Usage: bash verify_image_netbilling.sh

IMG="us-east1-docker.pkg.dev/dgen-466702/dgen-repo-east1/dgen:latest"

echo "Pulling ${IMG} ..."
docker pull --platform linux/amd64 -q "$IMG" >/dev/null

echo
echo "1) FORCE_NET_BILLING is env-var overridable (not a hardcoded False):"
docker run --rm --platform linux/amd64 "$IMG" \
  grep -n "^FORCE_NET_BILLING" /opt/dgen_os/python/financial_functions.py

echo
echo "2) schema name carries the _nb tag when the flag is set:"
docker run --rm --platform linux/amd64 "$IMG" \
  grep -n '_nb' /opt/dgen_os/python/data_functions.py

echo
echo "3) the flag actually flips inside the image:"
for v in "" "1"; do
  docker run --rm --platform linux/amd64 -e FORCE_NET_BILLING="$v" "$IMG" \
    bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate dg3n &&
              cd /opt/dgen_os/python &&
              python -c "import os,financial_functions as f; print(\"  FORCE_NET_BILLING env=%r -> %s\" % (os.environ.get(\"FORCE_NET_BILLING\"), f.FORCE_NET_BILLING))"' 2>/dev/null
done

echo
echo "If (3) prints None->False and '1'->True, the image is good. Then submit:"
echo "    bash submit_all_netbilling.sh 0.05"
