#!/usr/bin/env bash
set -euo pipefail

# Submit the full 49-state national run under NET BILLING, as the twin of an existing
# net-metering run, to isolate the effect of the export-compensation assumption.
#
# Why this exists
# ---------------
# `financial_functions.process_tariff` sets ur_metering_option from each agent's own
# tariff, and essentially every agent tariff carries 0 (net metering). So every dGen
# result to date credits exports at FULL RETAIL, and the wholesale export series the
# model builds and hands to SAM as `ur_ts_sell_rate` is never consulted -- verified by
# scaling wholesale_prices 0x/1x/10x and seeing bill and NPV move by exactly $0.00.
# FORCE_NET_BILLING=1 sets ur_metering_option = 2, which activates those wholesale
# prices. Nothing else changes.
#
# Usage:
#   bash submit_all_netbilling.sh 0.05     # match synapse_attachrate_5 (the net-metering control)
#
# Compare the resulting `_a5_nb` schemas against the existing `_a5` ones. Both env vars
# are injected into TEMP copies of the yamls at submit time; the checked-in yamls are
# never modified. Schemas are tagged `_a<pct>_nb` so the two arms cannot be confused.

RATE="${1:?usage: bash submit_all_netbilling.sh <attachment rate in [0,1]>   e.g. 0.05}"
PCT=$(python3 -c "r=float('$RATE'); assert 0<=r<=1, 'rate must be in [0,1]'; print(int(round(r*100)))")

LOCATION="us-east1"
JOB_TS=$(date -u +"%Y%m%d-%H%M%S")
PROVISIONING="STANDARD"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# label | yaml | machine-type   (mirrors submit_all_attach.sh: 11 jobs = all 48 states)
JOBS=(
  "mid-r1|dgen-batch-job-mid-states.yaml|c2d-highcpu-16"
  "mid-large-r1|dgen-batch-job-mid-large-states.yaml|c2d-highcpu-32"
  "large-r1|dgen-batch-job-large-states.yaml|c2d-highcpu-32"
  "ca-r2|dgen-batch-job-ca.yaml|c2d-highcpu-32"
  "large-r2|dgen-batch-job-large-states-r2.yaml|c2d-highcpu-32"
  "mid-large-r2a|dgen-batch-job-mid-large-states-r2a.yaml|c2d-highcpu-32"
  "mid-large-r2b|dgen-batch-job-mid-large-states-r2b.yaml|c2d-highcpu-32"
  "mid-r2a|dgen-batch-job-mid-states-r2a.yaml|c2d-highcpu-16"
  "mid-r2b|dgen-batch-job-mid-states-r2b.yaml|c2d-highcpu-16"
  "small-r2a|dgen-batch-job-small-states-r2a.yaml|c2d-highcpu-16"
  "small-r2b|dgen-batch-job-small-states-r2b.yaml|c2d-highcpu-16"
)

echo "Submitting national run | NET BILLING | FLAT_STORAGE_ATTACHMENT_RATE=${RATE} (a${PCT}_nb) | ts=${JOB_TS}"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r label yaml mtype <<< "$spec"
  src="batch_job_yamls/${yaml}"
  tmp="${TMPDIR}/${yaml}"

  FLAT_RATE="$RATE" python3 - "$src" "$tmp" <<'PYEOF'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
rate = os.environ["FLAT_RATE"]
out, injected = [], False
for ln in open(src):
    out.append(ln)
    if (not injected) and "LOCAL_CORES:" in ln:
        indent = ln[: len(ln) - len(ln.lstrip())]
        out.append(f'{indent}FLAT_STORAGE_ATTACHMENT_RATE: "{rate}"\n')
        out.append(f'{indent}FORCE_NET_BILLING: "1"\n')
        injected = True
assert injected, f"LOCAL_CORES anchor not found in {src}"
open(dst, "w").writelines(out)
PYEOF

  gcloud batch jobs submit "dgen-nb${PCT}-${label}-${JOB_TS}" \
    --location="${LOCATION}" \
    --config="${tmp}" \
    --machine-type="${mtype}" \
    --provisioning-model="${PROVISIONING}"
done

echo "Done: submitted net-billing run (${#JOBS[@]} jobs). Schemas tagged _a${PCT}_nb."
