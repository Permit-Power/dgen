#!/usr/bin/env bash
# Submit the net-billing re-scan: 49 tasks, one per state, both scenarios each.
#
# Deliberately on-demand rather than spot. Spot is cheaper, but there was no
# c2d-highcpu-32 spot capacity in this zone and the job cycled between QUEUED and
# SCHEDULED for half an hour without starting a single task. On-demand gets
# machines immediately, which is worth more here than the saving.
#
# 16-core machines: on-demand vCPU quota in us-east1 is 300, so 10 parallel tasks
# at 16 cores is 160 and fits. At 32 cores it would be 320 and would not.
set -euo pipefail

LOCATION="us-east1"
PROJECT="dgen-466702"
MACHINE="c2d-highcpu-16"
PROVISIONING="STANDARD"
CONFIG="batch_job_yamls/dgen-batch-job-netbilling-scan.yaml"
JOB="nb-scan-$(date -u +%Y%m%d-%H%M%S)"

if [ ! -f "${CONFIG}" ]; then
  echo "Run this from the repo root: ${CONFIG} not found." >&2
  exit 1
fi

# A leftover job from a previous attempt holds quota and confuses the log watch.
EXISTING=$(gcloud batch jobs list --location="${LOCATION}" --project="${PROJECT}" \
  --filter="name~nb-scan" --format="value(name.basename(),status.state)" 2>/dev/null || true)
if [ -n "${EXISTING}" ]; then
  echo "There are already nb-scan jobs present:"
  echo "${EXISTING}"
  echo "Delete them first, then re-run this script:"
  echo "  gcloud batch jobs delete <name> --location=${LOCATION} --project=${PROJECT} --quiet"
  exit 1
fi

echo "Submitting ${JOB}"
echo "  config:  ${CONFIG}"
echo "  machine: ${MACHINE} (${PROVISIONING})"
gcloud batch jobs submit "${JOB}" \
  --location="${LOCATION}" \
  --project="${PROJECT}" \
  --config="${CONFIG}" \
  --machine-type="${MACHINE}" \
  --provisioning-model="${PROVISIONING}"

echo
echo "Submitted. Watch it with:"
echo "  gcloud batch jobs describe ${JOB} --location=${LOCATION} --project=${PROJECT} --format='value(status.state)'"
echo "Results land in gs://dgen-assets/netbilling_scan/<STATE>/synapse_netbilling/"
