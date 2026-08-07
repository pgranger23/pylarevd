#!/bin/bash
# Run a command inside the SL7 container with dunesw set up.
#
# Only needed to export a detector geometry (see SETUP.md section 6) or to
# rebuild the shim -- never to display anything.
#
#   ./inlar.sh "bash shim/build_shim.sh"
#   ./inlar.sh "python -m pylarevd.export_geometry --fcl <geo.fcl> --out <out.npz>"
set -euo pipefail
DUNESW_VER=${DUNESW_VER:-v10_22_00d00}
DUNESW_QUAL=${DUNESW_QUAL:-e26:prof}
IMAGE=${IMAGE:-/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl7:latest}
WORKDIR=${WORKDIR:-$PWD}

exec apptainer exec -B /cvmfs -B /tmp -B /var/tmp -B "$HOME" -B "$WORKDIR" \
  --env HTTP_PROXY="${HTTP_PROXY:-}" --env HTTPS_PROXY="${HTTPS_PROXY:-}" \
  --env http_proxy="${HTTP_PROXY:-}" --env https_proxy="${HTTPS_PROXY:-}" \
  --env KRB5CCNAME="${KRB5CCNAME:-}" \
  "$IMAGE" \
  bash -c "source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh >/dev/null 2>&1
           setup dunesw $DUNESW_VER -q $DUNESW_QUAL >/dev/null 2>&1 || { echo 'dunesw setup failed'; exit 9; }
           cd '$WORKDIR'
           $*"
