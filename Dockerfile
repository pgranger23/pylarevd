# Container image of the pylarevd browser viewer (python -m pylarevd.web), for CERN PaaS
# (OpenShift) or any other container platform. See deploy/paas/README.md.
FROM registry.cern.ch/docker.io/library/python:3.12-slim

# git: the pylario dependency is installed from GitHub; libkrb5: XRootD's Kerberos login
# to EOS (the ticket itself comes from a sidecar, see deploy/paas/deployment.yaml).
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libkrb5-3 libgssapi-krb5-2 \
 && rm -rf /var/lib/apt/lists/*
COPY deploy/paas/krb5.conf /etc/krb5.conf

WORKDIR /opt/pylarevd
COPY pyproject.toml README.md ./
COPY pylarevd ./pylarevd
RUN pip install --no-cache-dir ".[xrootd]" xrootd

# OpenShift runs the container under an arbitrary uid with gid 0: HOME, matplotlib's
# config and the bundle cache must be writable by that group.
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib PYLAREVD_CACHE=/cache PYTHONUNBUFFERED=1
RUN mkdir -p /cache && chgrp -R 0 /cache && chmod -R g=u /cache
USER 1001
EXPOSE 8080
ENTRYPOINT ["python", "-m", "pylarevd.web", "--host", "0.0.0.0", "--port", "8080"]
