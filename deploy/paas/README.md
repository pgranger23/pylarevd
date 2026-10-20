# Running the viewer on CERN PaaS (OpenShift)

The viewer (`python -m pylarevd.web`) runs as one pod, with the CERN SSO proxy in front of it.

```
browser ── https://pylarevd.web.cern.ch ── SSO proxy (Helm chart cern-auth-proxy)
                                               │  only logged-in users with the allowed role
                                               ▼
                                      Service pylarevd:8080 ── viewer pod
                                                               ├─ /cache  CephFS bundle cache
                                                               ├─ /data   CephFS copies (optional)
                                                               └─ root://eos...  over xrootd,
                                                                  Kerberos ticket from a sidecar
```

## One-time setup

Log in (2FA, device flow):

```sh
oc sso-login paas
oc project pylarevd
```

Create the image build and the volumes, build from this checkout, and start the viewer:

```sh
oc apply -f deploy/paas/build.yaml -f deploy/paas/storage.yaml
oc start-build pylarevd --from-dir=. --follow
oc apply -f deploy/paas/app.yaml
```

Then put the SSO proxy in front of it:

```sh
helm repo add paas https://gitlab.cern.ch/api/v4/projects/124069/packages/helm/master
helm install sso paas/cern-auth-proxy -f deploy/paas/sso-values.yaml
```

Who may log in is set on the Application Portal (link in the project's
`lifecycle.webservices.cern.ch/applicationPortalLink` annotation). By default `default-role` is
every CERN and eduGAIN account; restrict it to an e-group there. `route.public` in
`sso-values.yaml` decides whether the site is reachable from outside the CERN network.

## Data

The `args` of the viewer container in `app.yaml` list what it serves: files or directories,
local or `root://` URLs (a URL ending in `/` stands for the `.root` files inside it).
`--allow-open-under` limits what the "open" box can reach.

- **Copies in the cluster:** mount `pylarevd-data` read-write in a temporary pod and `oc cp`
  the files to `/data`.
- **EOS over xrootd:** create the secret of a *service* account that can read the files, then add
  the ticket sidecar:

  ```sh
  oc create secret generic eos-credentials \
      --from-literal=KEYTAB_USER=<service account> --from-literal=KEYTAB_PWD=<password>
  oc patch deployment pylarevd --type=strategic --patch-file deploy/paas/eos-kerberos.yaml
  ```

  and list `root://eosuser.cern.ch//eos/...` paths in `app.yaml`.

## Limits

The project quota caps the *requested* resources at 1 CPU and 4 GiB in total, so the viewer
requests 0.6 CPU / 3 GiB and bursts to 6 CPU / 16 GiB. Each decode worker holds about 2 GB; keep
`--workers` at 4 or below. The route timeout is raised to 180 s for cold decodes and GIF exports.

## Updating

```sh
oc start-build pylarevd --from-dir=. --follow     # the deployment restarts on the new image
```

The bundle cache survives restarts; `BUNDLE_VERSION` in `pylarevd/web/cache.py` invalidates it.
