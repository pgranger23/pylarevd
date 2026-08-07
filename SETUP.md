# Setting up pylarevd

A LArSoft event display in pure python. At display time there is **no ROOT, no
art, no LArSoft and no compiled extension** — it reads art-ROOT files directly
with `uproot`.

If you are beta testing, please read [Reporting problems](#reporting-problems)
at the end: what to include makes the difference between a fixable report and a
guess.

---

## 1. Requirements

Python 3.10+, and:

| Package | For |
|---|---|
| `numpy`, `uproot` | reading the files (required) |
| `matplotlib` | static images (PNG/PDF/SVG) |
| `plotly` | interactive HTML |
| `dash` | the browser event browser (optional) |
| `fsspec-xrootd` + `XRootD` | reading `root://` / EOS files (optional) |

Nothing else. LArSoft is needed **only** to export a new detector geometry
(section 6), which most people never have to do.

**At any point, ask pylarevd what it has:**

```bash
python -m pylarevd --check
```

It lists every dependency, whether it is installed and at what version, what
each one buys you, the bundled geometries, and the exact command to install
anything that is absent. It is also the single most useful thing to paste into
a bug report.

---

## 2. Install

### Option A — a CERN or DUNE machine with CVMFS (nothing to install)

The LCG view already has everything including XRootD:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
git clone https://github.com/pgranger23/pylarevd.git
export PYTHONPATH=$PWD/pylarevd:$PYTHONPATH
```

Append `:$PYTHONPATH` rather than overwriting it — clobbering it hides the
view's own site-packages and nothing will import.

### Option B — your own machine

```bash
git clone https://github.com/pgranger23/pylarevd.git
cd pylarevd
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

That gives you a `pylarevd` command as well as the importable package.

Pick a narrower set if you prefer — the extras are:

| Install | Adds |
|---|---|
| `pip install -e .` | the display: `numpy`, `uproot`, `matplotlib`, `plotly` |
| `pip install -e ".[app]"` | **`dash`** — the interactive browser |
| `pip install -e ".[xrootd]"` | `fsspec-xrootd`, for `root://` and `/eos` files |
| `pip install -e ".[test]"` | `pytest`, `pillow` |
| `pip install -e ".[all]"` | all of the above |

Or, if you prefer a plain requirements file:

```bash
pip install -r requirements.txt
```

> The **`XRootD`** client itself is not pip-installed by any of these: it needs
> the XRootD C++ libraries and a compiler. The CVMFS LCG view already has it
> (Option A). Without it you can still read local files normally — only
> `root://` and `/eos` paths are affected.

### Check it works

```bash
python -m pylarevd --check
```

which should end with `everything pylarevd can use is installed.` — or tell you
exactly what to install and how.

---

## 3. First run

```bash
# what is in the file?
python -m pylarevd reco.root --list

# a static image of event 0
python -m pylarevd reco.root -e 0

# events 0-2, with truth overlaid, plus interactive HTML
python -m pylarevd reco.root -e 0-2 --truth --html -o out/
```

From python:

```python
from pylarevd import EventFile

f = EventFile("reco.root")        # geometry auto-detected from the file
ev = f[0]
print(ev.display().summary())

ev.display(truth=True).save("event0.png")
ev.display_3d().save("event0_3d.png")
ev.display_optical().save("event0_pds.png")

h = ev.hits()                     # numpy columns, physical coordinates attached
h.w, h.x, h.integral, h.view
bright = h[h.integral > 200]      # boolean-mask selection
```

Views: `display()` (2-D physical), `display(space="readout")` (channel/tick),
`display_3d()`, `display_optical()`, `display_flashes_3d()`.

---

## 4. The interactive browser — the main way to use this

It is a small web app. You run it **on the machine that holds the files**, and
view it **in the browser on your laptop** through an SSH tunnel. No data is
copied to your laptop and nothing renders there.

**Step 1 — on the machine with the files.** Leave this running; it prints the
tunnel command for you.

```bash
python -m pylarevd.app reco.root [more.root ...] --port 8050

# or start with nothing and open files from the browser:
python -m pylarevd.app --port 8050
```

**Step 2 — on your laptop, in a second terminal.** This prints nothing and does
not return: it is holding the tunnel open. Leave it running too.

```bash
ssh -N -L 8050:localhost:8050 <your-host>
```

**Step 3 — open <http://localhost:8050>.**

Use `localhost`, not the remote hostname — the tunnel makes the remote port look
like a local one. The server **binds to loopback only**, so it is not reachable
over the network; that is why the tunnel is required rather than a convenience.

If the port is taken, pass `--port 8051` and change *both* numbers in the tunnel
command to match.

Once it is up: step through events with ◀ ▶ or the **left/right arrow keys**,
switch between the five views, colour hits by charge or by owning object, toggle
truth/reco/radiological overlays, and change theme and colormap.

Every control is mirrored into the URL, so you can paste a link to exactly what
you are looking at. The link carries `rse=run:subrun:event`, which outranks the
entry number — so it still lands correctly when opened against a different
reconstruction pass of the same data.

Files on the command line are **optional**. With none, the app starts empty and
you open files from the **open by path** box — paste a full path and press
Enter. It joins the dropdown, so you can flip between several. A bare CERN EOS
path works too:

```
/eos/user/<u>/<user>/reco.root
```

which is rewritten to the right xrootd redirector when `/eos` is not mounted.

> Opening files by path lets the browser read anything your account can read.
> That is fine on localhost, which is why it is enabled there. If you bind
> elsewhere with `--host`, the box is disabled unless you also pass
> `--allow-remote-open`.

---

## 5. Reading remote files

Files on CERN EOS work without a mount:

```bash
python -m pylarevd "root://eosuser.cern.ch//eos/user/<u>/<user>/reco.root" -e 0
```

A bare `/eos/...` path is rewritten to the right redirector automatically when
`/eos` is not mounted, so you can paste a path from a listing and it just works.
You need a valid Kerberos ticket (`kinit <user>@CERN.CH`) and outbound access to
port 1094.

---

## 6. Geometry

Wire positions come from **LArSoft's own channel map**, exported once per
geometry into a small `.npz`. Two are bundled:

| File | Detector | Channels |
|---|---|---|
| `pylarevd/geom/dune10kt_v6_1x2x6.npz` | 1x2x6 workspace | 30 720 |
| `pylarevd/geom/dune10kt_v6_full.npz` | full 10 kt module | 384 000 |

**You do not normally choose one.** art files record the `Geometry` service
they were produced with, and that name is matched against the exported files.
If your sample needs a geometry that is not bundled you get a clear error
naming it, rather than a display full of NaN.

To export one you need LArSoft, inside the SL7 container:

```bash
./inlar.sh "bash shim/build_shim.sh"
./inlar.sh "python -m pylarevd.export_geometry \
    --fcl geom_dune10kt_1x2x6.fcl --out pylarevd/geom/<name>.npz"
```

The `.fcl` needs `Geometry` and `WireReadout` blocks matching your sample; copy
one of the two in the repo root and change the geometry it includes. The
detector `Name` in the fcl must match what the files record.

---

## 7. Running the tests

Sample files are not distributed. Point the suite at your own:

```bash
export PYLAREVD_ROCKMU=/path/to/a/cosmic_reco.root
export PYLAREVD_ATMNU=/path/to/a/neutrino_reco.root
python -m pytest tests/ -q
```

Without them the suite still runs and passes — the data-dependent tests skip
(33 pass, 103 skip, ~2 s). With both, expect 136 passing in ~3 min.

`PYLAREVD_ATMNU` should be a sample with neutrino truth, reconstruction
(Pandora tracks/showers) and optical data; `PYLAREVD_ROCKMU` a cosmic sample
with no neutrino.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `this file was produced with the '<name>' geometry, which is not exported` | Your sample needs a geometry that is not bundled. Export it (section 6), or pass `geometry=...` to use a bundled one anyway. |
| `! ... looks NOT disambiguated ... physical positions will be wrong` | You selected a raw hit collection (e.g. `gaushit`). In a wrapped-wire detector its hits sit on an arbitrary candidate wire — a median 3.9 cm from truth against 0.05 cm for a disambiguated one. Use `-t hitfd` or whatever your sample's disambiguated collection is called. |
| An event looks empty, or has very few hits | Often real: the interaction happened at or beyond the edge of the active volume and most of it escaped. The title says `[vertex OUTSIDE active volume]` and the block beneath gives the visible-energy fraction. |
| No truth block at all | The sample has no `simb::MCTruth` with a neutrino set — cosmic and radiological samples are like this. Expected, not a failure. |
| Static export of a 2-D interactive figure comes out empty | `write_image` on the plotly 2-D figures needs WebGL, which a headless machine may not have. Use `.save()` (matplotlib) for static output; the interactive HTML is fine in a real browser. |
| `no such file` for a `/eos/...` path | `/eos` is not mounted and you have no Kerberos ticket — run `kinit`. See section 5. |
| Browser shows `ERROR — ...` in the summary band | The message is the actual exception; a stale entry number after switching files is the usual cause. |
| `pylarevd: the interactive browser needs the 'dash' package` | `pip install "pylarevd[app]"`, or source the LCG view. Run `python -m pylarevd --check` to see everything at once. |
| Port 8050 already in use | `--port 8051`, and change **both** numbers in the `ssh -L` command to match. |
| Browser says "unable to connect" | The tunnel is not up, or you opened the remote hostname instead of `localhost:8050`. The `ssh -N -L` terminal must stay open. |
| Tunnel opens but the page is blank/spinning | The server is not running on the remote side any more — check the terminal from step 1. |

---

## Reporting problems

This is beta software and the decoder is doing something unusual — reading art's
member-wise serialisation directly. When something looks wrong, the most useful
report includes:

1. **What you ran** — the exact command or python, including the hit tag and view.
2. **The summary text**, not just the picture: `print(ev.display(...).summary())`.
   It carries the geometry name, hit counts per panel, the truth block, the
   containment line and any warnings.
3. **`python -m pylarevd <file> --list`** output, so I can see which products
   the file actually has.
4. **Whether it reproduces from a fresh `EventFile`** — some things are
   order-dependent, and knowing that narrows it enormously.
5. The file, or its path if I can reach it.

"The display is wrong" is hard to act on; "hits are 4 cm off the truth
trajectory in the V view on this file" is a bug I can find.
