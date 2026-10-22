"""Trigger primitives / activities / candidates for the web viewer's 2-D panels.

A trigger primitive (TP) knows only its readout channel. In a wrapped-wire
detector an induction channel is bonded to several wire segments, so placing a
TP in the physical (w, x) view needs the same disambiguation a hit does:

* the channel has one wire segment in an active drift volume: that one;
* it has several: the segment of the disambiguated hit on the same channel at
  the same time, if there is one (``PLACE_HIT``);
* otherwise the TP has no physical position and is drawn in the readout
  (channel, tick) view only.

Collection channels always have a single segment. Activities (TAs) become boxes
around their primitives, candidates (TCs) time bands across every panel.
"""
from __future__ import annotations

import numpy as np

NO_PANEL = 255
#: how a primitive got its wire (``tp/place``)
PLACE_UNIQUE, PLACE_HIT, PLACE_AMBIGUOUS, PLACE_INACTIVE = 0, 1, 2, 3
PLACE_NAMES = ("unique wire", "matched hit", "ambiguous", "no active wire")
#: a hit matches a TP when its peak, less the plane's hit-TP offset, lies within
#: the TP's time over threshold widened by this many ticks on each side; the
#: nearest to the TP peak wins. The offset is measured per event and plane
#: (the TP peaks on the raw pulse, the hit on the deconvolved one: +10, +5 and
#: -2 ticks on U, V and Z in ProtoDUNE-HD MC, at every drift time).
MATCH_MARGIN = 10.0
#: hit-TP pairs further apart than this do not enter the offset estimate
OFFSET_WINDOW = 20.0
#: a TPC whose drift length is below this fraction of the longest one is the
#: thin volume behind an APA, not a place where tracks are reconstructed
ACTIVE_DRIFT_FRACTION = 0.25


def _f32(a):
    return np.asarray(a, np.float32)


def _active_tpcs(g) -> np.ndarray:
    z = g._z
    drift = np.asarray(z["t_xmax"]) - np.asarray(z["t_xmin"])
    keep = drift >= ACTIVE_DRIFT_FRACTION * drift.max()
    return np.asarray(z["t_tpc"])[keep]


def _disambiguated_hits(ev, hits):
    """A hit collection whose WireIDs resolve wrapped channels, or None."""
    g = ev.geometry
    if hits is not None and (not g.wrapped_channels or hits.channels_split_across_wires):
        return hits
    for product in ev._src.hit_products():
        tag = product.split("_")[1]
        try:
            h = ev.hits(tag)
        except Exception:
            continue
        if h.channels_split_across_wires:
            return h
    return None


def hit_tp_offsets(tr, h, chan_view) -> dict:
    """Median (hit peak - nearest TP peak on the same channel) per view, in ticks."""
    if not tr.n_tp or h is None or not len(h):
        return {}
    order = np.lexsort((tr.tp_tick_peak, tr.tp_channel))
    key = tr.tp_channel[order].astype(np.int64) * 1e6 + tr.tp_tick_peak[order]
    hk = np.asarray(h.channel, np.int64) * 1e6 + np.asarray(h.tick, np.float64)
    j = np.clip(np.searchsorted(key, hk), 1, len(key) - 1)
    d = np.where(np.abs(key[j] - hk) < np.abs(key[j - 1] - hk), hk - key[j], hk - key[j - 1])
    ok = np.abs(d) < OFFSET_WINDOW
    out = {}
    for v in np.unique(chan_view):
        m = ok & (chan_view == v)
        if m.sum() >= 50:
            out[int(v)] = float(np.median(d[m]))
    return out


def place_primitives(ev, tr, hits):
    """Wire-table row of every TP (-1 if none) and how it was chosen."""
    g = ev.geometry
    n = tr.n_tp
    row = np.full(n, -1, np.int64)
    place = np.full(n, PLACE_INACTIVE, np.uint8)
    if not n:
        return row, place, None, {}

    # candidate segments per channel, active volumes only
    seg = np.flatnonzero(np.isin(g.w_tpc, _active_tpcs(g)))
    seg = seg[np.argsort(g.w_chan[seg], kind="stable")]
    seg_chan = g.w_chan[seg].astype(np.int64)
    ch = tr.tp_channel.astype(np.int64)
    lo = np.searchsorted(seg_chan, ch, "left")
    hi = np.searchsorted(seg_chan, ch, "right")
    k = hi - lo
    one = k == 1
    row[one] = seg[lo[one]]
    place[one] = PLACE_UNIQUE
    place[k > 1] = PLACE_AMBIGUOUS

    amb = np.flatnonzero(k > 1)
    h = _disambiguated_hits(ev, hits) if len(amb) else None
    off = {}
    if h is not None and len(h):
        off = hit_tp_offsets(tr, h, np.asarray(h.chan_view))
        tp_view = g.channel_view[np.clip(ch, 0, len(g.channel_view) - 1)]
        shift = np.array([off.get(int(v), 0.0) for v in tp_view[amb]])
        hrow = g.wire_rows(h.cryo, h.tpc, h.plane, h.wire)
        ok = (hrow >= 0) & (np.asarray(h.channel) >= 0)
        hc = np.asarray(h.channel, np.int64)[ok]
        ht = np.asarray(h.tick, np.float64)[ok]
        hrow = hrow[ok]
        order = np.lexsort((ht, hc))
        hc, ht, hrow = hc[order], ht[order], hrow[order]
        t0 = tr.tp_tick_start[amb] + shift - MATCH_MARGIN
        t1 = tr.tp_tick_start[amb] + tr.tp_tot[amb] + shift + MATCH_MARGIN
        # channel and tick packed into one sortable key (ticks < 1e6)
        key = hc * 1e6 + ht
        a = np.searchsorted(key, ch[amb] * 1e6 + t0, "left")
        b = np.searchsorted(key, ch[amb] * 1e6 + t1, "right")
        want = tr.tp_tick_peak[amb] + shift
        for i in np.flatnonzero(b > a):
            j = a[i] + int(np.argmin(np.abs(ht[a[i]:b[i]] - want[i])))
            r = hrow[j]
            if g.w_chan[r] == ch[amb[i]] and np.isin(g.w_tpc[r], _active_tpcs(g)):
                row[amb[i]] = r
                place[amb[i]] = PLACE_HIT
    return row, place, h, off


def export_trigger(ev, hits, displays, arrays, meta):
    """Fill ``tp/*``, ``ta/*``, ``tc/*`` and their 2-D boxes; ``displays`` is
    ``[(suffix, EventDisplay), ...]`` for the wire-space panel groupings."""
    from pylario.event import TA_ALGORITHMS, TC_ALGORITHMS, TC_TYPES, TP_ALGORITHMS
    tr = ev.trigger()
    g = ev.geometry
    row, place, mhits, offsets = place_primitives(ev, tr, hits)
    placed = row >= 0
    rr = np.maximum(row, 0)
    prow = np.where(placed, g.plane_rows(g.w_cryo[rr], g.w_tpc[rr], g.w_plane[rr]), -1)
    w = np.where(placed, g.wire_coord[rr], np.nan)
    x = np.where(prow >= 0, g.ticks_to_x(tr.tp_tick_peak, prow), np.nan)
    pr = np.maximum(prow, 0)
    dvdt = np.where(prow >= 0, g.p_slope[pr], np.nan)
    view = np.where(prow >= 0, g.p_view[pr], -1)
    dsign = np.where(prow >= 0, g.drift_sign[pr], 0)
    keys = {"orientation": np.where(prow >= 0, g.orientation_key[pr], 999),
            "view": view, "none": g.panel_of(view, dsign)}

    arrays["tp/channel"] = tr.tp_channel.astype(np.uint32)
    arrays["tp/tick"] = _f32(tr.tp_tick_peak)
    arrays["tp/tick0"] = _f32(tr.tp_tick_start)
    arrays["tp/tot"] = _f32(tr.tp_tot)
    arrays["tp/adc"] = _f32(tr.tp_adc_integral)
    arrays["tp/peak"] = _f32(tr.tp_adc_peak)
    arrays["tp/ta"] = tr.tp_ta.astype(np.int32)
    arrays["tp/chan_view"] = np.asarray(
        g.channel_view[np.clip(tr.tp_channel, 0, len(g.channel_view) - 1)], np.int8)
    arrays["tp/h"] = _f32(w)
    arrays["tp/v"] = _f32(x)
    arrays["tp/dvdt"] = _f32(dvdt)
    arrays["tp/place"] = place

    arrays["ta/tick0"] = _f32(tr.ta_tick_start)
    arrays["ta/tick1"] = _f32(tr.ta_tick_end)
    arrays["ta/tick_peak"] = _f32(tr.ta_tick_peak)
    arrays["ta/adc"] = _f32(tr.ta_adc_integral)
    arrays["ta/adc_peak"] = _f32(tr.ta_adc_peak)
    arrays["ta/ch_peak"] = tr.ta_channel_peak.astype(np.int32)
    arrays["ta/alg"] = tr.ta_algorithm.astype(np.int32)
    arrays["ta/tc"] = tr.ta_tc.astype(np.int32)
    arrays["ta/n_tp"] = np.array([len(t) for t in tr.ta_tps], np.uint32)
    arrays["tc/tick0"] = _f32(tr.tc_tick_start)
    arrays["tc/tick1"] = _f32(tr.tc_tick_end)
    arrays["tc/tick"] = _f32(tr.tc_tick_candidate)
    arrays["tc/type"] = tr.tc_type.astype(np.int32)
    arrays["tc/alg"] = tr.tc_algorithm.astype(np.int32)
    arrays["tc/n_ta"] = np.array([len(a) for a in tr.tc_tas], np.uint32)

    # readout boxes of the activities: one per (activity, plane) its TPs touch
    chv = arrays["tp/chan_view"]
    rbox, rpan, ridx = [], [], []
    for i, tps in enumerate(tr.ta_tps):
        tps = tps[(tps >= 0) & (tps < tr.n_tp)]
        for v in np.unique(chv[tps]):
            if v < 0:
                continue
            c = tr.tp_channel[tps[chv[tps] == v]]
            rbox.append([c.min() - 0.5, c.max() + 0.5, tr.ta_tick_start[i], tr.ta_tick_end[i]])
            rpan.append(v)
            ridx.append(i)
    arrays["ta_ro/box"] = _f32(np.reshape(rbox, (-1, 4)))
    arrays["ta_ro/panel"] = np.asarray(rpan, np.uint8)
    arrays["ta_ro/index"] = np.asarray(ridx, np.int32)

    wire_prow = None                    # readout plane of every wire, when needed
    active = set(_active_tpcs(g).tolist())
    drift_range = {b.tpc: (b.xmin, b.xmax) for b in g.tpcs if b.tpc in active}
    for sf, d in displays:
        pidx = np.full(tr.n_tp, NO_PANEL, np.uint8)
        k_of = keys[d.merge]
        for k, p in enumerate(d.panels):
            pidx[placed & (k_of == p.key)] = k
        arrays[f"tp/panel{sf}"] = pidx

        # activities: a box per (activity, panel, readout plane) around its TPs,
        # spanning the activity's time window in that plane's drift coordinate
        box, pan, idx = [], [], []
        for i, tps in enumerate(tr.ta_tps):
            tps = tps[(tps >= 0) & (tps < tr.n_tp)]
            tps = tps[pidx[tps] != NO_PANEL]
            for k, r in {(int(pidx[t]), int(prow[t])) for t in tps}:
                sel = tps[(pidx[tps] == k) & (prow[tps] == r)]
                v0, v1 = g.ticks_to_x(np.array([tr.ta_tick_start[i], tr.ta_tick_end[i]]),
                                      np.array([r, r]))
                box.append([w[sel].min(), w[sel].max(), min(v0, v1), max(v0, v1)])
                pan.append(k)
                idx.append(i)
        arrays[f"ta2d{sf}/box"] = _f32(np.reshape(box, (-1, 4)))
        arrays[f"ta2d{sf}/panel"] = np.asarray(pan, np.uint8)
        arrays[f"ta2d{sf}/index"] = np.asarray(idx, np.int32)

        # candidates: the time window in every active readout plane of every panel,
        # clipped to that volume's drift range; planes side by side along the
        # wires (same drift range) share one band
        bands: dict = {}
        if tr.n_tc:
            if wire_prow is None:
                wire_prow = g.plane_rows(g.w_cryo, g.w_tpc, g.w_plane)
            for k, p in enumerate(d.panels):
                hp = d.hits
                rows = set(np.unique(g.plane_rows(hp.cryo[p.mask], hp.tpc[p.mask],
                                                  hp.plane[p.mask])).tolist())
                rows |= set(np.unique(prow[pidx == k]).tolist())
                for r in sorted(rows - {-1}):
                    if int(g.p_tpc[r]) not in drift_range:
                        continue
                    xlo, xhi = drift_range[int(g.p_tpc[r])]
                    wr = g.wire_coord[wire_prow == r]
                    wr = wr[np.isfinite(wr)]
                    if not len(wr):
                        continue
                    for j in range(tr.n_tc):
                        t = np.array([tr.tc_tick_start[j], tr.tc_tick_end[j],
                                      tr.tc_tick_candidate[j]])
                        v = g.ticks_to_x(t, np.full(3, r))
                        v0, v1 = max(min(v[0], v[1]), xlo), min(max(v[0], v[1]), xhi)
                        if v0 >= v1:
                            continue
                        vc = v[2] if xlo <= v[2] <= xhi else np.nan
                        key = (k, j, round(v0, 2), round(v1, 2))
                        b = bands.get(key)
                        bands[key] = ([wr.min(), wr.max(), v0, v1, vc] if b is None else
                                      [min(b[0], wr.min()), max(b[1], wr.max()), v0, v1, vc])
        rows5 = np.reshape(list(bands.values()), (-1, 5))
        arrays[f"tc2d{sf}/box"] = _f32(rows5[:, :4])
        arrays[f"tc2d{sf}/vc"] = _f32(rows5[:, 4])
        arrays[f"tc2d{sf}/panel"] = np.asarray([k[0] for k in bands], np.uint8)
        arrays[f"tc2d{sf}/index"] = np.asarray([k[1] for k in bands], np.int32)

    counts = {PLACE_NAMES[c]: int((place == c).sum()) for c in range(len(PLACE_NAMES))}
    name = lambda table, c: table.get(int(c), f"code {int(c)}")
    meta["trigger"] = {
        "label": tr.label, "n_tp": tr.n_tp, "n_ta": tr.n_ta, "n_tc": tr.n_tc,
        "ticks_per_dts": tr.ticks_per_dts, "placement": counts,
        "match_hits": mhits.product if mhits is not None else None,
        "hit_tp_offset": {"UVZ"[v] if 0 <= v < 3 else str(v): round(o, 2)
                          for v, o in offsets.items()},
        "tp_algorithms": sorted({name(TP_ALGORITHMS, a) for a in tr.tp_algorithm}),
        "ta_algorithm_names": [name(TA_ALGORITHMS, a) for a in tr.ta_algorithm],
        "tc_type_names": [name(TC_TYPES, t) for t in tr.tc_type],
        "tc_algorithm_names": [name(TC_ALGORITHMS, a) for a in tr.tc_algorithm],
    }
    return summary_lines(tr, counts, mhits)


def summary_lines(tr, counts, mhits) -> list[str]:
    from pylario.event import TA_ALGORITHMS, TC_TYPES
    out = [f"trigger: {tr.n_tp:,} TPs, {tr.n_ta} TAs, {tr.n_tc} TCs "
           f"({tr.label.split(' + ')[0].split('_')[1]})"]
    n_amb = counts[PLACE_NAMES[PLACE_AMBIGUOUS]]
    out.append(f"  TP wires: {counts[PLACE_NAMES[PLACE_UNIQUE]]:,} unique, "
               f"{counts[PLACE_NAMES[PLACE_HIT]]:,} from "
               f"{mhits.product.split('_')[1] if mhits is not None else 'hits'}, "
               f"{n_amb:,} ambiguous (readout view only)")
    for j in range(tr.n_tc):
        out.append(f"  TC {j}: {TC_TYPES.get(int(tr.tc_type[j]), tr.tc_type[j])} "
                   f"at tick {tr.tc_tick_candidate[j]:.0f}, window "
                   f"{tr.tc_tick_start[j]:.0f}-{tr.tc_tick_end[j]:.0f}, "
                   f"{len(tr.tc_tas[j])} TA")
    for i in range(tr.n_ta):
        out.append(f"  TA {i}: {TA_ALGORITHMS.get(int(tr.ta_algorithm[i]), tr.ta_algorithm[i])}, "
                   f"{len(tr.ta_tps[i])} TPs, ADC sum {tr.ta_adc_integral[i]:.3g}, "
                   f"ticks {tr.ta_tick_start[i]:.0f}-{tr.ta_tick_end[i]:.0f}")
    return out
