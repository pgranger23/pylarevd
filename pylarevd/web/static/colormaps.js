// colormaps.js -- colour tables and theme palettes for the web viewer.
// Colormap LUTs were generated once from matplotlib; the palettes below mirror
// pylarevd/theme.py (DARK / LIGHT). Everything is plain sRGB; the renderer runs
// with ColorManagement disabled so these values reach the screen unchanged.
import * as THREE from './vendor/three.module.js';

export const LUT_B64 = {
 "turbo": "MBI7MhVDMxhKNBtRNR5YNiFfNyRmOCdtOSpzOi15Oy+APDKGPTWLPjiRPzuXPz6cQECiQUOnQUasQkmxQku1Q066RFG/RFTDRFbHRVnLRVzPRV7TRmHWRmTaRmbdRmngRmvjR27mR3HpR3PrR3buR3jwR3vyRn30RoD2RoL4RoX6Rof7RYr8RYz9RI/+Q5H+QpT/QZb/QJn/Ppv+PZ7+O6D9OqP8OKX7N6j6Nav4M633Ma/1L7L0LrTyLLfwKrnuKLzrJ77pJcDnI8PkIsXiIMffH8ndHsvaHM3YG9DVGtLSGtTQGdXNGNfKGNnIGNvFGN3CGN7AGOC9GeK7GeO5GuS2HOa0HeeyH+mvIOqsIuuqJeynJ+6kKu+hLPCeL/GbMvKYNfOUOPSRPPWOP/aKQ/eHRviESviATvl9Uvp6Vfp2WftzXfxvYfxsZf1paf1mbf5icf5fdf5cef5Zff9WgP9ThP9RiP9Oi/9Lj/9Jkv9Hlv5Emf5CnP5An/0/of09pPw8p/w6qfs5rPs4r/o3sfk2tPg2t/c1ufY1vPU0vvQ0wfM0w/E0xvA0yO80y+00zew00Oo00uk11Oc11+U12eQ22+I23eA339834d0349s45dk459c56dU569M57NE67s8678068cs68sk69Mc69cU69sM698E6+L45+bw5+ro5+7g4+7Y3/LM2/LE2/a41/aw0/qkz/qcy/qQx/qEw/p4v/pst/pks/pYr/pMq/pAp/Y0n/Yom/Icl/IQj+4Ei+34h+nsf+Xge+XUd+HIc928a9mwZ9WkY9GYX82MV8mAU8V0T8FsS71gR7VUQ7FMP61AO6k4N6EsM50kM5UcL5EUK4kMK4UEJ3z8I3T0I3DsH2jkH2DcG1jUG1DMF0jEF0C8Fzi0EzCsEyioEyCgDxSYDwyUDwSMCviECvCACuR4Ctx0CtBsBshoBrxgBrBcBqRYBpxQBpBMBoRIBnhABmw8BmA4BlQ0BkgsBjgoBiwkCiAgChQcCgQYCfgUCegQD",
 "magma": "AAAEAQAFAQEGAQEIAgEJAgILAgINAwMPAwMSBAQUBQQWBgUYBgUaBwYcCAceCQcgCggiCwkkDAkmDQopDgsrEAstEQwvEg0xEw00FA42FQ44Fg87GA89GRA/GhBCHBBEHRFHHhFJIBFLIRFOIhFQJBJTJRJVJxJYKRFaKhFcLBFfLRFhLxFjMRFlMxBnNBBpNhBrOBBsOQ9uOw9wPQ9xPw9yQA90Qg91RA92RRB3RxB4SRB4ShB5TBF6ThF7TxJ7URJ8UhN8VBN9VhR9VxV+WRV+WhZ+XBZ/XRd/Xxh/YBiAYhmAZBqAZRqAZxuAaByBahyBax2BbR2Bbh6BcB+Bch+BcyCBdSGBdiGBeCKBeSKCeyOCfCOCfiSCgCWCgSWBgyaBhCaBhieBiCeBiSiBiymBjCmBjiqBkCqBkSuBkyuAlCyAliyAmC2AmS2Amy5/nC5/ni9/oC9/oTB+ozB+pTF+pjF9qDJ9qjN9qzN8rTR8rjR7sDV7sjV7szZ6tTZ6tzd5uDd5ujh4vDl4vTl3vzp3wDp2wjt1xDx1xTx0xz1zyD5zyj5yzD9xzUBxz0Bw0EFv0kJv00Nu1URt1kVs2EVs2UZr20dq3Ehp3klo30po4Exn4k1m405l5E9k5VBk51Jj6FNi6VRi6lZh61dg7Fhg7Vpf7lte711e8F9e8WBd8mJd8mRc82Vc9Gdc9Glc9Wtc9mxc9m5c93Bc93Jc+HRc+HZc+Xhd+Xld+Xtd+n1e+n9e+oFf+4Nf+4Vg+4dh/Ilh/Ipi/Ixj/I5k/JBl/ZJm/ZRn/ZZo/Zhp/Zpq/Ztr/p1s/p9t/qFu/qNv/qVx/qdy/qlz/qp0/qx2/q53/rB4/rJ6/rR7/rZ8/rd+/rl//ruB/r2C/r+E/sGF/sKH/sSI/saK/siM/sqN/syP/s2Q/s+S/tGU/tOV/tWX/teZ/tia/dqc/dye/d6g/eCh/eKj/eOl/eWn/eep/emq/eus/Oyu/O6w/PCy/PK0/PS2/Pa4/Pe5/Pm7/Pu9/P2/",
 "viridis": "RAFURAJWRQRXRQVZRgdaRghcRgpdRgteRw1gRw5hRxBjRxFkRxNlSBRnSBZoSBdpSBhqSBpsSBttSBxuSB1vSB9wSCBxSCFzSCN0SCR1SCV2SCZ3SCh4SCl5Ryp6Ryx6Ry17Ry58Ry99RjB+RjJ+RjN/RjSARTWBRTeBRTiCRDmDRDqDRDuEQz2EQz6FQj+FQkCGQkGGQUKHQUSHQEWIQEaIP0eIP0iJPkmJPkqJPkyKPU2KPU6KPE+KPFCLO1GLO1KLOlOLOlSMOVWMOVaMOFiMOFmMN1qMN1uNNlyNNl2NNV6NNV+NNGCNNGGNM2KNM2ONMmSOMmWOMWaOMWeOMWiOMGmOMGqOL2uOL2yOLm2OLm6OLm+OLXCOLXGOLHGOLHKOLHOOK3SOK3WOKnaOKneOKniOKXmOKXqOKXuOKHyOKH2OJ36OJ3+OJ4COJoGOJoKOJoKOJYOOJYSOJYWOJIaOJIeOI4iOI4mOI4qNIouNIoyNIo2NIY6NIY+NIZCNIZGMIJKMIJKMIJOMH5SMH5WLH5aLH5eLH5iLH5mKH5qKHpuKHpyJHp2JH56JH5+IH6CIH6GIH6GHH6KHIKOGIKSGIaWFIaaFIqeFIqiEI6mDJKqDJauCJayCJq2BJ62BKK6AKa9/KrB/LLF+LbJ9LrN8L7R8MbV7MrZ6NLZ5Nbd5N7h4OLl3Orp2O7t1Pbx0P7xzQL1yQr5xRL9wRsBvSMFuSsFtTMJsTsNrUMRqUsVpVMVoVsZnWMdlWshkXMhjXsliYMpgY8tfZcteZ8xcac1bbM1abs5YcM9Xc9BWddBUd9FTetFRfNJQf9NOgdNNhNRLhtVJidVIi9ZGjtZFkNdDk9dBldhAmNg+m9k8ndk7oNo5oto3pds2qNs0qtwyrdwwsN0vst0ttd4ruN4put4ovd8mwN8lwt8jxeAhyOAgyuEfzeEd0OEc0uIb1eIa2OIZ2uMZ3eMY3+MY4uQY5eQZ5+QZ6uUa7OUb7+Uc8eUd9OYe9uYg+OYh++cj/ecl",
 "cividis": "ACJOACNPACRRACVTACVUACZWACdYAChZAChbACldACpfACphACtiACxkACxmAC1oAC5qAC5sAC9tADBvADBwADFwADFxATJxBTNxCDNwDDRwDzVwEjVwFDZwFjdwGDdvGjhvHDlvHjpvIDpvITtuIzxuJDxuJj1uJz5uKT9uKj9tK0BtLUFtLkFtL0JtMUNtMkNtM0RtNEVsNUVsNkZsOEdsOUhsOkhsO0lsPEpsPUpsPktsP0xsQExsQU1sQk5sQ05sRE9sRVBsRlFsR1FsSFJsSVNsSlNsS1RsTFVsTVVsTlZsT1dsUFdsUVhtUlltU1ptVFptVVttVVxtVlxtV11tWF5tWV5uWl9uW2BuXGFuXWFuXmJuXmNvX2NvYGRvYWVvYmVvY2ZwZGdwZWhwZWhwZmlwZ2pxaGpxaWtxamxxa21ybG1ybG5ybW9ybm9zb3BzcHFzcXJ0cnJ0cnN0c3R1dHR1dXV1dnZ2d3d2d3d3eHh3eXl3enp4e3p4fHt4fXx4fnx4fn14f354gH94gX94goB5g4F5hIJ5hYJ5hoN5h4R4iIV4iYV4ioZ4i4d4jIh4jYh4jol4j4p4kIt4kYt4kox4ko14k454lI53lY93lpB3l5F3mJJ3mZJ3mpN2m5R2nJV2nZV2npZ2n5d1oJh1oZl1opl1o5p0pJt0pZx0ppx0p51zqJ5zqZ9zqqBzq6ByrKFyraJyrqNxr6RxsKVxsaVws6ZwtKdvtahvtqlvt6luuKpuuattuqxtu61tvK5sva5svq9rv7BrwLFqwbJqwrNpw7NpxLRoxbVoxrZnx7dnyLhmyblly7llzLpkzbtjzrxjz71i0L5i0b9h0sBg08Bf1MFf1cJe1sNd18Rc2cVc2sZb28da3MhZ3chY3slY38pX4MtW4cxV4s1U5M5T5c9S5tBR59FQ6NJP6dNO6tNM69RL7dVK7tZJ79dI8NhG8dlF8tpE89tC9dxB9t0/994++N88+eA6++E4/OI2/eM0/uQ0/uU1/uY2/ug4",
 "inferno": "AAAEAQAFAQEGAQEIAgEKAgIMAgIOAwIQBAMSBAMUBQQXBgQZBwUbCAUdCQYfCgciCwckDAgmDQgpDgkrEAktEQowEgoyFAs0FQs3Fgs5GAw8GQw+GwxBHAxDHgxFHwxIIQxKIwxMJAxPJgxRKAtTKQtVKwtXLQtZLwpbMQpcMgpeNApfNglhOAliOQljOwlkPQllPglmQApnQgpoRApoRQppRwtqSQtqSgxrTAxrTQ1sTw1sUQ5sUg5tVA9tVQ9tVxBuWRBuWhFuXBJuXRJuXxNuYRNuYhRuZBVuZRVuZxZuaRZuahdubBhubRhubxlucRluchpudBpudRtudxxteBxteh1tfB1tfR5tfx5sgB9sgiBshCBrhSFrhyFriCJqiiJqjCNpjSNpjyRpkCVokiVokyZnlSZnlydmmCdmmihlmylknSlknypjoCpjoitioyxhpSxgpi1gqC5fqS5eqy9erTBdrjBcsDFbsTJaszJatDNZtjRYtzVXuTVWujZVvDdUvThTvzlSwDpRwTpQwztPxDxOxj1Nxz5MyD9LykBKy0FJzEJIzkNHz0RG0EVF0kZE00dD1EhC1UpB10s/2Ew+2U092k4821A73VE63lI431M34FU24VY14lc041kz5Fox5Vww5l0v514u6GAt6WEr6mMq62Qp62Yo7Gcm7Wkl7mok72wj724h8G8g8XEf8XMd8nQc83Yb83gZ9HkY9XsX9X0V9n4U9oAT94IS94QQ+IUP+IcO+IkM+YsL+YwK+Y4J+pAI+pIH+pQH+5YG+5cG+5kG+5sG+50H/J8H/KEI/KMJ/KUK/KYM/KgN/KoP/KwR/K4S/LAU/LIW/LQY+7Ya+7gd+7of+7wh+74j+sAm+sIo+sQq+sYt+ccv+cky+cs1+M03+M8699E999NA9tVD9tdG9dlJ9dtM9N1P9N9T9OFW8+Na8+Vd8uZh8uhl8upp8ext8e1x8e918fF58vJ98vSC8/WG8/aK9PiO9fmS9vqW+Pua+fyd+v2h/P+k",
 "plasma": "DQiHEAeIEweJFgeKGQaMGwaNHQaOIAaPIgaQJAaRJgWRKAWSKgWTLAWULgWVLwWWMQWXMwWXNQSYNwSZOASaOgSaPASbPgScPwScQQSdQwOeRAOeRgOfSAOfSQOgSwOhTAKhTgKiUAKiUQKjUwKjVQKkVgGkWAGkWQGlWwGlXAGmXgGmYAGmYQCnYwCnZACnZgCnZwCoaQCoagCobACobgCobwCocQCocgGodAGodQGodwGoeAGoegKoewKofQOofgOogASogQSngwWnhAWnhgamhwemiAimigmliwqljQuljgykjw2kkQ6jkg+jlBCilRGhlhOhmBSgmRWfmhafnBeenRidnhmdoBqcoRuboh2aox6apR+ZpiCYpyGXqCKWqiOVqySUrCaUrSeTriiSsCmRsSqQsiuPsyyOtC6NtS+MtjCLtzGKuDKJujOIuzSIvDWHvTeGvjiFvzmEwDqDwTuCwjyBwz2AxD5/xUB+xkF9x0J8yEN7yUR6ykV6y0Z5zEd4zEl3zUp2zkt1z0x00E1z0U5y0k9x01Fx1FJw1VNv1VRu1lVt11Zs2Fdr2Vhq2lpq2ltp21xo3F1n3V5m3l9l3mFk32Jj4GNj4WRi4mVh4mZg42hf5Gle5Wpd5Wtd5mxc525b529a6HBZ6XFY6XJX6nRX63VW63ZV7HdU7XlT7XpS7ntR73xR735Q8H9P8IBO8YFN8YNM8oRL84VL84dK9IhJ9IlI9YtH9YxG9o1F9o9E95BE95FD95NC+JRB+JVA+Zc/+Zg++Zo++ps9+pw8+p47+586+6E5+6I4/KM4/KU3/KY2/Kg1/Kk0/asz/awz/a4y/a8x/bEw/bIv/bQv/bUu/rct/rgs/ros/rsr/r0q/r4q/sAp/cIp/cMo/cUn/cYn/cgn/com/csm/M0l/M4l/NAl/NIl+9Mk+9Uk+9ck+tgk+tok+dwk+d0l+N8l+OEl9+Il9+Ql9uYm9ugm9ekm9esn9O0n8+4n8/An8vIn8fQm8fUl8Pck8Pkh"
};

export const CMAP_NAMES = ['turbo', 'magma', 'viridis', 'cividis', 'inferno', 'plasma'];

const _lutCache = new Map();

/** 256*3 Uint8Array of RGB for the named colormap. */
export function getLut(name) {
  let lut = _lutCache.get(name);
  if (!lut) {
    const b64 = LUT_B64[name] || LUT_B64.turbo;
    const bin = atob(b64);
    lut = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) lut[i] = bin.charCodeAt(i);
    _lutCache.set(name, lut);
  }
  return lut;
}

/** [r,g,b] in 0..1 at t in [0,1]. Allocates; for DOM/UI use, not render loops. */
export function sampleCmap(name, t) {
  const lut = getLut(name);
  const i = Math.max(0, Math.min(255, Math.round(t * 255)));
  return [lut[3 * i] / 255, lut[3 * i + 1] / 255, lut[3 * i + 2] / 255];
}

/** 256x1 RGBA DataTexture for a colormap (filled in place by updateLutTexture). */
export function makeLutTexture(name) {
  const data = new Uint8Array(256 * 4);
  const tex = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  updateLutTexture(tex, name);
  return tex;
}

export function updateLutTexture(tex, name) {
  const lut = getLut(name);
  const d = tex.image.data;
  for (let i = 0; i < 256; i++) {
    d[4 * i] = lut[3 * i]; d[4 * i + 1] = lut[3 * i + 1]; d[4 * i + 2] = lut[3 * i + 2]; d[4 * i + 3] = 255;
  }
  tex.needsUpdate = true;
}

/** CSS linear-gradient string for a colormap (used by colourbars). */
export function cmapGradientCss(name, lo = 0, hi = 1) {
  const stops = [];
  for (let k = 0; k <= 8; k++) {
    const t = lo + (hi - lo) * k / 8;
    const [r, g, b] = sampleCmap(name, t);
    stops.push(`rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${(k * 12.5)}%`);
  }
  return `linear-gradient(to right, ${stops.join(',')})`;
}

export function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}

// ---- themes (pylarevd/theme.py) -------------------------------------------
export const THEMES = {
  dark: {
    name: 'dark', fig_bg: '#0f172a', axes_bg: '#020617', legend_bg: '#1e293b', legend_edge: '#475569',
    fg: '#e2e8f0', fg_muted: '#94a3b8', grid: '#4f5e75', track: '#ffffff', vertex: '#ff2d78',
    shower: '#c084fc', mctrack: '#00d5ff', truth: '#8a94a6', unassociated: '#4e5d76',
    truth_vertex: '#facc15', wire: '#64748b', pd_edge: '#0b1220', pd_base: '#1e293b',
    categorical: ['#5b9bd5', '#e15759', '#59a14f', '#b07aa1', '#f28e2b', '#76b7b2', '#edc948', '#ff9da7'],
    cmap_floor: 0.08, cmap_ceiling: 1.0,
  },
  light: {
    name: 'light', fig_bg: '#ffffff', axes_bg: '#ffffff', legend_bg: '#f8fafc', legend_edge: '#94a3b8',
    fg: '#0f172a', fg_muted: '#475569', grid: '#8394ac', track: '#111827', vertex: '#be185d',
    shower: '#7c3aed', mctrack: '#0369a1', truth: '#6b7280', unassociated: '#8993a3',
    truth_vertex: '#b45309', wire: '#94a3b8', pd_edge: '#cbd5e1', pd_base: '#e2e8f0',
    categorical: ['#1f6fb2', '#c0392b', '#2e7d32', '#7b4397', '#d2691e', '#006670', '#9a7d0a', '#c34a6c'],
    cmap_floor: 0.0, cmap_ceiling: 0.92,
  },
};

// ---- truth particle classes (colour by pdg) -------------------------------
export const PDG_CLASSES = [
  { name: 'mu',      dark: '#38bdf8', light: '#0369a1' },
  { name: 'e',       dark: '#4ade80', light: '#15803d' },
  { name: 'gamma',   dark: '#fde047', light: '#a16207' },
  { name: 'p',       dark: '#fb7185', light: '#be123c' },
  { name: 'n',       dark: '#9ca3af', light: '#6b7280' },
  { name: 'pi',      dark: '#c4b5fd', light: '#6d28d9' },
  { name: 'K',       dark: '#fb923c', light: '#c2410c' },
  { name: 'nu',      dark: '#f0abfc', light: '#a21caf' },
  { name: 'other',   dark: '#e2e8f0', light: '#334155' },
];

/** Map a PDG code to an index into PDG_CLASSES. */
export function pdgClass(pdg) {
  const a = Math.abs(pdg);
  switch (a) {
    case 13: return 0;
    case 11: return 1;
    case 22: return 2;
    case 2212: return 3;
    case 2112: return 4;
    case 211: case 111: return 5;
    case 321: case 311: case 130: case 310: return 6;
    case 12: case 14: case 16: return 7;
    default: return 8;
  }
}

// ---- particle names ---------------------------------------------------------
const SYM = {
  11: 'e-', '-11': 'e+', 13: 'mu-', '-13': 'mu+', 15: 'tau-', '-15': 'tau+', 22: 'gamma',
  211: 'pi+', '-211': 'pi-', 111: 'pi0', 221: 'eta', 2212: 'p', '-2212': 'pbar', 2112: 'n', '-2112': 'nbar',
  321: 'K+', '-321': 'K-', 130: 'K0L', 310: 'K0S', 311: 'K0', '-311': 'K0bar',
  12: 'nu_e', 14: 'nu_mu', 16: 'nu_tau', '-12': 'anti-nu_e', '-14': 'anti-nu_mu', '-16': 'anti-nu_tau',
  3122: 'Lambda', 3222: 'Sigma+', 3212: 'Sigma0', 3112: 'Sigma-', 3322: 'Xi0', 3312: 'Xi-',
  411: 'D+', '-411': 'D-', 421: 'D0', 431: 'Ds+', 4122: 'Lambda_c+', 443: 'J/psi',
  1000010020: 'H2', 1000010030: 'H3', 1000020030: 'He3', 1000020040: 'alpha',
};
const ELEM = ['n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
  'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
  'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
  'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U'];

/** Short particle name for a PDG code: mu-, gamma, p, Ar39, ... (nuclei 10LZZZAAAI decoded). */
export function pdgSymbol(p) {
  if (SYM[p] !== undefined) return SYM[p];
  if (p > 1000000000) {
    const z = Math.floor(p / 10000) % 1000, a = Math.floor(p / 10) % 1000, iso = p % 10;
    return `${ELEM[z] || 'Z' + z}${a}${iso ? 'm' : ''}`;
  }
  return `pdg ${p}`;
}
