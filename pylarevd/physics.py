"""Names for the integer codes stored in truth records.

A neutrino interaction is written as a pile of bare integers -- ``fCCNC=0``,
``fMode=2``, ``fInteractionType=1091``, ``fTarget=1000180400`` -- which say
nothing at a glance.  This module turns them into the labels a physicist would
say out loud, and falls back to the raw number rather than inventing a name it
does not know.

Code values follow ``nusimdata/SimulationBase`` (``MCNeutrino.h``,
``MCTruth.h``), which is what LArSoft writes.
"""

from __future__ import annotations

#: simb::curr_type_
CC, NC = 0, 1

#: simb::int_type_ -- the coarse interaction mode.
MODES: dict[int, str] = {
    -1: "unknown",
    0: "QE", 1: "RES", 2: "DIS", 3: "COH", 4: "COH elastic",
    5: "e- scattering", 6: "IMD annihilation", 7: "inverse beta decay",
    8: "Glashow resonance", 9: "AMnuGamma", 10: "MEC", 11: "diffractive",
    12: "EM", 13: "weak mixing",
}

#: The finer "nuance" code.  Only the ones that actually turn up in DUNE
#: samples are named; anything else is reported as its number.
INTERACTIONS: dict[int, str] = {
    0: "unknown",
    1001: "CC QE", 1002: "NC QE",
    1003: "CC RES nu p pi+", 1004: "CC RES nu n pi+", 1005: "CC RES nu n pi0",
    1006: "NC RES nu p pi0", 1007: "NC RES nu p pi+", 1008: "NC RES nu n pi0",
    1009: "NC RES nu n pi-",
    1091: "CC DIS", 1092: "NC DIS",
    1095: "CC QE hyperon", 1096: "NC COH", 1097: "CC COH",
    1098: "nu-e elastic", 1099: "inverse mu decay", 1100: "MEC 2p2h",
}

#: simb::Origin_t
ORIGINS: dict[int, str] = {
    0: "unknown", 1: "beam neutrino", 2: "cosmic ray",
    3: "supernova neutrino", 4: "single particle",
}

#: GENIE/HepMC status as LArSoft stores it. 1 is the one that matters: those
#: are the particles actually handed to the detector simulation.
STATUS: dict[int, str] = {
    0: "initial state", 1: "final state", 2: "intermediate",
    3: "decayed", 11: "struck nucleon", 12: "DIS pre-fragmentation",
    14: "hadronic system", 15: "nuclear remnant",
}

FINAL_STATE = 1

_PDG_NAMES: dict[int, str] = {
    11: "e-", -11: "e+", 12: "nu_e", -12: "nubar_e",
    13: "mu-", -13: "mu+", 14: "nu_mu", -14: "nubar_mu",
    15: "tau-", -15: "tau+", 16: "nu_tau", -16: "nubar_tau",
    22: "gamma", 111: "pi0", 211: "pi+", -211: "pi-",
    130: "K0_L", 310: "K0_S", 311: "K0", -311: "K0bar",
    321: "K+", -321: "K-", 221: "eta", 331: "eta'",
    2112: "n", -2112: "nbar", 2212: "p", -2212: "pbar",
    3122: "Lambda", 3222: "Sigma+", 3112: "Sigma-", 3212: "Sigma0",
    3322: "Xi0", 3312: "Xi-", 3334: "Omega-",
    1000010020: "deuteron", 1000010030: "triton",
    1000020030: "He3", 1000020040: "alpha",
}

#: Element symbols, for naming nuclear PDG codes (10LZZZAAAI).
_ELEMENTS = (
    "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
)

#: GENIE's own pseudo-particles, which appear in a part list but are not
#: physical. Naming them stops "2000000001" showing up as an unknown code.
_GENIE_PSEUDO = {
    2000000001: "hadronic system",
    2000000002: "hadronic blob",
    2000000101: "bindino",
    # 2p2h/MEC struck-nucleon pairs: these show up as the "hit nucleon" of an
    # MEC interaction, where the target is two nucleons rather than one.
    2000000200: "nn cluster",
    2000000201: "np cluster",
    2000000202: "pp cluster",
    2000000300: "compound nucleus",
}


def nucleus_name(pdg: int) -> str | None:
    """Name a nuclear PDG code (``10LZZZAAAI``), e.g. 1000180400 -> ``Ar-40``."""
    if not 1000000000 <= pdg < 2000000000:
        return None
    z = (pdg // 10000) % 1000
    a = (pdg // 10) % 1000
    if z < len(_ELEMENTS):
        return f"{_ELEMENTS[z]}-{a}"
    return f"Z{z}-{a}"


def particle_name(pdg: int) -> str:
    """Readable name for a PDG code, or the code itself when unknown."""
    pdg = int(pdg)
    if pdg in _PDG_NAMES:
        return _PDG_NAMES[pdg]
    if pdg in _GENIE_PSEUDO:
        return _GENIE_PSEUDO[pdg]
    nuc = nucleus_name(pdg)
    if nuc:
        return nuc
    if abs(pdg) // 1000000 == 0 and abs(pdg) > 1000:
        return f"hadron {pdg}"
    return str(pdg)


def mode_name(mode: int) -> str:
    return MODES.get(int(mode), f"mode {int(mode)}")


def interaction_name(code: int) -> str:
    return INTERACTIONS.get(int(code), f"interaction {int(code)}")


def origin_name(code: int) -> str:
    return ORIGINS.get(int(code), f"origin {int(code)}")


def status_name(code: int) -> str:
    return STATUS.get(int(code), f"status {int(code)}")


def current_name(ccnc: int) -> str:
    return "CC" if int(ccnc) == CC else "NC"
