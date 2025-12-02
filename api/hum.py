"""
Internal hum management for Maven brains.

Each brain has a logical oscillator (phase) that runs continuously.  This module
handles phase updates, sampling of the current hum value and computation of
system coherence (Kuramoto order parameter).  Phases are persisted to
`reports/system/hum_state.json` so that hum continues across runs.

The configuration for the hum lives in CFG['hum'] in api/utils.py.  It
should contain at least:

    {
        "enabled": true,
        "K": 0.02,             # global coupling constant
        "dt_sec": 0.25,       # default timestep for updates
        "freq": {
            "sensorium": 5.2,
            ...
        }
    }

When coupling K is zero, each oscillator advances independently at its
natural frequency.  When K > 0, each oscillator is weakly attracted to the
average phase of all oscillators.

The main entry points are:

    tick(dt): update all phases by dt seconds
    sample(brain): return the current hum value (sin of phase) for a brain
    coherence(): return the Kuramoto order parameter (0..1)
"""

from __future__ import annotations

import json, time, math, cmath
from typing import Dict, Any

from api.utils import CFG
from brains.maven_paths import get_reports_path

# Location to persist hum state (relative to maven root).  See STATE_PATH below.

# The HUM state file lives under reports/system.  Use the central path helper to
# guarantee writes stay inside the Maven project root.
STATE_PATH = get_reports_path("system", "hum_state.json")

def _default_state() -> Dict[str, Any]:
    """
    Return a default state dict with zero phases and current timestamp.
    Uses the list of brain names from CFG['hum']['freq'] keys.
    """
    freqs = (CFG.get("hum") or {}).get("freq") or {}
    phases = {b: 0.0 for b in freqs}
    return {"phases": phases, "last_ts": time.time()}

def _load_state() -> Dict[str, Any]:
    """
    Load hum state from disk.  If the file doesn't exist or cannot be parsed,
    return a default state.
    """
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        # Validate presence of phases
        if not isinstance(data.get("phases"), dict):
            raise ValueError("Malformed hum state")
        return data
    except Exception:
        return _default_state()

def _save_state(state: Dict[str, Any]):
    """
    Persist the given state dict to disk.  Parent directories are created if
    necessary.  Failures to write are ignored.
    """
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass

def tick(dt: float | None = None):
    """
    Advance all oscillator phases by dt seconds.  If dt is None, falls back
    to CFG['hum']['dt_sec'].  Coupling is applied via the global K value.

    Args:
        dt: Time increment in seconds.  If omitted, use CFG['hum']['dt_sec'].
    """
    cfg = CFG.get("hum") or {}
    if not cfg.get("enabled", True):
        return
    if dt is None:
        try:
            dt = float(cfg.get("dt_sec", 0.25))
        except Exception:
            dt = 0.25
    state = _load_state()
    phases = state.get("phases", {})
    freqs: Dict[str, float] = cfg.get("freq") or {}
    # If no frequencies defined, nothing to update
    if not freqs:
        return
    K = float(cfg.get("K", 0.0))
    # Precompute coupling sum per oscillator
    brains = list(freqs.keys())
    N = len(brains)
    # Use a copy so updates don't affect coupling calcs mid-loop
    new_phases = dict(phases)
    for b in brains:
        theta = float(phases.get(b, 0.0))
        omega = float(freqs.get(b, 0.0))
        # Coupling term: average of differences between all other phases
        coupling = 0.0
        if K != 0.0 and N > 1:
            for j in brains:
                if j == b:
                    continue
                theta_j = float(phases.get(j, 0.0))
                coupling += math.sin(theta_j - theta)
            coupling /= (N - 1)
        dtheta = omega + K * coupling
        # Update phase and wrap to [0, 2π)
        new_phase = theta + dtheta * dt
        new_phases[b] = new_phase % (2.0 * math.pi)
    state["phases"] = new_phases
    state["last_ts"] = time.time()
    _save_state(state)

def sample(brain: str) -> float:
    """
    Return the current hum value (sin of phase) for the specified brain.  If
    the brain isn't configured, returns 0.0.

    Args:
        brain: Name of the cognitive brain (e.g. "language").

    Returns:
        A float in [-1, 1] representing the hum amplitude.
    """
    cfg = CFG.get("hum") or {}
    if not cfg.get("enabled", True):
        return 0.0
    state = _load_state()
    phase = float(state.get("phases", {}).get(brain, 0.0))
    return math.sin(phase)

def coherence() -> float:
    """
    Compute the Kuramoto order parameter R (0..1), which measures phase
    synchrony across all oscillators.  If no oscillators are defined,
    returns 1.0 (fully coherent by default).

    Returns:
        A float in [0, 1] indicating the degree of synchrony.
    """
    cfg = CFG.get("hum") or {}
    if not cfg.get("enabled", True):
        return 1.0
    freqs: Dict[str, float] = cfg.get("freq") or {}
    if not freqs:
        return 1.0
    state = _load_state()
    phases = state.get("phases", {})
    # Compute complex order parameter
    z = sum(cmath.exp(1j * phases.get(b, 0.0)) for b in freqs) / len(freqs)
    return abs(z)