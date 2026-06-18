"""Helpers for SNOMIN nitrogen dynamics (adapted from NUE_PCSE-Gym)."""

m2_to_ha = 1e-4

# Default discrete fertilizer rates for WOFOST SNOMIN (kg N/ha per application)
SNOMIN_NITROGEN_LEVELS = [0.0, 30.0, 60.0, 90.0]
# Dutch winter wheat standard practice: ~60 kg N/ha at each of 3 split applications
STANDARD_PRACTICE_KG_N = 60.0


def get_nitrogen_levels(n_levels=None, levels=None):
    """Discrete fertilizer levels in kg N/ha for SNOMIN.

    Default: 0, 30, 60, 90 kg N/ha.
    """
    if levels is not None:
        return list(levels)
    default = SNOMIN_NITROGEN_LEVELS
    if n_levels is None:
        return default.copy()
    if n_levels <= len(default):
        return default[:n_levels]
    step = default[-1] / (n_levels - 1)
    return [round(i * step, 1) for i in range(n_levels)]


def get_standard_practice_action_index(levels=None):
    """Discrete action index closest to STANDARD_PRACTICE_KG_N (default 60 kg N/ha)."""
    levels = levels or SNOMIN_NITROGEN_LEVELS
    return min(range(len(levels)), key=lambda i: abs(levels[i] - STANDARD_PRACTICE_KG_N))
