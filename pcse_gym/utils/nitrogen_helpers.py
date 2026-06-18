"""Helpers for SNOMIN nitrogen dynamics (adapted from NUE_PCSE-Gym)."""

m2_to_ha = 1e-4


def get_nitrogen_levels(n_levels=5):
    """Discrete fertilizer levels in kg N/ha (0, 4, 6, 8, 10 for n_levels=5)."""
    if n_levels <= 1:
        return [0.0]
    levels = [0.0, 4.0]
    for i in range(2, n_levels):
        levels.append(4.0 + (i - 1) * 0.5)
    return levels
