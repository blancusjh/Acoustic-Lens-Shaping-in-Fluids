"""Compose existing height/spot observations with a full static response matrix."""


def replace_mechanical_compliance(observe, stiffness, coupled_compliance):
    """O K^-1 -> O (K-A)^-1; offsets and physical observation units are unchanged."""
    return observe @ stiffness @ coupled_compliance
