"""SANAD Adaptive dimensionality-agnostic research core (not frozen RC3)."""
from .core import ArrayGeometry, DetectorConfig, AdaptiveCore, covariance_from_csi, discover_arrays
__all__ = ["ArrayGeometry", "DetectorConfig", "AdaptiveCore", "covariance_from_csi", "discover_arrays"]
