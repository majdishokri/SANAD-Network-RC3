"""Experimental real-RF conditioning for SANAD RC3 research; never replaces the frozen detector."""
from .conditioning import (Observation, WindowQuality, WindowAligner, AlignmentPolicy,
                           BaselineNormalizer, ObservationFilter, validate_run)
from .shadow import OneClassThreshold, ShadowHead
__all__ = ["Observation", "WindowQuality", "WindowAligner", "AlignmentPolicy",
           "BaselineNormalizer", "ObservationFilter", "validate_run",
           "OneClassThreshold", "ShadowHead"]
