"""SANAD Adaptive — independent research prototype, not frozen RC3."""
from .engine import ArraySpec, RFMeasurement, AdaptiveConfig, ArrayDecision, SANADAdaptiveNetwork
from .geometry import Decomposition, decompose
from .covariance import CovarianceProxy, covariance_to_proxy

__all__ = ['ArraySpec','RFMeasurement','AdaptiveConfig','ArrayDecision',
           'SANADAdaptiveNetwork','Decomposition','decompose',
           'CovarianceProxy','covariance_to_proxy']
__version__ = '0.1.0'
