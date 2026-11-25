"""Distribution transport & manifold prediction layer (Section 4)."""
from flow_dmo.transport.change_detector import ChangeDetectionResult, PopulationChangeDetector
from flow_dmo.transport.flow_matching import ConditionalVelocityNet, FlowMatchingTransport
from flow_dmo.transport.kalman_momentum import KalmanMomentumFilter
from flow_dmo.transport.portfolio_gate import PortfolioGate
from flow_dmo.transport.sliced_ot import SlicedOTTransport

__all__ = [
    "ChangeDetectionResult",
    "ConditionalVelocityNet",
    "FlowMatchingTransport",
    "KalmanMomentumFilter",
    "PopulationChangeDetector",
    "PortfolioGate",
    "SlicedOTTransport",
]
