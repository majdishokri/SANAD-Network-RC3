"""A deliberately changed multipath field must lead to review, not a fault claim."""
from sanad_adaptive.spectral import SpectralPhaseMonitor
from test_spectral_v02 import make_windows


def test_frequency_selective_motion_causes_review():
    engine=SpectralPhaseMonitor()
    reference=make_windows(n=64,seed=71)
    engine.register('east',reference[:32],reference[32:])
    motion=make_windows(n=1,seed=525,channel_change=1.)[0]
    verdict=engine.evaluate({'east':motion})['decisions']['east']
    assert verdict['status']=='CHANNEL_SHIFT_REVIEW'
    assert verdict['channel_shift'] is True
    assert not any(verdict['phase_alerts'])
