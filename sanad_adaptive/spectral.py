"""SANAD Adaptive v0.2: experimental spectral phase-evidence gate.

Does not replace or re-train frozen RC3. Accepts array-wise phase coherent CSI.
The model is *not* a proof of identifying phase calibration from arbitrary CSI:
a channel change can be observationally equivalent to diagonal phase drift.

One matrix per array is [elements, elements, subcarriers]. The code supports
arbitrary numbers of arrays and independent M_i, K_i for each array.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Mapping
import numpy as np

_EPS = 1.e-12


def spectral_coherence(csi: np.ndarray) -> np.ndarray:
    """Turn raw *within-array phase-synchronized* CSI [records,M,K] into [M,M,K].

    Removes common packet phase; never forms cross-array phase products.
    Deep-fade tones are masked using per-antenna record magnitudes.
    """
    h = np.asarray(csi)
    if h.ndim != 3 or h.shape[1] < 2 or h.shape[2] < 2 or not np.iscomplexobj(h):
        raise ValueError('Expected complex CSI [records,M>=2,K>=2]')
    if not np.isfinite(h).all():
        raise ValueError('CSI must be finite')
    mag = np.abs(h)
    floor = np.maximum(np.median(mag, axis=(0,2), keepdims=True) * .035, _EPS)
    valid = mag > floor
    u = np.where(valid, h / np.maximum(mag, _EPS), 0.)
    # Averages per-tone relative phases across packet records. Poorly observed
    # tones naturally obtain lower magnitude, and hence lower later weight.
    return np.einsum('tmk,tnk->mnk', u, np.conjugate(u), optimize=True) / len(h)


def transform_recording(raw: np.ndarray, window: int = 16) -> np.ndarray:
    """CSI [T,N,M,K] -> [W,N,M,M,K] for uniform arrays only."""
    a = np.asarray(raw)
    if a.ndim != 4 or not np.iscomplexobj(a) or window < 2:
        raise ValueError('Expected complex CSI [T,N,M,K], window>=2')
    nw = len(a) // window
    if nw < 1:
        raise ValueError('Insufficient records')
    return np.stack([np.stack([spectral_coherence(a[w*window:(w+1)*window,n])
                                  for n in range(a.shape[1])]) for w in range(nw)])


def _validate_spectrum(spec: np.ndarray) -> np.ndarray:
    s = np.asarray(spec)
    if s.ndim != 3 or s.shape[0] != s.shape[1] or s.shape[0] < 2 or s.shape[2] < 2:
        raise ValueError('Expected spectrum [M,M,K] with M,K>=2')
    if not np.iscomplexobj(s) or not np.isfinite(s).all():
        raise ValueError('Spectrum must be finite and complex')
    if not np.allclose(s, s.swapaxes(0,1).conj(), atol=1e-5, rtol=1e-5):
        raise ValueError('Spectrum is not Hermitian at each tone')
    return s.astype(np.complex128, copy=False)


@dataclass(frozen=True)
class SpectralConfig:
    phase_window: int = 8
    persistence: int = 3
    phase_floor_deg: float = 1.75
    phase_quantile: float = .95
    phase_multiplier: float = 1.2
    shift_quantile: float = .98
    shift_multiplier: float = 1.25
    shift_floor: float = .045
    min_pair_concentration: float = .4
    min_calibration_windows: int = 24

    def __post_init__(self):
        if not 3 <= self.phase_window <= self.min_calibration_windows // 2:
            raise ValueError('Insufficient calibration relative to phase_window')
        if not 1 <= self.persistence <= self.phase_window:
            raise ValueError('Invalid persistence')
        if not (0 < self.phase_quantile < 1 and 0 < self.shift_quantile < 1):
            raise ValueError('Quantiles must be between 0 and 1')
        if min(self.phase_floor_deg, self.shift_floor, self.phase_multiplier,
               self.shift_multiplier, self.min_pair_concentration) <= 0:
            raise ValueError('Positive configuration constants required')
        if self.min_pair_concentration >= 1:
            raise ValueError('Invalid concentration')


def _connected(m, ii, jj):
    seen = {0}
    for _ in range(m):
        for i, j in zip(ii,jj):
            if int(i) in seen: seen.add(int(j))
            if int(j) in seen: seen.add(int(i))
    return len(seen) == m


def _phase_and_shift(ref: np.ndarray, obs: np.ndarray, conf: SpectralConfig):
    """Graph-based relative hardware phase and post-fit channel spectral mismatch.

    Each pair contributes a phase-vs-frequency signature. Robust phase fitting
    cannot remove frequency-selective channel distortion, which raises mismatch.
    """
    m,_,k=ref.shape
    ii,jj=np.triu_indices(m,1)
    d=obs[ii,jj,:]*np.conj(ref[ii,jj,:])
    w=np.abs(obs[ii,jj,:])*np.abs(ref[ii,jj,:])
    # Unit-circle residual per subcarrier; aggregate edge phase circularly.
    unit=d/np.maximum(np.abs(d),_EPS)
    z=np.sum(w*unit,axis=1)
    wsum=np.sum(w,axis=1)
    concentration=np.abs(z)/np.maximum(wsum,_EPS)
    useful=(concentration>=conf.min_pair_concentration)&(wsum>_EPS)
    if not _connected(m,ii[useful],jj[useful]):
        return np.full(m,np.nan),float('inf'),False
    edge_angle=np.angle(z[useful]); ia=ii[useful];jb=jj[useful]
    base=np.sqrt(np.maximum(wsum[useful],_EPS))*concentration[useful]
    base=base/np.maximum(np.median(base),_EPS)
    weight=np.clip(base,.05,10.)
    ph=np.zeros(m,float)
    for _ in range(4):
        lap=np.zeros((m,m));rhs=np.zeros(m)
        np.add.at(lap,(ia,ia),weight);np.add.at(lap,(jb,jb),weight)
        np.add.at(lap,(ia,jb),-weight);np.add.at(lap,(jb,ia),-weight)
        np.add.at(rhs,ia,weight*edge_angle);np.add.at(rhs,jb,-weight*edge_angle)
        ph=np.linalg.solve(lap+np.ones((m,m))/m+1e-8*np.eye(m),rhs)
        residual=np.angle(np.exp(1j*(edge_angle-(ph[ia]-ph[jb]))))
        weight=np.maximum(.02,base*np.minimum(1.,.35/np.maximum(np.abs(residual),1e-8)))
    ph=np.rad2deg(np.angle(np.exp(1j*(ph-np.median(ph)))))
    # Robust channel proxy: mismatch remaining *across frequencies* after
    # removal of a single diagonal hardware phase. Healthy calibration sets
    # the rejection threshold; no threshold tuning on measured evaluation.
    correction=np.exp(-1j*(np.deg2rad(ph[ii]-ph[jj])))[:,None]
    err=np.abs(1-unit*correction)
    mismatch=float(np.sum(w*err)/np.maximum(w.sum(),_EPS))
    return ph,mismatch,True


def _circular_mean_deg(rows: np.ndarray):
    a=np.deg2rad(np.asarray(rows,float))
    return np.rad2deg(np.angle(np.mean(np.exp(1j*a),axis=0)))


class SpectralPhaseMonitor:
    """Independent per-array spectral detector; no pretrained model/RC3 weights.

    The caller must provide distinct known-healthy reference & calibration
    recordings and preserve array identity. Outputs are research only.
    """
    def __init__(self, config: SpectralConfig|None=None):
        self.conf=config or SpectralConfig()
        self.baselines={}
        self._hist={}
        self._score_hist={}

    def register(self, name: str, reference: np.ndarray, calibration: np.ndarray):
        name=str(name)
        if name in self.baselines:
            raise ValueError('Array already registered')
        r=np.asarray(reference);c=np.asarray(calibration)
        if r.ndim!=4 or c.ndim!=4 or len(r)<self.conf.min_calibration_windows or len(c)<self.conf.min_calibration_windows:
            raise ValueError('Expected >=24 reference and calibration spectral windows [T,M,M,K]')
        if r.shape[1:]!=c.shape[1:]:
            raise ValueError('Reference and calibration dimensions differ')
        for x in r: _validate_spectrum(x)
        for x in c: _validate_spectrum(x)
        base=r.mean(axis=0)
        phases=[];drifts=[];valid=[]
        for x in c:
            ph,d,ok=_phase_and_shift(base,x,self.conf)
            phases.append(ph);drifts.append(d);valid.append(ok)
        phases=np.asarray(phases)
        if not np.all(valid):
            # Do not pretend unobservable phase can be calibrated.
            threshold=np.full(r.shape[1],np.inf)
        else:
            smooth=np.asarray([_circular_mean_deg(phases[i:i+self.conf.phase_window])
                               for i in range(len(phases)-self.conf.phase_window+1)])
            threshold=np.maximum(self.conf.phase_floor_deg,
                self.conf.phase_multiplier*np.quantile(np.abs(smooth),self.conf.phase_quantile,axis=0))
        finite=np.asarray(drifts)[np.isfinite(drifts)]
        shift_thr=max(self.conf.shift_floor,self.conf.shift_multiplier*np.quantile(finite,self.conf.shift_quantile)) if len(finite) else 0.
        self.baselines[name]={
            'reference':base,'elements':int(r.shape[1]),'subcarriers':int(r.shape[3]),
            'phase_threshold_deg':np.asarray(threshold), 'channel_threshold':float(shift_thr),
            'phase_calibration_valid':bool(np.all(valid))}
        self._hist[name]=deque(maxlen=self.conf.phase_window)
        self._score_hist[name]=deque(maxlen=self.conf.persistence)

    def register_many(self, reference: Mapping[str,np.ndarray],calibration: Mapping[str,np.ndarray]):
        if set(reference)!=set(calibration):raise ValueError('Array ID mismatch')
        for name in reference:self.register(name,reference[name],calibration[name])

    def reset_history(self):
        for q in self._hist.values():q.clear()
        for q in self._score_hist.values():q.clear()

    def evaluate(self,snapshot: Mapping[str,np.ndarray]):
        if not isinstance(snapshot,Mapping) or not snapshot:
            raise ValueError('Use {array_id: [M,M,K] spectrum}')
        out={}
        for name,spec in snapshot.items():
            if name not in self.baselines:raise ValueError(f'Unknown array {name}; register first')
            a=_validate_spectrum(spec);b=self.baselines[name]
            if a.shape!=b['reference'].shape:raise ValueError('Array dimensions changed; recalibrate')
            phase,drift,observable=_phase_and_shift(b['reference'],a,self.conf)
            shift=bool(not np.isfinite(drift) or drift>b['channel_threshold'])
            # Do not accumulate possible channel changes into the phase detector.
            if shift or not observable:
                self._hist[name].clear();self._score_hist[name].clear()
                alerts=np.zeros(b['elements'],bool)
                scores=np.zeros(b['elements'],float)
                status='CHANNEL_SHIFT_REVIEW' if shift else 'PHASE_UNOBSERVABLE'
            else:
                self._hist[name].append(phase)
                if len(self._hist[name])<self.conf.phase_window:
                    scores=np.zeros(b['elements'],float)
                    alerts=np.zeros(b['elements'],bool)
                    status='WARMUP'
                else:
                    smoothed=_circular_mean_deg(np.stack(self._hist[name]))
                    scores=np.abs(smoothed)/b['phase_threshold_deg']
                    hit=scores>=1.
                    self._score_hist[name].append(hit)
                    alerts=(np.sum(np.stack(self._score_hist[name]),axis=0)>=self.conf.persistence
                            if len(self._score_hist[name])==self.conf.persistence
                            else np.zeros(b['elements'],bool))
                    status='PHASE_CANDIDATE' if np.any(alerts) else 'MONITORING'
            if b['elements']<3:
                alerts[:]=False;status='UNDERIDENTIFIED_ARRAY'
            out[name]={
                'phase_deg':phase.tolist() if observable else [None]*b['elements'],
                'phase_score':scores.tolist(),'phase_alerts':alerts.tolist(),
                'phase_threshold_deg':b['phase_threshold_deg'].tolist(),
                'channel_mismatch':float(drift) if np.isfinite(drift) else None,
                'channel_threshold':b['channel_threshold'],
                'channel_shift':shift,'status':status,'phase_observable':observable}
        return {'engine':'SANAD_ADAPTIVE_SPECTRAL_V02_EXPERIMENTAL','decisions':out,
                'arrays_observed':len(out),'research_only':True,'frozen_rc3_used':False}
