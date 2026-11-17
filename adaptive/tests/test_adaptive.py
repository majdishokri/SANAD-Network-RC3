import unittest
import numpy as np
from sanad_adaptive import (ArraySpec, RFMeasurement, AdaptiveConfig, SANADAdaptiveNetwork,
                            covariance_to_proxy, decompose)


class AdaptiveTests(unittest.TestCase):
    def test_variable_sizes_and_localization(self):
        rng = np.random.default_rng(12)
        specs=[ArraySpec('a4',np.arange(4,dtype=float)[:,None]),
               ArraySpec('a8',np.arange(8,dtype=float)[:,None],('a4',)),
               ArraySpec('a16',np.arange(16,dtype=float)[:,None],('a8',))]
        model=SANADAdaptiveNetwork(specs)
        for t in range(44):
            xs={}
            for s in specs:
                g=rng.normal(0,0.003,s.elements)
                p=rng.normal(0,0.02,s.elements)
                if t>=30 and s.array_id=='a8':
                    g[3]+=0.85
                    p[3]+=7.0
                xs[s.array_id]=RFMeasurement(g,p)
            out=model.update(xs)
        self.assertEqual([out[s.array_id].element_count for s in specs],[4,8,16])
        self.assertEqual(out['a8'].status,'LOCAL_RESEARCH_ALERT')
        self.assertEqual(out['a8'].candidate_element,3)
        self.assertEqual(out['a4'].status,'MONITORING')
        self.assertEqual(out['a16'].status,'MONITORING')
        self.assertIsNotNone(out['a8'].neighbor_score_contrast)

    def test_mask_and_missing_array(self):
        m=SANADAdaptiveNetwork([ArraySpec('A',np.arange(6,dtype=float)[:,None]),
                                ArraySpec('B',np.arange(4,dtype=float)[:,None])])
        for _ in range(27):
            out=m.update({'A':RFMeasurement(np.zeros(6),np.zeros(6)),
                          'B':RFMeasurement(np.zeros(4),np.zeros(4))})
        for _ in range(9):
            out=m.update({'A':RFMeasurement(np.zeros(6),phase_deg=np.full(6,np.nan),
                                             valid=np.array([1,1,1,1,0,1],bool))})
        self.assertTrue(all(not a for a in out['A'].phase_observed))
        self.assertGreaterEqual(out['A'].valid_elements,4)
        self.assertEqual(out['B'].status,'INSUFFICIENT_DATA')

    def test_permutation_equivariance(self):
        rng=np.random.default_rng(4)
        p=np.arange(8,dtype=float)[:,None]
        perm=np.array([6,2,5,0,4,7,1,3])
        a=SANADAdaptiveNetwork([ArraySpec('a',p)])
        b=SANADAdaptiveNetwork([ArraySpec('a',p[perm])])
        for t in range(45):
            g=rng.normal(0,0.002,8)
            g[2]+=(0.8 if t>=30 else 0.)
            x=a.update({'a':RFMeasurement(g)})['a']
            y=b.update({'a':RFMeasurement(g[perm])})['a']
        self.assertEqual(x.status,'LOCAL_RESEARCH_ALERT')
        self.assertEqual(y.status,'LOCAL_RESEARCH_ALERT')
        self.assertEqual(x.candidate_element,2)
        self.assertEqual(y.candidate_element,int(np.where(perm==2)[0][0]))
        np.testing.assert_allclose(np.asarray(x.element_scores)[perm],y.element_scores,atol=1e-6)

    def test_common_shift_not_local_fault(self):
        m=SANADAdaptiveNetwork([ArraySpec('a',np.arange(8,dtype=float)[:,None])])
        for t in range(44):
            common=0.8 if t>=30 else 0.
            out=m.update({'a':RFMeasurement(np.full(8,common),np.zeros(8))})['a']
        self.assertEqual(out.status,'COMMON_MODE_REVIEW')
        self.assertIsNone(out.candidate_element)

    def test_dynamic_registration_without_reset(self):
        m=SANADAdaptiveNetwork([ArraySpec('first',np.arange(5,dtype=float)[:,None])])
        for i in range(28):
            m.update({'first':RFMeasurement(np.zeros(5))})
        self.assertIsNotNone(m.states['first'].reference)
        m.register(ArraySpec('second',np.arange(16,dtype=float)[:,None]))
        out=m.update({'first':RFMeasurement(np.zeros(5)),
                      'second':RFMeasurement(np.zeros(16))})
        self.assertEqual(out['second'].status,'LEARNING')
        self.assertNotEqual(out['first'].status,'LEARNING')

    def test_covariance_adapter_relative_calibration(self):
        n=8
        ref=np.fromfunction(lambda i,j: 0.65**np.abs(i-j), (n,n),dtype=float).astype(complex)
        # Relative phase and gain of element 3 (0-based) vs other seven.
        g=np.zeros(n);g[3]=0.8
        phase=np.zeros(n);phase[3]=7.
        d=10**(g/20)*np.exp(1j*np.deg2rad(phase))
        observed=d[:,None]*ref*d[None,:].conj()
        q=covariance_to_proxy(observed,ref)
        self.assertTrue(q.phase_observed.all())
        self.assertAlmostEqual(q.gain_db[3]-q.gain_db[0],0.8,places=5)
        self.assertAlmostEqual(q.phase_deg[3]-q.phase_deg[0],7.0,places=3)

    def test_unobservable_small_array(self):
        m=SANADAdaptiveNetwork([ArraySpec('single',np.array([[0.]]))])
        for t in range(42):
            out=m.update({'single':RFMeasurement(np.array([.8 if t>=30 else 0.]))})['single']
        self.assertEqual(out.status,'INSUFFICIENT_DATA')
        self.assertIsNone(out.candidate_element)

    def test_baseline_cannot_relearn_after_missing_commissioning(self):
        m=SANADAdaptiveNetwork([ArraySpec('a',np.arange(5,dtype=float)[:,None])])
        for t in range(48):
            x={} if t<24 else {'a':RFMeasurement(np.ones(5))}
            out=m.update(x)['a']
        self.assertEqual(out.status,'INSUFFICIENT_COMMISSIONING')
        self.assertIsNone(m.states['a'].reference)

    def test_solver_runs_on_irregular_geometry_and_missing_channels(self):
        positions=np.array([[0.,0.],[1.,0.],[2.,.7],[3.,1.],[4.,0.]])
        y=np.zeros((5,2));y[2,0]=2.;y[2,1]=1.5;y[1,1]=np.nan
        result=decompose(y,positions)
        self.assertTrue(result.converged)
        self.assertEqual(result.sparse.shape,(5,2))
        self.assertFalse(result.valid[1,1])
        self.assertTrue(np.isfinite(result.sparse[2,0]))
        self.assertGreater(abs(result.sparse[2,0]),.5)


    def test_100_arrays_heterogeneous_topologies(self):
        sizes=[4,8,16,32,6]
        specs=[ArraySpec(f'array-{k}',np.arange(sizes[k%5],dtype=float)[:,None],
                         (f'array-{k-1}',) if k>0 else ()) for k in range(101)]
        model=SANADAdaptiveNetwork(specs)
        for t in range(30):
            output=model.update({s.array_id:RFMeasurement(np.zeros(s.elements)) for s in specs})
        self.assertEqual(len(output),101)
        self.assertEqual([output[f'array-{k}'].element_count for k in (0,1,2,3,100)],
                         [4,8,16,32,4])
        self.assertEqual(output['array-100'].status,'MONITORING')

    def test_incoherent_reference_withholds_phase(self):
        c=np.eye(8,dtype=np.complex128)
        q=covariance_to_proxy(c,c)
        self.assertFalse(q.phase_observed.any())
        self.assertTrue(np.isnan(q.phase_deg).all())
        self.assertEqual(q.coherence_edges,0)

    def test_phase_wrap_not_a_spurious_360_degree_fault(self):
        model=SANADAdaptiveNetwork([ArraySpec('a',np.arange(8,dtype=float)[:,None])])
        for t in range(42):
            p=np.full(8,179.,dtype=float)
            if t>=31:
                p[4]=-174.  # +7 degrees, NOT -353 degrees
            out=model.update({'a':RFMeasurement(np.zeros(8),p)})['a']
        self.assertEqual(out.candidate_element,4)
        self.assertEqual(out.status,'LOCAL_RESEARCH_ALERT')
        self.assertTrue(all(out.phase_observed))
        self.assertEqual(out.actuation,'NONE_RESEARCH_ONLY')

    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError):
            ArraySpec('a',np.array([[0.],[0.]]))
        m=SANADAdaptiveNetwork([ArraySpec('a',np.arange(8,dtype=float)[:,None])])
        with self.assertRaises(ValueError):
            m.update({'a':RFMeasurement(np.zeros(7))})
        with self.assertRaises(ValueError):
            m.update({'not-registered':RFMeasurement(np.zeros(3))})


if __name__=='__main__':
    unittest.main()
