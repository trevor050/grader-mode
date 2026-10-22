import unittest
from closed_loop_core import calibrate,chunk_stats,retain,monitor_value
import numpy as np

class Contracts(unittest.TestCase):
    def test_calibration_fixed_orientation_and_ties(self):
        rows=[{'cond':c,'score':s} for c,ss in [('rl_tutor',[2,4]),('student_tutor',[0,2])] for s in ss]
        r=calibrate(rows)
        self.assertEqual((r['threshold'],r['auc'],r['tp'],r['fp']),(2,.875,2,1))
    def test_exact_boundary_and_lookahead(self):
        k,n,l,e,b=retain(list(range(32))+[99],99,{100},4096)
        self.assertEqual((len(k),n,l,e,b),(32,32,1,False,False))
        self.assertEqual(retain([1,99],99,{100},4096),([1,99],1,0,False,True))
        with self.assertRaises(AssertionError):retain([99,2],99,{100},4096)
    def test_cap_and_retrigger(self):
        self.assertEqual(retain([1,2,3],99,{100},2)[0],[1,2])
        cs=[{'push':s,'reasoning_tokens':32,'prefix_tokens':100} for s in [True,False,True,False]]
        self.assertEqual(chunk_stats(cs),dict(push_on_chunks=2,pushed_tokens=64,switches=3,retrigger=True,reasoning_chunks=4,hooked_prefix_tokens=200))
    def test_positions_identity_and_missing(self):
        rows=[(4,[1]*60),(2,[3]*60)]
        self.assertEqual(monitor_value(rows,[2,4])[:,0].tolist(),[3,1])
        with self.assertRaises(KeyError):monitor_value(rows,[3])
        with self.assertRaises(AssertionError):monitor_value([(2,[float('nan')]*60)],[2])

if __name__=='__main__':unittest.main()

class SchedulerLayouts(unittest.TestCase):
    def test_new_runner_batch_is_bound_before_forward(self):
        from types import SimpleNamespace
        from closed_loop_core import install_batch_observer,batch_layout
        batch=SimpleNamespace(num_reqs=2,req_ids=['second','first'],query_start_loc_np=np.array([0,3,8]),num_computed_tokens_np=np.array([14,9]))
        class V2:
            def prepare_inputs(self,*args,**kwargs):return batch
        runner=V2()
        self.assertFalse(hasattr(runner,'input_batch'))
        self.assertEqual(install_batch_observer(runner),'v2_prepare_inputs')
        runner.prepare_inputs(None)
        ids,starts,computed=batch_layout(runner)
        self.assertEqual((ids,starts.tolist(),computed.tolist()),(['second','first'],[0,3,8],[14,9]))
    def test_legacy_layout_and_padding(self):
        from types import SimpleNamespace
        from closed_loop_core import install_batch_observer,batch_layout
        runner=SimpleNamespace(input_batch=SimpleNamespace(num_reqs=1,req_ids=['x',None],num_computed_tokens_cpu=np.array([10,0])),query_start_loc=SimpleNamespace(np=np.array([0,8,8])))
        self.assertEqual(install_batch_observer(runner),'v1_input_batch')
        ids,starts,computed=batch_layout(runner)
        self.assertEqual((ids,starts.tolist(),computed.tolist()),(['x'],[0,8],[10]))
