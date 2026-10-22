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
