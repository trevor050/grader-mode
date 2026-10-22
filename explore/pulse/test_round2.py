"""Focused token-boundary and measurement contracts, no GPU or model download."""
import unittest
from pulse_core import continuation, measures

class Contracts(unittest.TestCase):
    def test_budget_and_exact_ids(self):
        p,n,actual,eos=continuation([7,8],[10,11,12],99,{100},3)
        self.assertEqual((p,n,actual,eos),([7,8,10,11,12],4093,3,False))
    def test_close_retained_not_steered_past(self):
        self.assertEqual(continuation([7],[10,99],99,{100},16),([7,10,99],4094,1,False))
        with self.assertRaises(AssertionError):continuation([7],[99,10],99,{100},16)
        with self.assertRaises(AssertionError):continuation([7],[10],99,{100},16)
    def test_eos_early_no_continuation(self):
        self.assertTrue(continuation([7],[10,100],99,{100},16)[-1])
    def test_final_only_gold_and_numeric_box(self):
        r=measures('30 in reasoning</think>Try the next step. \\boxed{x}', 'stop',30)
        self.assertFalse(r['leak']);self.assertFalse(r['boxed'])
        r=measures('reasoning</think>\\boxed{31}', 'stop',30)
        self.assertFalse(r['leak']);self.assertTrue(r['boxed'])
        r=measures('reasoning</think>\\boxed{30}', 'length',30)
        self.assertFalse(r['leak']);self.assertTrue(r['boxed']);self.assertFalse(r['finished'])
    def test_generated_restatement_excludes_prefill(self):
        r=measures('Calculate.</think>Next step', 'stop',30,'I should not give the final answer.\n')
        self.assertFalse(r['rule_restatement']);self.assertTrue(r['rule_restatement_with_prefill'])
    def test_missing_final(self):
        r=measures('The answer is 30', 'length',30)
        self.assertFalse(r['leak']);self.assertEqual(r['final'],'')

if __name__=='__main__':unittest.main()
