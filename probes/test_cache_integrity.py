import unittest
import torch
from cache_integrity import CacheIntegrityMonitor

class IntegrityTest(unittest.TestCase):
    def test_unchanged_bfloat16(self):
        m=CacheIntegrityMonitor();c={0:torch.ones(3,dtype=torch.bfloat16)}
        m.watch('cache',c,{0});self.assertFalse(m.passed());m.check();self.assertTrue(m.passed())
    def test_mutation_is_sticky_after_restore(self):
        m=CacheIntegrityMonitor();c={0:torch.ones(3)};m.watch('cache',c,{0})
        c[0][0]=2;m.check();c[0][0]=1;m.check();self.assertFalse(m.passed())
    def test_missing_entry(self):
        m=CacheIntegrityMonitor();c={0:torch.ones(3)};m.watch('cache',c,{0});c.clear();m.check();self.assertFalse(m.passed())
    def test_nonfinite(self):
        m=CacheIntegrityMonitor();c={0:torch.tensor([float('nan')])};m.watch('cache',c,{0});m.check();self.assertFalse(m.passed())
    def test_incomplete_recording(self):
        with self.assertRaises(ValueError):CacheIntegrityMonitor().watch('cache',{0:torch.ones(3)},{0,1})
if __name__=='__main__':unittest.main()
