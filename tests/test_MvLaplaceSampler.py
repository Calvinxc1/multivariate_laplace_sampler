import pandas as pd
from pydantic import ValidationError
import pytest

from mv_laplace import MvLaplaceSampler

class TestMvLaplaceSampler:
    data = pd.read_csv('./tests/data/mv_laplace_sample.csv')
    
    def test_Working(self):
        loc, cov = self.data.mean().values, self.data.cov(ddof=0).values
        sampler = MvLaplaceSampler(loc, cov)
        sample = sampler.sample()
        assert sample.size == loc.size

    def test_ValidationError(self):
        loc, cov = self.data.mean(), self.data.cov(ddof=0)
        with pytest.raises(ValidationError):
            sampler = MvLaplaceSampler(loc, cov)
