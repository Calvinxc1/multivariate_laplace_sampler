import numpy as np
from pydantic import validate_arguments
from scipy import stats
from typing import Union

from .PydanticConfig import Config

class MvLaplaceSampler:
    @validate_arguments(config=Config)
    def __init__(self, loc:Union[list,np.ndarray], cov:Union[list,np.ndarray]):
        if type(loc) is list: loc = np.array(loc)
        if type(cov) is list: cov = np.array(cov)
        
        self._mv_normal = stats.multivariate_normal(mean=loc, cov=cov)
        self._normal = stats.norm(loc=loc, scale=np.sqrt(np.diag(cov)))
        self._laplace = stats.laplace(loc=loc, scale=np.sqrt(np.diag(cov) / 2))
        
    @validate_arguments(config=Config)
    def sample(self, sample_size:Union[int,None]=None) -> np.array:
        mv_samples = self._mv_normal.rvs(sample_size)
        cdf_samples = self._normal.cdf(mv_samples)
        laplace_samples = self._laplace.ppf(cdf_samples)
        return laplace_samples
