from typing import Generator, List, Optional, Tuple
import numpy as np

from .nptypes import NByOneNDArray, OneDimNDArray
from .pmodels import ModelFunction 
from curvefit_estimator import CurvefitEstimator
from sklearn.utils.validation import check_is_fitted

from .utils import argsort_1d
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.exceptions import NotFittedError

def check_not_fitted(method): 

    def inner(*args, **kwargs): 
        try: 
            return method(*args, **kwargs)
        except AttributeError as err:
            raise NotFittedError(
                f'This Estimator is not fitted yet. Call `fit` with X and y') from err 
    
    return inner 

class EnergyChangepointEstimator(BaseEstimator, RegressorMixin): 
    """A container object for a changepoint model. After a model is fit you can access scores and 
    load calculations via propeties. 
    """

    def __init__(self, model: Optional[ModelFunction]=None): 
        self.model = model  

    @classmethod
    def sort_X_y(cls, X: NByOneNDArray, y: OneDimNDArray): 
        return argsort_1d(X, y)

    @classmethod 
    def fit_many(cls, 
        models: List[ModelFunction], 
        X: NByOneNDArray, 
        y: OneDimNDArray, 
        sigma: Optional[OneDimNDArray]=None, 
        absolute_sigma: Optional[bool]=None, 
        sort: bool=True, 
        fail_silently: bool=True,
        **estimator_kwargs) -> Generator[Tuple[str, Optional['EnergyChangepointEstimator']], None, None]:
        
        if sort: 
            X, y = cls.sort_X_y(X, y)

        for m in models: 
            est = cls(m)
            try:
                yield est.name, est.fit(X, y, sigma, absolute_sigma, **estimator_kwargs)
            except Exception: # XXX this is bad... need to figure out exactly what to catch here .. prob LinAlgError? or something from skl...
                if fail_silently:
                    yield m.name, None
                else:
                    raise 


    @classmethod 
    def fit_one(cls, model: ModelFunction, 
        X: NByOneNDArray, 
        y: OneDimNDArray, 
        sigma: Optional[OneDimNDArray]=None, 
        absolute_sigma: Optional[bool]=None, 
        sort: bool=True, 
        fail_silently: bool=True,
        **estimator_kwargs) -> Tuple[str, Optional['EnergyChangepointEstimator']]: 
        """Fits a single model. Will sort data if needed and optionally fail silently.

        Args:
            model (AbstractEnergyParameterModel): [description]
            data (CurvefitEstimatorData): [description]
            sort (bool, optional): [description]. Defaults to True.

        Returns:
            Optional[AbstractEnergyParameterModel]: [description]
        """
        if sort:
            X, y = cls.sort_X_y(X, y)
        
        est = cls(model)
        try: 
            return est.name, est.fit(X, y, sigma, absolute_sigma, **estimator_kwargs)
        except Exception: # XXX  same as above... Note I can't really combine these methods or we'd be sorting the multiple times for no reason
            if fail_silently:
                return model.name, None 
            raise

    def fit(self, 
        X: NByOneNDArray, 
        y: OneDimNDArray, 
        sigma: Optional[OneDimNDArray]=None, 
        absolute_sigma: Optional[bool]=None, 
        **estimator_kwargs) -> 'EnergyChangepointEstimator':
        """This is wrapper around CurvefitEstimator.fit and allows interoperability with sklearn
        NOTE: THIS METHOD DOES NOT SORT THE DATA! Input data should be sorted appropriately beforehand. You can 
        use changepointmodel.utils.argsort_1d for standard changepoint model data.

        Args:
            data (CurvefitEstimatorInputData): [description]
            reshape (Optional[Tuple[int]], optional): [description]. Defaults to None.
        """
        self.estimator_ = CurvefitEstimator(model_func=self.model.f, bounds=self.model.bounds, **estimator_kwargs)
        self.pred_y_ = self.estimator_.fit(X, y, sigma, absolute_sigma).predict(X)  # call estimator fit  # XXX scipy/skl error in our own here.
        self.X_, self.y_ = self.estimator_.X_, self.estimator_.y_  # XXX I think these need to be here for interop with skl cv
        self.sigma_ = sigma 
        self.absolute_sigma_ = absolute_sigma
        
        return self


    def predict(self, X: NByOneNDArray) -> OneDimNDArray:  
        """Proxy a call to estimator.predict in order to use the model to generate a changepoint model with 
        different X vals on the fit estimator. This also allows interoperability with sklearn.

        Args:
            data (CurvefitEstimatorRegressorData): [description]

        Returns:
            np.array : predicted y values
        """  
        check_is_fitted(self)
        return self.estimator_.predict(X) 


    def adjust(self, other: 'EnergyChangepointEstimator') -> OneDimNDArray:
        """ A convenience method that predicts using the X values of another EnergyChangepointEstimator. 
        In option-c methodology this other would be the post retrofit model, making this calling instance the pre model.

        Args:
            other (EnergyChangepointEstimator): [description]

        Returns:
            OneDimNDArray: [description]
        """
        check_is_fitted(self)
        return self.predict(other.X)


    @property 
    def name(self):
        if self.model is None:
            raise ValueError('Cannot access name of model that is not set.') 
        return self.model.name


    @property
    @check_not_fitted 
    def X(self) -> NByOneNDArray:
        return self.X_


    @property 
    @check_not_fitted
    def y(self) -> OneDimNDArray:
        return self.y_

    @property 
    @check_not_fitted
    def coeffs(self) -> Tuple[float, ...]: # tuple
        return self.estimator_.popt_


    @property 
    @check_not_fitted
    def cov(self) -> Tuple[float, ...]: 
        return self.estimator_.pcov_


    @property 
    @check_not_fitted
    def pred_y(self) -> OneDimNDArray: 
        return self.pred_y_

    @property
    @check_not_fitted
    def sigma(self) -> Optional[OneDimNDArray]: 
        return self.sigma_ 
    

    @property 
    @check_not_fitted
    def absolute_sigma(self) -> Optional[bool]: 
        return self.absolute_sigma_

    @check_not_fitted
    def n_params(self) -> int: 
        return len(self.coeffs)

    @check_not_fitted
    def total_pred_y(self) -> float: 
        return np.sum(self.pred_y_)

    @check_not_fitted
    def total_y(self) -> float: 
        return np.sum(self.estimator_.y_)

    @check_not_fitted
    def len_y(self) -> int: 
        return len(self.estimator_.y_)



