from typing import Any, List, NamedTuple, Optional, TypeVar
import pydantic 
import datetime
import numpy as np

import re

# we make these schemas composable to leave room for future work to model with multiple independent variables

# we distinguish between targets and regressors. In our case the target is the dep var.. 
# We use timestamps to join the data together at runtime based on their datestamp

# below are somewhat abstract models that cover input values 

Metadata = TypeVar('Metadata', Optional[Any])

_ym_re = re.compile(r'[0-9]{4}-[0-9]{2}')

class YearMonth(NamedTuple):
    """A custom YearMonth type. Can validate against YYYY-MM, ('YYYY', 'MM') or (int, int). 
    Checks that year and month are valid ISO 8601 ranges.
    """
    year: int
    month: int 

    @classmethod
    def _parse_yearmonth_str(cls, v):
        if not _ym_re.match(v): 
            raise ValueError(f'str format not correct. Must be {_ym_re.pattern}')
        return [int(x) for x in v.split('-')]

    @classmethod 
    def _parse_yearmonth_tuple(cls, v): 
        return (int(x) for x in v)    

    @classmethod 
    def _validate_ym(cls, year, month): 
        if year < 1 or month < 1 or month > 12:
            raise ValueError(f'{year}-{month} is out of range')

    @classmethod 
    def __get_validators__(cls): 
        yield cls.validate 

    
    @classmethod 
    def validate(cls, v): 
        if isinstance(v, str): 
            year, month = cls._parse_yearmonth_str(v)
        elif isinstance(v, tuple):
            year, month = cls._parse_yearmonth_tuple(v)
        elif isinstance(v, YearMonth): 
            year, month = v 
        cls._validate_ym(year, month)
        return cls(year=year, month=month)  
        
        
    def __repr__(self):
        return f'YearMonth(year={self.year}, month={self.month})'


class TypedArray(np.ndarray):
    """A type for a typed numpy array based on 
    https://github.com/samuelcolvin/pydantic/issues/380
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return np.array(v, dtype=cls.inner_type)


class _ArrayMeta(type): # override the metaclass here to define an inner type on the class so pydantic can validate
    def __getitem__(self, t):
        return type('Array', (TypedArray,), {'inner_type': t})


class NumpyArray(np.ndarray, metaclass=_ArrayMeta):
    """ A numpy array type"""
    pass


class MonthlyTimeSeries(pydantic.BaseModel):
    """ defines a monthly series """
    obs: List[YearMonth]

    @pydantic.validator('obs')
    def validate_obs(self, v: List[YearMonth]) -> List[YearMonth]: 
        if len(v) < 12: 
            raise ValueError(f'Need at least 12 points for a monthly series.')

