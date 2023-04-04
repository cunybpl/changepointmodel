""" Exception wrapper for bema changepoint.
"""
from typing import Dict
import sys
import traceback

from typing import Optional


class BemaChangepointException(Exception):
    def __init__(self, info: Optional[Dict[str, str]] = None, message: str = ""):
        self.info = info
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.message}"

    def __repr__(self) -> str:
        return str(vars(self))


def bema_changepoint_exception_wrapper(
    err: Exception, message: str, **info_kwargs: str
) -> BemaChangepointException:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    info = {
        **info_kwargs,
        "exc": err.__class__.__name__,
        "tb": repr(traceback.format_exception(exc_type, exc_value, exc_traceback)),
    }

    e = BemaChangepointException(info=info, message=message)
    return e
