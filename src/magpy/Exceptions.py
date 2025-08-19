"""
magpy exception objects
"""


class MagpyException(Exception):
    """Base class for all exceptions raised by Magpy."""
    ...


class MagpyFileException(MagpyException):
    """Base class for all exceptions raised related to file operations."""
    ...


class MagpyFileNotFoundError(MagpyFileException):
    """Exception raised when a file is not found."""
    ...


class MagpyInvalidObjectError(MagpyFileException):
    """Exception raised when a ROOT object is invalid."""
    ...


class MagpySplineException(MagpyException):
    """Base class for all exceptions raised related to spline operations."""
    ...


class MagpyProbabilityException(MagpyException):
    """Exception raised when a probability calculation fails."""
    ...

class MagpyBinException(MagpyException):
    """Base class for all exceptions raised related to bin operations."""
    ...