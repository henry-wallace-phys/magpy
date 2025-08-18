'''
magpy exception objects
'''

class MagpyException(Exception):
    """Base class for all exceptions raised by Magpy."""
    pass

class MagpyFileException(MagpyException):
    """Base class for all exceptions raised related to file operations.
    """
    ...


class MagpyFileNotFoundError(MagpyFileException):
    """Exception raised when a file is not found."""
    pass

class MagpyInvalidObjectError(MagpyFileException):
    """Exception raised when a ROOT object is invalid."""
    pass

class MagpySplineException(MagpyException):
    """Base class for all exceptions raised related to spline operations."""
    pass
