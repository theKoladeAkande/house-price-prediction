#errors.py
"""custom exception """
class BaseError(Exception):
    """ Base package error"""

class InvalidModelInput(BaseError):
    """Invalid input for model"""