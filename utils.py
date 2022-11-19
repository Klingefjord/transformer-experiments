class Config:
    """A class to hold dynamic configuration parameters for a model."""

    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
