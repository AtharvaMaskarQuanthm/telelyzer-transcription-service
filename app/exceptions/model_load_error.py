class ModelLoadError(Exception):
    def __init__(self, message, model_name=None):
        super().__init__(message)
        self.model_name = model_name