import numpy as np

class DownsampleOutput:
    def __init__(self, left_channel : np.ndarray, right_channel : np.ndarray) -> None:
        self.left_channel = left_channel
        self.right_channel = right_channel