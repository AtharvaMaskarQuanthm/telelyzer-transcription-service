import numpy as np
from pydantic import BaseModel, field_validator
from typing import List

from dataclasses import dataclass

@dataclass
class LoadAudioFormat:
    audio : np.ndarray
    sampling_rate : int
    channels : int

class AudioWaveFormFormat(BaseModel):
    audio_waveform: List[float]  # Changed from np.ndarray
    sampling_rate: int
    
    @field_validator('audio_waveform', mode='before')
    @classmethod
    def convert_numpy_to_list(cls, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    
    # Helper method to get as numpy array when needed
    def get_numpy_array(self) -> np.ndarray:
        return np.array(self.audio_waveform)

@dataclass
class DownsampleOutput:
    left_channel : np.ndarray
    right_channel : np.ndarray

# --- SPEECH TIMESTAMPS 
@dataclass
class SpeechTimestampsChunking:
    max_duration: float = 25.0
    min_duration: float = 3.0
    gap_threshold: float = 1.0
    short_chunk_fallback: float = 0.01
    verbose: bool = False

@dataclass
class SplitAudio:
    sampling_rate: int = 16000
    silence_duration: float = 0.5
    leading_extension = float = 0.5
    trailing_extension = float = 1.0