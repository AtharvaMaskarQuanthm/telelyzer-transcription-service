import numpy as np

from dataclasses import dataclass

@dataclass
class LoadAudioFormat:
    audio : np.ndarray
    sampling_rate : int
    channels : int

@dataclass
class AudioWaveFormFormat:
    audio_waveform : np.ndarray
    sampling_rate : int

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