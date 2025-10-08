from dataclasses import dataclass

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