from dataclasses import dataclass

@dataclass
class TranscriptModel:
    expected_sampling_rate : int = 16000
    expected_channels : int = 2
    mono_channel : int = 1
    mono_sampling_rate : int = 16000

@dataclass
class WhisperModel:
    model_id: str = "openai/whisper-large-v3"
    device: str = "cuda"