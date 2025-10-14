from dataclasses import dataclass
from typing import List

# ---  WHISPER MODEL DATA CLASSES ---
@dataclass
class TranscriptModel:
    expected_sampling_rate : int = 16000
    expected_channels : int = 2
    mono_channel : int = 1
    mono_sampling_rate : int = 16000

@dataclass
class WhisperModel:
    # model_id: str = "AtharvaMaskarQuanthm/whisper-large-v2-hindi-lora-finetuned" # "openai/whisper-large-v3"
    model_id: str = "AtharvaMaskarQuanthm/whisper-large-v2-hindi-lora-finetuned-merged" # "openai/whisper-large-v3"
    device: str = "cuda"
    language : str = "hi"
    task : str = "trabscribe"
    num_beams : int  = 5

# --- TRANSCRIPTION SERVICE DATA CLASSES --- 
@dataclass
class TranscriptStatus:
    SUCCESS = "success"
    SILENT_AUDIO = "silent_audio"
    ONLY_RIGHT_CHANNEL_AUDIO = "ONLY_RIGHT_CHANNEL_AUDIO"
    ONLY_LEFT_CHANNEL_AUDIO = "ONLY_LEFT_CHANNEL_AUDIO"
    TRANSCRIPTION_ERROR = "transcription_error"

@dataclass
class TranscriptFormat:
    start_timestamp: float
    end_timestamp: float
    text: str
    speaker_label: str

@dataclass
class TranscriptionServiceOutput:
    transcript: List[TranscriptFormat]
    status: TranscriptStatus
    channels: int
    sampling_rate: int
