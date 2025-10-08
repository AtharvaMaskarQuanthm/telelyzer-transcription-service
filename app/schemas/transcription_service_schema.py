from enum import Enum
from typing import List
from pydantic import BaseModel

class TranscriptStatus(str, Enum):  # Make it a string Enum for cleaner output
    SUCCESS = "success"
    SILENT_AUDIO = "silent_audio"
    ONLY_AGENT_SPOKE = "only_agent_spoke"
    ONLY_CUSTOMER_SPOKE = "only_customer_spoke"
    TRANSCRIPTION_ERROR = "transcription_error"


class TranscriptFormat(BaseModel):
    start_timestamp: float
    end_timestamp: float
    text: str
    speaker_label: str

    
class TranscriptionServiceOutput(BaseModel):
    transcript: List[TranscriptFormat]
    status: TranscriptStatus
    channels: int
    sampling_rate: int
