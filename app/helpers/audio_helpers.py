import librosa
import numpy as np
import os
import time

from dataclasses import dataclass
from dotenv import load_dotenv
from langsmith import traceable
from langfuse import observe, Langfuse

from app.models.transcription_service import TranscriptModel
from app.utils.logger import get_logger

logger = get_logger()
load_dotenv()

@dataclass
class SplitChannelsOutput:
    left_channel : np.ndarray
    right_channel : np.ndarray

@dataclass
class DownsampleOutput:
    downsampled_left_channel : np.ndarray
    downsampled_right_channel : np.ndarray

LANGFUSE_SECRET_KEY = "sk-lf-db4ef20a-6683-4c06-8a6a-bc3880af1bcb"
LANGFUSE_PUBLIC_KEY = "pk-lf-322d9ee4-d538-4bf2-8fca-e58ffa272855"
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"

langfuse_client = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    base_url=LANGFUSE_BASE_URL # US region: https://us.cloud.langfuse.com
)


print(os.getenv("YOUR_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY"))

@observe(name="Split Channels")
def split_channels(audio: np.ndarray) -> SplitChannelsOutput:
    """
    This helper function split the channels into 2 channels. 

    Parameters:
        - audio (np.ndarray) : An array you wanna split. 

    Returns:
        - split_channels (SplitChannelsOutput) : Channels split in 2
    """

    return SplitChannelsOutput(
        left_channel = audio[0], 
        right_channel = audio[1]
    )

@observe()
def downsample_audio(audio_left_channel : np.ndarray, audio_right_channel : np.ndarray, original_sampling_rate : int) -> DownsampleOutput:
    
    start_time = time.time()
    resampled_left_channel = librosa.resample(audio_left_channel, orig_sr=original_sampling_rate, target_sr=TranscriptModel.expected_sampling_rate)
    resampled_right_channel = librosa.resample(audio_right_channel, orig_sr=original_sampling_rate, target_sr=TranscriptModel.expected_sampling_rate)
        
    logger.info(f":Audio resampling and channel splitting completed in {time.time() - start_time}s")

    return DownsampleOutput(
            downsampled_left_channel = resampled_left_channel,
            downsampled_right_channel = resampled_right_channel
    )
