import librosa
import numpy as np
import os
import time

from dataclasses import dataclass
from dotenv import load_dotenv
from langsmith import traceable
from langfuse import observe

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

@observe
def downsample_audio(audio_left_channel : np.ndarray, audio_right_channel : np.ndarray, original_sampling_rate : int) -> DownsampleOutput:
    
    start_time = time.time()
    resampled_left_channel = librosa.resample(audio_left_channel, orig_sr=original_sampling_rate, target_sr=TranscriptModel.expected_sampling_rate)
    resampled_right_channel = librosa.resample(audio_right_channel, orig_sr=original_sampling_rate, target_sr=TranscriptModel.expected_sampling_rate)
        
    logger.info(f":Audio resampling and channel splitting completed in {time.time() - start_time}s")

    return DownsampleOutput(
            downsampled_left_channel = resampled_left_channel,
            downsampled_right_channel = resampled_right_channel
    )
