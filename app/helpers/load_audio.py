import numpy as np

from typing import Optional

from app.helpers.load_audio_from_url import load_audio_from_url
from app.models.audio import LoadAudioFormat, AudioWaveFormFormat

def load_audio(audio_url : Optional[str] = None, audio_waveform : Optional[AudioWaveFormFormat] = None) -> LoadAudioFormat:
    """
    This function loads the audio from the audio url or waveform so it can be used by telelyzer

    Parameters:
        - audio_url (Optiona[str]) -> audio url we wanna use (Can be optional if audio waveform is provided)
        - audio_waveform (Optional[str]) -> audio waveform we wanna use (Can be optional if audio url is provided)

    Returns:
        - audio_data (Tuple[np.ndarray, int, int]) -> audio data we return, consists of the librosa loaded array, sampling rate and 
    """

    if audio_url and audio_waveform is not None:
        raise ValueError("Provide only one of `audio_url` or `audio_waveform`, not both.")

    if not audio_url and audio_waveform is None:
        raise ValueError("One of `audio_url` or `audio_waveform` must be provided.")
    
    # 1. If the audio is in the waveform format
    if audio_waveform is not None:
        print(audio_waveform.__dict__.keys())
        
        return LoadAudioFormat(
            audio=np.array(audio_waveform.audio_waveform), 
            sampling_rate=audio_waveform.sampling_rate, 
            channels=int(np.array(audio_waveform.audio_waveform).ndim)
        )

    # 2. If the audio is in the audio file
    audio_waveform, sampling_rate = load_audio_from_url(audio_file_url=audio_url)

    if not isinstance(audio_waveform, np.ndarray):
        raise ValueError("Audio waveform generated is not a numpy array")

    return LoadAudioFormat(
        audio=audio_waveform, 
        sampling_rate=sampling_rate, 
        channels=int(audio_waveform.ndim)
    )

    