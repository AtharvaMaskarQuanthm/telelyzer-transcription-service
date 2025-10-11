import io
import librosa
import requests

import numpy as np

from typing import Tuple

def load_audio_from_url(audio_file_url: str) -> Tuple[np.ndarray, int]:
    try:
        response = requests.get(audio_file_url)
        response.raise_for_status()
        audio_bytes = io.BytesIO(response.content)
        audio, sr = librosa.load(audio_bytes, sr=None, mono=False)
        return audio, sr
        
    except Exception as e:
        raise ValueError(f"Failed to load audio from url : {e}")