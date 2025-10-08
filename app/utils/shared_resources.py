import torch
from silero_vad import load_silero_vad
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from app.models.transcription_model import WhisperModel


class SharedResources:
    """
    Lazily loads and caches shared resources like models and processors
    to avoid repeated loading and memory overhead.
    """
    _vad_model = None
    _whisper_model = None
    _processor = None

    @classmethod
    def vad_model(cls):
        if cls._vad_model is None:
            cls._vad_model = load_silero_vad()
        return cls._vad_model

    @classmethod
    def whisper_model(cls):
        if cls._whisper_model is None:
            cls._whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WhisperModel.model_id
            ).to(WhisperModel.device)
            cls._whisper_model.eval()
        return cls._whisper_model

    @classmethod
    def processor(cls):
        if cls._processor is None:
            cls._processor = AutoProcessor.from_pretrained(WhisperModel.model_id)
        return cls._processor
