import torch
from silero_vad import load_silero_vad
from transformers import AutoProcessor
import ctranslate2
from app.exceptions.model_load_error import ModelLoadError
from app.models.transcription_service import WhisperModel

class SharedResources:
    _vad_model = None
    _whisper_model = None  # CTranslate2 Whisper
    _processor = None

    @classmethod
    def vad_model(cls):
        if cls._vad_model is None:
            try:
                cls._vad_model = load_silero_vad()
            except Exception as e:
                raise ModelLoadError("Failed to load model", model_name="VAD") from e
        return cls._vad_model

    @classmethod
    def whisper_model(cls):
        # Load CTranslate2 Whisper Model instead of HuggingFace
        if cls._whisper_model is None:
            try:
                cls._whisper_model = ctranslate2.models.Whisper(
                    WhisperModel.model_path, device=WhisperModel.device  # use correct model_path and device
                )
            except Exception as e:
                raise ModelLoadError("Failed to load model", model_name="CTranslate2 Whisper") from e
        return cls._whisper_model

    @classmethod
    def processor(cls):
        # Huggingface processor for feature extraction
        if cls._processor is None:
            try:
                cls._processor = AutoProcessor.from_pretrained(WhisperModel.model_id)
            except Exception as e:
                raise ModelLoadError("Failed to load model", model_name="Whisper Processor") from e
        return cls._processor