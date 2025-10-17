import torch
from silero_vad import load_silero_vad
from transformers import AutoProcessor
from faster_whisper import WhisperModel
from app.exceptions.model_load_error import ModelLoadError
from app.models.transcription_service import WhisperModelData

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
                cls._whisper_model = WhisperModel(WhisperModelData.model_id, device="cuda", compute_type="float16")

            except Exception as e:
                raise ModelLoadError(f"Failed to load model CTranslate2 Whisper {e}", model_name="CTranslate2 Whisper") from e
        return cls._whisper_model

    @classmethod
    def processor(cls):
        # Huggingface processor for feature extraction
        if cls._processor is None:
            try:
                cls._processor = AutoProcessor.from_pretrained(WhisperModelData.model_id)
            except Exception as e:
                raise ModelLoadError("Failed to load model", model_name="Whisper Processor") from e
        return cls._processor