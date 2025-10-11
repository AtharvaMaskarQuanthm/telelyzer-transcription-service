import torch
from silero_vad import load_silero_vad
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from app.exceptions.model_load_error import ModelLoadError

from app.models.transcription_service import WhisperModel

class SharedResources:
    _vad_model = None
    _whisper_model = None
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
        if cls._whisper_model is None:
            try:
                cls._whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    WhisperModel.model_id
                ).to(WhisperModel.device)
                cls._whisper_model.eval()
            except Exception as e:
                raise ModelLoadError("Failed to load model", model_name="Whisper") from e
        return cls._whisper_model

    @classmethod
    def processor(cls):
        if cls._processor is None:
            try:
                cls._processor = AutoProcessor.from_pretrained(WhisperModel.model_id)
            except Exception as e:
                raise ModelLoadError("Failed to load model", model_name="Whisper Processor") from e
        return cls._processor

