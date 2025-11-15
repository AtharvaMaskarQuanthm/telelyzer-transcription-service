from dataclasses import asdict
from fastapi import FastAPI, Query, HTTPException
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel
from typing import Optional, List

from app.services.transcription_service import TranscriptionService
from app.models.transcription_service import TranscriptStatus, TranscriptionServiceOutput
from app.models.audio import AudioWaveFormFormat

from app.utils.logger import get_logger

import uvicorn
app = FastAPI()

# Response model - Create as Pydantic model, not inheriting from dataclass
class TranscriptionResponse(BaseModel):
    transcript: List[dict]  # Adjust based on your actual structure
    status: str
    channels: int
    sampling_rate: int
    success: bool = True

# Request Model
class TranscribeRequest(BaseModel):
    audio_waveform: List[float]
    sampling_rate: int

# Health check response model
class HealthCheckResponse(BaseModel):
    status: str
    message: str

logger = get_logger()

@app.get("/health", response_model=HealthCheckResponse)
@traceable
async def health_check():
    """
    Simple health check endpoint.
    Returns 200 if the service is up.
    """
    return HealthCheckResponse(status="ok", message="Service is running")

@app.get("/transcribe-url", response_model=TranscriptionResponse)
@traceable(run_name="Transcription Service")
async def transcribe_url(url: str = Query(..., description="Audio file URL to transcribe")):
    with traceable(run_name="Transcribe URL"):
        try:
            transcription_service = TranscriptionService(audio_url=url)
            transcripts = await transcription_service.process()

        except Exception as e:
            logger.error(f"Error loading audio from provided URL - Check if the URL is valid. Error : {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return TranscriptionResponse(
            **asdict(transcripts),  # Use asdict() for dataclass
            success=transcripts.status != TranscriptStatus.TRANSCRIPTION_ERROR
        )
 
@app.post("/transcribe-waveform", response_model=TranscriptionResponse)
@traceable(run_name="Transcription Service")
async def transcribe_waveform(request: TranscribeRequest):
    with traceable(run_name="Transcribe Waveform"):
        try:
            # Create AudioWaveFormFormat object from request
            audio_format = AudioWaveFormFormat(
                audio_waveform=request.audio_waveform,
                sampling_rate=request.sampling_rate
            )
            
            transcription_service = TranscriptionService(
                audio_waveform=audio_format  # Pass the AudioWaveFormFormat object
            )
            
            transcripts = await transcription_service.process()

        except Exception as e:
            logger.error(f"Error loading audio from provided input. Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return TranscriptionResponse(
            **asdict(transcripts),  # Use asdict() for dataclass
            success=transcripts.status != TranscriptStatus.TRANSCRIPTION_ERROR
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)