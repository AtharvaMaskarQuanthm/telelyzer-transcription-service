from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.transcription_service import TranscriptionService
from app.models.transcription_service import TranscriptStatus, TranscriptionServiceOutput
from app.models.audio import AudioWaveFormFormat

from app.utils.logger import get_logger

import uvicorn
app = FastAPI()

# Response model
class TranscriptionResponse(TranscriptionServiceOutput):
    success: bool = True

# Request Model
class TranscribeRequest():
    audio_waveform : Optional[AudioWaveFormFormat]

# Health check response model
class HealthCheckResponse(BaseModel):
    status: str
    message: str

logger = get_logger()

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Simple health check endpoint.
    Returns 200 if the service is up.
    """
    return HealthCheckResponse(status="ok", message="Service is running")

@app.get("/transcribe-url", response_model=TranscriptionResponse)
async def transcribe_url(url: str = Query(..., description="Audio file URL to transcribe")):

    try:
        transcription_service = TranscriptionService(audio_url=url)

        transcripts = await transcription_service.process()

    except Exception as e:
        logger.error(f"Error loading audio from provided URL - Check if the URL is valid. Error : {e}")

        raise

    return TranscriptionResponse(**transcripts.model_dump(), success=transcripts.status != TranscriptStatus.TRANSCRIPTION_ERROR)
 
@app.post("/transcribe-waveform", response_model=TranscriptionResponse)
async def transcribe_waveform(request : TranscribeRequest):
    if not request.audio_waveform:
        raise HTTPException(
            status_code=400,
            detail="'audio_waveform' must be provided"
        )
    
    try:
        transcription_service = TranscriptionService(
            audio_url=request.url,
            audio_waveform=request.audio_waveform
        )
        
        transcripts = await transcription_service.process()

    except Exception as e:
        logger.error(f"Error loading audio from provided input. Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return TranscriptionResponse(
        **transcripts.model_dump(),
        success=transcripts.status != TranscriptStatus.TRANSCRIPTION_ERROR
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)