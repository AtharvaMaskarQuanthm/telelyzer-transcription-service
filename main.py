from fastapi import FastAPI, Query
from pydantic import BaseModel
from app.services.transcription_service import TranscriptionService
from app.schemas.transcription_service_schema import TranscriptStatus, TranscriptionServiceOutput

from app.utils.logger import get_logger

import uvicorn
app = FastAPI()


# Response model
class TranscriptionResponse(TranscriptionServiceOutput):
    success: bool = True

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

@app.get("/transcribe", response_model=TranscriptionResponse)
async def transcribe(url: str = Query(..., description="Audio file URL to transcribe")):

    try:
        transcription_service = TranscriptionService(audio_url=url)

    except Exception as e:
        logger.error(f"Error loading audio from provided URL - Check if the URL is valid. Error : {e}")

        return TranscriptionResponse(
            transcript=None, 
            status=None, 
            channels=None, 
            sampling_rate=None, 
            success=False
            )
        
    transcripts = await transcription_service.process()

    return TranscriptionResponse(**transcripts.model_dump(), success=transcripts.status != TranscriptStatus.TRANSCRIPTION_ERROR)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)