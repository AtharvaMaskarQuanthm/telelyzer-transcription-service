# handler.py - Updated with librosa for base64 option
import runpod
import base64
import tempfile
import os
import librosa

from dataclasses import asdict
from dotenv import load_dotenv
from typing import Dict, Any
from langsmith import traceable, Client

import os

from langfuse import observe, propagate_attributes, Langfuse

from app.services.transcription_service import TranscriptionService
from app.models.transcription_service import TranscriptStatus
from app.models.audio import AudioWaveFormFormat
from app.utils.logger import get_logger

logger = get_logger()
load_dotenv()

langfuse_client = Langfuse(
    public_key="pk-lf-99830cbb-597c-4d14-9087-3f015ed8d67f",
    secret_key="sk-lf-cb623f62-07aa-4ba3-9089-85fcf2c75236",
    base_url="https://cloud.langfuse.com" # US region: https://us.cloud.langfuse.com
)

print(os.getenv("YOUR_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY"))

@observe(name="Handler")
async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for transcription.
    
    Accepts 3 input formats:
    
    1. URL-based:
       {"input": {"audio_url": "https://example.com/audio.mp3"}}
    
    2. Waveform-based:
       {"input": {"audio_waveform": [...], "sampling_rate": 16000}}
    
    3. Base64 audio file:
       {"input": {"audio_base64": "base64_encoded_audio_data", "filename": "audio.mp3"}}
    """

    with propagate_attributes(
        metadata={"service": "Transcription Service"}
    ):

        try:
            job_input = job["input"]
            
            # Option 1: URL-based transcription
            if "audio_url" in job_input:
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name="Process audio url",
                ) as span:
                    audio_url = job_input["audio_url"]
                    logger.info(f"Processing audio from URL: {audio_url}")
                    
                    transcription_service = TranscriptionService(audio_url=audio_url)
                    transcripts = await transcription_service.process()

                    span.update(output = {"transcript": transcripts})
            
            # Option 2: Waveform-based transcription
            elif "audio_waveform" in job_input and "sampling_rate" in job_input:
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name="Process audio url",
                ) as span:
                        logger.info("Processing audio from waveform")
                        
                        audio_format = AudioWaveFormFormat(
                            audio_waveform=job_input["audio_waveform"],
                            sampling_rate=job_input["sampling_rate"]
                        )
                        
                        transcription_service = TranscriptionService(audio_waveform=audio_format)
                        transcripts = await transcription_service.process()

                        span.update(output = {"transcript": transcripts})
            
            # Option 3: Base64 encoded audio file
            elif "audio_base64" in job_input:
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name="Process audio url",
                ) as span:
                    logger.info("Processing audio from base64 encoded file")
                    
                    # Decode base64 audio
                    audio_data = base64.b64decode(job_input["audio_base64"])
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                        temp_file.write(audio_data)
                        temp_file_path = temp_file.name
                    
                    try:
                        # Load the audio with librosa
                        logger.info(f"Loading audio from temp file: {temp_file_path}")
                        audio_waveform, sampling_rate = librosa.load(temp_file_path, sr=None)
                        
                        logger.info(f"Loaded audio: {len(audio_waveform)} samples at {sampling_rate}Hz")
                        
                        # Create AudioWaveFormFormat object
                        audio_format = AudioWaveFormFormat(
                            audio_waveform=audio_waveform.tolist(),
                            sampling_rate=int(sampling_rate)
                        )
                        
                        # Use the waveform approach
                        transcription_service = TranscriptionService(audio_waveform=audio_format)
                        transcripts = await transcription_service.process()

                        span.update(output = {"transcript": transcripts})
                        
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                            logger.info("Cleaned up temporary file")
            
            else:
                return {
                    "error": "Invalid input. Provide one of: 'audio_url', 'audio_waveform' + 'sampling_rate', or 'audio_base64'",
                    "success": False
                }
        
            # Return the transcription result
            return {
                **asdict(transcripts),
                "success": transcripts.status != TranscriptStatus.TRANSCRIPTION_ERROR
            }
        
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            return {
                "error": str(e),
                "success": False
            }


@observe
def main():
    runpod.serverless.start({"handler": handler})

if __name__ == "__main__":
    main()