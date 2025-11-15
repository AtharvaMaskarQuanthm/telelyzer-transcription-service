import asyncio
import ctranslate2
import librosa
import os
import torch

import numpy as np

from dotenv import load_dotenv
from silero_vad import get_speech_timestamps
from langsmith import traceable
from langfuse import observe, propagate_attributes, Langfuse

from typing import Dict, List, Literal, Optional, Tuple

from app.exceptions.model_load_error import ModelLoadError

from app.helpers.audio_helpers import downsample_audio, split_channels
from app.helpers.load_audio import load_audio

from app.models.audio import AudioWaveFormFormat, DownsampleOutput, SplitAudio, SpeechTimestampsChunking
from app.models.transcription_service import TranscriptModel, TranscriptionServiceOutput, TranscriptStatus, WhisperModelData

from app.utils.logger import get_logger
from app.utils.shared_resources import SharedResources

logger = get_logger()
load_dotenv()

LANGFUSE_SECRET_KEY = "sk-lf-db4ef20a-6683-4c06-8a6a-bc3880af1bcb"
LANGFUSE_PUBLIC_KEY = "pk-lf-322d9ee4-d538-4bf2-8fca-e58ffa272855"
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"

langfuse_client = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    base_url=LANGFUSE_BASE_URL # US region: https://us.cloud.langfuse.com
)

print(os.getenv("YOUR_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY"))

class TranscriptionService:
    """
    This class handles the Transcription Service
    """

    def __init__(self, audio_url : Optional[str] = None, audio_waveform : Optional[AudioWaveFormFormat] = None):
        
        # 1. Load the audio file
        try:
            self.audio_data = load_audio(audio_url=audio_url, audio_waveform=audio_waveform)
            self.sr = self.audio_data.sampling_rate
            self.channels = self.audio_data.channels
            self.audio = self.audio_data.audio
            logger.info(f"Audio Data loaded successfully")
            logger.info(f"Sampling rate of the audio file is: {self.audio_data.sampling_rate}")
            logger.info(f"Number of channels in the audio file is: {self.audio_data.channels}")
        except Exception as e:
            logger.error(e)
            raise

        # 2. Load the shared resources. 
        try:
            self.vad_model = SharedResources.vad_model()
            logger.info("VAD Model loaded successfully")
            self.whisper_model = SharedResources.whisper_model()
            logger.info("Whisper Model Loaded successfully")
        except ModelLoadError as e:
            logger.error(e)
            raise

    # ---- HELPER FUNCTIONS
    def _ensure_16k_mono(self, waveform: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Return a mono, 16 kHz float32 waveform.
        Accepts (n,), (channels, n), or (n, channels).
        """
        # Make (channels, n) if 2D and probably (n, channels)
        if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.T  # -> (channels, n)

        # Collapse to mono
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)

        target_sr = 16000
        if sr != target_sr:
            waveform = librosa.resample(waveform.astype(float), orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        return waveform, sr
    
    @observe(name="Generate Speech Timestamps")
    async def generate_speech_timestamps(self, audio_16k_mono: np.ndarray) -> List[Dict]:
        """
        Wrapper over Silero VAD. Expects 16 kHz mono array of float32.
        Returns a List[{'start': float, 'end': float}] in seconds.
        """
        try:
            ts = get_speech_timestamps(
                    audio_16k_mono, 
                    self.vad_model,
                    sampling_rate=16000,
                    threshold=0.3,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=2000,
                    speech_pad_ms=500,
                    return_seconds=True
                )
            if not isinstance(ts, list):
                logger.error("VAD returned non-list: %s", type(ts))
                return []

            norm = []
            for seg in ts:
                if isinstance(seg, dict) and ('start' in seg) and ('end' in seg):
                    norm.append({'start': float(seg['start']), 'end': float(seg['end'])})
                else:
                    logger.warning("Skipping malformed VAD segment: %s", seg)

            if norm:
                total_speech = sum(s['end'] - s['start'] for s in norm)
                logger.info(f"VAD found {len(norm)} segments, total speech: {total_speech:.2f}s")
            else:
                logger.warning("VAD found no speech segments!")
            
            return norm
        except Exception as e:
            logger.error(f"generate_speech_timestamps failed: {e}")
            raise

    # ========== NEW METHOD: VAD vs Transcription Comparison ==========
    @observe(name="Compare VAD & Transcription")
    def _compare_vad_vs_transcription(
        self,
        vad_segments: List[Dict],
        transcribed_results: List[Dict]
    ) -> Dict:
        """
        Compare VAD detected speech vs what was actually transcribed
        
        Args:
            vad_segments: Raw VAD output [{'start': float, 'end': float}, ...]
            transcribed_results: Transcription results [{'start_timestamp': float, 'end_timestamp': float, ...}, ...]
        
        Returns:
            Dict with coverage metrics
        """
        if not vad_segments:
            logger.info("No VAD segments to compare")
            return {'coverage_percent': 0, 'status': 'no_vad_segments'}
        
        if not transcribed_results:
            logger.warning("No transcription results to compare!")
            total_vad = sum(s['end'] - s['start'] for s in vad_segments)
            return {
                'vad_detected_duration': total_vad,
                'transcribed_duration': 0,
                'coverage_percent': 0,
                'missed_speech_duration': total_vad,
                'status': 'nothing_transcribed'
            }
        
        # Convert transcription results to same format as VAD
        transcribed_segments = [
            {
                'start': result['start_timestamp'],
                'end': result['end_timestamp']
            }
            for result in transcribed_results
        ]
        
        # Calculate totals
        total_vad_duration = sum(seg['end'] - seg['start'] for seg in vad_segments)
        total_transcribed_duration = sum(seg['end'] - seg['start'] for seg in transcribed_segments)
        
        # Find how much of VAD-detected speech was actually transcribed
        covered_duration = 0
        missed_segments = []
        
        for vad_seg in vad_segments:
            vad_duration = vad_seg['end'] - vad_seg['start']
            seg_covered = 0
            
            # Check overlap with all transcribed segments
            for trans_seg in transcribed_segments:
                overlap_start = max(vad_seg['start'], trans_seg['start'])
                overlap_end = min(vad_seg['end'], trans_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                seg_covered += overlap
            
            covered_duration += seg_covered
            
            # If less than 50% of this VAD segment was transcribed, it's likely missed
            if seg_covered < vad_duration * 0.5:

                missed_segments.append({
                    'start': vad_seg['start'],
                    'end': vad_seg['end'],
                    'duration': vad_duration,
                    'covered': seg_covered,
                    'coverage_ratio': seg_covered / vad_duration if vad_duration > 0 else 0
                })
        
        missed_duration = total_vad_duration - covered_duration
        coverage_percent = (covered_duration / total_vad_duration * 100) if total_vad_duration > 0 else 0
        missed_percent = (missed_duration / total_vad_duration * 100) if total_vad_duration > 0 else 0
        
        # Log results
        langfuse_client.update_current_span(
                    metadata={
                        "total_vad_duration": total_vad_duration, 
                        "total_vad_segments": vad_segments, 
                        "total_transcribed_duration": total_transcribed_duration, 
                        "total_transcribed_segments": transcribed_segments, 
                        "speech_coverage_percentage": coverage_percent, 
                        "missed_speech_duration": missed_duration, 
                        "missed_speech_percentage": missed_percent
                    }
                )
        logger.info(f"="*60)
        logger.info(f"VAD vs TRANSCRIPTION COMPARISON")
        logger.info(f"="*60)
        logger.info(f"VAD Detected Speech:    {total_vad_duration:.2f}s ({len(vad_segments)} segments)")
        logger.info(f"Actually Transcribed:   {total_transcribed_duration:.2f}s ({len(transcribed_segments)} chunks)")
        logger.info(f"Speech Coverage:        {coverage_percent:.2f}%")
        logger.info(f"Missed Speech:          {missed_duration:.2f}s ({missed_percent:.2f}%)")
        
        if missed_percent < 2:
            logger.info(f"✅ EXCELLENT: < 2% speech missed")
        elif missed_percent < 5:
            logger.info(f"✅ GOOD: < 5% speech missed")
        elif missed_percent < 10:
            logger.warning(f"⚠️ FAIR: 5-10% speech missed - consider lowering VAD threshold to 0.25")
        else:
            logger.warning(f"❌ POOR: > 10% speech missed - VAD parameters need adjustment!")
        
        if missed_segments:
            logger.info(f"Missed segments: {len(missed_segments)}")
            # Log top 3 missed segments
            sorted_missed = sorted(missed_segments, key=lambda x: x['duration'], reverse=True)[:3]
            for i, seg in enumerate(sorted_missed, 1):
                logger.info(f"  {i}. {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s) "
                           f"- {seg['coverage_ratio']*100:.0f}% covered")
        
        logger.info(f"="*60)
        
        return {
            'vad_detected_duration': total_vad_duration,
            'transcribed_duration': total_transcribed_duration,
            'covered_duration': covered_duration,
            'coverage_percent': coverage_percent,
            'missed_duration': missed_duration,
            'missed_percent': missed_percent,
            'missed_segments': missed_segments,
            'status': 'analyzed'
        }

    @observe(name="Speech Timestamp Chunking Algorithm")
    async def _speech_timestamps_chunking_algorithm(
        self,
        speech_timestamps: List[Dict],
        max_duration: float = SpeechTimestampsChunking.max_duration,
        min_duration: float = SpeechTimestampsChunking.min_duration,
        gap_threshold: float = SpeechTimestampsChunking.gap_threshold,
        short_chunk_fallback: float = SpeechTimestampsChunking.short_chunk_fallback,
        padding: float = 0.2,
        overlap: float = 0.1,
        verbose: bool = SpeechTimestampsChunking.verbose,
        audio_duration: float = None
    ) -> List[Dict]:
        """
        Merge VAD segments into chunks obeying duration and gap rules.
        Input/Output: List of dicts with 'start','end' (seconds).
        """
        try:
            if not speech_timestamps:
                return []

            if not isinstance(speech_timestamps, list):
                raise TypeError(f"speech_timestamps must be a list, got {type(speech_timestamps)}")

            if speech_timestamps and not isinstance(speech_timestamps[0], dict):
                raise TypeError(f"speech_timestamps items must be dict, got {type(speech_timestamps[0])}")

            # Step 1: Add padding to all segments
            padded_segments = []
            for seg in speech_timestamps:
                seg = seg.copy()
                seg['start'] = max(0, seg['start'] - padding)
                if audio_duration:
                    seg['end'] = min(audio_duration, seg['end'] + padding)
                else:
                    seg['end'] = seg['end'] + padding
                
                seg_duration = seg['end'] - seg['start']
                if seg_duration <= 0:
                    if verbose:
                        print(f"Skipping non-positive duration seg: {seg}")
                    continue
                    
                padded_segments.append(seg)
            
            if not padded_segments:
                return []

            # Step 2: Merge segments into chunks
            chunks = []
            current = padded_segments[0].copy()

            for seg in padded_segments[1:]:
                gap = seg['start'] - current['end']
                combined_duration = seg['end'] - current['start']

                if gap <= gap_threshold and combined_duration <= max_duration:
                    current['end'] = seg['end']
                else:
                    chunks.append(current)
                    current = seg.copy()

            if current:
                chunks.append(current)

            # Step 3: Ensure minimum duration by merging small chunks
            merged_chunks = []
            i = 0
            while i < len(chunks):
                chunk = chunks[i].copy()
                chunk_duration = chunk['end'] - chunk['start']
                
                while chunk_duration < min_duration and i + 1 < len(chunks):
                    next_chunk = chunks[i + 1]
                    gap = next_chunk['start'] - chunk['end']
                    
                    if gap <= max_duration:
                        if verbose:
                            print(f"Merging short chunk [{chunk['start']:.2f}, {chunk['end']:.2f}] "
                                f"with next [{next_chunk['start']:.2f}, {next_chunk['end']:.2f}]")
                        chunk['end'] = next_chunk['end']
                        i += 1
                        chunk_duration = chunk['end'] - chunk['start']
                    else:
                        break
                
                if chunk_duration < min_duration:
                    if chunk_duration >= short_chunk_fallback:
                        if verbose:
                            print(f"Keeping short chunk [{chunk['start']:.2f}, {chunk['end']:.2f}] "
                                f"({chunk_duration:.2f}s) - meets fallback threshold")
                        merged_chunks.append(chunk)
                    else:
                        if merged_chunks:
                            prev_gap = chunk['start'] - merged_chunks[-1]['end']
                            if prev_gap <= max_duration * 2:
                                if verbose:
                                    print(f"Rescuing short chunk [{chunk['start']:.2f}, {chunk['end']:.2f}] "
                                        f"by merging with previous")
                                merged_chunks[-1]['end'] = chunk['end']
                            else:
                                if verbose:
                                    print(f"WARNING: Forced to keep very short chunk [{chunk['start']:.2f}, "
                                        f"{chunk['end']:.2f}] ({chunk_duration:.2f}s) - isolated segment")
                                merged_chunks.append(chunk)
                        else:
                            if verbose:
                                print(f"WARNING: Keeping very short first chunk [{chunk['start']:.2f}, "
                                    f"{chunk['end']:.2f}] ({chunk_duration:.2f}s)")
                            merged_chunks.append(chunk)
                else:
                    merged_chunks.append(chunk)
                
                i += 1

            # Step 4: Add overlap between chunks
            final_chunks = []
            for i, chunk in enumerate(merged_chunks):
                chunk = chunk.copy()
                
                if i > 0 and overlap > 0:
                    chunk['start'] = max(merged_chunks[i-1]['end'] - overlap, chunk['start'])
                
                if i < len(merged_chunks) - 1 and overlap > 0:
                    chunk['end'] = min(merged_chunks[i+1]['start'] + overlap, chunk['end'])
                
                final_chunks.append(chunk)

            if verbose:
                print(f"\nChunking summary:")
                print(f"  Input segments: {len(speech_timestamps)}")
                print(f"  Output chunks: {len(final_chunks)}")
                total_speech = sum(c['end'] - c['start'] for c in final_chunks)
                print(f"  Total speech duration: {total_speech:.2f}s")

            return final_chunks

        except Exception as e:
            logger.error(f"Error in Speech Timestamps algorithm: {e}")
            raise
    
    @observe(name="Split Audio Channels")
    async def _split_audio(
        self,
        audio: np.ndarray,
        start_timestamp: float,
        end_timestamp: float,
        sampling_rate: int = TranscriptModel.expected_sampling_rate,
        silence_duration: float = 0.1,
        leading_extension: float = 0.0,
        trailing_extension: float = 0.0,
    ) -> Dict:

        def normalize_rms(a: np.ndarray, target_dBFS: float = -20.0) -> np.ndarray:
            rms = np.sqrt(np.mean(a ** 2))
            if rms == 0:
                return a
            current_dBFS = 20 * np.log10(rms)
            factor = 10 ** ((target_dBFS - current_dBFS) / 20)
            return a * factor

        try:
            start_sample = max(0, int(start_timestamp * sampling_rate))
            end_sample = min(len(audio), int(end_timestamp * sampling_rate))

            if end_sample <= start_sample:
                logger.warning(f"Invalid split range: start={start_sample}, end={end_sample}")
                return None

            chunk = audio[start_sample:end_sample]
            chunk = normalize_rms(chunk)

            silence = np.zeros(int(silence_duration * sampling_rate), dtype=audio.dtype)

            return {
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "audio_chunk": np.concatenate([silence, chunk, silence])
            }
        except Exception as e:
            logger.error(f"Error splitting audio into chunks: {e}")
            raise
    
    async def _transcribe_audio(self, audio_chunk_data: dict, speaker_label: str = "") -> dict:
        try:
            if not audio_chunk_data:
                return None

            ct2_model = SharedResources.whisper_model()
            audio_chunk = audio_chunk_data['audio_chunk']
            segments, info = ct2_model.transcribe(audio_chunk, beam_size=5, language="hi", vad_filter=False)
            text = ' '.join([s.text for s in segments])

            return {
                "start_timestamp": audio_chunk_data['start_timestamp'],
                "end_timestamp": audio_chunk_data['end_timestamp'],
                "text": text.strip(),
                "speaker_label": speaker_label,
            }
        except Exception as e:
            logger.error(f"Error translating audio using Whisper Module : {e}")
            raise
    
    @observe(name="Transcribe Stereo Calls")
    async def _transcribe_stereo_calls(self):
        """
        This function analyzes the stereo calls
        """
        try:
            left = self.downsampled_audio.downsampled_left_channel
            right = self.downsampled_audio.downsampled_right_channel
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="ensure audio is 16K mono",
            ) as span:
                left_16k, _ = self._ensure_16k_mono(left, TranscriptModel.expected_sampling_rate if self.sr is None else self.sr)
                right_16k, _ = self._ensure_16k_mono(right, TranscriptModel.expected_sampling_rate if self.sr is None else self.sr)

            # Generate speech timestamps - SAVE THESE!
            left_ts_task = self.generate_speech_timestamps(left_16k)
            right_ts_task = self.generate_speech_timestamps(right_16k)
            left_ts, right_ts = await asyncio.gather(left_ts_task, right_ts_task)

            if not left_ts and not right_ts:
                return TranscriptionServiceOutput(
                    transcript=[],
                    status=TranscriptStatus.SILENT_AUDIO,
                    channels=TranscriptModel.expected_channels,
                    sampling_rate=TranscriptModel.expected_sampling_rate
                )

            # Chunking
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="Speech Timestamp Chunking Algorithm",
            ) as span:
                left_chunks_task = self._speech_timestamps_chunking_algorithm(left_ts) if left_ts else asyncio.sleep(0, result=[])
                right_chunks_task = self._speech_timestamps_chunking_algorithm(right_ts) if right_ts else asyncio.sleep(0, result=[])
                left_chunks, right_chunks = await asyncio.gather(left_chunks_task, right_chunks_task)

                left_chunks = left_chunks or []
                right_chunks = right_chunks or []

            langfuse_client.update_current_span(
                metadata={
                    "num_left_chunk": len(left_chunks), 
                    "num_right_chunk": len(right_chunks), 
                    "total_chunks": len(left_chunks) + len(right_chunks)
                }
            )

            # Split into audio chunks
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="Split Audio Chunks",
            ) as span:
                left_split_tasks = [self._split_audio(left_16k, c['start'], c['end']) for c in left_chunks]
                right_split_tasks = [self._split_audio(right_16k, c['start'], c['end']) for c in right_chunks]
                left_splits, right_splits = await asyncio.gather(
                    asyncio.gather(*left_split_tasks),
                    asyncio.gather(*right_split_tasks)
                )
                left_splits = [c for c in (left_splits or []) if c is not None]
                right_splits = [c for c in (right_splits or []) if c is not None]

            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="Transcribe Audio",
            ) as span:
                # Transcribe
                left_tx_tasks = [self._transcribe_audio(c, speaker_label="Left Channel") for c in left_splits]
                right_tx_tasks = [self._transcribe_audio(c, speaker_label="Right Channel") for c in right_splits]

                left_results, right_results = await asyncio.gather(
                    asyncio.gather(*left_tx_tasks) if left_tx_tasks else asyncio.sleep(0, result=[]),
                    asyncio.gather(*right_tx_tasks) if right_tx_tasks else asyncio.sleep(0, result=[])
                )

                left_results = [r for r in (left_results or []) if r is not None]
                right_results = [r for r in (right_results or []) if r is not None]

            # ========== NEW: Compare VAD vs Transcription ==========
            if left_ts and left_results:
                self._compare_vad_vs_transcription(left_ts, left_results)
            if right_ts and right_results:
                self._compare_vad_vs_transcription(right_ts, right_results)

            if not left_results and right_results:
                return TranscriptionServiceOutput(
                    transcript=right_results,
                    status=TranscriptStatus.ONLY_RIGHT_CHANNEL_AUDIO,
                    channels=TranscriptModel.expected_channels,
                    sampling_rate=TranscriptModel.expected_sampling_rate
                )

            if left_results and not right_results:
                return TranscriptionServiceOutput(
                    transcript=left_results,
                    status=TranscriptStatus.ONLY_LEFT_CHANNEL_AUDIO,
                    channels=TranscriptModel.expected_channels,
                    sampling_rate=TranscriptModel.expected_sampling_rate
                )

            all_chunks = left_results + right_results
            sorted_transcripts = sorted(all_chunks, key=lambda x: x['start_timestamp'])

            return TranscriptionServiceOutput(
                transcript=sorted_transcripts,
                status=TranscriptStatus.SUCCESS,
                channels=TranscriptModel.expected_channels,
                sampling_rate=TranscriptModel.expected_sampling_rate
            )

        except Exception as e:
            logger.error(f"Error transcribing stereo calls : {e}")
            raise
    
    @observe(name="Transcribe Mono Calls")
    async def _transcribe_mono_calls(self):
        """
        This function transcribes mono calls
        """
        try:


            mono_16k, _ = self._ensure_16k_mono(self.downsampled_audio, self.sr)

            # 1) VAD - SAVE THIS!
            vad_segments = await self.generate_speech_timestamps(mono_16k)
            
            if not vad_segments:
                return TranscriptionServiceOutput(
                    transcript=[],
                    status=TranscriptStatus.SILENT_AUDIO,
                    channels=TranscriptModel.mono_channel,
                    sampling_rate=TranscriptModel.mono_sampling_rate
                )

            # 2) Chunking
            audio_duration = len(mono_16k) / 16000
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="Speech Timestamps Chunking Algorithm",
            ) as span:
                chunked = await self._speech_timestamps_chunking_algorithm(
                    vad_segments,  # Pass VAD segments
                    audio_duration=audio_duration,
                    verbose=True
                )

            # Log coverage
            if chunked:
                total_covered = sum(c['end'] - c['start'] for c in chunked)
                coverage = (total_covered / audio_duration) * 100
                logger.info(f"Audio coverage: {coverage:.1f}% ({len(chunked)} chunks)")
                
                if coverage < 90:
                    logger.warning(f"LOW COVERAGE: Only {coverage:.1f}% of audio will be transcribed!")

            if not chunked:
                return TranscriptionServiceOutput(
                    transcript=[],
                    status=TranscriptStatus.SILENT_AUDIO,
                    channels=TranscriptModel.mono_channel,
                    sampling_rate=TranscriptModel.mono_sampling_rate
                )

            # 3) Split audio
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="Split Audio",
            ) as span:
                audio_chunk_tasks = [self._split_audio(mono_16k, c['start'], c['end']) for c in chunked]
                self.audio_chunks = await asyncio.gather(*audio_chunk_tasks)
                self.audio_chunks = [c for c in (self.audio_chunks or []) if c is not None]

            if not self.audio_chunks:
                return TranscriptionServiceOutput(
                    transcript=[],
                    status=TranscriptStatus.SILENT_AUDIO,
                    channels=TranscriptModel.mono_channel,
                    sampling_rate=TranscriptModel.mono_sampling_rate
                )

            # 4) Transcribe
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="Transcribe Audio",
            ) as span:
                transcribe_audio_chunks_tasks = [self._transcribe_audio(chunk, speaker_label="") for chunk in self.audio_chunks]
                transcription_results = await asyncio.gather(*transcribe_audio_chunks_tasks)
                transcription_results = [r for r in (transcription_results or []) if r is not None]

            # ========== NEW: Compare VAD vs Transcription ==========
            if vad_segments and transcription_results:
                self._compare_vad_vs_transcription(vad_segments, transcription_results)

            return TranscriptionServiceOutput(
                transcript=transcription_results,
                status=TranscriptStatus.SUCCESS,
                channels=TranscriptModel.mono_channel,
                sampling_rate=TranscriptModel.mono_sampling_rate
            )

        except Exception as e:
            logger.error(f"Error transcribing mono audio : {e}")
            raise
    
    @observe()
    async def process(self) -> TranscriptionServiceOutput:
        """
        This function processes the audio file
        """

        if self.sr == TranscriptModel.expected_sampling_rate and self.channels == TranscriptModel.expected_channels:
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="Transcribe Mono 16k calls",
            ) as span:
                with propagate_attributes(
                    metadata={"sampling_rate": self.sr, "channels": self.channels}
                ):
                    self.audio_split_channels = split_channels(audio=self.audio)
                    self.downsampled_audio = DownsampleOutput(
                        downsampled_left_channel=self.audio_split_channels.left_channel,
                        downsampled_right_channel=self.audio_split_channels.right_channel
                    )
                    self.raw_transcripts = await self._transcribe_stereo_calls()

                    span.update(output=self.raw_transcripts)

                    return self.raw_transcripts

        elif self.sr > TranscriptModel.expected_sampling_rate and self.channels == TranscriptModel.expected_channels:
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="Transcribe Higher Sampling rate Stereo",
            ) as span:
                with propagate_attributes(
                    metadata={"sampling_rate": self.sr, "channels": self.channels}
                ):  
                    self.audio_split_channels = split_channels(audio=self.audio)
                    self.downsampled_audio = downsample_audio(
                        audio_left_channel=self.audio_split_channels.left_channel,
                        audio_right_channel=self.audio_split_channels.right_channel,
                        original_sampling_rate=self.sr
                    )

                    if self.downsampled_audio is None:
                        return TranscriptionServiceOutput(
                            transcript=[],
                            status=TranscriptStatus.TRANSCRIPTION_ERROR,
                            channels=self.channels,
                            sampling_rate=self.sr
                        )

                    self.raw_transcripts = await self._transcribe_stereo_calls()
                    span.update(output=self.raw_transcripts)

                    return self.raw_transcripts

        elif self.sr < TranscriptModel.expected_sampling_rate and self.channels < TranscriptModel.expected_channels:
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="Transcribe Low Sampling rate Mono",
            ) as span:
                with propagate_attributes(
                    metadata={"sampling_rate": self.sr, "channels": self.channels}
                ):  
                    self.downsampled_audio = self.audio
                    self.raw_transcripts = await self._transcribe_mono_calls()

                    span.update(output=self.raw_transcripts)

                    return self.raw_transcripts
        
        elif self.sr > TranscriptModel.expected_sampling_rate and self.channels < TranscriptModel.expected_channels:
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name="Transcribe High Sampling rate Mono",
            ) as span:
                with propagate_attributes(
                    metadata={"sampling_rate": self.sr, "channels": self.channels}
                ):  
                    self.downsampled_audio = self.audio

                    self.raw_transcripts = await self._transcribe_mono_calls()

                    span.update(output=self.raw_transcripts)

                    return self.raw_transcripts

        else:
            logger.error(f"Unsupported file format")
            raise ValueError(f"Unsupported audio format: sr={self.sr}, channels={self.channels}")