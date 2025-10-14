import asyncio
import librosa
import torch

import numpy as np

from silero_vad import get_speech_timestamps
from typing import Dict, List, Literal, Optional, Tuple

from app.exceptions.model_load_error import ModelLoadError

from app.helpers.audio_helpers import downsample_audio, split_channels
from app.helpers.load_audio import load_audio

from app.models.audio import AudioWaveFormFormat, DownsampleOutput, SplitAudio, SpeechTimestampsChunking
from app.models.transcription_service import TranscriptModel, TranscriptionServiceOutput, TranscriptStatus, WhisperModel

from app.utils.logger import get_logger
from app.utils.shared_resources import SharedResources

logger = get_logger()

class TranscriptionService:
    """
    This class handles the Transcription Service
    """

    def __init__(self, audio_url : Optional[str] = None, audio_waveform : Optional[AudioWaveFormFormat] = None):
        
        # 1. Load the audio file
        try:
            self.audio_data = load_audio(audio_url=audio_url, audio_waveform=audio_waveform)
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
            self.processor = SharedResources.processor()
            logger.info("Whisper Processor Loaded successfully")
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

    async def generate_speech_timestamps(self, audio_16k_mono: np.ndarray) -> List[Dict]:
        """
        Wrapper over Silero VAD. Expects 16 kHz mono array of float32.
        Returns a List[{'start': float, 'end': float}] in seconds.
        """
        try:
            ts = get_speech_timestamps(audio_16k_mono, self.vad_model, sampling_rate=16000, return_seconds=True)
            if not isinstance(ts, list):
                logger.error("VAD returned non-list: %s", type(ts))
                return []

            norm = []
            for seg in ts:
                if isinstance(seg, dict) and ('start' in seg) and ('end' in seg):
                    # Ensure seconds (float)
                    norm.append({'start': float(seg['start']), 'end': float(seg['end'])})
                else:
                    logger.warning("Skipping malformed VAD segment: %s", seg)
            return norm
        except Exception as e:
            logger.error(f"generate_speech_timestamps failed: {e}")
            raise

    async def _speech_timestamps_chunking_algorithm(
        self,
        speech_timestamps: List[Dict],
        max_duration: float = SpeechTimestampsChunking.max_duration,
        min_duration: float = SpeechTimestampsChunking.min_duration,
        gap_threshold: float = SpeechTimestampsChunking.gap_threshold,
        short_chunk_fallback: float = SpeechTimestampsChunking.short_chunk_fallback,
        verbose: bool = SpeechTimestampsChunking.verbose
    ) -> List[Dict]:
        """
        Merge VAD segments into chunks obeying duration and gap rules.
        Input/Output: List of dicts with 'start','end' (seconds).
        """
        try:
            if not speech_timestamps:
                return []

            # Defensive check on shape
            if not isinstance(speech_timestamps, list):
                raise TypeError(f"speech_timestamps must be a list, got {type(speech_timestamps)}")

            if speech_timestamps and not isinstance(speech_timestamps[0], dict):
                raise TypeError(f"speech_timestamps items must be dict, got {type(speech_timestamps[0])}")

            chunks = []
            current = None

            for seg in speech_timestamps:
                seg = seg.copy()
                seg_duration = seg['end'] - seg['start']
                if seg_duration <= 0:
                    if verbose:
                        print(f"Skipping non-positive duration seg: {seg}")
                    continue

                if current is None:
                    current = seg
                    continue

                gap = round(seg['start'] - current['end'], 6)
                combined_duration = seg['end'] - current['start']

                if gap <= gap_threshold and combined_duration <= max_duration:
                    current['end'] = seg['end']
                elif combined_duration <= min_duration and gap <= gap_threshold:
                    # still merge small additions
                    current['end'] = seg['end']
                else:
                    current_duration = current['end'] - current['start']

                    if current_duration >= min_duration:
                        chunks.append(current)
                    else:
                        if chunks and (current['start'] - chunks[-1]['end']) <= gap_threshold:
                            if verbose:
                                print(f"Backward-merged short chunk [{current['start']}, {current['end']}] into previous.")
                            chunks[-1]['end'] = current['end']
                        elif current_duration >= short_chunk_fallback:
                            chunks.append(current)
                            if verbose:
                                print(f"Preserved short chunk [{current['start']}, {current['end']}]")
                        else:
                            if verbose:
                                print(f"Dropped too-short chunk [{current['start']}, {current['end']}]")

                    current = seg

            if current:
                final_duration = current['end'] - current['start']
                if final_duration >= min_duration:
                    chunks.append(current)
                elif chunks and (current['start'] - chunks[-1]['end']) <= gap_threshold:
                    chunks[-1]['end'] = current['end']
                    if verbose:
                        print(f"Final chunk backward-merged [{current['start']}, {current['end']}]")
                elif final_duration >= short_chunk_fallback:
                    chunks.append(current)
                    if verbose:
                        print(f"Final short chunk preserved [{current['start']}, {current['end']}]")
                elif verbose:
                    print(f"Final chunk dropped [{current['start']}, {current['end']}]")

            return chunks

        except Exception as e:
            logger.error(f"Error in Speech Timestamps algorithm: {e}")
            raise

    async def _split_audio(
        self,
        audio: np.ndarray,
        start_timestamp: float,
        end_timestamp: float,
        sampling_rate: int = TranscriptModel.expected_sampling_rate,
        silence_duration: float = SplitAudio.silence_duration,
        leading_extension: float = SplitAudio.leading_extension,
        trailing_extension: float = SplitAudio.trailing_extension
    ) -> Dict:

        def normalize_rms(a: np.ndarray, target_dBFS: float = -20.0) -> np.ndarray:
            rms = np.sqrt(np.mean(a ** 2))
            if rms == 0:
                return a
            current_dBFS = 20 * np.log10(rms)
            factor = 10 ** ((target_dBFS - current_dBFS) / 20)
            return a * factor

        try:
            start_sample = max(0, int((start_timestamp - leading_extension) * sampling_rate))
            end_sample = min(len(audio), int((end_timestamp + trailing_extension) * sampling_rate))

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

    async def _transcribe_audio(self, audio_chunk_data: Dict, speaker_label: Literal["Left Channel", "Right Channel", ""]) -> Dict:
        try:
            if not audio_chunk_data:
                return None

            audio_chunk = audio_chunk_data['audio_chunk']

            # Whisper feature extractor expects 16k
            inputs = self.processor(audio_chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(WhisperModel.device) for k, v in inputs.items()}

            generate_kwargs = {
                "forced_decoder_ids": self.processor.get_decoder_prompt_ids(
                    language=WhisperModel.language,
                    task=WhisperModel.task
                ),
                "num_beams": WhisperModel.num_beams
            }

            with torch.no_grad():
                generated_ids = self.whisper_model.generate(**inputs, **generate_kwargs)

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return {
                "start_timestamp": audio_chunk_data['start_timestamp'],
                "end_timestamp": audio_chunk_data['end_timestamp'],
                "text": text.strip(),
                "speaker_label": speaker_label
            }

        except Exception as e:
            logger.error(f"Error translating audio using Whisper Module : {e}")
            raise
        
    async def _transcribe_stereo_calls(self):
        """
        This function analyzes the stereo calls
        """
        try:
            # self.downsampled_audio is a DownsampleOutput with .downsampled_left_channel / .downsampled_right_channel
            left = self.downsampled_audio.downsampled_left_channel
            right = self.downsampled_audio.downsampled_right_channel

            # Ensure both sides are 16k mono arrays (some splitters may preserve shapes differently)
            left_16k, _ = self._ensure_16k_mono(left, TranscriptModel.expected_sampling_rate if self.sr is None else self.sr)
            right_16k, _ = self._ensure_16k_mono(right, TranscriptModel.expected_sampling_rate if self.sr is None else self.sr)

            # Generate speech timestamps
            left_ts_task = self.generate_speech_timestamps(left)
            right_ts_task = self.generate_speech_timestamps(right)
            left_ts, right_ts = await asyncio.gather(left_ts_task, right_ts_task)

            if not left_ts and not right_ts:
                return TranscriptionServiceOutput(
                    transcript=[],
                    status=TranscriptStatus.SILENT_AUDIO,
                    channels=TranscriptModel.expected_channels,
                    sampling_rate=TranscriptModel.expected_sampling_rate
                )

            # Chunking (each returns List[Dict] or None)
            left_chunks_task = self._speech_timestamps_chunking_algorithm(left_ts) if left_ts else asyncio.sleep(0, result=[])
            right_chunks_task = self._speech_timestamps_chunking_algorithm(right_ts) if right_ts else asyncio.sleep(0, result=[])
            left_chunks, right_chunks = await asyncio.gather(left_chunks_task, right_chunks_task)

            left_chunks = left_chunks or []
            right_chunks = right_chunks or []

            # Split into audio chunks
            left_split_tasks = [self._split_audio(left_16k, c['start'], c['end']) for c in left_chunks]
            right_split_tasks = [self._split_audio(right_16k, c['start'], c['end']) for c in right_chunks]
            left_splits, right_splits = await asyncio.gather(
                asyncio.gather(*left_split_tasks),
                asyncio.gather(*right_split_tasks)
            )
            left_splits = [c for c in (left_splits or []) if c is not None]
            right_splits = [c for c in (right_splits or []) if c is not None]

            # Transcribe
            left_tx_tasks = [self._transcribe_audio(c, speaker_label="Left Channel") for c in left_splits]
            right_tx_tasks = [self._transcribe_audio(c, speaker_label="Right Channel") for c in right_splits]

            left_results, right_results = await asyncio.gather(
                asyncio.gather(*left_tx_tasks) if left_tx_tasks else asyncio.sleep(0, result=[]),
                asyncio.gather(*right_tx_tasks) if right_tx_tasks else asyncio.sleep(0, result=[])
            )

            left_results = [r for r in (left_results or []) if r is not None]
            right_results = [r for r in (right_results or []) if r is not None]

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

    async def _transcribe_mono_calls(self):
        """
        This function transcribes mono calls
        """
        try:
            # Ensure 16k mono BEFORE everything (8 kHz input was your crash source)
            mono_16k, _ = self._ensure_16k_mono(self.downsampled_audio, self.sr)

            # 1) VAD
            ts = await self.generate_speech_timestamps(mono_16k)  # List[Dict]
            if not ts:
                return TranscriptionServiceOutput(
                    transcript=[],
                    status=TranscriptStatus.SILENT_AUDIO,
                    channels=TranscriptModel.mono_channel,
                    sampling_rate=TranscriptModel.mono_sampling_rate
                )

            # 2) Chunking
            chunked = await self._speech_timestamps_chunking_algorithm(ts)
            if not chunked:
                return TranscriptionServiceOutput(
                    transcript=[],
                    status=TranscriptStatus.SILENT_AUDIO,
                    channels=TranscriptModel.mono_channel,
                    sampling_rate=TranscriptModel.mono_sampling_rate
                )

            # 3) Split audio
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

            # 4) Transcribe (FIXED: use audio_chunks, not left_channel_chunks)
            transcribe_audio_chunks_tasks = [self._transcribe_audio(chunk, speaker_label="") for chunk in self.audio_chunks]
            transcription_results = await asyncio.gather(*transcribe_audio_chunks_tasks)
            transcription_results = [r for r in (transcription_results or []) if r is not None]

            return TranscriptionServiceOutput(
                transcript=transcription_results,
                status=TranscriptStatus.SUCCESS,
                channels=TranscriptModel.mono_channel,
                sampling_rate=TranscriptModel.mono_sampling_rate
            )

        except Exception as e:
            logger.error(f"Error transcribing mono audio : {e}")

            raise

    async def process(self) -> TranscriptionServiceOutput:
        """
        This function processes the audio file
        """

        # 1. Check if there's a need for downsampling
    
        # 1.1 If the audio is 16k stereo, no resample needed
        if self.sr == TranscriptModel.expected_sampling_rate and self.channels == TranscriptModel.expected_channels:
            # Split but keep native 16k
            self.audio_split_channels = split_channels(audio=self.audio)

            # Make it match what downstream expects
            self.downsampled_audio = DownsampleOutput(
                downsampled_left_channel=self.audio_split_channels.left_channel,
                downsampled_right_channel=self.audio_split_channels.right_channel
            )

            self.raw_transcripts = await self._transcribe_stereo_calls()
            return self.raw_transcripts

        elif self.sr > TranscriptModel.expected_sampling_rate and self.channels == TranscriptModel.expected_channels:
            # Need for downsampling and splitting to 16k
            self.audio_split_channels = split_channels(audio=self.audio)

            # Use your existing util to resample to 16k per channel
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
            return self.raw_transcripts

        elif self.sr < TranscriptModel.expected_sampling_rate and self.channels < TranscriptModel.expected_channels:
            # Mono and lower than 16k — handle as mono and upsample inside the mono pipeline
            self.downsampled_audio = self.audio  # raw mono; _ensure_16k_mono() will fix SR
            self.raw_transcripts = await self._transcribe_mono_calls()
            return self.raw_transcripts

        else:
            logger.error(f"Unsupported file format")
            raise ValueError(f"Unsupported audio format: sr={self.sr}, channels={self.channels}")