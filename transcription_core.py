"""
Core transcription functionality extracted for reuse
"""

import os
import sys
import tempfile
import shutil
import re
import threading
import logging
from pathlib import Path
from faster_whisper import WhisperModel
import structlog
from datetime import datetime, timezone

# Set up logging
logger = structlog.get_logger()


class TranscriptionService:
    """Core transcription service that can be used by both CLI and web interface"""
    
    def __init__(self, model_name="small"):
        self.model_name = model_name
        self.model = None
        self._model_lock = threading.Lock()
    
    def _load_model(self):
        """Lazy load the Whisper model (thread-safe)"""
        if self.model is None:
            with self._model_lock:
                # Double-check pattern to avoid race conditions
                if self.model is None:
                    logger.info("Loading Whisper model", model=self.model_name)
                    print(f"[{datetime.now(timezone.utc).isoformat()}] Loading Whisper model: {self.model_name}", flush=True)
                    self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                    logger.info("Whisper model loaded successfully", model=self.model_name)
                    print(f"[{datetime.now(timezone.utc).isoformat()}] Whisper model loaded: {self.model_name}", flush=True)
        return self.model
    
    def detect_language_from_filename(self, filename):
        """
        Detect language from filename pattern like file_en.m4a, file_de.m4a, etc.
        Returns language code or None if not detected
        """
        pattern = r'_([a-z]{2})\.[^.]+$'
        match = re.search(pattern, filename.lower())
        return match.group(1) if match else None
    
    def format_timestamp(self, seconds):
        """Convert seconds to SRT/VTT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def transcribe_file(self, audio_file_path, language=None, streaming=False, original_filename=None):
        """
        Transcribe an audio file and return structured results
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code or None for auto-detection
            streaming: Whether to return segments as they're processed
            original_filename: Original filename for language detection
            
        Returns:
            dict: {
                'text': str,
                'segments': list,
                'language': str,
                'language_probability': float
            }
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file '{audio_file_path}' not found")
        
        model = self._load_model()
        
        # Determine language settings
        if not language:
            # Check if language is specified in filename
            filename_to_check = original_filename or audio_file_path
            detected_lang = self.detect_language_from_filename(filename_to_check)
            if detected_lang:
                language = detected_lang
        
        # Transcribe with faster-whisper
        segments, info = model.transcribe(
            audio_file_path,
            beam_size=5,
            word_timestamps=True,
            language=language
        )
        
        # Process segments
        all_segments = []
        full_text_parts = []
        
        # Convert generator to list for processing
        for segment in segments:
            text = segment.text.strip()
            if text:
                segment_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': text
                }
                all_segments.append(segment_data)
                full_text_parts.append(text)
        
        # Return final result
        result = {
            'text': ' '.join(full_text_parts),
            'segments': all_segments,
            'language': info.language,
            'language_probability': info.language_probability
        }
        
        return result
    
    def transcribe_buffer(self, audio_buffer, filename=None, language=None, streaming=False):
        """
        Transcribe audio from a buffer/bytes object
        
        Args:
            audio_buffer: Audio data as bytes
            filename: Original filename for language detection
            language: Language code or None for auto-detection
            streaming: Whether to return segments as they're processed
            
        Returns:
            dict: Same as transcribe_file
        """
        # Determine file extension from original filename or use generic
        if filename:
            import os
            _, ext = os.path.splitext(filename)
            if ext.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.mp4', '.mov', '.avi', '.mkv']:
                suffix = ext.lower()
            else:
                suffix = '.wav'  # Default to WAV for unknown extensions
        else:
            suffix = '.wav'  # Default when no filename provided
            
        # Create temporary file with correct extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            temp_file.write(audio_buffer)
            temp_file.close()
            
            return self.transcribe_file(
                temp_file.name,
                language=language,
                streaming=streaming,
                original_filename=filename
            )
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def export_to_format(self, result, output_format, base_filename):
        """
        Export transcription result to different formats
        
        Args:
            result: Transcription result dict
            output_format: Format (txt, json, srt, vtt, tsv)
            base_filename: Base filename for output
            
        Returns:
            str: Formatted content
        """
        if output_format == "txt":
            return result["text"]
        
        elif output_format == "json":
            import json
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        elif output_format == "srt":
            content = []
            for i, segment in enumerate(result["segments"], 1):
                start = self.format_timestamp(segment["start"])
                end = self.format_timestamp(segment["end"])
                text = segment["text"].strip()
                content.append(f"{i}\n{start} --> {end}\n{text}\n")
            return "\n".join(content)
        
        elif output_format == "vtt":
            content = ["WEBVTT\n"]
            for segment in result["segments"]:
                start = self.format_timestamp(segment["start"])
                end = self.format_timestamp(segment["end"])
                text = segment["text"].strip()
                content.append(f"{start} --> {end}\n{text}\n")
            return "\n".join(content)
        
        elif output_format == "tsv":
            content = ["start\tend\ttext"]
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip().replace("\t", " ")
                content.append(f"{start:.3f}\t{end:.3f}\t{text}")
            return "\n".join(content)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")