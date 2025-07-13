#!/usr/bin/env python3
"""
Whisper CLI transcription tool
Self-contained script for transcribing audio files using OpenAI Whisper
"""

import argparse
import sys
import os
import tempfile
import shutil
from pathlib import Path
from transcription_core import TranscriptionService


def transcribe_audio_streaming(audio_file, model_name="small", output_format="txt", output_dir=None, language=None, no_stream=False):
    """
    Stream transcription results as segments with timestamps.
    """
    temp_file = None
    try:
        # Handle stdin input
        if audio_file == "/dev/stdin":
            print("Reading audio from stdin...", file=sys.stderr)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
            shutil.copyfileobj(sys.stdin.buffer, temp_file)
            temp_file.close()
            audio_file = temp_file.name
            print(f"Audio saved to temporary file: {audio_file}", file=sys.stderr)
        
        if not os.path.exists(audio_file):
            print(f"Error: Audio file '{audio_file}' not found.", file=sys.stderr)
            return False
        
        print(f"Loading Whisper model: {model_name}", file=sys.stderr)
        service = TranscriptionService(model_name)
        
        print(f"Transcribing: {audio_file}", file=sys.stderr)
        print("Processing audio...", file=sys.stderr)
        
        # Get original filename for language detection
        original_filename = audio_file
        if temp_file:
            original_filename = os.environ.get('ORIGINAL_FILENAME', audio_file)
        
        # Determine language
        if not language:
            detected_lang = service.detect_language_from_filename(original_filename)
            if detected_lang:
                language = detected_lang
                print(f"Language detected from filename: {language}", file=sys.stderr)
            else:
                print("Using default languages: German and English", file=sys.stderr)
        
        print("Starting streaming transcription...", file=sys.stderr)
        
        # Transcribe using the service
        result = service.transcribe_file(
            audio_file,
            language=language,
            streaming=False,
            original_filename=original_filename
        )
        
        # Show detected language
        print(f"Detected language: {result['language']}", file=sys.stderr)
        print(f"Detected language probability: {result['language_probability']:.2f}", file=sys.stderr)
        
        if not no_stream:
            print("Streaming transcription:", file=sys.stderr)
            # Output segments with timestamps
            for segment in result['segments']:
                print(f"[{segment['start']:.1f}s] {segment['text']}", flush=True)
                sys.stdout.flush()
        else:
            # Output all text at once
            if result['text']:
                print(result['text'], flush=True)
        
        # Save to file if needed
        if output_format != "txt" or output_dir:
            save_output_file(result, audio_file, output_format, output_dir)
        
        print("Transcription complete!", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}", file=sys.stderr)
        return False
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def save_output_file(result, audio_file, output_format, output_dir):
    """Save transcription to file using TranscriptionService"""
    audio_path = Path(audio_file)
    base_name = audio_path.stem
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = audio_path.parent
    
    # Use TranscriptionService to export the format
    service = TranscriptionService()
    content = service.export_to_format(result, output_format, base_name)
    
    # Save to file
    output_file = output_path / f"{base_name}.{output_format}"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Output saved to: {output_file}", file=sys.stderr)


def transcribe_audio(audio_file, model_name="small", output_format="txt", output_dir=None, language=None):
    """
    Transcribe an audio file using OpenAI Whisper
    """
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.", file=sys.stderr)
        return False
    
    try:
        print(f"Loading Whisper model: {model_name}", file=sys.stderr)
        service = TranscriptionService(model_name)
        
        print(f"Transcribing: {audio_file}", file=sys.stderr)
        print("Processing audio...", file=sys.stderr)
        
        # Determine language settings
        if not language:
            original_filename = os.environ.get('ORIGINAL_FILENAME', audio_file)
            detected_lang = service.detect_language_from_filename(original_filename)
            if detected_lang:
                language = detected_lang
                print(f"Language detected from filename: {language}", file=sys.stderr)
            else:
                print("Using default languages: German and English", file=sys.stderr)
        
        # Transcribe using the service
        result = service.transcribe_file(
            audio_file,
            language=language,
            streaming=False,
            original_filename=original_filename
        )
        
        print("Transcription complete!", file=sys.stderr)
        
        # Prepare output filename
        audio_path = Path(audio_file)
        base_name = audio_path.stem
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = audio_path.parent
        
        # Save transcription using service export
        content = service.export_to_format(result, output_format, base_name)
        output_file = output_path / f"{base_name}.{output_format}"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"Output saved to: {output_file}", file=sys.stderr)
        print(result['text'])
        
        return True
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}", file=sys.stderr)
        return False




def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py audio.wav                    # Stream to stdout (auto-detect language)
  python transcribe.py meeting_en.m4a               # Auto-detect English from filename
  python transcribe.py presentation_de.wav          # Auto-detect German from filename
  python transcribe.py audio.mp3 --language en      # Force English
  python transcribe.py audio.mp3 --format srt       # Save SRT file
  python transcribe.py audio.wav --output-dir ./out # Save to directory
  
Available models (by size and accuracy):
  tiny    - ~39 MB, fastest, least accurate
  base    - ~142 MB, good balance
  small   - ~466 MB, better accuracy (default)
  medium  - ~1.5 GB, high accuracy
  large   - ~2.9 GB, highest accuracy
        """
    )
    
    parser.add_argument(
        "audio_file", 
        help="Path to the audio file to transcribe"
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Whisper model to use (default: small)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "json", "srt", "vtt", "tsv"],
        default="txt",
        help="Output format (default: txt)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save output files (default: same as input file)"
    )
    
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (output all text at once)"
    )
    
    parser.add_argument(
        "--language", "-l",
        help="Force language (e.g., 'en', 'de'). If not specified, auto-detects from filename or defaults to German/English"
    )
    
    args = parser.parse_args()
    
    # Use streaming by default for txt output to stdout
    if args.format == "txt" and not args.output_dir:
        success = transcribe_audio_streaming(
            args.audio_file,
            args.model,
            args.format,
            args.output_dir,
            args.language,
            args.no_stream
        )
    else:
        success = transcribe_audio(
            args.audio_file,
            args.model,
            args.format,
            args.output_dir,
            args.language
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()