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
from faster_whisper import WhisperModel


def detect_language_from_filename(filename):
    """
    Detect language from filename pattern like file_en.m4a, file_de.m4a, etc.
    Returns language code or None if not detected
    """
    import re
    # Pattern to match _<langcode>.<ext> at the end of filename
    pattern = r'_([a-z]{2})\.[^.]+$'
    match = re.search(pattern, filename.lower())
    return match.group(1) if match else None


def transcribe_audio_streaming(audio_file, model_name="small", output_format="txt", output_dir=None, language=None, no_stream=False):
    """
    Stream transcription results as segments with timestamps.
    
    Note: Due to Whisper's architecture, true real-time streaming isn't possible.
    This function processes the entire audio file and then displays segments with
    timestamps to simulate streaming behavior.
    """
    temp_file = None
    try:
        # Handle stdin input
        if audio_file == "/dev/stdin":
            print("Reading audio from stdin...", file=sys.stderr)
            # Create a temporary file to store the audio data
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
            shutil.copyfileobj(sys.stdin.buffer, temp_file)
            temp_file.close()
            audio_file = temp_file.name
            print(f"Audio saved to temporary file: {audio_file}", file=sys.stderr)
        
        if not os.path.exists(audio_file):
            print(f"Error: Audio file '{audio_file}' not found.", file=sys.stderr)
            return False
        
        print(f"Loading Whisper model: {model_name}", file=sys.stderr)
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        
        print(f"Transcribing: {audio_file}", file=sys.stderr)
        print("Processing audio...", file=sys.stderr)
        
        # Determine language settings
        if not language:
            # Check if language is specified in filename
            original_filename = audio_file
            if temp_file:
                # Get the original filename from environment variable
                original_filename = os.environ.get('ORIGINAL_FILENAME', audio_file)
            
            detected_lang = detect_language_from_filename(original_filename)
            if detected_lang:
                language = detected_lang
                print(f"Language detected from filename: {language}", file=sys.stderr)
            else:
                # Default to German and English
                language = None  # Let Whisper auto-detect
                print("Using default languages: German and English", file=sys.stderr)
        
        # True streaming with faster-whisper
        try:
            print("Starting streaming transcription...", file=sys.stderr)
            
            # Use faster-whisper for true streaming
            segments, info = model.transcribe(
                audio_file,
                beam_size=5,
                word_timestamps=True,
                language=language
            )
            
            # Show detected language
            print(f"Detected language: {info.language}", file=sys.stderr)
            print(f"Detected language probability: {info.language_probability:.2f}", file=sys.stderr)
            
            if not no_stream:
                print("Streaming transcription:", file=sys.stderr)
            
            # Iterate over segments as they are processed (TRUE STREAMING!)
            all_segments = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    all_segments.append(segment)
                    if not no_stream:
                        # Output with timestamp - this happens as segments are processed!
                        print(f"[{segment.start:.1f}s] {text}", flush=True)
                        sys.stdout.flush()
                    else:
                        # For no-stream mode, we'll collect all segments first
                        pass
            
            # For no-stream mode, output all text at once
            if no_stream:
                full_text = ' '.join(segment.text.strip() for segment in all_segments)
                if full_text:
                    print(full_text, flush=True)
            
            # Create result object for file saving
            result = {
                'text': ' '.join(segment.text.strip() for segment in all_segments),
                'segments': [{'start': s.start, 'end': s.end, 'text': s.text} for s in all_segments],
                'language': info.language
            }
                
        except Exception as e:
            print(f"Error during streaming: {e}", file=sys.stderr)
            # Fallback to basic processing
            segments, info = model.transcribe(audio_file, language=language)
            full_text = ' '.join(segment.text.strip() for segment in segments)
            if full_text:
                print(full_text, flush=True)
            result = {'text': full_text, 'segments': [], 'language': info.language}
        
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
    """Save transcription to file"""
    audio_path = Path(audio_file)
    base_name = audio_path.stem
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = audio_path.parent
    
    # Save based on format
    if output_format == "json":
        import json
        output_file = output_path / f"{base_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    elif output_format == "srt":
        output_file = output_path / f"{base_name}.srt"
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    elif output_format == "vtt":
        output_file = output_path / f"{base_name}.vtt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in result["segments"]:
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"{start} --> {end}\n{text}\n\n")
    elif output_format == "tsv":
        output_file = output_path / f"{base_name}.tsv"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("start\tend\ttext\n")
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip().replace("\t", " ")
                f.write(f"{start:.3f}\t{end:.3f}\t{text}\n")
    
    if output_format != "txt":
        print(f"Output saved to: {output_file}", file=sys.stderr)


def transcribe_audio(audio_file, model_name="small", output_format="txt", output_dir=None, language=None):
    """
    Transcribe an audio file using OpenAI Whisper
    
    Args:
        audio_file: Path to the audio file
        model_name: Whisper model to use (tiny, base, small, medium, large)
        output_format: Output format (txt, json, srt, vtt, tsv)
        output_dir: Directory to save output files
    """
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.", file=sys.stderr)
        return False
    
    try:
        print(f"Loading Whisper model: {model_name}", file=sys.stderr)
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        
        print(f"Transcribing: {audio_file}", file=sys.stderr)
        print("Processing audio...", file=sys.stderr)
        
        # Determine language settings (same as streaming function)
        if not language:
            original_filename = os.environ.get('ORIGINAL_FILENAME', audio_file)
            detected_lang = detect_language_from_filename(original_filename)
            if detected_lang:
                language = detected_lang
                print(f"Language detected from filename: {language}", file=sys.stderr)
            else:
                print("Using default languages: German and English", file=sys.stderr)
        
        # Transcribe with faster-whisper
        segments, info = model.transcribe(audio_file, language=language)
        print("Transcription complete!", file=sys.stderr)
        
        # Convert to old format for compatibility
        result = {
            'text': ' '.join(segment.text.strip() for segment in segments),
            'segments': [{'start': s.start, 'end': s.end, 'text': s.text} for s in segments],
            'language': info.language
        }
        
        # Prepare output filename
        audio_path = Path(audio_file)
        base_name = audio_path.stem
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = audio_path.parent
        
        # Save transcription based on format
        if output_format == "txt":
            output_file = output_path / f"{base_name}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
        
        elif output_format == "json":
            import json
            output_file = output_path / f"{base_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        elif output_format == "srt":
            output_file = output_path / f"{base_name}.srt"
            with open(output_file, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result["segments"], 1):
                    start = format_timestamp(segment["start"])
                    end = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        
        elif output_format == "vtt":
            output_file = output_path / f"{base_name}.vtt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for segment in result["segments"]:
                    start = format_timestamp(segment["start"])
                    end = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"{start} --> {end}\n{text}\n\n")
        
        elif output_format == "tsv":
            output_file = output_path / f"{base_name}.tsv"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("start\tend\ttext\n")
                for segment in result["segments"]:
                    start = segment["start"]
                    end = segment["end"]
                    text = segment["text"].strip().replace("\t", " ")
                    f.write(f"{start:.3f}\t{end:.3f}\t{text}\n")
        
        print(f"Output saved to: {output_file}", file=sys.stderr)
        print(result['text'])
        
        return True
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}", file=sys.stderr)
        return False


def format_timestamp(seconds):
    """Convert seconds to SRT/VTT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


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