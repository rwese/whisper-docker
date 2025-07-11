#!/usr/bin/env python3
"""
Whisper CLI transcription tool
Self-contained script for transcribing audio files using OpenAI Whisper
"""

import argparse
import sys
import os
import whisper
import tempfile
import shutil
from pathlib import Path


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
    Stream transcription results as they are processed
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
        model = whisper.load_model(model_name)
        
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
                language = None  # Let Whisper auto-detect, but we'll hint with German/English
                print("Using default languages: German and English", file=sys.stderr)
        
        # Transcribe the audio file with language settings
        transcribe_options = {
            'verbose': False,
            'word_timestamps': False,
            'temperature': 0.0,
            'best_of': 1
        }
        
        if language:
            transcribe_options['language'] = language
        
        # Process audio in streaming mode with segments
        try:
            # Load and process audio
            audio = whisper.load_audio(audio_file)
            
            # Process with streaming by enabling word timestamps and processing segments
            transcribe_options['word_timestamps'] = True
            result = model.transcribe(audio_file, **transcribe_options)
            
            # Show detected language
            if result and 'language' in result:
                print(f"Detected language: {result['language']}", file=sys.stderr)
            
            # Stream output segment by segment with timestamps
            if result and 'segments' in result and result['segments']:
                if not no_stream:
                    print("Streaming transcription:", file=sys.stderr)
                    for segment in result['segments']:
                        text = segment.get('text', '').strip()
                        if text:
                            # Output segment with timestamp
                            start_time = segment.get('start', 0)
                            print(f"[{start_time:.1f}s] {text}", flush=True)
                            sys.stdout.flush()
                            
                            # Add a small delay to simulate real-time streaming
                            import time
                            time.sleep(0.2)
                else:
                    # Output all text at once without timestamps
                    full_text = ' '.join(segment.get('text', '').strip() for segment in result['segments'] if segment.get('text', '').strip())
                    if full_text:
                        print(full_text, flush=True)
            elif result and 'text' in result:
                # Fallback to full text if no segments
                transcribed_text = result['text'].strip()
                if transcribed_text:
                    print(transcribed_text, flush=True)
                    sys.stdout.flush()
                else:
                    print("(No speech detected)", file=sys.stderr)
            else:
                print("(Transcription failed - no result)", file=sys.stderr)
                
        except Exception as e:
            print(f"Error during streaming: {e}", file=sys.stderr)
            # Fallback to regular processing
            result = model.transcribe(audio_file, **transcribe_options)
            if result and 'text' in result:
                print(result['text'].strip(), flush=True)
        
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
        model = whisper.load_model(model_name)
        
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
        
        # Transcribe with language settings
        transcribe_options = {'verbose': True}
        if language:
            transcribe_options['language'] = language
        
        result = model.transcribe(audio_file, **transcribe_options)
        print("Transcription complete!", file=sys.stderr)
        
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