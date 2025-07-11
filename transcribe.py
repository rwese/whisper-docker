#!/usr/bin/env python3
"""
Whisper CLI transcription tool
Self-contained script for transcribing audio files using OpenAI Whisper
"""

import argparse
import sys
import os
import whisper
from pathlib import Path


def transcribe_audio_streaming(audio_file, model_name="small", output_format="txt", output_dir=None):
    """
    Stream transcription results as they are processed
    """
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.", file=sys.stderr)
        return False
    
    try:
        print(f"Loading Whisper model: {model_name}", file=sys.stderr)
        model = whisper.load_model(model_name)
        
        print(f"Transcribing: {audio_file}", file=sys.stderr)
        print("Streaming transcription...", file=sys.stderr)
        
        # Load audio and get segments
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Detect the spoken language
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        print(f"Detected language: {detected_lang}", file=sys.stderr)
        
        # Decode the audio
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        
        # Stream the result immediately
        print(result.text)
        
        # Save to file if needed
        if output_format != "txt" or output_dir:
            # For non-streaming output formats, we need the full result
            full_result = model.transcribe(audio_file, verbose=False)
            save_output_file(full_result, audio_file, output_format, output_dir)
        
        return True
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}", file=sys.stderr)
        return False


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


def transcribe_audio(audio_file, model_name="small", output_format="txt", output_dir=None):
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
        
        # Stream transcription results
        segments = model.transcribe(audio_file, verbose=True)
        print("Transcription complete!", file=sys.stderr)
        
        # For streaming, we'll output segments as they're processed
        result = segments
        
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
  python transcribe.py audio.wav                    # Stream to stdout
  python transcribe.py audio.m4a --model small      # Stream with small model
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
        "--stream", "-s",
        action="store_true",
        help="Stream transcription results in real-time (default: enabled)"
    )
    
    args = parser.parse_args()
    
    # Use streaming by default for txt output to stdout
    if args.format == "txt" and not args.output_dir:
        success = transcribe_audio_streaming(
            args.audio_file,
            args.model,
            args.format,
            args.output_dir
        )
    else:
        success = transcribe_audio(
            args.audio_file,
            args.model,
            args.format,
            args.output_dir
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()