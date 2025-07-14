"""
Python client for Whisper Transcription API
Uses async endpoints for all transcription operations
"""

import requests
import json
from typing import Optional, Dict, Any, Union, BinaryIO
from pathlib import Path


class WhisperAPIError(Exception):
    """Exception raised for API errors"""
    pass


class WhisperClient:
    """Python client for Whisper Transcription API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize Whisper API client
        
        Args:
            base_url: Base URL of the Whisper API service
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            if not response.ok:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                except:
                    error_detail = response.text or f"HTTP {response.status_code}"
                raise WhisperAPIError(f"API request failed: {error_detail}")
            return response
        except requests.RequestException as e:
            raise WhisperAPIError(f"Network error: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status
        
        Returns:
            Dict containing health status information
        """
        response = self._make_request('GET', '/health')
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """
        List available Whisper models
        
        Returns:
            Dict containing list of available models with descriptions
        """
        response = self._make_request('GET', '/models')
        return response.json()
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information
        
        Returns:
            Dict containing API name, version, and endpoints
        """
        response = self._make_request('GET', '/api')
        return response.json()
    
    def transcribe(
        self,
        audio_file: Union[str, Path, BinaryIO],
        model: str = "small",
        language: Optional[str] = None,
        output_format: str = "json",
        streaming: bool = False
    ) -> Union[Dict[str, Any], str]:
        """
        Transcribe an audio file
        
        Args:
            audio_file: Path to audio file or file-like object
            model: Whisper model to use (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'de') or None for auto-detection
            output_format: Output format (json, txt, srt, vtt, tsv)
            streaming: Return segments as they're processed (only for json format)
        
        Returns:
            Transcription result as dict (if json format) or string (other formats)
        
        Raises:
            WhisperAPIError: If transcription fails
            FileNotFoundError: If audio file doesn't exist
        """
        # Prepare file for upload
        if isinstance(audio_file, (str, Path)):
            audio_file = Path(audio_file)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
            with open(audio_file, 'rb') as f:
                files = {'file': (audio_file.name, f, self._get_mime_type(audio_file))}
                return self._do_transcribe(files, model, language, output_format, streaming)
        else:
            # Assume it's a file-like object
            filename = getattr(audio_file, 'name', 'audio_file')
            files = {'file': (filename, audio_file, 'audio/wav')}
            return self._do_transcribe(files, model, language, output_format, streaming)
    
    def _do_transcribe(self, files, model, language, output_format, streaming):
        """Internal method to perform transcription"""
        data = {
            'model': model,
            'output_format': output_format,
            'streaming': streaming
        }
        
        if language:
            data['language'] = language
        
        response = self._make_request('POST', '/transcribe', files=files, data=data)
        
        # Return appropriate format based on content type
        content_type = response.headers.get('content-type', '')
        if 'application/json' in content_type:
            return response.json()
        else:
            return response.text
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for audio file"""
        extension = file_path.suffix.lower()
        mime_types = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/m4a',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg',
            '.wma': 'audio/x-ms-wma',
            '.aac': 'audio/aac',
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska'
        }
        return mime_types.get(extension, 'application/octet-stream')
    
    def transcribe_file(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Union[Dict[str, Any], str]:
        """
        Convenience method to transcribe a file by path
        
        Args:
            file_path: Path to the audio file
            **kwargs: Additional arguments for transcribe()
        
        Returns:
            Transcription result
        """
        return self.transcribe(file_path, **kwargs)
    
    def transcribe_to_text(
        self,
        audio_file: Union[str, Path, BinaryIO],
        model: str = "small",
        language: Optional[str] = None
    ) -> str:
        """
        Convenience method to transcribe and return plain text
        
        Args:
            audio_file: Path to audio file or file-like object
            model: Whisper model to use
            language: Language code or None for auto-detection
        
        Returns:
            Transcribed text as string
        """
        result = self.transcribe(audio_file, model=model, language=language, output_format="txt")
        return result if isinstance(result, str) else result.get('text', '')
    
    def transcribe_to_srt(
        self,
        audio_file: Union[str, Path, BinaryIO],
        model: str = "small",
        language: Optional[str] = None
    ) -> str:
        """
        Convenience method to transcribe and return SRT subtitles
        
        Args:
            audio_file: Path to audio file or file-like object
            model: Whisper model to use
            language: Language code or None for auto-detection
        
        Returns:
            SRT formatted subtitles as string
        """
        return self.transcribe(audio_file, model=model, language=language, output_format="srt")
    
    def transcribe_with_segments(
        self,
        audio_file: Union[str, Path, BinaryIO],
        model: str = "small",
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to transcribe and return detailed results with segments
        
        Args:
            audio_file: Path to audio file or file-like object
            model: Whisper model to use
            language: Language code or None for auto-detection
        
        Returns:
            Detailed transcription result with segments
        """
        result = self.transcribe(audio_file, model=model, language=language, output_format="json")
        if not isinstance(result, dict):
            raise WhisperAPIError("Expected JSON response but got text")
        return result


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = WhisperClient("http://localhost:8000")
    
    # Test connection
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
        
        models = client.list_models()
        print(f"Available models: {[m['name'] for m in models['models']]}")
        
        # Example transcription (uncomment to test with actual file)
        # result = client.transcribe_to_text("audio.mp3", model="small")
        # print(f"Transcription: {result}")
        
    except WhisperAPIError as e:
        print(f"API Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
