"""
API tests for Whisper transcription service
"""

import pytest
import requests
import io
import json
import time
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_AUDIO_FILE = "test.m4a"  # Assumes test file exists


class TestWhisperAPI:
    """Test suite for Whisper API endpoints"""
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data
    
    def test_models_endpoint(self):
        """Test models listing endpoint"""
        response = requests.get(f"{BASE_URL}/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 5
        
        model_names = [model["name"] for model in data["models"]]
        expected_models = ["tiny", "base", "small", "medium", "large"]
        assert set(model_names) == set(expected_models)
    
    def test_api_info_endpoint(self):
        """Test API info endpoint"""
        response = requests.get(f"{BASE_URL}/api")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Whisper Transcription API"
        assert "version" in data
        assert "endpoints" in data
    
    def test_openapi_schema(self):
        """Test OpenAPI schema endpoint"""
        response = requests.get(f"{BASE_URL}/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "/transcribe" in schema["paths"]
    
    def test_docs_endpoint(self):
        """Test that docs endpoint is accessible"""
        response = requests.get(f"{BASE_URL}/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_frontend_endpoint(self):
        """Test that frontend is served at root"""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Whisper Transcription Service" in response.text
    
    @pytest.mark.skipif(not Path(TEST_AUDIO_FILE).exists(), 
                       reason=f"Test audio file {TEST_AUDIO_FILE} not found")
    def test_transcribe_json_format(self):
        """Test transcription with JSON output format"""
        with open(TEST_AUDIO_FILE, 'rb') as f:
            files = {'file': (TEST_AUDIO_FILE, f, 'audio/m4a')}
            data = {
                'model': 'tiny',  # Use fastest model for testing
                'output_format': 'json',
                'language': 'en'
            }
            
            response = requests.post(f"{BASE_URL}/transcribe", files=files, data=data)
            
        assert response.status_code == 200
        result = response.json()
        
        # Check required fields
        assert "text" in result
        assert "segments" in result
        assert "language" in result
        assert "language_probability" in result
        assert "model" in result
        assert "filename" in result
        
        # Check data types
        assert isinstance(result["text"], str)
        assert isinstance(result["segments"], list)
        assert isinstance(result["language"], str)
        assert isinstance(result["language_probability"], float)
        assert result["model"] == "tiny"
    
    @pytest.mark.skipif(not Path(TEST_AUDIO_FILE).exists(), 
                       reason=f"Test audio file {TEST_AUDIO_FILE} not found")
    def test_transcribe_text_format(self):
        """Test transcription with text output format"""
        with open(TEST_AUDIO_FILE, 'rb') as f:
            files = {'file': (TEST_AUDIO_FILE, f, 'audio/m4a')}
            data = {
                'model': 'tiny',
                'output_format': 'txt'
            }
            
            response = requests.post(f"{BASE_URL}/transcribe", files=files, data=data)
            
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        assert len(response.text) > 0
    
    @pytest.mark.skipif(not Path(TEST_AUDIO_FILE).exists(), 
                       reason=f"Test audio file {TEST_AUDIO_FILE} not found")
    def test_transcribe_srt_format(self):
        """Test transcription with SRT output format"""
        with open(TEST_AUDIO_FILE, 'rb') as f:
            files = {'file': (TEST_AUDIO_FILE, f, 'audio/m4a')}
            data = {
                'model': 'tiny',
                'output_format': 'srt'
            }
            
            response = requests.post(f"{BASE_URL}/transcribe", files=files, data=data)
            
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        
        # Check SRT format characteristics
        content = response.text
        assert "-->" in content  # SRT timestamp format
        assert content.strip()  # Not empty
    
    def test_transcribe_invalid_model(self):
        """Test transcription with invalid model"""
        # Create a dummy audio file for testing
        dummy_audio = b"dummy audio data"
        files = {'file': ('test.mp3', io.BytesIO(dummy_audio), 'audio/mp3')}
        data = {'model': 'invalid_model', 'output_format': 'json'}
        
        response = requests.post(f"{BASE_URL}/transcribe", files=files, data=data)
        assert response.status_code == 400
        error = response.json()
        assert "Invalid model" in error["detail"]
    
    def test_transcribe_invalid_format(self):
        """Test transcription with invalid output format"""
        dummy_audio = b"dummy audio data"
        files = {'file': ('test.mp3', io.BytesIO(dummy_audio), 'audio/mp3')}
        data = {'model': 'tiny', 'output_format': 'invalid_format'}
        
        response = requests.post(f"{BASE_URL}/transcribe", files=files, data=data)
        assert response.status_code == 400
        error = response.json()
        assert "Invalid output format" in error["detail"]
    
    def test_transcribe_no_file(self):
        """Test transcription without file"""
        data = {'model': 'tiny', 'output_format': 'json'}
        
        response = requests.post(f"{BASE_URL}/transcribe", data=data)
        assert response.status_code == 422  # Validation error


def test_create_dummy_audio_file():
    """Create a small dummy audio file for testing if none exists"""
    if not Path(TEST_AUDIO_FILE).exists():
        # Create a minimal M4A file structure (this won't actually work for transcription)
        # In a real test environment, you'd want to include a proper test audio file
        dummy_content = b'\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00M4A mp42isom\x00\x00\x00\x08wide'
        with open(TEST_AUDIO_FILE, 'wb') as f:
            f.write(dummy_content)
        print(f"Created dummy test file: {TEST_AUDIO_FILE}")


if __name__ == "__main__":
    # Run tests
    print("Running Whisper API tests...")
    print(f"Testing against: {BASE_URL}")
    
    # Create dummy file if needed
    test_create_dummy_audio_file()
    
    # Basic connectivity test
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health check: {response.status_code}")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the API is running with: docker-compose up whisper-web")
        exit(1)
    
    # Run pytest
    pytest.main([__file__, "-v"])