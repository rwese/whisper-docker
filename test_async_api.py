"""
Tests for async transcription API endpoints
"""

import pytest
import time
import json
import hashlib
from pathlib import Path
import tempfile
import requests
from typing import Optional


class TestAsyncAPI:
    """Test async transcription API endpoints"""
    
    BASE_URL = "http://localhost:8000"
    
    @classmethod
    def setup_class(cls):
        """Setup test class with sample audio file"""
        # Create a small test audio file (empty file for testing)
        cls.test_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        cls.test_file.write(b"RIFF" + b"\x00" * 44)  # Minimal WAV header
        cls.test_file.close()
        cls.test_filename = Path(cls.test_file.name).name
        
        # Calculate expected file hash
        with open(cls.test_file.name, 'rb') as f:
            cls.test_file_hash = hashlib.sha256(f.read()).hexdigest()
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test files"""
        Path(cls.test_file.name).unlink(missing_ok=True)
    
    def test_health_endpoint(self):
        """Test that the health endpoint is working"""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_models_endpoint(self):
        """Test that the models endpoint returns available models"""
        response = requests.get(f"{self.BASE_URL}/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
        
        # Check for expected models
        model_names = [model["name"] for model in data["models"]]
        expected_models = ["tiny", "base", "small", "medium", "large"]
        for model in expected_models:
            assert model in model_names
    
    def test_submit_async_task(self):
        """Test submitting an async transcription task"""
        with open(self.test_file.name, 'rb') as f:
            files = {'file': (self.test_filename, f, 'audio/wav')}
            data = {
                'model': 'tiny',
                'language': 'en',
                'output_format': 'json'
            }
            
            response = requests.post(f"{self.BASE_URL}/transcribe/async", files=files, data=data)
            
        assert response.status_code == 200
        result = response.json()
        
        # Check response structure
        assert "task_id" in result
        assert "filename" in result
        assert "file_hash" in result
        assert "created_at" in result
        assert "duplicate" in result
        
        # Check values
        assert result["filename"] == self.test_filename
        assert result["file_hash"] == self.test_file_hash
        assert isinstance(result["duplicate"], bool)
        
        return result["task_id"]
    
    def test_get_task_status(self):
        """Test getting task status"""
        # First submit a task
        task_id = self.test_submit_async_task()
        
        # Get task status
        response = requests.get(f"{self.BASE_URL}/tasks/{task_id}")
        assert response.status_code == 200
        
        status = response.json()
        
        # Check status structure
        required_fields = [
            "task_id", "filename", "file_hash", "status", "model",
            "language", "output_format", "created_at", "ttl_expires_at",
            "file_size"
        ]
        for field in required_fields:
            assert field in status
        
        # Check values
        assert status["task_id"] == task_id
        assert status["filename"] == self.test_filename
        assert status["file_hash"] == self.test_file_hash
        assert status["status"] in ["pending", "processing", "completed", "failed"]
        assert status["model"] == "tiny"
        assert status["language"] == "en"
        assert status["output_format"] == "json"
        
        return task_id, status
    
    def test_wait_for_task_completion(self):
        """Test waiting for task completion and getting result"""
        # Submit task and get initial status
        task_id, initial_status = self.test_get_task_status()
        
        # Wait for completion (with timeout)
        max_wait = 30  # 30 seconds timeout
        wait_interval = 2
        waited = 0
        
        while waited < max_wait:
            response = requests.get(f"{self.BASE_URL}/tasks/{task_id}")
            assert response.status_code == 200
            
            status = response.json()
            
            if status["status"] == "completed":
                break
            elif status["status"] == "failed":
                pytest.fail(f"Task failed: {status.get('error_message', 'Unknown error')}")
            
            time.sleep(wait_interval)
            waited += wait_interval
        
        # Check that task completed
        assert status["status"] == "completed"
        assert status["started_at"] is not None
        assert status["completed_at"] is not None
        assert status["processing_duration_seconds"] is not None
        
        return task_id
    
    def test_get_task_result(self):
        """Test getting task result"""
        # Wait for a task to complete
        task_id = self.test_wait_for_task_completion()
        
        # Get result
        response = requests.get(f"{self.BASE_URL}/tasks/{task_id}/result")
        assert response.status_code == 200
        
        result = response.json()
        
        # Check result structure
        required_fields = ["text", "segments", "language", "language_probability", "character_count", "task_metadata"]
        for field in required_fields:
            assert field in result
        
        # Check task metadata
        task_meta = result["task_metadata"]
        assert task_meta["task_id"] == task_id
        assert task_meta["status"] == "completed"
        
        # Check transcription data
        assert isinstance(result["text"], str)
        assert isinstance(result["segments"], list)
        assert isinstance(result["language"], str)
        assert isinstance(result["language_probability"], float)
        assert isinstance(result["character_count"], int)
        
        return result
    
    def test_get_task_result_different_formats(self):
        """Test getting task result in different formats"""
        # Wait for a task to complete
        task_id = self.test_wait_for_task_completion()
        
        # Test different format outputs
        formats_to_test = ["txt", "srt", "vtt", "tsv"]
        
        for format_type in formats_to_test:
            response = requests.get(f"{self.BASE_URL}/tasks/{task_id}/result?format={format_type}")
            assert response.status_code == 200
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if format_type == "txt":
                assert "text/plain" in content_type
            elif format_type in ["srt", "vtt"]:
                assert "text/plain" in content_type
            elif format_type == "tsv":
                assert "text/tab-separated-values" in content_type
            
            # Check that we got text content
            content = response.text
            assert isinstance(content, str)
            assert len(content) > 0
    
    def test_duplicate_file_handling(self):
        """Test that duplicate files return existing task"""
        # Submit first task
        with open(self.test_file.name, 'rb') as f:
            files = {'file': (self.test_filename, f, 'audio/wav')}
            data = {'model': 'tiny', 'output_format': 'json'}
            response1 = requests.post(f"{self.BASE_URL}/transcribe/async", files=files, data=data)
        
        assert response1.status_code == 200
        result1 = response1.json()
        first_task_id = result1["task_id"]
        
        # Submit same file again
        with open(self.test_file.name, 'rb') as f:
            files = {'file': (self.test_filename, f, 'audio/wav')}
            data = {'model': 'tiny', 'output_format': 'json'}
            response2 = requests.post(f"{self.BASE_URL}/transcribe/async", files=files, data=data)
        
        assert response2.status_code == 200
        result2 = response2.json()
        
        # Should return the same task or indicate duplicate
        assert result2["file_hash"] == result1["file_hash"]
        
        # If it's a new task, it should be marked as duplicate
        # If it's the same task, task_id should match
        if result2["task_id"] != first_task_id:
            assert result2.get("duplicate", False) == True
    
    def test_invalid_task_id(self):
        """Test handling of invalid task ID"""
        invalid_task_id = "00000000-0000-0000-0000-000000000000"
        
        # Test status endpoint
        response = requests.get(f"{self.BASE_URL}/tasks/{invalid_task_id}")
        assert response.status_code == 404
        
        # Test result endpoint
        response = requests.get(f"{self.BASE_URL}/tasks/{invalid_task_id}/result")
        assert response.status_code == 404
    
    def test_pending_task_result(self):
        """Test getting result for pending task"""
        # Submit a task but don't wait for completion
        task_id = self.test_submit_async_task()
        
        # Immediately try to get result (should return 202)
        response = requests.get(f"{self.BASE_URL}/tasks/{task_id}/result")
        
        # Should be 202 (Accepted) if still pending/processing, or 200 if very quick
        assert response.status_code in [202, 200]
        
        if response.status_code == 202:
            error_data = response.json()
            assert "detail" in error_data
            assert "pending" in error_data["detail"] or "processing" in error_data["detail"]
    
    def test_api_info_endpoint(self):
        """Test that API info includes async endpoints"""
        response = requests.get(f"{self.BASE_URL}/api")
        assert response.status_code == 200
        
        data = response.json()
        assert "endpoints" in data
        
        endpoints = data["endpoints"]
        assert "POST /transcribe/async" in endpoints
        assert "GET /tasks/{task_id}" in endpoints
        assert "GET /tasks/{task_id}/result" in endpoints


if __name__ == "__main__":
    # Run tests manually (requires running web service)
    print("Running async API tests...")
    print("Make sure the web service is running on http://localhost:8000")
    
    # Basic connectivity test
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Web service not responding correctly")
            exit(1)
        print("✅ Web service is running")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to web service at http://localhost:8000")
        print("   Please start the service with: python web_service.py")
        exit(1)
    
    # Run the test class
    test_instance = TestAsyncAPI()
    test_instance.setup_class()
    
    try:
        # Run tests in order
        print("\n🧪 Running tests...")
        
        test_instance.test_health_endpoint()
        print("✅ Health endpoint test passed")
        
        test_instance.test_models_endpoint()
        print("✅ Models endpoint test passed")
        
        test_instance.test_submit_async_task()
        print("✅ Submit async task test passed")
        
        test_instance.test_get_task_status()
        print("✅ Get task status test passed")
        
        test_instance.test_wait_for_task_completion()
        print("✅ Task completion test passed")
        
        test_instance.test_get_task_result()
        print("✅ Get task result test passed")
        
        test_instance.test_get_task_result_different_formats()
        print("✅ Different format results test passed")
        
        test_instance.test_duplicate_file_handling()
        print("✅ Duplicate file handling test passed")
        
        test_instance.test_invalid_task_id()
        print("✅ Invalid task ID test passed")
        
        test_instance.test_api_info_endpoint()
        print("✅ API info endpoint test passed")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        test_instance.teardown_class()