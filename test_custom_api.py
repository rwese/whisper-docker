#!/usr/bin/env python3
"""
Test script for the custom transcription API endpoint
"""

import json
import os
import sys

import requests


def test_transcribe_endpoint():
    """Test the /transcribe endpoint"""

    # Test API URL
    api_url = "http://localhost:8000/transcribe"

    # Test file (you would need to provide a real audio file for testing)
    test_file = "test_audio.wav"

    if not os.path.exists(test_file):
        print(
            f"Warning: Test file {test_file} not found. Please provide a test audio file."
        )
        return

    # Test data
    test_data = {
        "model": "tiny",
        "language": "en",
        "prompt": "This is a test recording",
        "temperature": 0.2,
    }

    # Test with authentication if API key is set
    headers = {}
    api_key = os.environ.get("TRANSCRIPTION_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with open(test_file, "rb") as f:
            files = {"audio": f}
            response = requests.post(
                api_url, data=test_data, files=files, headers=headers
            )

        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print(f"Status: {result['status']}")
            print(f"Transcription: {result['transcription']}")
            print(f"Language: {result['language']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Duration: {result['duration']}")
            print(f"Model: {result['model']}")
            print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
        else:
            print("ERROR!")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw response: {response.text}")

    except Exception as e:
        print(f"Test failed with exception: {e}")


def test_error_cases():
    """Test error handling"""

    api_url = "http://localhost:8000/transcribe"

    # Test 1: No file
    print("\n=== Test 1: No file ===")
    try:
        response = requests.post(api_url, data={"model": "tiny"})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Invalid model
    print("\n=== Test 2: Invalid model ===")
    try:
        # Create a dummy file
        with open("dummy.wav", "wb") as f:
            f.write(b"dummy audio data")

        with open("dummy.wav", "rb") as f:
            files = {"audio": f}
            response = requests.post(api_url, data={"model": "invalid"}, files=files)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

        # Cleanup
        os.remove("dummy.wav")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Invalid temperature
    print("\n=== Test 3: Invalid temperature ===")
    try:
        # Create a dummy file
        with open("dummy.wav", "wb") as f:
            f.write(b"dummy audio data")

        with open("dummy.wav", "rb") as f:
            files = {"audio": f}
            response = requests.post(
                api_url, data={"model": "tiny", "temperature": 2.0}, files=files
            )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

        # Cleanup
        os.remove("dummy.wav")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Testing Custom Transcription API")
    print("=" * 40)

    # Test main endpoint
    test_transcribe_endpoint()

    # Test error cases
    test_error_cases()

    print("\nTest completed!")
