#!/usr/bin/env python3
"""
Concurrent testing script for the multithreaded Whisper service
"""

import asyncio
import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path

import aiohttp

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_AUDIO_FILE = "test_audio.wav"  # You need to provide this
CONCURRENT_REQUESTS = 5
TIMEOUT = 60  # Increased timeout for async operations


def create_test_audio():
    """Create a simple test audio file if it doesn't exist"""
    if not Path(TEST_AUDIO_FILE).exists():
        print(
            f"Warning: {TEST_AUDIO_FILE} not found. Please provide a test audio file."
        )
        return False
    return True


async def test_single_request(session, request_id, model="tiny"):
    """Test a single async transcription request"""
    start_time = time.time()

    try:
        # Submit async task
        with open(TEST_AUDIO_FILE, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename=TEST_AUDIO_FILE)
            data.add_field("model", model)
            data.add_field("output_format", "json")

            async with session.post(
                f"{BASE_URL}/transcribe/async", data=data, timeout=TIMEOUT
            ) as response:
                if response.status == 200:
                    task_info = await response.json()
                    task_id = task_info["task_id"]

                    # Poll for completion
                    for i in range(30):  # Wait up to 30 seconds
                        await asyncio.sleep(1)

                        async with session.get(
                            f"{BASE_URL}/tasks/{task_id}"
                        ) as status_response:
                            if status_response.status == 200:
                                status_data = await status_response.json()

                                if status_data["status"] == "completed":
                                    # Get result
                                    async with session.get(
                                        f"{BASE_URL}/tasks/{task_id}/result"
                                    ) as result_response:
                                        if result_response.status == 200:
                                            result = await result_response.json()
                                            duration = time.time() - start_time
                                            return {
                                                "request_id": request_id,
                                                "status": "success",
                                                "duration": duration,
                                                "text_length": len(
                                                    result.get("text", "")
                                                ),
                                                "segments": len(
                                                    result.get("segments", [])
                                                ),
                                                "language": result.get(
                                                    "language", "unknown"
                                                ),
                                                "processing_duration": status_data.get(
                                                    "processing_duration_seconds", 0
                                                ),
                                            }
                                        else:
                                            error_text = await result_response.text()
                                            return {
                                                "request_id": request_id,
                                                "status": "error",
                                                "duration": time.time() - start_time,
                                                "error": f"Failed to get result: {result_response.status} - {error_text}",
                                            }
                                elif status_data["status"] == "failed":
                                    return {
                                        "request_id": request_id,
                                        "status": "error",
                                        "duration": time.time() - start_time,
                                        "error": f"Task failed: {status_data.get('error_message', 'Unknown error')}",
                                    }
                                # Continue polling if still processing
                            else:
                                status_error = await status_response.text()
                                return {
                                    "request_id": request_id,
                                    "status": "error",
                                    "duration": time.time() - start_time,
                                    "error": f"Status check failed: {status_response.status} - {status_error}",
                                }
                    else:
                        return {
                            "request_id": request_id,
                            "status": "error",
                            "duration": time.time() - start_time,
                            "error": "Task did not complete within timeout",
                        }
                else:
                    error_text = await response.text()
                    return {
                        "request_id": request_id,
                        "status": "error",
                        "duration": time.time() - start_time,
                        "error": f"HTTP {response.status}: {error_text}",
                    }
    except Exception as e:
        return {
            "request_id": request_id,
            "status": "error",
            "duration": time.time() - start_time,
            "error": str(e),
        }


async def test_concurrent_requests():
    """Test concurrent async transcription requests"""
    print(f"Testing {CONCURRENT_REQUESTS} concurrent async requests...")

    async with aiohttp.ClientSession() as session:
        # Test health endpoint first
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status != 200:
                    print(f"Health check failed: {response.status}")
                    return
                health_data = await response.json()
                print(f"Health check: {health_data}")
        except Exception as e:
            print(f"Cannot connect to service: {e}")
            return

        # Run concurrent requests
        start_time = time.time()

        tasks = []
        for i in range(CONCURRENT_REQUESTS):
            task = asyncio.create_task(test_single_request(session, i))
            tasks.append(task)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_duration = time.time() - start_time

        # Process results
        successful = 0
        failed = 0
        total_transcription_time = 0

        print(f"\nResults after {total_duration:.2f} seconds:")
        print("-" * 60)

        for result in results:
            if isinstance(result, Exception):
                print(f"Exception: {result}")
                failed += 1
            elif result["status"] == "success":
                successful += 1
                total_transcription_time += result["duration"]
                print(
                    f"Request {result['request_id']}: Success in {result['duration']:.2f}s, "
                    f"text: {result['text_length']} chars, "
                    f"segments: {result['segments']}, "
                    f"language: {result['language']}, "
                    f"processing: {result.get('processing_duration', 0):.2f}s"
                )
            else:
                failed += 1
                print(
                    f"Request {result['request_id']}: Failed in {result['duration']:.2f}s - {result['error']}"
                )

        print("-" * 60)
        print(f"Summary:")
        print(f"  Total requests: {CONCURRENT_REQUESTS}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total wall time: {total_duration:.2f}s")
        print(
            f"  Average request time: {total_transcription_time / successful:.2f}s"
            if successful > 0
            else "  No successful requests"
        )
        print(
            f"  Concurrency benefit: {(total_transcription_time / total_duration):.2f}x"
            if total_duration > 0
            else "  N/A"
        )


async def test_async_endpoints():
    """Test async transcription endpoints"""
    print("\nTesting async endpoints...")

    async with aiohttp.ClientSession() as session:
        # Submit async transcription
        try:
            with open(TEST_AUDIO_FILE, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=TEST_AUDIO_FILE)
                data.add_field("model", "tiny")
                data.add_field("output_format", "json")

                async with session.post(
                    f"{BASE_URL}/transcribe/async", data=data
                ) as response:
                    if response.status == 200:
                        task_info = await response.json()
                        task_id = task_info["task_id"]
                        print(f"Async task submitted: {task_id}")

                        # Poll for completion
                        for i in range(30):  # Wait up to 30 seconds
                            await asyncio.sleep(1)

                            async with session.get(
                                f"{BASE_URL}/tasks/{task_id}"
                            ) as status_response:
                                if status_response.status == 200:
                                    status_data = await status_response.json()
                                    print(
                                        f"Status check {i+1}: {status_data['status']}"
                                    )

                                    if status_data["status"] == "completed":
                                        # Get result
                                        async with session.get(
                                            f"{BASE_URL}/tasks/{task_id}/result"
                                        ) as result_response:
                                            if result_response.status == 200:
                                                result = await result_response.json()
                                                print(
                                                    f"Async result: {len(result.get('text', ''))} characters"
                                                )
                                                print(
                                                    f"Processing time: {status_data.get('processing_duration_seconds', 0):.2f}s"
                                                )
                                                break
                                    elif status_data["status"] == "failed":
                                        print(
                                            f"Async task failed: {status_data.get('error_message', 'Unknown error')}"
                                        )
                                        break
                        else:
                            print("Async task did not complete in time")
                    else:
                        error_text = await response.text()
                        print(
                            f"Failed to submit async task: {response.status} - {error_text}"
                        )

        except Exception as e:
            print(f"Async test failed: {e}")


def main():
    if not create_test_audio():
        print("Please provide a test audio file named 'test_audio.wav'")
        return

    print("Starting concurrent testing...")
    print(f"Target URL: {BASE_URL}")
    print(f"Test file: {TEST_AUDIO_FILE}")

    try:
        # Test concurrent requests (now all async)
        asyncio.run(test_concurrent_requests())

        # Test a single async endpoint for comparison
        asyncio.run(test_async_endpoints())

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main()
