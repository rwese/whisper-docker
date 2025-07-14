#!/bin/bash

# Simple concurrent test using curl
# Make sure the service is running and you have a test audio file

BASE_URL="http://localhost:8000"
TEST_FILE="test_audio.wav"
CONCURRENT_REQUESTS=5

echo "Testing concurrent requests to Whisper service..."
echo "Base URL: $BASE_URL"
echo "Test file: $TEST_FILE"
echo "Concurrent requests: $CONCURRENT_REQUESTS"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file $TEST_FILE not found!"
    echo "Please provide a test audio file or create a small one for testing."
    exit 1
fi

# Test health endpoint
echo "Testing health endpoint..."
curl -s "$BASE_URL/health" | jq . || echo "Health check failed"

echo -e "\nStarting concurrent test..."
start_time=$(date +%s)

# Function to make a single request
make_request() {
    local request_id=$1
    local start=$(date +%s%3N)

    response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/transcribe" \
        -F "file=@$TEST_FILE" \
        -F "model=tiny" \
        -F "output_format=json")

    local end=$(date +%s%3N)
    local duration=$((end - start))
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n -1)

    if [ "$http_code" = "200" ]; then
        local text_length=$(echo "$body" | jq -r '.text' | wc -c)
        local segments=$(echo "$body" | jq '.segments | length')
        local language=$(echo "$body" | jq -r '.language')

        echo "Request $request_id: SUCCESS in ${duration}ms - $text_length chars, $segments segments, language: $language"
    else
        echo "Request $request_id: FAILED (HTTP $http_code) in ${duration}ms"
        echo "  Error: $(echo "$body" | head -c 100)"
    fi
}

# Run concurrent requests
echo "Starting $CONCURRENT_REQUESTS concurrent requests..."
for i in $(seq 1 $CONCURRENT_REQUESTS); do
    make_request $i &
done

# Wait for all background jobs to complete
wait

end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo -e "\nTest completed in ${total_duration} seconds"

# Test async endpoint
echo -e "\nTesting async endpoint..."
async_response=$(curl -s -X POST "$BASE_URL/transcribe/async" \
    -F "file=@$TEST_FILE" \
    -F "model=tiny" \
    -F "output_format=json")

if [ $? -eq 0 ]; then
    task_id=$(echo "$async_response" | jq -r '.task_id')
    echo "Async task submitted: $task_id"

    # Poll for completion
    for i in {1..30}; do
        sleep 1
        status_response=$(curl -s "$BASE_URL/tasks/$task_id")
        status=$(echo "$status_response" | jq -r '.status')

        echo "Status check $i: $status"

        if [ "$status" = "completed" ]; then
            result_response=$(curl -s "$BASE_URL/tasks/$task_id/result")
            text_length=$(echo "$result_response" | jq -r '.text' | wc -c)
            processing_time=$(echo "$status_response" | jq -r '.processing_duration_seconds')
            echo "Async result: $text_length characters, processed in ${processing_time}s"
            break
        elif [ "$status" = "failed" ]; then
            error_msg=$(echo "$status_response" | jq -r '.error_message')
            echo "Async task failed: $error_msg"
            break
        fi
    done
else
    echo "Failed to submit async task"
fi

echo "Test completed!"
