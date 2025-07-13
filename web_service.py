"""
FastAPI web service for Whisper transcription
"""

import os
import tempfile
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from transcription_core import TranscriptionService
from async_storage import storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    # Startup
    task = asyncio.create_task(storage.start_processing_loop())
    yield
    # Shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Whisper Transcription API",
    description="Upload audio files for transcription using OpenAI Whisper",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global transcription services for different models
transcription_services = {}

def get_transcription_service(model_name: str) -> TranscriptionService:
    """Get or create a transcription service for the specified model"""
    if model_name not in transcription_services:
        transcription_services[model_name] = TranscriptionService(model_name)
    return transcription_services[model_name]


class TranscriptionResponse(BaseModel):
    text: str
    segments: List[dict]
    language: str
    language_probability: float
    model: str
    filename: str


class HealthResponse(BaseModel):
    status: str
    message: str


class ModelsResponse(BaseModel):
    models: List[dict]


class AsyncTaskResponse(BaseModel):
    task_id: str
    filename: str
    file_hash: str
    created_at: str
    duplicate: bool


class TaskStatusResponse(BaseModel):
    task_id: str
    filename: str
    file_hash: str
    status: str
    model: str
    language: Optional[str]
    output_format: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    ttl_expires_at: str
    file_size: int
    error_message: Optional[str]
    processing_duration_seconds: Optional[float]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Whisper transcription service is running"
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available Whisper models"""
    models = [
        {"name": "tiny", "size": "~39 MB", "description": "Fastest, least accurate"},
        {"name": "base", "size": "~142 MB", "description": "Good balance"},
        {"name": "small", "size": "~466 MB", "description": "Better accuracy (default)"},
        {"name": "medium", "size": "~1.5 GB", "description": "High accuracy"},
        {"name": "large", "size": "~2.9 GB", "description": "Highest accuracy"}
    ]
    return ModelsResponse(models=models)


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(default="small"),
    language: Optional[str] = Form(default=None),
    output_format: str = Form(default="json"),
    streaming: bool = Form(default=False)
):
    """
    Transcribe an uploaded audio file
    
    Args:
        file: Audio file to transcribe
        model: Whisper model to use (tiny, base, small, medium, large)
        language: Language code (e.g., 'en', 'de') or None for auto-detection
        output_format: Output format (json, txt, srt, vtt, tsv)
        streaming: Return segments as they're processed (only for json format)
    """
    
    # Validate model
    valid_models = ["tiny", "base", "small", "medium", "large"]
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Must be one of: {', '.join(valid_models)}"
        )
    
    # Validate output format
    valid_formats = ["json", "txt", "srt", "vtt", "tsv"]
    if output_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output format '{output_format}'. Must be one of: {', '.join(valid_formats)}"
        )
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Common audio file extensions
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.mp4', '.mov', '.avi', '.mkv']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_extension}'. Supported types: {', '.join(valid_extensions)}"
        )
    
    try:
        # Read file content
        audio_content = await file.read()
        
        # Get transcription service
        service = get_transcription_service(model)
        
        # Transcribe audio
        if streaming and output_format == "json":
            # Streaming mode - return segments as they're processed
            # Note: This is a simplified implementation
            # In a real-world scenario, you'd want to use WebSockets or Server-Sent Events
            result = service.transcribe_buffer(
                audio_content,
                filename=file.filename,
                language=language,
                streaming=False  # We'll collect all segments for now
            )
        else:
            # Non-streaming mode
            result = service.transcribe_buffer(
                audio_content,
                filename=file.filename,
                language=language,
                streaming=False
            )
        
        # Format response based on output_format
        if output_format == "json":
            response_data = TranscriptionResponse(
                text=result["text"],
                segments=result["segments"],
                language=result["language"],
                language_probability=result["language_probability"],
                model=model,
                filename=file.filename
            )
            return response_data
        
        else:
            # Return formatted text response
            formatted_content = service.export_to_format(
                result,
                output_format,
                os.path.splitext(file.filename)[0]
            )
            
            # Set appropriate content type
            if output_format == "txt":
                media_type = "text/plain"
            elif output_format in ["srt", "vtt"]:
                media_type = "text/plain"
            elif output_format == "tsv":
                media_type = "text/tab-separated-values"
            else:
                media_type = "text/plain"
            
            return PlainTextResponse(
                content=formatted_content,
                media_type=media_type
            )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/")
async def root():
    """Serve the frontend HTML"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


@app.post("/transcribe/async", response_model=AsyncTaskResponse)
async def submit_async_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="small"),
    language: Optional[str] = Form(default=None),
    output_format: str = Form(default="json")
):
    """
    Submit an audio file for async transcription
    
    Returns task metadata including task_id for status checking
    """
    
    # Validate model
    valid_models = ["tiny", "base", "small", "medium", "large"]
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Must be one of: {', '.join(valid_models)}"
        )
    
    # Validate output format
    valid_formats = ["json", "txt", "srt", "vtt", "tsv"]
    if output_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output format '{output_format}'. Must be one of: {', '.join(valid_formats)}"
        )
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Common audio file extensions
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.mp4', '.mov', '.avi', '.mkv']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_extension}'. Supported types: {', '.join(valid_extensions)}"
        )
    
    try:
        # Read file content
        audio_content = await file.read()
        
        # Create async task
        task_info = storage.create_task(
            filename=file.filename,
            file_content=audio_content,
            model=model,
            language=language,
            output_format=output_format
        )
        
        return AsyncTaskResponse(**task_info)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status and metadata of a transcription task
    """
    task_status = storage.get_task_status(task_id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatusResponse(**task_status)


@app.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str, format: Optional[str] = None):
    """
    Get the result of a completed transcription task
    
    Args:
        task_id: Task identifier
        format: Optional format override (json, txt, srt, vtt, tsv)
    """
    result = storage.get_task_result(task_id)
    
    if not result:
        # Check if task exists but is not completed
        task_status = storage.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        elif task_status['status'] == 'failed':
            raise HTTPException(status_code=500, detail=f"Task failed: {task_status.get('error_message', 'Unknown error')}")
        elif task_status['status'] in ['pending', 'processing']:
            raise HTTPException(status_code=202, detail=f"Task is {task_status['status']}, result not ready yet")
        else:
            raise HTTPException(status_code=500, detail="Result not available")
    
    # Use format from task or override
    output_format = format or result['task_metadata']['output_format']
    
    if output_format == "json":
        return JSONResponse(content=result)
    else:
        # Get transcription service for formatting
        service = get_transcription_service(result['task_metadata']['model'])
        
        try:
            formatted_content = service.export_to_format(
                result,
                output_format,
                os.path.splitext(result['task_metadata']['filename'])[0]
            )
            
            # Set appropriate content type
            if output_format == "txt":
                media_type = "text/plain"
            elif output_format in ["srt", "vtt"]:
                media_type = "text/plain"
            elif output_format == "tsv":
                media_type = "text/tab-separated-values"
            else:
                media_type = "text/plain"
            
            return PlainTextResponse(
                content=formatted_content,
                media_type=media_type
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to format result: {str(e)}")


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "name": "Whisper Transcription API",
        "version": "1.0.0",
        "description": "Upload audio files for transcription using OpenAI Whisper",
        "endpoints": {
            "POST /transcribe": "Upload and transcribe audio files (synchronous)",
            "POST /transcribe/async": "Submit audio file for async transcription",
            "GET /tasks/{task_id}": "Get task status and metadata",
            "GET /tasks/{task_id}/result": "Get transcription result",
            "GET /health": "Health check",
            "GET /models": "List available models",
            "GET /docs": "Interactive API documentation",
            "GET /openapi.json": "OpenAPI schema"
        }
    }




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "web_service:app",
        host=host,
        port=port,
        reload=False,
        access_log=True
    )