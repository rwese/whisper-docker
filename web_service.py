"""
FastAPI web service for Whisper transcription
"""

import os
import sys
import tempfile
import asyncio
import threading
import time
import psutil
import logging
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from concurrent.futures import ThreadPoolExecutor
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = None
    CONTENT_TYPE_LATEST = "text/plain"
    def generate_latest():
        return "# Prometheus client not available\n"

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

from datetime import datetime, timezone

from transcription_core import TranscriptionService
from async_storage import storage

# Configure structured logging
if STRUCTLOG_AVAILABLE:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger()
else:
    # Fallback to basic logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

# Prometheus metrics - use a function to initialize metrics safely
def get_or_create_metrics():
    """Get or create Prometheus metrics, handling registry conflicts"""
    if not PROMETHEUS_AVAILABLE:
        return None
    
    # Create new metrics with error handling
    try:
        # Clear any existing metrics first
        REGISTRY._collector_to_names.clear()
        REGISTRY._names_to_collectors.clear()
        
        metrics = {
            'request_count': Counter(
                'whisper_requests_total',
                'Total number of requests',
                ['method', 'endpoint', 'status_code']
            ),
            'request_duration': Histogram(
                'whisper_request_duration_seconds',
                'Request duration in seconds',
                ['method', 'endpoint']
            ),
            'transcription_duration': Histogram(
                'whisper_transcription_duration_seconds',
                'Transcription duration in seconds',
                ['model']
            ),
            'file_size_histogram': Histogram(
                'whisper_file_size_bytes',
                'Size of uploaded files in bytes'
            ),
            'active_requests': Gauge(
                'whisper_active_requests',
                'Number of active requests'
            ),
            'loaded_models': Gauge(
                'whisper_loaded_models',
                'Number of loaded models'
            ),
            'thread_pool_active': Gauge(
                'whisper_thread_pool_active',
                'Number of active threads in thread pool'
            ),
            'task_queue_size': Gauge(
                'whisper_task_queue_size',
                'Size of async task queue'
            ),
            'model_load_duration': Histogram(
                'whisper_model_load_duration_seconds',
                'Model loading duration in seconds',
                ['model']
            ),
            'error_count': Counter(
                'whisper_errors_total',
                'Total number of errors',
                ['error_type', 'endpoint']
            )
        }
        return metrics
    except (ValueError, AttributeError) as e:
        # If metrics already exist or there's an error, return None to indicate we should skip metrics
        if STRUCTLOG_AVAILABLE:
            logger.warning("Metrics already registered or error occurred, disabling Prometheus metrics", error=str(e))
        else:
            logger.warning(f"Metrics already registered or error occurred, disabling Prometheus metrics: {e}")
        return None

# Initialize metrics
metrics = get_or_create_metrics()

# Helper function to safely update metrics
def safe_metric_update(metric_name, operation, *args, **kwargs):
    """Safely update a metric, handling cases where metrics are disabled"""
    if metrics and metrics.get(metric_name):
        try:
            operation(*args, **kwargs)
        except Exception as e:
            if STRUCTLOG_AVAILABLE:
                logger.warning(f"Failed to update metric {metric_name}", error=str(e))
            else:
                logger.warning(f"Failed to update metric {metric_name}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    # Startup
    logger.info("Starting Whisper transcription service", timestamp=datetime.now(timezone.utc).isoformat())
    task = asyncio.create_task(storage.start_processing_loop())
    yield
    # Shutdown
    logger.info("Shutting down Whisper transcription service", timestamp=datetime.now(timezone.utc).isoformat())
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

# Global transcription services for different models with thread safety
transcription_services = {}
transcription_services_lock = threading.Lock()

def get_transcription_service(model_name: str) -> TranscriptionService:
    """Get or create a transcription service for the specified model (thread-safe)"""
    with transcription_services_lock:
        if model_name not in transcription_services:
            logger.info("Creating new transcription service", model=model_name)
            start_time = time.time()
            transcription_services[model_name] = TranscriptionService(model_name)
            safe_metric_update('model_load_duration', lambda: metrics['model_load_duration'].labels(model=model_name).observe(time.time() - start_time))
            safe_metric_update('loaded_models', lambda: metrics['loaded_models'].set(len(transcription_services)))
            logger.info("Transcription service created", model=model_name, total_services=len(transcription_services))
        return transcription_services[model_name]

# Thread pool for formatting tasks only (reduced since no sync transcription)
transcription_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="formatting-")

# Global stats for monitoring
app_start_time = time.time()
request_counter = 0
request_counter_lock = threading.Lock()

# Middleware for logging and metrics
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Log request start
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    # Update active requests gauge
    safe_metric_update('active_requests', lambda: metrics['active_requests'].inc())
    
    try:
        response = await call_next(request)
        
        # Calculate request duration
        duration = time.time() - start_time
        
        # Update metrics
        endpoint = request.url.path
        safe_metric_update('request_count', lambda: metrics['request_count'].labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code
        ).inc())
        
        safe_metric_update('request_duration', lambda: metrics['request_duration'].labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration))
        
        # Log request completion
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration_seconds=duration,
            client_ip=request.client.host if request.client else "unknown"
        )
        
        return response
        
    except Exception as e:
        # Calculate request duration for failed requests
        duration = time.time() - start_time
        
        # Update error metrics
        endpoint = request.url.path
        safe_metric_update('error_count', lambda: metrics['error_count'].labels(
            error_type=type(e).__name__,
            endpoint=endpoint
        ).inc())
        
        safe_metric_update('request_count', lambda: metrics['request_count'].labels(
            method=request.method,
            endpoint=endpoint,
            status_code=500
        ).inc())
        
        # Log error
        logger.error(
            "Request failed",
            method=request.method,
            url=str(request.url),
            error=str(e),
            error_type=type(e).__name__,
            duration_seconds=duration,
            client_ip=request.client.host if request.client else "unknown",
            exc_info=True
        )
        
        raise
    
    finally:
        # Update active requests gauge
        safe_metric_update('active_requests', lambda: metrics['active_requests'].dec())

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    try:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        return Response("# Metrics temporarily unavailable\n", media_type=CONTENT_TYPE_LATEST)


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
    uptime_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    active_threads: Optional[int] = None
    thread_pool_active: Optional[int] = None


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
    """Enhanced health check endpoint with system metrics"""
    try:
        # Get system metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        # Get thread information
        active_threads = threading.active_count()
        thread_pool_active = len([t for t in threading.enumerate() if t.name.startswith("formatting-")])
        
        # Calculate uptime
        uptime = time.time() - app_start_time
        
        return HealthResponse(
            status="healthy",
            message="Whisper transcription service is running",
            uptime_seconds=uptime,
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            cpu_usage_percent=cpu_percent,
            active_threads=active_threads,
            thread_pool_active=thread_pool_active
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            message=f"Health check partially failed: {str(e)}"
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


@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    global request_counter
    
    with request_counter_lock:
        current_request_count = request_counter
    
    # Get loaded models
    loaded_model_stats = []
    with transcription_services_lock:
        for model_name, service in transcription_services.items():
            loaded_model_stats.append({
                "name": model_name,
                "loaded": service.model is not None
            })
    
    # Get thread pool stats
    thread_pool_stats = {
        "max_workers": transcription_executor._max_workers,
        "active_threads": len([t for t in threading.enumerate() if t.name.startswith("formatting-")]),
        "total_threads": threading.active_count()
    }
    
    # Update Prometheus gauges
    safe_metric_update('loaded_models', lambda: metrics['loaded_models'].set(len(loaded_model_stats)))
    safe_metric_update('thread_pool_active', lambda: metrics['thread_pool_active'].set(thread_pool_stats["active_threads"]))
    safe_metric_update('task_queue_size', lambda: metrics['task_queue_size'].set(storage.processing_queue.qsize()))
    
    return {
        "uptime_seconds": time.time() - app_start_time,
        "total_requests": current_request_count,
        "loaded_models": loaded_model_stats,
        "thread_pool": thread_pool_stats,
        "async_storage_stats": {
            "max_workers": storage.executor._max_workers,
            "queue_size": storage.processing_queue.qsize()
        }
    }




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
        
        # Log async task creation
        logger.info(
            "Creating async transcription task",
            filename=file.filename,
            file_size=len(audio_content),
            model=model,
            language=language,
            output_format=output_format
        )
        
        # Print to stdout for host monitoring
        print(f"[{datetime.now(timezone.utc).isoformat()}] Creating async task: {file.filename} ({len(audio_content)} bytes) with model {model}", flush=True)
        
        # Create async task
        task_info = storage.create_task(
            filename=file.filename,
            file_content=audio_content,
            model=model,
            language=language,
            output_format=output_format
        )
        
        # Update task queue metrics
        safe_metric_update('task_queue_size', lambda: metrics['task_queue_size'].set(storage.processing_queue.qsize()))
        
        logger.info(
            "Async transcription task created",
            task_id=task_info['task_id'],
            filename=file.filename,
            queue_size=storage.processing_queue.qsize()
        )
        
        # Print to stdout for host monitoring
        print(f"[{datetime.now(timezone.utc).isoformat()}] Async task created: {task_info['task_id']} for {file.filename}", flush=True)
        
        return AsyncTaskResponse(**task_info)
        
    except Exception as e:
        logger.error(
            "Failed to create async task",
            filename=file.filename,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        print(f"[{datetime.now(timezone.utc).isoformat()}] ERROR: Failed to create async task: {file.filename} - {str(e)}", file=sys.stderr, flush=True)
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
            "POST /transcribe/async": "Submit audio file for async transcription",
            "GET /tasks/{task_id}": "Get task status and metadata",
            "GET /tasks/{task_id}/result": "Get transcription result",
            "GET /health": "Health check",
            "GET /models": "List available models",
            "GET /metrics": "Prometheus metrics endpoint",
            "GET /docs": "Interactive API documentation",
            "GET /openapi.json": "OpenAPI schema"
        }
    }




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WORKERS", 1))  # Default to 1 for compatibility
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    
    logger.info(
        "Starting Whisper web service",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level
    )
    
    # Configure uvicorn with better concurrency settings
    uvicorn.run(
        "web_service:app",
        host=host,
        port=port,
        workers=workers if workers > 1 else None,  # Use multiprocessing only if workers > 1
        reload=False,
        access_log=True,
        log_level=log_level,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30,
        limit_concurrency=100,  # Limit concurrent requests
        limit_max_requests=1000  # Restart workers after 1000 requests to prevent memory leaks
    )