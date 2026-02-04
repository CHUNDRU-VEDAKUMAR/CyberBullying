"""
Production FastAPI Application for Advanced Cyberbullying Detection
Implements REST API with:
- Request validation
- Response serialization
- Error handling
- Logging
- Health checks
- Batch processing support
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import logging
import time
import json
from datetime import datetime
from functools import lru_cache

from src.main_system import CyberbullyingSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Cyberbullying Detection API",
    description="100% Context-Aware, Severity-Based, Explainable Detection",
    version="2.0.0"
)

# ============================================================================
# Pydantic Models
# ============================================================================

class TextRequest(BaseModel):
    """Single text detection request."""
    text: str = Field(..., min_length=1, max_length=10000, 
                      description="Text to analyze for cyberbullying")
    include_explanation: bool = Field(True, description="Include LIME explanation")
    use_ensemble: bool = Field(False, description="Use 3-model ensemble (slower but more accurate)")
    use_advanced_context: bool = Field(True, description="Use advanced spaCy context analysis")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()


class BatchRequest(BaseModel):
    """Batch detection request."""
    texts: List[str] = Field(..., min_items=1, max_items=100,
                             description="List of texts to analyze")
    include_explanations: bool = Field(False, description="Include explanations (slower)")
    use_ensemble: bool = Field(False, description="Use ensemble")
    
    @validator('texts')
    def texts_not_empty(cls, v):
        for text in v:
            if not text.strip():
                raise ValueError('Texts cannot contain empty items')
        return [t.strip() for t in v]


class DetectionResult(BaseModel):
    """Single detection result."""
    text: str
    is_bullying: bool
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    detected_types: List[str]
    scores: Dict[str, float]
    explanation: str
    action: str  # SUSPEND, HIDE, WARN, MONITOR
    confidence: float
    highlighted_words: List[List]
    context_info: Dict
    processing_time_ms: float


class BatchDetectionResult(BaseModel):
    """Batch detection result."""
    results: List[DetectionResult]
    total_texts: int
    bullying_count: int
    critical_count: int
    processing_time_ms: float
    avg_text_length: float


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    timestamp: str


# ============================================================================
# Global State (cached systems)
# ============================================================================

@lru_cache(maxsize=3)
def get_system(use_ensemble: bool = False, use_advanced_context: bool = True) -> CyberbullyingSystem:
    """Get or create cyberbullying system (cached for performance)."""
    logger.info(f"Loading system: ensemble={use_ensemble}, advanced_context={use_advanced_context}")
    try:
        system = CyberbullyingSystem(
            model_name='unitary/toxic-bert',
            use_ensemble=use_ensemble,
            use_advanced_context=use_advanced_context
        )
        logger.info("System loaded successfully")
        return system
    except Exception as e:
        logger.error(f"Failed to load system: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")


# ============================================================================
# Routes
# ============================================================================

@app.get("/health", response_model=HealthStatus)
def health_check():
    """Health check endpoint."""
    try:
        system = get_system(use_ensemble=False, use_advanced_context=True)
        model_loaded = system.engine is not None
    except:
        model_loaded = False
    
    return HealthStatus(
        status="healthy" if model_loaded else "degraded",
        version="2.0.0",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.post("/detect", response_model=DetectionResult)
async def detect_bullying(request: TextRequest):
    """
    Detect cyberbullying in a single text.
    
    **Response Fields:**
    - `is_bullying`: Whether text contains cyberbullying
    - `severity`: CRITICAL/HIGH/MEDIUM/LOW
    - `detected_types`: Labels (toxic, severe_toxic, obscene, threat, insult, identity_hate)
    - `scores`: Confidence scores per label
    - `explanation`: Human-readable summary
    - `action`: Recommended action (SUSPEND/HIDE/WARN/MONITOR)
    - `highlighted_words`: Words contributing to detection
    - `context_info`: Negation, sarcasm, target type analysis
    """
    start_time = time.time()
    
    try:
        # Get system
        system = get_system(
            use_ensemble=request.use_ensemble,
            use_advanced_context=request.use_advanced_context
        )
        
        # Analyze
        logger.info(f"Analyzing text: {request.text[:50]}...")
        result = system.analyze(request.text)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log detection
        logger.info(json.dumps({
            'event': 'detection',
            'is_bullying': result['is_bullying'],
            'severity': result['severity'],
            'labels': result['detected_types'],
            'latency_ms': processing_time,
            'context_aware': result['context_info'].get('advanced_context', False),
            'ensemble': request.use_ensemble
        }))
        
        # Return structured response
        return DetectionResult(
            text=request.text,
            is_bullying=result['is_bullying'],
            severity=result['severity'],
            detected_types=result['detected_types'],
            scores=result['scores'],
            explanation=result['explanation'],
            action=result['action'],
            confidence=result.get('confidence', 0.0),
            highlighted_words=result.get('highlighted_words', []),
            context_info=result['context_info'],
            processing_time_ms=processing_time
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")


@app.post("/detect-batch", response_model=BatchDetectionResult)
async def detect_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Detect cyberbullying in multiple texts (batch processing).
    
    **Benefits:**
    - Batch processing is more efficient than individual requests
    - Model is loaded once and reused
    - Optimal for high-throughput scenarios
    
    **Limitations:**
    - Max 100 texts per request
    - Explanations not included by default (too slow)
    """
    start_time = time.time()
    
    try:
        system = get_system(
            use_ensemble=request.use_ensemble,
            use_advanced_context=True
        )
        
        results = []
        bullying_count = 0
        critical_count = 0
        
        logger.info(f"Processing batch of {len(request.texts)} texts")
        
        for i, text in enumerate(request.texts):
            result = system.analyze(text)
            
            results.append(DetectionResult(
                text=text,
                is_bullying=result['is_bullying'],
                severity=result['severity'],
                detected_types=result['detected_types'],
                scores=result['scores'],
                explanation=result['explanation'],
                action=result['action'],
                confidence=result.get('confidence', 0.0),
                highlighted_words=[] if not request.include_explanations else result.get('highlighted_words', []),
                context_info=result['context_info'],
                processing_time_ms=0  # Filled in batch result
            ))
            
            if result['is_bullying']:
                bullying_count += 1
                if result['severity'] == 'CRITICAL':
                    critical_count += 1
        
        processing_time = (time.time() - start_time) * 1000
        avg_text_length = sum(len(t) for t in request.texts) / len(request.texts)
        
        # Log batch
        logger.info(json.dumps({
            'event': 'batch_detection',
            'total': len(request.texts),
            'bullying_count': bullying_count,
            'critical_count': critical_count,
            'latency_ms': processing_time,
            'throughput_texts_per_sec': len(request.texts) / (processing_time / 1000)
        }))
        
        return BatchDetectionResult(
            results=results,
            total_texts=len(request.texts),
            bullying_count=bullying_count,
            critical_count=critical_count,
            processing_time_ms=processing_time,
            avg_text_length=avg_text_length
        )
    
    except Exception as e:
        logger.error(f"Batch detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {e}")


@app.get("/models")
def list_models():
    """List available models and configurations."""
    return {
        "models": [
            {
                "name": "unitary/toxic-bert",
                "description": "BERT base fine-tuned on Jigsaw toxicity data",
                "size": "110M",
                "languages": ["en"],
                "labels": 6
            },
            {
                "name": "roberta-base",
                "description": "RoBERTa base with better contextual understanding",
                "size": "125M",
                "languages": ["en"],
                "labels": 6
            },
            {
                "name": "ensemble (advanced)",
                "description": "3-model weighted ensemble (DeBERTa v3 + RoBERTa-large + DistilBERT)",
                "size": "800M+",
                "languages": ["en"],
                "labels": 6,
                "accuracy": "Highest"
            }
        ],
        "labels": [
            {"name": "toxic", "description": "General toxicity", "severity": "MEDIUM"},
            {"name": "severe_toxic", "description": "Severe toxicity", "severity": "CRITICAL"},
            {"name": "obscene", "description": "Obscene language", "severity": "MEDIUM"},
            {"name": "threat", "description": "Threats or violence", "severity": "CRITICAL"},
            {"name": "insult", "description": "Insults or personal attacks", "severity": "HIGH"},
            {"name": "identity_hate", "description": "Hate speech", "severity": "CRITICAL"}
        ]
    }


@app.get("/stats")
def get_stats():
    """Get API statistics (placeholder)."""
    return {
        "uptime_seconds": 0,  # Would be calculated
        "total_requests": 0,  # Would be tracked
        "avg_latency_ms": 0,
        "p99_latency_ms": 0,
        "bullying_detection_rate": 0.0
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize models on startup."""
    logger.info("🚀 Starting Cyberbullying Detection API v2.0")
    try:
        # Pre-load default system
        system = get_system(use_ensemble=False, use_advanced_context=True)
        logger.info("✓ Default system loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load system: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down API")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
