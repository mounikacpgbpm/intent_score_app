from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import uvicorn
import numpy as np
import librosa
import torch
import soundfile as sf
from pydantic import BaseModel
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import logging
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Audio & Text Analysis API",
    description="Upload WAV audio files or TXT text files for sentiment and intent analysis",
    version="1.0.0"
)

# Response model
class AnalysisResponse(BaseModel):
    sentiment: str
    confidence: float
    intent_score: dict
    text: str
    source_type: str

class BatchAudioResponse(BaseModel):
    results: List[AnalysisResponse]
    total_files: int
    failed_files: List[dict] = []

class ModelManager:
    def __init__(self):
        self.intent_model = None
        self.intent_processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_models(self):
        """Load all required models"""
        try:
            logger.info("Loading intent model...")
            model_name = "facebook/wav2vec2-base-960h"
            self.intent_processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.intent_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=10,
                ignore_mismatched_sizes=True
            )
            self.intent_model.to(self.device)
            self.intent_model.eval()
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    model_manager.load_models()

def preprocess_audio(audio_bytes: bytes):
    """Preprocess audio bytes for model input"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_path = temp_file.name
        
        try:
            # Use soundfile to read audio (doesn't need FFmpeg)
            audio_array, sr = sf.read(temp_path)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Normalize audio
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            return audio_array, sr
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        raise

def analyze_intent_from_audio(audio_array: np.ndarray):
    """Get intent scores from audio"""
    try:
        inputs = model_manager.intent_processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        inputs = {key: value.to(model_manager.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            logits = model_manager.intent_model(**inputs).logits
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Intent labels
        intent_labels = {
            0: "greeting", 1: "question", 2: "command", 3: "statement",
            4: "affirmation", 5: "negation", 6: "request", 7: "apology",
            8: "gratitude", 9: "farewell"
        }
        
        # Get all intent scores
        intent_scores = {}
        for i in range(probabilities.shape[1]):
            intent_name = intent_labels.get(i, f"intent_{i}")
            intent_scores[intent_name] = float(probabilities[0][i].item())
        
        return intent_scores
        
    except Exception as e:
        logger.error(f"Intent analysis error: {e}")
        # Return uniform distribution as fallback
        return {f"intent_{i}": 0.1 for i in range(10)}

def analyze_sentiment_from_text(text: str):
    """Analyze sentiment from text"""
    try:
        text_lower = text.lower()
        
        # Simple rule-based sentiment
        positive_words = ['hello', 'hi', 'good', 'great', 'thanks', 'thank', 'please', 'happy', 'love']
        negative_words = ['bad', 'terrible', 'hate', 'angry', 'sad', 'upset', 'wrong', 'problem']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.5 + (positive_count * 0.1), 0.95)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.5 + (negative_count * 0.1), 0.95)
        else:
            sentiment = "neutral"
            confidence = 0.5
            
        return sentiment, confidence
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return "neutral", 0.5

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Audio & Text Analysis API",
        "version": "1.0.0",
        "description": "Upload WAV audio files or TXT text files for sentiment and intent analysis",
        "endpoints": {
            "POST /analyze/audio": "Upload a WAV file for analysis",
            "POST /analyze/text": "Upload a TXT file for analysis",
            "GET /health": "Check API health status"
        },
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": model_manager.intent_model is not None,
        "device": str(model_manager.device)
    }

@app.post("/analyze/audio", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze a single WAV audio file
    """
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    try:
        # Read and process audio
        contents = await file.read()
        audio_array, sample_rate = preprocess_audio(contents)
        
        # Since we can't transcribe without FFmpeg, we'll use a placeholder
        # You can add a lightweight ASR model here if needed
        transcribed_text = "Audio processed successfully (transcription not available)"
        
        # Analyze sentiment (simplified for now)
        sentiment, confidence = "neutral", 0.5
        
        # Get intent scores from audio
        intent_scores = analyze_intent_from_audio(audio_array)
        
        return AnalysisResponse(
            sentiment=sentiment,
            confidence=confidence,
            intent_score=intent_scores,
            text=transcribed_text,
            source_type="audio"
        )
        
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(file: UploadFile = File(...)):
    """
    Analyze a single TXT text file
    """
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only TXT files are supported")
    
    try:
        # Read text file
        contents = await file.read()
        text = contents.decode('utf-8').strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text file is empty")
        
        # Analyze sentiment from text
        sentiment, confidence = analyze_sentiment_from_text(text)
        
        # For text files, return placeholder intent scores
        # You can add text intent classification here if needed
        intent_scores = {
            "greeting": 0.1, "question": 0.1, "command": 0.1, "statement": 0.1,
            "affirmation": 0.1, "negation": 0.1, "request": 0.1, "apology": 0.1,
            "gratitude": 0.1, "farewell": 0.1
        }
        
        return AnalysisResponse(
            sentiment=sentiment,
            confidence=confidence,
            intent_score=intent_scores,
            text=text,
            source_type="text"
        )
        
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )