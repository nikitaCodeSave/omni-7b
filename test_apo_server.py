"""
Модуль FastAPI для сервиса автоматической транскрипции аудио с помощью модели Qwen2.5-Omni-7B.

Основные возможности:
- Загрузка и инициализация модели и процессора Qwen2.5-Omni
- Обработка аудиофайлов с разбиением на чанки и последующей транскрипцией
- REST API для получения транскрипции аудио
- Очистка GPU-памяти и управление ресурсами

Автор: (указать автора при необходимости)
"""
import logging
import os
import time
import tempfile
import shutil
import gc

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import torch
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor

import librosa
import soundfile as sf
import numpy as np

from contextlib import asynccontextmanager

# Import your inference logic from a separate module (should be properly refactored/typed there too)
from inference import inference

# --- Utils for local models ---

def load_qwen25_omni_model(cache_dir, model_name, **kwargs):
    """
    Загрузка модели и процессора Qwen2.5-Omni из локального кэша или репозитория.
    
    Параметры:
        cache_dir (str): Путь к директории с кэшем моделей.
        model_name (str): Имя или путь к модели.
        **kwargs: Дополнительные параметры для from_pretrained.
    
    Возвращает:
        model, processor: Загруженные объекты модели и процессора.
    """
    model = Qwen2_5OmniModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype="auto",
        device_map="auto",
        **kwargs
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    return model, processor

# --- Configuration ---
class Settings(BaseModel):
    """
    Конфигурация сервиса транскрипции.
    """
    model_id: str = "Qwen/Qwen2.5-Omni-7B"
    cache_dir: str = "./models"
    chunk_duration: int = 30
    overlap_seconds: int = 2
    batch_size: int = 8
    prompt: str = 'Transcribe the Russian audio into text with correct punctuation. The audio is from a single speaker. Write in natural, readable Russian.'
    sys_prompt: str = 'You are a highly accurate speech recognition model specialized in transcribing single-speaker Russian audio. Your transcription must include correct punctuation and be easy to read, preserving the natural flow of conversation.'

    class Config:
        arbitrary_types_allowed = True

settings = Settings()

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("stt_api")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)

# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекст жизненного цикла приложения FastAPI.
    Загружает модель при старте и очищает ресурсы при завершении.
    """
    logger.info("Loading Qwen2.5-Omni model and processor...")
    clear_gpu_memory()
    try:
        model, processor = load_qwen25_omni_model(cache_dir=settings.cache_dir, model_name=settings.model_id, attn_implementation="flash_attention_2")
        app.state.model = model
        app.state.processor = processor
        logger.info("Model and processor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app.state.model = None
        app.state.processor = None
    yield
    logger.info("Shutting down and cleaning up resources.")
    if hasattr(app.state, "model"):
        del app.state.model
    if hasattr(app.state, "processor"):
        del app.state.processor
    clear_gpu_memory()

app = FastAPI(
    title="Qwen-Omni-7B STT API",
    description="API for audio transcription using Qwen/Qwen2.5-Omni-7B",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utilities ---
def clear_gpu_memory():
    """
    Очистка памяти GPU и сборка мусора для предотвращения утечек памяти.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

def prepare_audio_chunks(
    audio_data: np.ndarray, 
    sr: int, 
    chunk_duration: int, 
    overlap_seconds: int, 
    temp_dir: str
):
    """
    Разбивает аудиоданные на чанки заданной длины с перекрытием.
    
    Параметры:
        audio_data (np.ndarray): Массив аудиосэмплов.
        sr (int): Частота дискретизации.
        chunk_duration (int): Длительность чанка в секундах.
        overlap_seconds (int): Перекрытие между чанками в секундах.
        temp_dir (str): Временная директория для сохранения чанков.
    
    Возвращает:
        List[str]: Пути к сохранённым аудиочанкам.
    """
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap_seconds * sr)
    step_samples = chunk_samples - overlap_samples
    total_samples = len(audio_data)
    num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / step_samples)))
    chunk_paths = []
    for i in range(num_chunks):
        start_sample = i * step_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk_data = audio_data[start_sample:end_sample]
        chunk_file = os.path.join(temp_dir, f"chunk_{i+1}.wav")
        sf.write(chunk_file, chunk_data, sr, 'PCM_16')
        chunk_paths.append(chunk_file)
    return chunk_paths

def process_audio_in_chunks(
    audio_path: str,
    model,
    processor,
    prompt: str,
    sys_prompt: str,
    chunk_duration: int,
    overlap_seconds: int,
    batch_size: int
) -> str:
    """
    Обрабатывает аудиофайл по чанкам и выполняет транскрипцию каждого чанка.
    
    Параметры:
        audio_path (str): Путь к аудиофайлу.
        model: Загруженная модель Qwen2.5-Omni.
        processor: Процессор для модели.
        prompt (str): Промпт для транскрипции.
        sys_prompt (str): Системный промпт.
        chunk_duration (int): Длительность чанка.
        overlap_seconds (int): Перекрытие между чанками.
        batch_size (int): Размер батча (не используется явно).
    
    Возвращает:
        str: Итоговая склеенная транскрипция.
    """
    try:
        audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        raise ValueError(f"Could not load or process audio file: {e}")

    with tempfile.TemporaryDirectory(prefix="audio_chunks_") as temp_dir:
        chunk_paths = prepare_audio_chunks(
            audio_data, sr, chunk_duration, overlap_seconds, temp_dir
        )
        results = []
        for i, chunk_path in enumerate(chunk_paths):
            try:
                response = inference(
                    chunk_path,
                    prompt=prompt,
                    sys_prompt=sys_prompt,
                    model=model,
                    processor=processor
                )
                transcription = response[0].split("assistant\n")[-1].strip() if response else ""
                results.append(transcription)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Error processing chunk {i+1}: {e}")
                results.append("")
            clear_gpu_memory()
    return " ".join(filter(None, results))

# --- Pydantic models ---
class TranscriptionResponse(BaseModel):
    """
    Модель ответа API с результатом транскрипции и временем обработки.
    """
    transcription: str
    processing_time_seconds: float

# --- Routes ---
@app.post(
    "/transcribe/",
    response_model=TranscriptionResponse,
    summary="Transcribe audio file"
)
async def transcribe_audio(
    request: Request,
    audio_file: UploadFile = File(..., description="Audio file (wav, mp3, etc.)"),
    prompt: str = Form(settings.prompt, description="Prompt for transcription"),
    sys_prompt: str = Form(settings.sys_prompt, description="System prompt"),
    chunk_duration: int = Form(settings.chunk_duration, description="Chunk duration (seconds)"),
    overlap_seconds: int = Form(settings.overlap_seconds, description="Chunk overlap (seconds)"),
    batch_size: int = Form(settings.batch_size, description="Batch size"),
):
    """
    Эндпоинт для транскрипции аудиофайла. Принимает файл, параметры обработки и возвращает текстовую транскрипцию.
    """
    start_time = time.time()
    model = request.app.state.model
    processor = request.app.state.processor
    if not model or not processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model not available. Check server logs."
        )
    fd, temp_audio_path = tempfile.mkstemp(suffix=os.path.splitext(audio_file.filename)[1])
    os.close(fd)
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        transcription = process_audio_in_chunks(
            audio_path=temp_audio_path,
            model=model,
            processor=processor,
            prompt=prompt,
            sys_prompt=sys_prompt,
            chunk_duration=chunk_duration,
            overlap_seconds=overlap_seconds,
            batch_size=batch_size
        )
    except ValueError as ve:
        logger.error(f"Value error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass
        await audio_file.close()
    elapsed = round(time.time() - start_time, 2)
    return TranscriptionResponse(
        transcription=transcription,
        processing_time_seconds=elapsed
    )

@app.get("/", summary="Service health check")
async def root():
    """
    Эндпоинт для проверки работоспособности сервиса.
    """
    return {"message": "Qwen Omni STT API is running."}

if __name__ == "__main__":
    # Точка входа для локального запуска сервиса
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
