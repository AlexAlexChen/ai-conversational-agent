from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import io

from app.core.agent import AIAgent
from app.core.tts import TextToSpeech
from app.core.stt import SpeechToText
from app.services.llm_service import get_llm_service

router = APIRouter(prefix="/voice", tags=["voice"])

class SourceInfo(BaseModel):
    """Source information model"""
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None

class VoiceResponse(BaseModel):
    """Voice response model (text part)"""
    message_id: str
    text: str
    sources: List[SourceInfo] = []
    session_id: str

@router.post("/", response_model=VoiceResponse)
async def process_voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    llm_service = Depends(get_llm_service)
):
    """
    Process voice input and return text response

    - **audio**: Audio file with speech input
    - **session_id**: Session identifier for conversation context (optional)
    """
    # Generate a session ID if not provided
    session_id = session_id or str(uuid4())

    try:
        # Read audio file
        audio_data = await audio.read()

        # Convert speech to text
        stt = SpeechToText()
        text_input = stt.transcribe(audio_data)

        if not text_input:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        # Process text with AI agent
        agent = AIAgent(llm_service)
        response, sources = await agent.process_message(
            message=text_input,
            session_id=session_id
        )

        return VoiceResponse(
            message_id=str(uuid4()),
            text=response,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice: {str(e)}")

@router.post("/tts")
async def text_to_speech(text: str = Form(...), voice: Optional[str] = Form("default")):
    """
    Convert text to speech

    - **text**: Text to convert to speech
    - **voice**: Voice identifier (optional)
    """
    try:
        tts = TextToSpeech()
        audio_data = tts.synthesize(text, voice)

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mp3"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting text to speech: {str(e)}")

@router.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Convert speech to text

    - **audio**: Audio file with speech input
    """
    try:
        # Read audio file
        audio_data = await audio.read()

        # Convert speech to text
        stt = SpeechToText()
        text = stt.transcribe(audio_data)

        if not text:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting speech to text: {str(e)}")
