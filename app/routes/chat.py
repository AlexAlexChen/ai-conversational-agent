from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4

from app.core.agent import AIAgent
from app.services.llm_service import get_llm_service

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    session_id: Optional[str] = None

class SourceInfo(BaseModel):
    """Source information model"""
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None

class ChatResponse(BaseModel):
    """Chat response model"""
    message_id: str
    response: str
    sources: List[SourceInfo] = []
    session_id: str

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, llm_service = Depends(get_llm_service)):
    """
    Process a chat message and return a response

    - **message**: The user's message
    - **session_id**: Session identifier for conversation context (optional)
    """
    # Generate a session ID if not provided
    session_id = request.session_id or str(uuid4())

    # Create AI agent
    agent = AIAgent(llm_service)

    try:
        # Process message
        response, sources = await agent.process_message(
            message=request.message,
            session_id=session_id
        )

        return ChatResponse(
            message_id=str(uuid4()),
            response=response,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.delete("/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear chat history for a session

    - **session_id**: Session identifier
    """
    try:
        # Create AI agent and clear history
        agent = AIAgent(get_llm_service())
        agent.memory_store.clear_history(session_id)
        return {"message": f"Chat history cleared for session {session_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")
