from fastapi import FastAPI
from app.routes import chat, voice

app = FastAPI(
    title="AI Agent API",
    description="Humanized AI agent with natural language interaction in both text and voice",
    version="1.0.0"
)

# Include routers
app.include_router(chat.router)
app.include_router(voice.router)

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message"""
    return {"message": "Welcome to the Humanized AI Agent API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
