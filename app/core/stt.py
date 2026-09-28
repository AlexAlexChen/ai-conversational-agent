import logging
from typing import Optional, Dict, Any, Union
import io
import tempfile
import os

from app.config import STT_SERVICE, STT_API_KEY

logger = logging.getLogger(__name__)

class SpeechToText:
    """Speech-to-text service that converts audio to text"""

    def __init__(self):
        """Initialize the STT service"""
        self.service = STT_SERVICE
        self.api_key = STT_API_KEY

        if not self.api_key:
            logger.warning(f"No API key provided for STT service: {self.service}")

    def transcribe(self, audio_data: bytes, language: str = "en-US") -> str:
        """Transcribe audio to text

        Args:
            audio_data: Audio data as bytes
            language: Language code

        Returns:
            Transcribed text
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")

        logger.info(f"Transcribing audio with {self.service} STT (size: {len(audio_data)} bytes)")

        if self.service == "google":
            return self._google_stt(audio_data, language)
        elif self.service == "azure":
            return self._azure_stt(audio_data, language)
        elif self.service == "whisper":
            return self._whisper_stt(audio_data, language)
        else:
            raise ValueError(f"Unsupported STT service: {self.service}")

    def _google_stt(self, audio_data: bytes, language: str) -> str:
        """Google Cloud Speech-to-Text implementation

        Args:
            audio_data: Audio data
            language: Language code

        Returns:
            Transcribed text
        """
        try:
            from google.cloud import speech

            # Instantiate a client
            client = speech.SpeechClient()

            # Load the audio into memory
            audio = speech.RecognitionAudio(content=audio_data)

            # Configure the request
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MP3,
                sample_rate_hertz=16000,
                language_code=language
            )

            # Detects speech in the audio file
            response = client.recognize(config=config, audio=audio)

            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript + " "

            return transcript.strip()

        except Exception as e:
            logger.error(f"Error with Google STT: {str(e)}")
            return ""

    def _azure_stt(self, audio_data: bytes, language: str) -> str:
        """Azure Speech-to-Text implementation

        Args:
            audio_data: Audio data
            language: Language code

        Returns:
            Transcribed text
        """
        try:
            import azure.cognitiveservices.speech as speechsdk

            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            # Create a speech configuration with the subscription key and region
            speech_config = speechsdk.SpeechConfig(
                subscription=self.api_key,
                region="eastus"  # Replace with your Azure region
            )

            # Set the language
            speech_config.speech_recognition_language = language

            # Create an audio configuration for the temporary file
            audio_config = speechsdk.audio.AudioConfig(filename=temp_file_path)

            # Create a speech recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # Start recognition
            result = recognizer.recognize_once()

            # Clean up the temporary file
            os.remove(temp_file_path)

            # Check for success
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            else:
                logger.warning(f"Azure STT recognition failed: {result.reason}")
                return ""

        except Exception as e:
            logger.error(f"Error with Azure STT: {str(e)}")
            return ""

    def _whisper_stt(self, audio_data: bytes, language: str) -> str:
        """OpenAI Whisper implementation

        Args:
            audio_data: Audio data
            language: Language code

        Returns:
            Transcribed text
        """
        try:
            import openai

            # Set the API key
            openai.api_key = self.api_key

            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            # Open the file and send it to the Whisper API
            with open(temp_file_path, "rb") as audio_file:
                transcription = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    language=language.split("-")[0]  # Whisper expects ISO 639-1 language codes
                )

            # Clean up the temporary file
            os.remove(temp_file_path)

            # Return the transcribed text
            return transcription.get("text", "")

        except Exception as e:
            logger.error(f"Error with Whisper STT: {str(e)}")
            return ""
