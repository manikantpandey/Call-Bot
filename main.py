import os
import json
import base64
import audioop
import tempfile
import logging
import asyncio
from typing import Optional
import requests
import soundfile as sf
import librosa
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from deepgram import Deepgram
from TTS.api import TTS

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful and friendly phone assistant.")
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/glow-tts")

if not GROQ_API_KEY or not DEEPGRAM_API_KEY:
    raise RuntimeError("GROQ_API_KEY and DEEPGRAM_API_KEY must be set in .env")

TWILIO_SAMPLE_RATE = 8000
TWILIO_CHUNK_SIZE = 320  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("twilio-bidi")

deepgram_client = Deepgram(DEEPGRAM_API_KEY)
tts = TTS(TTS_MODEL).to("cpu")  

app = FastAPI()


def query_groq(prompt: str) -> str:
    """Simple Groq chat completion (synchronous)."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 512,
        "stream": False
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def transcribe_with_deepgram(raw_ulaw_bytes: bytes) -> str:
    """
    Transcribe μ-law raw bytes from Twilio using Deepgram prerecorded endpoint.
    We pass encoding params so Deepgram knows how to interpret the bytes.
    """
    try:
        source = {"buffer": raw_ulaw_bytes, "mimetype": "audio/mulaw"}
        options = {
            "model": "nova-2",       
            "encoding": "mulaw",
            "sample_rate": TWILIO_SAMPLE_RATE,
            "language": "en-US",
            "punctuate": True
        }
        res = await deepgram_client.transcription.prerecorded(source, options)
        transcript = ""
        try:
            transcript = res["results"]["channels"][0]["alternatives"][0]["transcript"]
        except Exception:
            logger.debug("Deepgram returned unexpected shape: %s", res)
        logger.info("Deepgram transcript: %s", transcript)
        return transcript
    except Exception as e:
        logger.exception("Deepgram transcription failed: %s", e)
        return ""


def synthesize_tts_to_wav_file(text: str) -> str:
    """
    Synthesize text to a WAV file with Coqui TTS and return the filepath.
    This is blocking (execute it in executor).
    """
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tf.close()
    out_path = tf.name
    logger.info("Synthesizing TTS to %s", out_path)
    tts.tts_to_file(text=text, file_path=out_path)
    return out_path


def wav_to_8k_mulaw_bytes(wav_path: str) -> bytes:
    """
    Read a WAV file (whatever sampling) and convert to 8kHz mono μ-law bytes.
    Returns raw μ-law bytes (NOT base64 encoded).
    """
    # soundfile reads into numpy array
    data, sr = sf.read(wav_path, dtype="int16")
    # make mono if needed
    if data.ndim > 1:
        data = data[:, 0]

    # Resample to 8k if needed
    if sr != TWILIO_SAMPLE_RATE:
        data = librosa.resample(data.astype(float), orig_sr=sr, target_sr=TWILIO_SAMPLE_RATE)
        # convert back to int16
        data = data.astype("int16")

    pcm_bytes = data.tobytes()
    # convert PCM16 -> μ-law
    mulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
    return mulaw_bytes


async def stream_mulaw_to_twilio(websocket: WebSocket, stream_sid: str, mulaw_bytes: bytes):
    """
    Chunk mulaw_bytes and send them to Twilio as repeated `media` events.
    Keep small chunk size so Twilio can play smoothly.
    """
    # Optionally send a mark before starting response
    try:
        await websocket.send_json({"event": "mark", "streamSid": stream_sid, "mark": {"name": "response_start"}})
    except Exception:
        pass

    # send in small chunks
    total = len(mulaw_bytes)
    logger.info("Streaming %d bytes of μ-law audio back to Twilio (streamSid=%s)", total, stream_sid)
    offset = 0
    while offset < total:
        chunk = mulaw_bytes[offset: offset + TWILIO_CHUNK_SIZE]
        payload_b64 = base64.b64encode(chunk).decode("utf-8")
        audio_delta = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload_b64}
        }
        # send and throttle slightly to avoid flooding
        await websocket.send_json(audio_delta)
        offset += len(chunk)
        # sleep a tiny bit to simulate streaming; tune as necessary
        await asyncio.sleep(0.03)

    # send a mark after finishing response
    try:
        await websocket.send_json({"event": "mark", "streamSid": stream_sid, "mark": {"name": "response_end"}})
    except Exception:
        pass


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Twilio will call this on incoming voice; return TwiML to connect to media stream.
    Ensure Twilio can reach your host (use ngrok or HTTPS publicly).
    """
    hostname = request.url.hostname
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{hostname}/media-stream"/>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Handle Twilio Media Stream events bidirectionally.
    - Receive Twilio µ-law audio chunks, buffer them
    - On `mark` (or when buffer threshold reached) run STT -> LLM -> TTS
    - Stream μ-law TTS audio back to Twilio in small media events
    """
    await websocket.accept()
    logger.info("Twilio connected to /media-stream")

    stream_sid: Optional[str] = None
    buffer_ulaw = bytearray()
    BUFFER_THRESHOLD_BYTES = 8000 * 1  

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                logger.debug("Received non-json ws message")
                continue

            event = data.get("event")
            # Twilio start event
            if event == "start":
                stream_sid = data["start"].get("streamSid")
                logger.info("Stream started: %s", stream_sid)
                buffer_ulaw = bytearray()

            # Twilio sends media events containing base64-encoded μ-law payload
            elif event == "media":
                payload_b64 = data["media"]["payload"]
                try:
                    chunk = base64.b64decode(payload_b64)
                    buffer_ulaw.extend(chunk)
                except Exception:
                    logger.exception("Failed to decode incoming media payload")

                # If buffer crosses threshold, process a segment asynchronously (so we don't block receiving)
                if len(buffer_ulaw) >= BUFFER_THRESHOLD_BYTES:
                    # snapshot and clear buffer (so new incoming audio is not lost)
                    audio_for_processing = bytes(buffer_ulaw)
                    buffer_ulaw.clear()

                    # process in background to keep receiving
                    asyncio.create_task(handle_segment_and_respond(websocket, stream_sid, audio_for_processing))

            # Twilio mark event — a good segmentation point
            elif event == "mark":
                logger.info("Received mark from Twilio: %s", data.get("mark"))
                # When Twilio sends a mark, process current buffer
                if buffer_ulaw:
                    audio_for_processing = bytes(buffer_ulaw)
                    buffer_ulaw.clear()
                    asyncio.create_task(handle_segment_and_respond(websocket, stream_sid, audio_for_processing))

            # Twilio stop event — caller ended the stream. Process leftover and keep connection lifecycle
            elif event == "stop":
                logger.info("Received stop event. streamSid=%s", stream_sid)
                if buffer_ulaw:
                    audio_for_processing = bytes(buffer_ulaw)
                    buffer_ulaw.clear()
                    await handle_segment_and_respond(websocket, stream_sid, audio_for_processing)
            
                break

            else:
                logger.debug("Unhandled event: %s", event)

    except Exception as e:
        logger.exception("media_stream websocket exception: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Twilio WebSocket disconnected")


async def handle_segment_and_respond(websocket: WebSocket, stream_sid: str, raw_ulaw_bytes: bytes):
    """
    Given a chunk of raw μ-law bytes from the caller:
    - Transcribe with Deepgram
    - Query Groq
    - Synthesize TTS
    - Stream μ-law response back to Twilio
    """
    if not raw_ulaw_bytes:
        logger.info("Empty segment received, skipping.")
        return

    logger.info("Handling audio segment (%d bytes) for streamSid=%s", len(raw_ulaw_bytes), stream_sid)

    # 1) Transcribe
    transcript = await transcribe_with_deepgram(raw_ulaw_bytes)
    if not transcript:
        logger.info("No transcript produced for this segment.")
        return

    # 2) Query LLM (blocking HTTP) -> run in executor to avoid blocking event loop
    loop = asyncio.get_running_loop()
    try:
        groq_reply = await loop.run_in_executor(None, query_groq, transcript)
    except Exception as e:
        logger.exception("Groq call failed: %s", e)
        groq_reply = "Sorry, I had trouble generating a response."

    logger.info("Groq reply: %s", groq_reply)

    # 3) TTS: synthesize to wav (blocking) in executor
    try:
        wav_path = await loop.run_in_executor(None, synthesize_tts_to_wav_file, groq_reply)
    except Exception as e:
        logger.exception("TTS synthesis failed: %s", e)
        return

    # 4) Convert to μ-law 8k
    try:
        mulaw_bytes = await loop.run_in_executor(None, wav_to_8k_mulaw_bytes, wav_path)
    except Exception as e:
        logger.exception("Failed to convert TTS wav to μ-law: %s", e)
        mulaw_bytes = b""

    # cleanup wav file
    try:
        os.remove(wav_path)
    except Exception:
        pass

    if not mulaw_bytes:
        logger.info("No μ-law bytes to stream back.")
        return

    # 5) Stream back to Twilio in chunks
    try:
        await stream_mulaw_to_twilio(websocket, stream_sid, mulaw_bytes)
        logger.info("Finished streaming response back to Twilio for streamSid=%s", stream_sid)
    except Exception as e:
        logger.exception("Failed while streaming audio to Twilio: %s", e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
