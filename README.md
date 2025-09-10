# Call Bot

A real-time, bidirectional phone assistant that uses Twilio, Deepgram, Groq LLM, and Coqui TTS to transcribe, generate, and synthesize speech for phone calls.

## Features

- Receives phone calls via Twilio and streams audio in real-time.
- Transcribes caller speech using Deepgram.
- Generates intelligent responses using Groq LLM.
- Synthesizes responses to speech using Coqui TTS.
- Streams synthesized speech back to the caller via Twilio.

## Requirements

- Python 3.10+
- Twilio account (for phone calls)
- Deepgram API key (for speech-to-text)
- Groq API key (for LLM)
- [ngrok](https://ngrok.com/) or public HTTPS endpoint (for Twilio webhooks)

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/manikantpandey/Call-Bot.git
   cd Call\ Bot
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**

   Copy `.env.example` to `.env` and fill in your API keys and desired settings:

   ```sh
   cp .env.example .env
   ```

   Edit `.env` and set:
   - `PORT` (e.g., 8000)
   - `GROQ_API_KEY`
   - `DEEPGRAM_API_KEY`
   - (Optional) `GROQ_MODEL`, `SYSTEM_PROMPT`, `TTS_MODEL`

## Usage

1. **Start the server:**

   ```sh
   python main.py
   ```

   Or with Uvicorn directly:

   ```sh
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Expose your server to the internet (for Twilio):**

   ```sh
   ngrok http 8000
   ```

   Use the HTTPS URL provided by ngrok for your Twilio webhook configuration.

3. **Configure Twilio:**

   - Set your Twilio phone number's Voice webhook to `https://<your-ngrok-domain>/incoming-call`.

## API Endpoints

- `POST /incoming-call`: Twilio webhook for incoming calls. Returns TwiML to connect to the media stream.
- `WS /media-stream`: WebSocket endpoint for Twilio Media Streams (handles audio in both directions).

## Environment Variables

See [.env.example](.env.example) for all available configuration options.

## File Structure

- [`main.py`](main.py): Main FastAPI application and all logic.
- `requirements.txt`: Python dependencies.
- `.env.example`: Example environment configuration.

## License

MIT License

---

**Note:** This project is for demonstration and prototyping purposes. Production use may require additional security, error handling,