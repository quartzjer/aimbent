# Audio/Video Streaming Experiments

Collection of scripts for testing different streaming and chat functionalities with OpenAI's realtime API.

## Setup

1. Install dependencies (recommend using a virtual environment):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
2. Create a `.env` and set your `OPENAI_API_KEY`

## Available Scripts

### chat.py
Audio file playback with chat interface. Plays an audio file while allowing text chat interaction.
- Usage: `python chat.py path/to/audio.mp4 [--verbose]`
- Requires: Audio file (mp4 format), ffmpeg installed

### observe.py
Like chat.py but trying to be a passive observer and automatic commenter.

### test.py
Microphone input with chat interface. Records from default microphone while allowing text chat.
- Usage: `python test.py [--verbose]`
- Requires: Working microphone

### whip_test.py
WHIP (WebRTC HTTP Ingestion Protocol) test server
- Usage: Run server: `python whip_test.py`
- Listens on port 8080 at /whip endpoint

### whip_client.py
Client for testing WHIP protocol
- Usage: `python whip_client.py http://server/whip path/to/video.mp4`

### rtmp_test.py, rtmp_relay.py, etc
_Failed_ RTMP streaming experiments
- rtmp_test.py: Basic RTMP server with audio playback
- rtmp_relay.py: RTMP server that relays audio to OpenAI API
- Usage: Run server and connect with RTMP client to port 1935

## Common Features

- All scripts using OpenAI's API require `OPENAI_API_KEY` in `.env`
- Most scripts support a `--verbose` flag for debug output
- Use Ctrl+C to gracefully exit any script
- Audio settings are standardized across scripts:
  - 24kHz sample rate
  - 16-bit PCM
  - Mono channel
