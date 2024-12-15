## Connection

Connect to the WebSocket endpoint:

```
wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key=YOUR_API_KEY
```

## Protocol Overview

The protocol follows a bidirectional communication pattern where:

1. Client establishes connection and sends setup message
2. Server acknowledges with setup completion
3. Client can then stream audio chunks or send text messages
4. Server responds with audio or text responses

All messages are JSON encoded.

## Message Types

### 1. Initial Setup

Client must send this message immediately after connection:

```json
{
  "setup": {
    "model": "models/gemini-2.0-flash-exp",
    "generationConfig": {
      "responseModalities": "audio",
      "speechConfig": {
        "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": "Aoede" } }
      }
    },
    "systemInstruction": {
      "parts": [
        {
          "text": "You are my helpful assistant."
        }
      ]
    },
    "tools": [
      { "googleSearch": {} },
      {
        "functionDeclarations": [
          {
            "name": "foo",
            "description": "Does foo.",
            "parameters": {
              "type": "object",
              "properties": {
                "bar": {
                  "type": "string",
                  "description": "bar stuff"
                }
              },
              "required": ["bar"]
            }
          }
        ]
      }
    ]
  }
}
```

Server acknowledges with:

```json
{ "setupComplete": {} }
```

### 2. Audio Streaming

#### Client Audio Input

- Audio must be PCM format, 16kHz sample rate, 16-bit
- Audio data must be base64 encoded
- Note: seems you can only send one chunk per message right now, got back a "Realtime user input must have exactly one m; then sent 1007 (invalid frame payload data)" error when batching multiple.

```json
{
  "realtimeInput": {
    "mediaChunks": [
      { "mimeType": "audio/pcm;rate=16000", "data": "AAAAAAAA...EAO7/1P/b/w==" }
    ]
  }
}
```

#### Server Audio Response

Server responds with PCM audio at 24kHz sample rate:

```json
{
  "serverContent": {
    "modelTurn": {
      "parts": [
        {
          "inlineData": {
            "mimeType": "audio/pcm;rate=24000",
            "data": "nOwc7FjqC6G...sOyAEF5gIvAev/"
          }
        }
      ]
    }
  }
}
```

When complete, server also sends:

```json
{
  "serverContent": {
    "turnComplete": true
  }
}
```

### 3. Text Messages (Optional)

Client can also send text messages:

```json
{
  "clientContent": {
    "turns": [{ "role": "user", "parts": [{ "text": "hello" }] }],
    "turnComplete": true
  }
}
```
