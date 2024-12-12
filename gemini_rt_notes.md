## client setup

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

server replies:

```json
{ "setupComplete": {} }
```

## client audio

Note: it seems you can only send one chunk per message right now, got back a "Realtime user input must have exactly one m; then sent 1007 (invalid frame payload data)" error when batching multiple.

```json
{
  "realtimeInput": {
    "mediaChunks": [
      { "mimeType": "audio/pcm;rate=16000", "data": "AAAAAAAA...EAO7/1P/b/w==" }
    ]
  }
}
```

## server audio

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

## client text

```json
{
  "clientContent": {
    "turns": [{ "role": "user", "parts": [{ "text": "hello" }] }],
    "turnComplete": true
  }
}
```
