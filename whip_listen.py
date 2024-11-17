#!/usr/bin/env python3

import asyncio
import json
import uuid
import av
import base64
import os
import aiohttp

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

from dotenv import load_dotenv
load_dotenv()

SERVER_PORT = 8080
WHIP_ENDPOINT = "/whip"
WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

class WHIPHandler:

    def __init__(self, offer: RTCSessionDescription):
        self.pc = RTCPeerConnection()
        self.offer = offer
        self.id = uuid.uuid4()
        self.connection_closed = asyncio.Event()
        self.audio_queue = asyncio.Queue()
        self.api_key = os.getenv("OPENAI_API_KEY")

        print(f"Initialized WHIPHandler with ID: {self.id}")

    async def handle_audio_track(self, track: MediaStreamTrack):
        print(f"Handling audio track: {track.kind}")
        print(f"Track settings: {track.kind} {getattr(track, 'id', 'N/A')}")

        resampler = av.AudioResampler(
            format='s16',
            layout='mono',
            rate=24000
        )

        buffer = b''
        min_buffer_size = 4800
        frame_count = 0

        while not self.connection_closed.is_set():
            try:
                frame = await track.recv()
                frame_count += 1

                resampled_frames = resampler.resample(frame)

                for resampled_frame in resampled_frames:
                    audio_data = bytes(resampled_frame.planes[0])
                    buffer += audio_data

                    if len(buffer) >= min_buffer_size:
                        await self.audio_queue.put(buffer)
                        buffer = b''

                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")

            except Exception as e:
                print(f"Error receiving audio frame: {e}")
                await self.audio_queue.put(None)  # Signal end of audio data
                self.connection_closed.set()
                break

    async def start_websocket(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        session_update_message = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": "You will be listening to a conversation, your job is to notice when you can be helpful. You are not part of the conversation, you are only an observer, and you don't need to transcribe, just focus on providing concise actionable suggestions.",
                "turn_detection": {"type": "server_vad", "threshold": 0.5},
                "temperature": 0.7,
                "max_response_output_tokens": 500
            }
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.ws_connect(WEBSOCKET_URL) as ws:
                self.websocket = ws
                await ws.send_json(session_update_message)
                print("WebSocket connection established and session updated.")

                send_task = asyncio.create_task(self.send_audio_data())
                receive_task = asyncio.create_task(self.receive_messages())

                await self.connection_closed.wait()

                send_task.cancel()
                receive_task.cancel()
                await ws.close()
                print("WebSocket connection closed.")

    async def send_audio_data(self):
        try:
            sent_count = 0
            while not self.connection_closed.is_set():
                audio_data = await self.audio_queue.get()
                if audio_data is None:
                    break
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                audio_event = {"type": "input_audio_buffer.append", "audio": audio_base64}
                await self.websocket.send_json(audio_event)
                sent_count += 1
                if sent_count % 100 == 0:
                    commit_event = {"type": "input_audio_buffer.commit"}
                    await self.websocket.send_json(commit_event)
                    print(f"Committed {sent_count} buffers")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in send_audio_data: {e}")

    async def receive_messages(self):
        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    event_type = event.get('type')
                    print(f"Received event: {event_type}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in receive_messages: {e}")

    async def run(self):
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state changed to: {self.pc.connectionState}")
            if self.pc.connectionState in ["failed", "closed"]:
                self.connection_closed.set()

        @self.pc.on("track")
        async def on_track(track):
            print(f"Received track: {track.kind}")
            if track.kind == "audio":
                asyncio.create_task(self.handle_audio_track(track))
            else:
                print(f"Ignoring track of kind: {track.kind}")

        await self.pc.setRemoteDescription(self.offer)
        print("Set remote description.")

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        print("Created and set local description (answer).")

        asyncio.create_task(self.start_websocket())

    async def close(self):
        await self.pc.close()
        print(f"WHIPHandler {self.id} closed.")

async def handle_whip(request):
    if request.method != "POST":
        return web.Response(status=405, text="Method Not Allowed")

    try:
        offer_json = await request.json()
        print(f"Received WHIP offer: {offer_json}")
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    handler = None

    try:
        offer = RTCSessionDescription(sdp=offer_json.get("sdp"), type=offer_json.get("type"))

        handler = WHIPHandler(offer)
        if handler.connection_closed.is_set():
            return web.Response(status=500, text="Server Error: API key not set.")

        await handler.run()

        answer = {
            "type": handler.pc.localDescription.type,
            "sdp": handler.pc.localDescription.sdp,
        }

        print(f"Sending WHIP answer: {answer}")

        return web.json_response(answer)

    except Exception as e:
        print(f"Error in WHIP handler: {e}")
        if handler:
            await handler.close()
        return web.Response(status=500, text=str(e))

def main():
    app = web.Application()
    app.router.add_post(WHIP_ENDPOINT, handle_whip)

    print(f"Starting WHIP server on port {SERVER_PORT}, endpoint {WHIP_ENDPOINT}")
    web.run_app(app, port=SERVER_PORT)

if __name__ == "__main__":
    main()