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

# Add global connection tracking
pcs = set()
pcs_by_resource_id = {}
handlers_by_resource_id = {}

class WHIPHandler:

    def __init__(self, offer: RTCSessionDescription):
        # Configure DTLS role
        self.pc = RTCPeerConnection()
        # Only add audio transceiver, set to recvonly
        self.pc.addTransceiver("audio", direction="recvonly")
        
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

        frame_count = 0

        while not self.connection_closed.is_set():
            try:
                frame = await track.recv()
                frame_count += 1

                resampled_frames = resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    audio_data = resampled_frame.to_ndarray().tobytes()
                    await self.audio_queue.put(audio_data)

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

        try:
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
        except Exception as e:
            print(f"Error in start_websocket: {e}")
            self.connection_closed.set()

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
                    #await self.websocket.send_json(commit_event)
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
                    print(f"Received event: {event_type}")#, msg.data)
                    if event_type == "response.text.done":
                        text = event.get("text", "")
                        if text:
                            print(f"Aimbe: {text}", end='', flush=True)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in receive_messages: {e}")
            self.connection_closed.set()

    async def _wait_for_ice_gathering(self):
        """Wait for ICE gathering to complete."""
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

    async def run(self):
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state changed to: {self.pc.connectionState}")
            if self.pc.connectionState in ["failed", "closed"]:
                self.connection_closed.set()

        @self.pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                print(f"Received audio track")
                asyncio.create_task(self.handle_audio_track(track))

        await self.pc.setRemoteDescription(self.offer)
        print("Set remote description.")

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        # Wait for ICE gathering to complete
        await self._wait_for_ice_gathering()
        
        print("Created and set local description (answer).")
        asyncio.create_task(self.start_websocket())

    async def close(self):
        await self.pc.close()
        pcs.discard(self.pc)
        pcs_by_resource_id.pop(str(self.id), None)
        handlers_by_resource_id.pop(str(self.id), None)
        print(f"WHIPHandler {self.id} closed.")

async def handle_whip(request):
    # Add CORS headers
    if request.method == "OPTIONS":
        response = web.Response(status=204)
        response.headers.update({
            'Access-Control-Allow-Methods': 'OPTIONS, POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '86400',
        })
        return response

    if request.method != "POST":
        return web.Response(status=405, text="Method Not Allowed")

    try:
        content_type = request.headers.get('Content-Type', '').lower()
        if content_type == 'application/sdp':
            sdp_content = await request.text()
            offer_json = {
                "type": "offer",
                "sdp": sdp_content
            }
        elif content_type == 'application/json':
            offer_json = await request.json()
        else:
            return web.Response(
                status=415,
                text=f"Unsupported Content-Type: {content_type}. Expected application/sdp or application/json"
            )

        print(f"Processed offer: {offer_json}")
    except Exception as e:
        print(f"Error processing request: {e}")
        return web.Response(status=400, text=str(e))

    handler = None

    try:
        offer = RTCSessionDescription(sdp=offer_json.get("sdp"), type=offer_json.get("type", "offer"))

        handler = WHIPHandler(offer)
        resource_id = str(handler.id)
        
        # Track connections
        pcs.add(handler.pc)
        pcs_by_resource_id[resource_id] = handler.pc
        handlers_by_resource_id[resource_id] = handler

        await handler.run()

        # No need to modify SDP since we're already configured for receive-only
        answer_sdp = handler.pc.localDescription.sdp
        
        # Return the answer with the correct content type and CORS headers
        response = web.Response(
            content_type='application/sdp',
            text=answer_sdp,
            status=201  # Created
        )
        response.headers.update({
            'Location': f'{WHIP_ENDPOINT}/{resource_id}',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Expose-Headers': 'Location'
        })
        return response

    except Exception as e:
        print(f"Error in WHIP handler: {e}")
        if handler:
            await handler.close()
        return web.Response(status=500, text=str(e))

async def handle_delete(request):
    resource_id = request.match_info['resource_id']
    handler = handlers_by_resource_id.get(resource_id)
    if handler:
        await handler.close()
        return web.Response(status=200)
    return web.Response(status=404)

async def on_shutdown(app):
    # Close all peer connections
    coros = [handler.close() for handler in handlers_by_resource_id.values()]
    await asyncio.gather(*coros)
    pcs.clear()
    pcs_by_resource_id.clear()
    handlers_by_resource_id.clear()

async def debug_request(request):
    print(f"\nDEBUG: {request.method} {request.path}")
    print("Headers:")
    for name, value in request.headers.items():
        print(f"  {name}: {value}")
    
    if request.can_read_body:
        body = await request.text()
        print(f"Body: {body}\n")
    
    return await handle_whip(request) if request.path == WHIP_ENDPOINT else web.Response(status=404)

def main():
    app = web.Application()
    app.router.add_post(WHIP_ENDPOINT, handle_whip)
    app.router.add_delete(f'{WHIP_ENDPOINT}/{{resource_id}}', handle_delete)
    app.on_shutdown.append(on_shutdown)
    app.router.add_route('*', '/{tail:.*}', debug_request)
    
    print(f"Starting WHIP server on port {SERVER_PORT}, endpoint {WHIP_ENDPOINT}")
    web.run_app(app, port=SERVER_PORT)

if __name__ == "__main__":
    main()