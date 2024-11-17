#!/usr/bin/env python3

import asyncio
import json
import uuid
import av
import pyaudio

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

SERVER_PORT = 8080
WHIP_ENDPOINT = "/whip"

class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True
        )
        print(f"Audio initialized with default output device")

    def play(self, audio_data):
        self.stream.write(audio_data)

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

class WHIPHandler:

    def __init__(self, offer: RTCSessionDescription):
        self.pc = RTCPeerConnection()
        self.offer = offer
        self.id = uuid.uuid4()
        self.connection_closed = asyncio.Event()
        self.audio_player = AudioPlayer()
        print(f"Initialized WHIPHandler with ID: {self.id}")

    async def handle_audio_track(self, track: MediaStreamTrack):
        print(f"Handling audio track: {track.kind}")
        print(f"Track settings: {track.kind}, ID: {getattr(track, 'id', 'N/A')}")

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
                if not resampled_frames:
                    continue

                for resampled_frame in resampled_frames:
                    audio_data = resampled_frame.to_ndarray().tobytes()
                    self.audio_player.play(audio_data)

                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")

            except Exception as e:
                print(f"Error receiving or processing audio frame: {e}")
                break

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

    async def close(self):
        self.audio_player.cleanup()
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