#!/usr/bin/env python3

import asyncio
import json
import uuid
import av
from pathlib import Path

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

# Configuration Constants
SERVER_PORT = 8080
WHIP_ENDPOINT = "/whip"
OUTPUT_FILE = "./recording.pcm"

class WHIPHandler:
    """
    Handles a single WHIP client connection.
    """

    def __init__(self, offer: RTCSessionDescription):
        self.pc = RTCPeerConnection()
        self.offer = offer
        self.id = uuid.uuid4()
        self.connection_closed = asyncio.Event()
        print(f"Initialized WHIPHandler with ID: {self.id}")

    async def handle_audio_track(self, track: MediaStreamTrack):
        """
        Handles incoming audio tracks by resampling and writing data to file.
        """
        print(f"Handling audio track: {track.kind}")

        # Open output file
        with open(OUTPUT_FILE, 'wb') as output_file:

            # Create an audio resampler
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=24000
            )

            frame_count = 0
            byte_count = 0

            while not self.connection_closed.is_set():
                try:
                    frame = await track.recv()
                    frame_count += 1

                    # Convert aiortc.AudioFrame to av.AudioFrame
                    av_frame = av.AudioFrame.from_ndarray(
                        frame.to_ndarray(),
                        format=frame.format.name,
                        layout=frame.layout.name
                    )
                    av_frame.sample_rate = frame.sample_rate

                    # Resample the frame
                    resampled_frames = resampler.resample(av_frame)

                    # Extract and write raw PCM data
                    for resampled_frame in resampled_frames:
                        for plane in resampled_frame.planes:
                            pcm_data = bytes(plane)
                            byte_count += len(pcm_data)
                            output_file.write(pcm_data)

                    # Provide progress updates every 100 frames
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count} frames, {byte_count} bytes written.")

                except Exception as e:
                    print(f"Error receiving audio frame: {e}")
                    break

    async def run(self):
        """
        Sets up the peer connection, handles negotiation, and processes incoming tracks.
        """
        # Handle incoming tracks
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

        # Set remote description
        await self.pc.setRemoteDescription(self.offer)
        print("Set remote description.")

        # Create answer
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        print("Created and set local description (answer).")

    async def close(self):
        """
        Closes the peer connection.
        """
        await self.pc.close()
        print(f"WHIPHandler {self.id} closed.")

async def handle_whip(request):
    """
    Handles incoming WHIP client requests.
    """
    if request.method != "POST":
        return web.Response(status=405, text="Method Not Allowed")

    try:
        offer_json = await request.json()
        print(f"Received WHIP offer: {offer_json}")
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    handler = None

    try:
        # Extract SDP offer
        offer = RTCSessionDescription(sdp=offer_json.get("sdp"), type=offer_json.get("type"))

        # Initialize WHIP Handler
        handler = WHIPHandler(offer)
        await handler.run()

        # Create answer JSON
        answer = {
            "type": handler.pc.localDescription.type,
            "sdp": handler.pc.localDescription.sdp,
        }

        print(f"Sending WHIP answer: {answer}")

        # Return the answer
        return web.json_response(answer)

    except Exception as e:
        print(f"Error in WHIP handler: {e}")
        if handler:
            await handler.close()
        return web.Response(status=500, text=str(e))

def main():
    # Create web application
    app = web.Application()
    app.router.add_post(WHIP_ENDPOINT, handle_whip)

    # Run the web server
    print(f"Starting WHIP server on port {SERVER_PORT}, endpoint {WHIP_ENDPOINT}")
    web.run_app(app, port=SERVER_PORT)

if __name__ == "__main__":
    main()