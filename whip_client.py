#!/usr/bin/env python3

import sys
import asyncio
import json
import uuid
from pathlib import Path

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer

async def send_offer(server_url, offer_json):
    async with aiohttp.ClientSession() as session:
        async with session.post(server_url, json=offer_json) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Failed to send offer: {response.status} {text}")
            answer_json = await response.json()
            return answer_json

async def run_whip_client(server_url, video_path):
    # Validate video file
    video_path = Path(video_path)
    if not video_path.is_file():
        print(f"Error: Video file '{video_path}' does not exist.")
        sys.exit(1)

    # Initialize media player
    player = MediaPlayer(str(video_path))

    # Create a new RTCPeerConnection
    pc = RTCPeerConnection()

    # Add video track
    if player.video:
        pc.addTrack(player.video)

    # Add audio track if available
    if player.audio:
        pc.addTrack(player.audio)

    # Create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Prepare WHIP offer JSON
    offer_json = {
        "type": pc.localDescription.type,
        "sdp": pc.localDescription.sdp
    }

    print("Sending offer to WHIP server...")

    try:
        # Send offer and receive answer
        answer_json = await send_offer(server_url, offer_json)
    except Exception as e:
        print(f"Error during WHIP negotiation: {e}")
        await pc.close()
        sys.exit(1)

    # Set remote description
    answer = RTCSessionDescription(sdp=answer_json["sdp"], type=answer_json["type"])
    await pc.setRemoteDescription(answer)

    print("WHIP negotiation successful. Streaming started.")

    try:
        # Keep the script running to maintain the connection
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Stopping streaming...")
    finally:
        await pc.close()
        player.stop()

def print_usage():
    print("Usage: python whip_client.py http://server/whip path/to/video.mp4")

def main():
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)

    server_url = sys.argv[1]
    video_path = sys.argv[2]

    asyncio.run(run_whip_client(server_url, video_path))

if __name__ == "__main__":
    main()