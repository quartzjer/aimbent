import asyncio
import uuid
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from aiortc.sdp import candidate_from_sdp
import av

# Dictionary to keep track of peer connections
pcs = {}


async def handle_post(request):
    # Check Content-Type
    if request.content_type != 'application/sdp':
        return web.Response(status=415, text='Unsupported Media Type')

    # Get SDP offer from body
    sdp = await request.text()
    offer = RTCSessionDescription(sdp=sdp, type='offer')

    # Create a new RTCPeerConnection
    pc = RTCPeerConnection()
    session_id = str(uuid.uuid4())
    pcs[session_id] = pc

    # Handle tracks
    @pc.on('track')
    def on_track(track):
        print('Track %s received' % track.kind)
        if track.kind == 'video':
            # Start task to process incoming video frames
            asyncio.create_task(save_frames(track))

    # Set remote description
    await pc.setRemoteDescription(offer)

    # Create and set local description
    await pc.setLocalDescription(await pc.createAnswer())

    # Wait for ICE gathering to complete
    while pc.iceGatheringState != 'complete':
        await asyncio.sleep(0.1)

    # Construct Location header with the unique endpoint URL
    location_url = str(request.url.with_path(f'/whip/{session_id}').with_query(''))

    # Return response with SDP answer
    headers = {
        'Content-Type': 'application/sdp',
        'Location': location_url
    }
    return web.Response(status=201, headers=headers, text=pc.localDescription.sdp)


async def handle_patch(request):
    # Get session ID from URL
    session_id = request.match_info['id']
    pc = pcs.get(session_id)
    if not pc:
        return web.Response(status=404, text='Not Found')

    # Check Content-Type
    if request.content_type != 'application/trickle-ice-sdpfrag':
        return web.Response(status=415, text='Unsupported Media Type')

    # Get ICE candidate from body
    candidate_sdpfrag = await request.text()
    lines = candidate_sdpfrag.strip().splitlines()
    for line in lines:
        if line.startswith('a=candidate:'):
            sdp_line = line[2:]  # Remove 'a='
            candidate = candidate_from_sdp(sdp_line)
            await pc.addIceCandidate(candidate)
        elif line.startswith('a=end-of-candidates'):
            # Signal end of candidates
            await pc.addIceCandidate(None)

    return web.Response(status=204)


async def handle_delete(request):
    session_id = request.match_info['id']
    pc = pcs.pop(session_id, None)
    if not pc:
        return web.Response(status=404, text='Not Found')

    # Close the peer connection
    await pc.close()
    return web.Response(status=204)


async def save_frames(track):
    counter = 0
    while True:
        try:
            frame = await track.recv()
            counter += 1
            if counter % 30 == 0:
                # Save frame to snapshot.png
                img = frame.to_image()
                img.save('snapshot.png')
        except av.AVError as e:
            print('Error receiving frame:', e)
            break


app = web.Application()
app.router.add_post('/whip', handle_post)
app.router.add_patch('/whip/{id}', handle_patch)
app.router.add_delete('/whip/{id}', handle_delete)

if __name__ == '__main__':
    web.run_app(app, port=8080)
