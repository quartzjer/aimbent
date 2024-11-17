import asyncio
import argparse
import logging
import time
import uuid
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

logging.basicConfig(level=logging.INFO)
pcs = set()
pcs_by_resource_id = {}


async def handle_whip(request):
    # Extract the SDP offer from the request
    offer_sdp = await request.text()
    logging.info("Received SDP offer")

    pc = RTCPeerConnection()

    resource_id = str(uuid.uuid4())
    pcs_by_resource_id[resource_id] = pc
    pcs.add(pc)

    total_bytes = {'value': 0}
    start_time = time.time()
    logging_task_started = False

    @pc.on("track")
    def on_track(track):
        logging.info(f"Track received: {track.kind}")
        asyncio.ensure_future(process_track(track, total_bytes))

        nonlocal logging_task_started
        if not logging_task_started:
            asyncio.ensure_future(log_bytes(total_bytes, start_time))
            logging_task_started = True

    @pc.on('connectionstatechange')
    async def on_connectionstatechange():
        logging.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ('failed', 'closed'):
            await pc.close()
            pcs.discard(pc)
            pcs_by_resource_id.pop(resource_id, None)

    # Set the remote description
    offer = RTCSessionDescription(sdp=offer_sdp, type='offer')
    await pc.setRemoteDescription(offer)

    # Create an answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Return the SDP answer with the required Location header
    response = web.Response(
        text=pc.localDescription.sdp,
        content_type='application/sdp',
        status=201
    )
    response.headers['Location'] = f'/whip/{resource_id}'
    return response


async def handle_delete(request):
    resource_id = request.match_info['resource_id']
    pc = pcs_by_resource_id.get(resource_id)
    if pc:
        await pc.close()
        pcs.discard(pc)
        del pcs_by_resource_id[resource_id]
        return web.Response(status=200)
    else:
        return web.Response(status=404)


async def process_track(track, total_bytes):
    while True:
        frame = await track.recv()
        if track.kind == 'video':
            # For video frames
            total_bytes['value'] += sum(len(p.buffer) for p in frame.planes)
        elif track.kind == 'audio':
            # For audio frames
            total_bytes['value'] += len(frame.data)


async def log_bytes(total_bytes, start_time):
    while True:
        await asyncio.sleep(1)
        elapsed = time.time() - start_time
        logging.info(f"Total bytes received: {total_bytes['value']} in {elapsed:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="WHIP Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to listen on')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    app = web.Application()
    app.router.add_post('/whip', handle_whip)
    app.router.add_delete('/whip/{resource_id}', handle_delete)

    async def on_shutdown(app):
        coros = [pc.close() for pc in pcs]
        await asyncio.gather(*coros)
        pcs.clear()
        pcs_by_resource_id.clear()

    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()