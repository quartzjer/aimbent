import asyncio, os, tempfile, io
from dbus_next.aio import MessageBus
from dbus_next.constants import BusType
import gi
gi.require_version('Gdk', '4.0')
from gi.repository import Gdk
from PIL import Image

async def take_screenshot():
    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    introspection = await bus.introspect('org.gnome.Shell.Screenshot', '/org/gnome/Shell/Screenshot')
    proxy_obj = bus.get_proxy_object('org.gnome.Shell.Screenshot',
                                     '/org/gnome/Shell/Screenshot',
                                     introspection)
    interface = proxy_obj.get_interface('org.gnome.Shell.Screenshot')
    temp_path = os.path.join(tempfile.gettempdir(), "screenshot.png")
    # Call the Screenshot method (non-interactive, no flash)
    await interface.call_screenshot(False, False, temp_path)
    with open(temp_path, "rb") as f:
        screenshot_bytes = f.read()
    os.remove(temp_path)
    return screenshot_bytes

def get_monitor_geometries():
    # Get the default display. If it is None, try opening one from the environment.
    display = Gdk.Display.get_default()
    if display is None:
        env_display = os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")
        if env_display is not None:
            display = Gdk.Display.open(env_display)
        if display is None:
            raise RuntimeError("No display available")
    # In GTK 4, get_monitors() returns a list of Gdk.Monitor objects.
    monitors = display.get_monitors()
    geometries = []
    for monitor in monitors:
        geom = monitor.get_geometry()  # geom is a Gdk.Rectangle with attributes: x, y, width, height
        geometries.append(geom)
    return geometries

# New: asynchronous snapshot helper that returns an array of monitor images.
async def screen_snap_async():
    # Take the screenshot for all monitors.
    screenshot_data = await take_screenshot()
    im = Image.open(io.BytesIO(screenshot_data))
    geometries = get_monitor_geometries()
    monitor_images = []
    for geom in geometries:
        box = (geom.x, geom.y, geom.x + geom.width, geom.y + geom.height)
        monitor_img = im.crop(box)
        monitor_images.append(monitor_img)
    return monitor_images

# New: synchronous wrapper to call the asynchronous snapshot.
def screen_snap():
    return asyncio.run(screen_snap_async())

def main():
    # Updated main: use screen_snap() and save full images.
    monitor_images = screen_snap()
    for idx, img in enumerate(monitor_images, start=1):
        filename = f"monitor_{idx}.png"
        img.save(filename)
        print(f"Saved {filename}")
    print(f"Processed {len(monitor_images)} monitor images.")

if __name__ == "__main__":
    main()
