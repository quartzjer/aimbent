#!/usr/bin/env python3
import dbus
import json
import sys
import argparse
from collections import defaultdict

# Add debug flag
DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def list_windows():
    debug_print("Connecting to D-Bus session...")
    bus = dbus.SessionBus()
    shell = bus.get_object("org.gnome.Shell", "/org/gnome/Shell")
    # JavaScript to retrieve an array of window info objects:
    # Each object contains:
    #   id    - the window ID (an integer)
    #   title - the window title
    #   app   - the application's WM_CLASS (its "app name")
    js_code = """
    JSON.stringify(global.get_window_actors().map(a => {
      let w = a.meta_window;
      return {id: w.get_id(), title: w.get_title(), app: w.get_wm_class()};
    }));
    """
    # Call the Eval method on GNOME Shell
    debug_print("Executing JavaScript in GNOME Shell...")
    
    # Fix: Swap the order of return values from shell.Eval()
    success, result = shell.Eval(js_code, dbus_interface="org.gnome.Shell")
    
    debug_print(f"Eval result type: {type(result)}, success: {success}")
    if not success:
        print("Error: Failed to evaluate JavaScript in GNOME Shell.")
        sys.exit(1)
    try:
        debug_print(f"Raw result: {result[:100]}..." if len(str(result)) > 100 else f"Raw result: {result}")
        
        # Fix: The result is already a JSON string, we need to handle it properly
        result_str = str(result)
        # Check if the string starts and ends with quotes (indicating a JSON string inside a string)
        if result_str.startswith('"') and result_str.endswith('"'):
            # Remove the outer quotes and unescape the inner JSON string
            result_str = json.loads(result_str)
            
        debug_print(f"Processed result (first 100): {result_str[:100]}..." if len(result_str) > 100 else f"Processed result: {result_str}")
        # Now parse the properly formatted JSON string
        windows = json.loads(result_str)
        debug_print(f"Found {len(windows)} windows")
        
        # Filter out invalid window entries
        valid_windows = []
        for i, win in enumerate(windows):
            if isinstance(win, dict) and "id" in win and "app" in win:
                valid_windows.append(win)
            else:
                debug_print(f"Skipping invalid window at index {i}: {type(win)}, value: {win}")
        
        windows = valid_windows
        debug_print(f"After filtering, {len(windows)} valid windows remain")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Result type: {type(result)}")
        if isinstance(result, str) and len(result) > 0:
            print(f"Result (first 100 chars): {result[:100]}...")
        sys.exit(1)
    return windows

def choose_app(windows):
    # Group windows by app name.
    apps = defaultdict(list)
    for win in windows:
        # At this point all windows should be valid dictionaries
        app_name = win.get("app")
        if app_name:
            apps[app_name].append(win)
    if not apps:
        print("No windows found from any application.")
        sys.exit(1)
    # List available app names (sorted alphabetically).
    app_list = sorted(apps.keys())
    print("Available Applications:")
    for i, app in enumerate(app_list):
        count = len(apps[app])
        print(f"{i}: {app} (has {count} window{'s' if count > 1 else ''})")
    try:
        choice = int(input("Choose an application by index: "))
    except ValueError:
        print("Invalid input. Exiting.")
        sys.exit(1)
    if choice < 0 or choice >= len(app_list):
        print("Invalid index. Exiting.")
        sys.exit(1)
    selected_app = app_list[choice]
    # Choose the first window from the selected app.
    selected_window = apps[selected_app][0]
    return selected_app, selected_window

def capture_window(window, output_path):
    debug_print(f"Capturing window with ID: {window['id']}")
    bus = dbus.SessionBus()
    # Get the GNOME Shell Screenshot object.
    screenshot_obj = bus.get_object("org.gnome.Shell.Screenshot", "/org/gnome/Shell/Screenshot")
    # Retrieve the window ID (convert to int if needed)
    window_id = int(window["id"])
    # Call ScreenshotWindow method.
    # Parameters: (window id, include_frame (bool), include_cursor (bool), filename (string))
    debug_print(f"Taking screenshot of window {window_id} to {output_path}")
    ret = screenshot_obj.ScreenshotWindow(window_id, False, False, output_path,
                                          dbus_interface="org.gnome.Shell.Screenshot")
    # ret is expected to be a tuple: (success (boolean), error_message (string))
    success, error_msg = ret
    debug_print(f"Screenshot result: success={success}, error_msg={error_msg}")
    if not success:
        print("Error capturing screenshot:", error_msg)
        sys.exit(1)

def main():
    global DEBUG
    
    # Add command-line argument for debug mode
    parser = argparse.ArgumentParser(description="Capture screenshots of Gnome application windows")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    DEBUG = args.debug
    debug_print("Debug mode enabled")
    
    windows = list_windows()
    selected_app, window = choose_app(windows)
    print(f"Selected application: {selected_app}\nWindow title: {window.get('title')}")
    # Save as <appname>.png (spaces in app names will be preserved)
    output_path = f"{selected_app}.png"
    capture_window(window, output_path)
    print(f"Screenshot saved to {output_path}")

if __name__ == "__main__":
    main()
