#!/usr/bin/env python3
from mutagen.oggvorbis import OggVorbis
import sys

def add_chapters(filename):
    try:
        audio = OggVorbis(filename)
    except Exception as e:
        print(f"Error opening {filename}: {e}")
        sys.exit(1)

    # Add chapter markers (using the Xiph Chapter Extension convention)
    audio["CHAPTER001"] = "00:00:00.000"      # Start time for Chapter 1
    audio["CHAPTER001NAME"] = "Chapter 1: Start"
    audio["CHAPTER002"] = "00:00:05.000"      # Start time for Chapter 2
    audio["CHAPTER002NAME"] = "Chapter 2: Five Seconds"

    # Save the changes back to the file
    audio.save()
    print(f"Chapters added successfully to {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_chapters.py <file.ogg>")
        sys.exit(1)
    add_chapters(sys.argv[1])
