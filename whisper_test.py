#!/usr/bin/env python3
import argparse
from faster_whisper import WhisperModel
import time
import sys
import torch
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Optional
import scipy.signal

# Define global static variable for sample rate
SAMPLE_RATE = 16000

@dataclass
class AudioChunk:
    data: np.ndarray
    timestamp: float

class StreamSimulator:
    def __init__(self, audio_path, chunk_size_ms=100):
        # Load audio file
        self.audio, self.original_sample_rate = sf.read(audio_path)
        if len(self.audio.shape) > 1:
            self.audio = self.audio.mean(axis=1)  # Convert stereo to mono
        
        # Resample to 16kHz if necessary
        if self.original_sample_rate != SAMPLE_RATE:
            # Calculate number of samples in resampled audio
            new_length = int(len(self.audio) * SAMPLE_RATE / self.original_sample_rate)
            self.audio = scipy.signal.resample(self.audio, new_length)
            self.sample_rate = SAMPLE_RATE
        else:
            self.sample_rate = self.original_sample_rate
        
        # Normalize audio
        self.audio = (self.audio * 32768).astype(np.float32)
        
        self.chunk_size = int(self.sample_rate * chunk_size_ms / 1000)
        self.position = 0
        print(f"# Loaded audio file with {len(self.audio)} samples at {self.sample_rate} Hz", file=sys.stderr)
        
    def get_chunk(self) -> Optional[AudioChunk]:
        if self.position >= len(self.audio):
            return None
            
        end_pos = min(self.position + self.chunk_size, len(self.audio))
        chunk = self.audio[self.position:end_pos]
        timestamp = self.position / self.sample_rate
        
        self.position = end_pos
        return AudioChunk(chunk, timestamp)

class StreamingTranscriber:
    def __init__(self, model_name="medium", device="cpu", compute_type="float32"):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.buffer = np.array([])
        self.sample_rate = SAMPLE_RATE  # Whisper's expected sample rate
        
    def process_chunk(self, chunk: AudioChunk, min_chunk_size=2.0):
        # Add chunk to buffer
        self.buffer = np.concatenate([self.buffer, chunk.data]) if len(self.buffer) > 0 else chunk.data
        
        # Only process if we have enough audio
        if len(self.buffer) >= self.sample_rate * min_chunk_size:
            print(f"# scanning buffer size {len(self.buffer)}")
            segments, _ = self.model.transcribe(
                self.buffer,
                word_timestamps=True,
                vad_filter=True
            )
            
            words_detected = False
            last_word_end = 0
            
            for segment in segments:
                for word in segment.words:
                    words_detected = True
                    print(f"{word.word}\t{word.probability:.3f}")
                    last_word_end = word.end
            
            if words_detected:
                # Keep audio after last word
                samples_to_keep = int(len(self.buffer) - (last_word_end * self.sample_rate))
                self.buffer = self.buffer[-samples_to_keep:] if samples_to_keep > 0 else np.array([])
                print(f"# keeping {samples_to_keep} samples")

def process_file(args):
    device = get_available_device() if args.device == "cuda" else "cpu"
    compute_type = "float32" if device == "cpu" else args.compute_type
    
    if args.streaming:
        print(f"# Processing in streaming mode with model: {args.model_size} on {device}", file=sys.stderr)
        start_time = time.time()
        
        simulator = StreamSimulator(args.input_file, chunk_size_ms=args.chunk_size)
        transcriber = StreamingTranscriber(args.model_size, device, compute_type)
        
        while True:
            chunk = simulator.get_chunk()
            if chunk is None:
                break
            transcriber.process_chunk(chunk)
                    
        print(f"# Streaming processing took {time.time() - start_time:.2f}s", file=sys.stderr)
        
    else:
        # Original non-streaming processing
        model = WhisperModel(args.model_size, device=device, compute_type=compute_type)
        print(f"# Processing with model: {args.model_size} on {device}", file=sys.stderr)
        start_time = time.time()
        
        segments, _ = model.transcribe(
            args.input_file,
            word_timestamps=True,
            vad_filter=args.vad,
            vad_parameters=dict(
                min_silence_duration_ms=args.min_silence,
                threshold=args.vad_threshold
            ),
            beam_size=args.beam_size,
            best_of=args.best_of,
            temperature=args.temperature
        )
        
        for segment in segments:
            for word in segment.words:
                if word.probability >= args.min_confidence:
                    print(f"{word.start:.3f}\t{word.word}\t{word.probability:.3f}")
                    
        print(f"# Processing took {time.time() - start_time:.2f}s", file=sys.stderr)

def get_available_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            return "cuda"
        except Exception as e:
            print(f"# CUDA error detected ({str(e)}), falling back to CPU", file=sys.stderr)
    return "cpu"

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio file with word-level timestamps')
    
    # Previous arguments remain the same...
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('--model-size', '-m', 
                       choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2'],
                       default='medium')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--compute-type', choices=['default', 'float16', 'float32'], default='float16')
    parser.add_argument('--beam-size', '-b', type=int, default=5)
    parser.add_argument('--best-of', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--vad', action='store_true')
    parser.add_argument('--min-silence', type=int, default=1000)
    parser.add_argument('--vad-threshold', type=float, default=0.35)
    parser.add_argument('--min-confidence', '-c', type=float, default=0.0)
    
    # New streaming-specific arguments
    parser.add_argument('--streaming', '-s', action='store_true',
                       help='Process file in streaming mode')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Chunk size in milliseconds for streaming mode')
    
    args = parser.parse_args()
    
    try:
        process_file(args)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()