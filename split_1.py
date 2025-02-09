import numpy as np
import webrtcvad
import librosa
import sounddevice as sd
from scipy.spatial.distance import cosine
from collections import deque
import argparse
from datetime import timedelta
import sys
import time

class AudioSegmenter:
    def __init__(self, sample_rate=16000, frame_duration_ms=30, 
                 vad_mode=3, speaker_change_threshold=0.3,
                 min_silence_duration_ms=500, speaker_embedding_similarity=0.8):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.speaker_change_threshold = speaker_change_threshold
        self.min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
        self.speaker_embedding_similarity = speaker_embedding_similarity
        
        self.vad = webrtcvad.Vad(vad_mode)
        self.audio_buffer = []
        self.feature_buffer = deque(maxlen=50)
        self.silence_counter = 0
        
        # New speaker tracking attributes
        self.speaker_embeddings = []  # List of known speaker embeddings
        self.current_speaker_id = None
        self.speaker_timeline = []  # List of (start_time, end_time, speaker_id) tuples
        self.segment_start_time = 0.0
    
    def _extract_features(self, audio_chunk):
        mfccs = librosa.feature.mfcc(
            y=audio_chunk,
            sr=self.sample_rate,
            n_mfcc=20,
            hop_length=int(self.frame_size/2)
        )
        return np.mean(mfccs, axis=1)
    
    def _identify_speaker(self, current_features):
        if not self.speaker_embeddings:
            self.speaker_embeddings.append(current_features)
            return 0
        
        # Compare with all known speakers
        similarities = [1 - cosine(current_features, emb) for emb in self.speaker_embeddings]
        max_similarity = max(similarities)
        best_match = np.argmax(similarities)
        
        # If similarity is too low, this is a new speaker
        if max_similarity < self.speaker_embedding_similarity:
            self.speaker_embeddings.append(current_features)
            return len(self.speaker_embeddings) - 1
        
        return best_match
    
    def _detect_speaker_change(self, current_features):
        if len(self.feature_buffer) < 2:
            return False
        recent_features = np.mean(list(self.feature_buffer), axis=0)
        return cosine(current_features, recent_features) > self.speaker_change_threshold
    
    def process_chunk(self, audio_chunk, current_time):
        audio_chunk = np.array(audio_chunk, dtype=np.float32)
        audio_chunk_int16 = (audio_chunk * 32768).astype(np.int16)
        
        is_voice = self.vad.is_speech(
            audio_chunk_int16.tobytes(),
            self.sample_rate
        )
        
        current_features = self._extract_features(audio_chunk)
        is_speaker_change = False
        speaker_id = None
        
        if is_voice:
            is_speaker_change = self._detect_speaker_change(current_features)
            
            if is_speaker_change or self.current_speaker_id is None:
                # Identify the speaker
                speaker_id = self._identify_speaker(current_features)
                
                # If there was a previous speaker, record their timeline
                if self.current_speaker_id is not None:
                    self.speaker_timeline.append((
                        self.segment_start_time,
                        current_time,
                        self.current_speaker_id
                    ))
                    self.segment_start_time = current_time
                
                self.current_speaker_id = speaker_id
            
            self.feature_buffer.append(current_features)
            self.silence_counter = 0
        else:
            self.silence_counter += 1
            
            # If silence is long enough, close the current speaker segment
            if self.silence_counter >= self.min_silence_frames and self.current_speaker_id is not None:
                self.speaker_timeline.append((
                    self.segment_start_time,
                    current_time,
                    self.current_speaker_id
                ))
                self.current_speaker_id = None
        
        self.audio_buffer.append(audio_chunk)
        
        should_segment = (
            is_speaker_change or 
            self.silence_counter >= self.min_silence_frames
        )
        
        if should_segment and len(self.audio_buffer) > 0:
            segment = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            self.silence_counter = 0
            return is_voice, is_speaker_change, segment, speaker_id
        
        return is_voice, is_speaker_change, None, speaker_id

    def reset(self):
        self.audio_buffer = []
        self.feature_buffer.clear()
        self.silence_counter = 0
        self.speaker_timeline = []
        self.current_speaker_id = None
        self.segment_start_time = 0.0

def play_audio_segment(segment, sample_rate):
    """Play an audio segment using sounddevice."""
    try:
        sd.play(segment, sample_rate)
        sd.wait()  # Wait until the audio is finished playing
    except Exception as e:
        print(f"Error playing audio segment: {e}", file=sys.stderr)

def format_timestamp(seconds):
    return str(timedelta(seconds=round(seconds, 2)))

def process_audio_file(file_path, frame_duration_ms=30, vad_mode=3,
                      speaker_change_threshold=0.3, min_silence_duration_ms=500,
                      play_segments=False):
    print(f"Loading audio file: {file_path}")
    try:
        # librosa can handle mp3 files automatically through ffmpeg
        audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    except Exception as e:
        print(f"Error loading audio file: {e}", file=sys.stderr)
        sys.exit(1)
    
    segmenter = AudioSegmenter(
        sample_rate=16000,
        frame_duration_ms=frame_duration_ms,
        vad_mode=vad_mode,
        speaker_change_threshold=speaker_change_threshold,
        min_silence_duration_ms=min_silence_duration_ms
    )
    
    frame_size = int(16000 * frame_duration_ms / 1000)
    total_frames = len(audio) // frame_size
    last_voice_state = False
    transitions = []
    
    print("\nProcessing audio...")
    print("Timestamp | Event | Speaker")
    print("-" * 60)
    
    for i in range(total_frames):
        chunk = audio[i * frame_size:(i + 1) * frame_size]
        current_time = i * frame_duration_ms / 1000
        is_voice, is_speaker_change, segment, speaker_id = segmenter.process_chunk(chunk, current_time)
        
        if is_voice != last_voice_state:
            event = "Voice Started" if is_voice else "Voice Ended"
            print(f"{format_timestamp(current_time)} | {event}")
            transitions.append((current_time, event, None))
            last_voice_state = is_voice
        
        if is_speaker_change and speaker_id is not None:
            print(f"{format_timestamp(current_time)} | Speaker Change | Speaker {speaker_id}")
            transitions.append((current_time, "Speaker Change", speaker_id))
        
        if segment is not None:
            segment_duration = len(segment) / 16000
            print(f"{format_timestamp(current_time)} | Segment End ({segment_duration:.2f}s)")
            transitions.append((current_time, f"Segment End ({segment_duration:.2f}s)", None))
            
            if play_segments:
                print("Playing segment...")
                play_audio_segment(segment, 16000)
                # Add a small pause between segments
                time.sleep(0.5)
    
    # Print speaker timeline
    print("\nSpeaker Timeline:")
    print("Start Time | End Time | Speaker")
    print("-" * 60)
    for start, end, speaker_id in segmenter.speaker_timeline:
        print(f"{format_timestamp(start)} | {format_timestamp(end)} | Speaker {speaker_id}")
    
    return transitions, segmenter.speaker_timeline

def main():
    parser = argparse.ArgumentParser(description='Process audio file for voice activity and speaker changes.')
    parser.add_argument('file_path', help='Path to the audio file')
    parser.add_argument('--frame-duration', type=int, default=30,
                      help='Frame duration in milliseconds (default: 30)')
    parser.add_argument('--vad-mode', type=int, choices=[0, 1, 2, 3], default=3,
                      help='VAD aggressiveness mode (0-3, default: 3)')
    parser.add_argument('--speaker-threshold', type=float, default=0.3,
                      help='Speaker change detection threshold (default: 0.3)')
    parser.add_argument('--min-silence', type=int, default=500,
                      help='Minimum silence duration in ms (default: 500)')
    parser.add_argument('-p', '--play-segments', action='store_true',
                      help='Play audio segments as they are detected')
    
    args = parser.parse_args()
    
    try:
        transitions, speaker_timeline = process_audio_file(
            args.file_path,
            frame_duration_ms=args.frame_duration,
            vad_mode=args.vad_mode,
            speaker_change_threshold=args.speaker_threshold,
            min_silence_duration_ms=args.min_silence,
            play_segments=args.play_segments
        )
        
        print("\nFinal Speaker Timeline:")
        print("Start Time | End Time | Speaker")
        print("-" * 60)
        for start, end, speaker_id in speaker_timeline:
            print(f"{format_timestamp(start)} | {format_timestamp(end)} | Speaker {speaker_id}")
        
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()