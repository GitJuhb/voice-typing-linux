#!/usr/bin/env python3
"""
Enhanced voice typing with pre-recording buffer
Combines faster-whisper with pre-buffer technique from RealtimeSTT
"""

import argparse
import numpy as np
import pyaudio
import webrtcvad
import collections
import subprocess
import sys
import signal
import time
import threading
import queue
from faster_whisper import WhisperModel

class VoiceTyping:
    def __init__(self, model_size="small", device="cpu", language="en"):
        # Audio settings
        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30  # ms
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        self.PRE_BUFFER_DURATION_SEC = 1.5  # Pre-recording buffer
        self.BUFFER_DURATION_SEC = 4.0      # Main buffer (longer for context)
        self.SILENCE_DURATION_SEC = 0.8     # Silence to trigger processing
        
        # VAD settings - less aggressive
        self.vad = webrtcvad.Vad(1)  # Least aggressive mode
        
        # Initialize Whisper
        print(f"Loading {model_size} model on {device}...")
        compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        
        # Warm up the model
        print("Warming up model...")
        dummy_audio = np.zeros(16000, dtype=np.float32)
        list(self.model.transcribe(dummy_audio, language=language))
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE
        )
        
        # Pre-recording circular buffer (always recording)
        self.pre_buffer = collections.deque(
            maxlen=int(self.PRE_BUFFER_DURATION_SEC * self.RATE / self.CHUNK_SIZE)
        )
        
        # Main recording buffer
        self.recording_buffer = []
        
        # State
        self.is_recording = False
        self.silence_chunks = 0
        self.speech_detected = False
        
        # Threading for continuous pre-buffering
        self.audio_queue = queue.Queue()
        self.running = True
        
    def audio_reader_thread(self):
        """Continuously read audio in a separate thread"""
        while self.running:
            try:
                chunk = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                self.audio_queue.put(chunk)
            except:
                pass
                
    def process_audio(self, audio_data):
        """Transcribe audio and type it out"""
        # Convert to numpy array
        audio_np = np.frombuffer(b''.join(audio_data), dtype=np.int16).astype(np.float32) / 32768.0
        
        # Skip if too short
        if len(audio_np) < 0.5 * self.RATE:  # Less than 0.5 seconds
            return
        
        # Transcribe with optimized settings
        segments, _ = self.model.transcribe(
            audio_np,
            language=self.language,
            beam_size=5,  # Better accuracy
            best_of=3,    # Multiple attempts
            temperature=0.0,
            without_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=400  # Padding around speech
            )
        )
        
        # Collect all text
        full_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        
        if full_text:
            # Type it out
            self.type_text(full_text)
            
    def type_text(self, text):
        """Type the text using ydotool or xdotool"""
        try:
            # Try ydotool first
            subprocess.run(['ydotool', 'type', '-d1', '-H1', text + ' '],
                         capture_output=True, text=True, check=True)
            print(f"✓ {text}")
        except:
            # Fallback to xdotool
            try:
                subprocess.run(['xdotool', 'type', '--delay', '10', text + ' '], 
                             capture_output=True, check=True)
                print(f"✓ {text}")
            except:
                print(f"Failed to type: {text}")
                
    def run(self):
        """Main recording loop"""
        print("\n🎤 Enhanced voice typing active!")
        print("Speak naturally - initial words won't be missed!")
        print("Press Ctrl+C to stop\n")
        
        # Start audio reader thread
        reader_thread = threading.Thread(target=self.audio_reader_thread)
        reader_thread.daemon = True
        reader_thread.start()
        
        try:
            while self.running:
                # Get audio chunk from queue
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Always add to pre-buffer (circular buffer)
                self.pre_buffer.append(chunk)
                
                # Check for speech
                is_speech = self.vad.is_speech(chunk, self.RATE)
                
                if is_speech and not self.is_recording:
                    # Start recording - include pre-buffer!
                    print("🎤 ", end='', flush=True)
                    self.is_recording = True
                    self.recording_buffer = list(self.pre_buffer)  # Include pre-buffer
                    self.recording_buffer.append(chunk)
                    self.silence_chunks = 0
                    
                elif self.is_recording:
                    # Continue recording
                    self.recording_buffer.append(chunk)
                    
                    if not is_speech:
                        self.silence_chunks += 1
                        silence_duration = self.silence_chunks * self.CHUNK_DURATION_MS / 1000
                        
                        if silence_duration >= self.SILENCE_DURATION_SEC:
                            # Process the recording
                            print("[processing] ", end='', flush=True)
                            self.process_audio(self.recording_buffer)
                            
                            # Reset
                            self.is_recording = False
                            self.recording_buffer = []
                            self.silence_chunks = 0
                    else:
                        self.silence_chunks = 0
                        
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        time.sleep(0.1)  # Let thread finish
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print("\n\n🛑 Voice typing stopped.")

def main():
    parser = argparse.ArgumentParser(description='Enhanced voice typing with pre-buffer')
    parser.add_argument('--model', default='small',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Model size (default: small)')
    parser.add_argument('--device', default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device (default: cpu)')
    parser.add_argument('--language', default='en',
                       help='Language code (default: en)')
    
    args = parser.parse_args()
    
    # Check CUDA if requested
    if args.device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA not available, using CPU")
                args.device = 'cpu'
        except ImportError:
            print("PyTorch not installed, using CPU")
            args.device = 'cpu'
    
    # Run voice typing
    vt = VoiceTyping(
        model_size=args.model,
        device=args.device,
        language=args.language
    )
    
    # Handle graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    
    vt.run()

if __name__ == '__main__':
    main()