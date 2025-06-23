#!/usr/bin/env python3
"""
Voice-to-Text Tool: Records audio while hotkey is pressed, transcribes via OpenAI, and pastes the text.

Usage:
    Run this script to start the global hotkey listener.
    Press and hold Option + Command + V to record audio (max 30 seconds).
    Release the hotkey to transcribe and paste the text.
"""

import os
import time
import signal
import tempfile
import subprocess
import threading
import logging
import sys
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import required dependencies
try:
    import pyperclip
    from pynput import keyboard
    from dotenv import load_dotenv
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError as e:
    logger.error(f"Failed to import required dependency: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install openai pyperclip pynput python-dotenv sounddevice soundfile numpy setuptools")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI API with version detection
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
    print("Create a .env file with your OpenAI API key: OPENAI_API_KEY=your_key_here")
    exit(1)

# Initialize OpenAI with version detection
try:
    try:
        # Try to use pkg_resources if available
        import pkg_resources
        import openai
        openai_version = pkg_resources.get_distribution("openai").version
        logger.info(f"Detected OpenAI package version: {openai_version}")
    except ImportError:
        # Fall back to directly checking the module
        import openai
        try:
            openai_version = openai.__version__
            logger.info(f"Detected OpenAI version from attribute: {openai_version}")
        except AttributeError:
            openai_version = "unknown"
            logger.warning("Could not determine OpenAI version, assuming legacy")
    
    # Handle different OpenAI API versions
    if str(openai_version).startswith("0."):
        # Legacy OpenAI API (v0.x)
        logger.info("Using OpenAI legacy API (v0.x)")
        openai.api_key = openai_api_key
        OPENAI_LEGACY = True
    else:
        # Modern OpenAI API (v1.x+)
        logger.info("Using OpenAI modern API (v1.x+)")
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        OPENAI_LEGACY = False
except Exception as e:
    # Fall back to legacy as default if detection fails
    logger.warning(f"Error detecting OpenAI version: {e}")
    logger.info("Falling back to legacy OpenAI API")
    import openai
    openai.api_key = openai_api_key
    OPENAI_LEGACY = True

# Constants
SAMPLE_RATE = 16000  # Hz
MAX_RECORDING_SECONDS = 30
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"

# Global variables
recording = False
audio_data = []
recording_thread = None
cache_dir = Path("~/Library/Caches/voice-paste").expanduser()


def ensure_cache_dir():
    """Ensure cache directory exists."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def record_audio():
    """Record audio from microphone while hotkey is held."""
    global recording, audio_data
    
    audio_data = []  # Reset audio data
    
    def audio_callback(indata, frames, time_info, status):
        """Callback for audio stream to collect recorded data."""
        if status:
            logger.warning(f"Audio status: {status}")
        if recording:
            audio_data.append(indata.copy())
    
    # Set up the audio stream
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            start_time = time.time()
            logger.info("Recording started...")
            
            while recording and (time.time() - start_time) < MAX_RECORDING_SECONDS:
                time.sleep(0.1)  # Sleep to reduce CPU usage
                
            elapsed = time.time() - start_time
            if elapsed >= MAX_RECORDING_SECONDS:
                logger.info(f"Maximum recording time reached ({MAX_RECORDING_SECONDS}s)")
            
            logger.info(f"Recording stopped after {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Error in audio recording: {str(e)}")
    
    if audio_data:
        logger.info(f"Recorded {len(audio_data)} audio chunks, processing...")
        process_audio()
    else:
        logger.warning("No audio data captured during recording")


def process_audio():
    """Save recorded audio to file and send to OpenAI for transcription."""
    if not audio_data:
        logger.warning("No audio recorded")
        return
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = ensure_cache_dir()
    temp_file = temp_dir / f"voice_recording_{timestamp}.wav"
    
    try:
        # Concatenate audio chunks and save to file
        audio_concat = np.concatenate(audio_data, axis=0)
        sf.write(temp_file, audio_concat, SAMPLE_RATE)
        logger.info(f"Audio saved to: {temp_file}")
        
        # Transcribe audio using OpenAI API
        transcribe_audio(temp_file)
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
    finally:
        # Cleanup even if transcription fails
        if temp_file.exists():
            try:
                os.remove(temp_file)
                logger.info(f"Temporary audio file deleted: {temp_file}")
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")


def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI's API and paste the result."""
    try:
        logger.info("Sending audio to OpenAI for transcription...")
        transcribed_text = None
        
        # Open the audio file
        with open(audio_file, "rb") as file:
            if OPENAI_LEGACY:
                # Handle legacy OpenAI API (v0.x)
                try:
                    response = openai.Audio.transcribe(
                        model=TRANSCRIPTION_MODEL,
                        file=file
                    )
                    transcribed_text = response.get("text", "")
                except Exception as e:
                    logger.error(f"OpenAI legacy API error: {str(e)}")
            else:
                # Handle modern OpenAI API (v1.x+)
                try:
                    response = client.audio.transcriptions.create(
                        model=TRANSCRIPTION_MODEL,
                        file=file
                    )
                    transcribed_text = response.text
                except Exception as e:
                    logger.error(f"OpenAI modern API error: {str(e)}")
        
        if transcribed_text:
            logger.info(f"Transcription received: {transcribed_text[:50]}...")
            
            # Copy to clipboard
            pyperclip.copy(transcribed_text)
            
            # Programmatically paste with Cmd+V
            paste_text()
        else:
            logger.warning("Received empty transcription from OpenAI")
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")


def paste_text():
    """Programmatically trigger Cmd+V to paste the clipboard contents."""
    try:
        # For macOS, we can use AppleScript to simulate Cmd+V
        cmd = ['osascript', '-e', 'tell application "System Events" to keystroke "v" using command down']
        subprocess.run(cmd, check=True)
        logger.info("Text pasted successfully")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pasting text: {e}")


def is_v_key(key):
    """Check if a key is the 'v' key, including different representations."""
    try:
        # Check for direct character match (v or V)
        if hasattr(key, 'char') and key.char and key.char.lower() == 'v':
            return True
        
        # Check for special symbols that might be v on some keyboard layouts (√, ◊, etc.)
        if hasattr(key, 'char') and key.char in ['√', '◊', 'v', 'V']:
            return True
        
        # Check for vk code on some systems
        if hasattr(key, 'vk') and key.vk == 86:  # 86 is the virtual key code for 'v'
            return True
            
        # Check KeyCode directly
        if key == keyboard.KeyCode.from_char('v') or key == keyboard.KeyCode.from_char('V'):
            return True
    except:
        pass
    
    return False


def on_key_press(key):
    """Handle key press event."""
    global recording, recording_thread
    
    # Check if Option + Command + V is pressed (macOS)
    try:
        cmd_keys = [keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r]
        alt_keys = [keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r]
        
        if (key in cmd_keys and any(k in alt_keys for k in currently_pressed_keys) and 
            any(is_v_key(k) for k in currently_pressed_keys)):
            start_recording()
        elif (key in alt_keys and any(k in cmd_keys for k in currently_pressed_keys) and 
              any(is_v_key(k) for k in currently_pressed_keys)):
            start_recording()
        elif (is_v_key(key) and any(k in cmd_keys for k in currently_pressed_keys) and 
              any(k in alt_keys for k in currently_pressed_keys)):
            start_recording()
    except:
        pass
    
    # Add key to currently pressed keys
    currently_pressed_keys.add(key)


def on_key_release(key):
    """Handle key release event."""
    global recording
    
    # Remove key from currently pressed keys
    try:
        currently_pressed_keys.remove(key)
    except KeyError:
        pass
    
    # Check if any part of the hotkey is released while recording
    try:
        cmd_keys = [keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r]
        alt_keys = [keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r]
        
        if recording and (key in cmd_keys or key in alt_keys or is_v_key(key)):
            recording = False
    except:
        pass


def start_recording():
    """Start recording if not already recording."""
    global recording, recording_thread
    
    if not recording:
        recording = True
        logger.info("Hotkey combination detected, starting recording")
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.daemon = True
        recording_thread.start()


def request_permissions():
    """Request necessary permissions on first run."""
    # Request microphone permission by attempting a quick audio capture
    try:
        logger.info("Requesting microphone permission...")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=1024):
            time.sleep(0.1)
        logger.info("Microphone permission granted")
    except Exception as e:
        logger.error(f"Error accessing microphone: {str(e)}")
        print("\n⚠️  Please grant microphone permission in System Settings\n")
    
    # For Accessibility permissions, we can only inform the user
    print("\n⚠️  Important: This tool needs Accessibility permissions to work properly.")
    print("   Please go to System Settings → Privacy & Security → Accessibility")
    print("   and add Python or Terminal to the allowed applications.\n")


def handle_exit(signum, frame):
    """Clean up on exit."""
    logger.info("Exiting voice-paste...")
    keyboard_listener.stop()
    sys.exit(0)


if __name__ == "__main__":
    print("🎙️ Voice-Paste starting...")
    print("Hold Option (⌥) + Command (⌘) + V to start recording (max 15s)")
    print("Release to transcribe and paste")
    print("Press Ctrl+C to quit")
    
    # Set up global key tracking
    currently_pressed_keys = set()
    
    # Request permissions on first run
    request_permissions()
    
    # Set up signal handlers for clean exit
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # Set up key listeners for press and release
    logger.info("Initializing keyboard listener")
    keyboard_listener = keyboard.Listener(
        on_press=on_key_press,
        on_release=on_key_release
    )
    keyboard_listener.start()
    logger.info("Keyboard listener started - waiting for hotkey ⌥ ⌘ V")
    
    try:
        # Keep the main thread alive
        keyboard_listener.join()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        keyboard_listener.stop()
