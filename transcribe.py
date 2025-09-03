import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# --- Settings ---
# THE ONLY CHANGE IS HERE: Switched to the multilingual "base" model
MODEL_TYPE = "base"
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
TEMP_AUDIO_FILE = "temp_audio.wav"

# --- Load the Whisper model ---
# This is done only once to save time
print("Loading Whisper model...")
model = whisper.load_model(MODEL_TYPE)
print("Model loaded.")

try:
    print("Starting real-time transcription. Press Ctrl+C to stop.")
    
    # This loop will run forever until you stop it
    while True:
        # 1. Record audio from the microphone
        print(f"\nListening for {RECORD_SECONDS} seconds...")
        audio_data = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
        sd.wait()  # Wait for the recording to complete

        # 2. Save the recorded audio to a temporary file
        write(TEMP_AUDIO_FILE, SAMPLE_RATE, audio_data)
        
        # 3. Use Whisper to transcribe the audio file
        print("Transcribing (auto-detecting language)...")
        result = model.transcribe(TEMP_AUDIO_FILE, fp16=False)
        transcribed_text = result['text']
        
        # 4. Print the transcribed text and the detected language
        if transcribed_text.strip():
            detected_language = result['language']
            print(f"--> Detected Language: {detected_language.upper()}")
            print(f"--> Transcription: {transcribed_text}")
        else:
            print("--> No speech detected.")

except KeyboardInterrupt:
    print("\nStopping transcription.")