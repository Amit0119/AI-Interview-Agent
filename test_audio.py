import whisper
import tempfile
import os

def transcribe_test():
    try:
        model = whisper.load_model("base")
        print("Model loaded.")
        
        # Create a small dummy WAV file
        import wave, struct
        dummy_wav = b""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                for _ in range(16000):
                    wav.writeframes(struct.pack('h', 0))
            temp_path = f.name
        
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
            
        print("Audio bytes length:", len(audio_bytes))
        
        # AudioProcessor logic
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f2:
            f2.write(audio_bytes)
            temp2_path = f2.name
            
        print("Loading audio with whisper...")
        audio_np = whisper.load_audio(temp2_path)
        print("Audio loaded, length:", len(audio_np))
        result = model.transcribe(audio_np, language="en", fp16=False)
        print("Result:", result.get("text", ""))
        
        os.remove(temp_path)
        os.remove(temp2_path)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    transcribe_test()
