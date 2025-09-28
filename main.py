import threading
import audio_utils
import eyetrack
import wavanalyse
import transcribe
from openai import OpenAI

def main():
    stop_event = threading.Event()
    eye_time = {}

    print("Press Enter to start recording and eye tracking...")
    input()
    print("Both started. Press Enter again to stop.")

    def run_audio():
        audio_utils.record(stop_event)

    def run_eyes():
        eye_time["value"] = eyetrack.track_eyes(stop_event)

    t1 = threading.Thread(target=run_audio)
    t2 = threading.Thread(target=run_eyes)
    t1.start()
    t2.start()

    input()  # second Enter stops both
    stop_event.set()

    t1.join()
    t2.join()

    pauses, duration = wavanalyse.analyze_wav()
    eyecontactratio = None
    if duration > 0:
        eyecontactratio = eye_time.get("value", 0) / duration

    client = OpenAI()
    with open("output.wav", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

    feedback = transcribe.generate_feedback(transcript, pauses, eyecontactratio)
    print("\n--- FEEDBACK ---\n")
    print(feedback)

if __name__ == "__main__":
    main()
