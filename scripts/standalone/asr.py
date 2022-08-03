"""
asr.py

Standalone script with automatic (streaming) speech recognition and text-to-speech (TTS) functionality. Speech
recognition is handled offline via Vosk, while TTS requires internet access (to ping Google's TTS API).

Note :: ASR requires `vosk and sounddevice` installed!
Note :: TTS requires `pydub and simpleaudio` installed!
"""
from io import BytesIO

import vosk
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from sounddevice import RawInputStream, query_devices
from vosk import KaldiRecognizer
from vosk import Model as VoskModel


# Suppress Log Level
vosk.SetLogLevel(50)

# Constants
MODE = "ASR"


def asr() -> None:
    if MODE == "ASR":
        print("[*] Dropping into microphone ASR loop...")

        # Load sample rate directly from Microphone (assume Microphone is Device ID = 0)
        samplerate = int(query_devices(0, "input")["default_samplerate"])

        # By default loads the "smallest" model with the language code we specify...
        model = VoskModel(lang="en-us")
        recognizer = KaldiRecognizer(model, samplerate)

        # Open Stream!
        print("[*] Dropping into While Loop...")
        with RawInputStream(samplerate=samplerate, blocksize=4096, device=0, dtype="int16", channels=1) as stream:
            while True:
                data, _ = stream.read(4096)
                if recognizer.AcceptWaveform(bytes(data)):
                    result = recognizer.Result()
                    print(recognizer.Result())

                    # Exit condition
                    if "quit" in result["text"]:
                        break

    elif MODE == "TTS":
        print("[*] Dropping into interactive TTS loop...")
        while True:
            utterance = input("\n[*] Enter text to speak aloud or (q)uit =>> ")
            if utterance in {"q", "quit"}:
                break

            # Modular "speak" function
            def speak(language: str) -> None:
                with BytesIO() as f:
                    gTTS(text=language, lang="en", tld="com.au").write_to_fp(f)
                    f.seek(0)

                    # Use PyDub to Play Audio...
                    audio = AudioSegment.from_file(f, format="mp3")
                    play(audio)

            # Speak!
            speak(utterance)

    elif MODE == "ECHO":
        print("[*] Call and Response Loop...")


if __name__ == "__main__":
    asr()
