"""
asr.py

Standalone script with automatic (streaming) speech recognition and text-to-speech (TTS) functionality. Speech
recognition is handled offline via Vosk, while TTS requires internet access (to ping Google's TTS API).

Note :: ASR requires `vosk and sounddevice` installed!
Note :: TTS requires `pydub and simpleaudio` installed!
"""
import time
from io import BytesIO

import vosk
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from sounddevice import RawInputStream, query_devices
from vosk import KaldiRecognizer
from vosk import Model as VoskModel


# Suppress Log Level
vosk.SetLogLevel(-1)

# Constants
MODE = "ECHO"


def asr() -> None:
    if MODE == "ASR":
        print("[*] Dropping into Microphone-based ASR...")

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

        # Load sample rate directly from Microphone (assume Microphone is Device ID = 0)
        samplerate = int(query_devices(0, "input")["default_samplerate"])

        # By default loads the "smallest" model with the language code we specify...
        model = VoskModel(lang="en-us")
        recognizer = KaldiRecognizer(model, samplerate)

        # Modular "listen" function (asyncio)
        def listen() -> str:
            with RawInputStream(samplerate=samplerate, blocksize=4096, device=0, dtype="int16", channels=1) as s:
                while True:
                    buffer, _ = s.read(4096)
                    if recognizer.AcceptWaveform(bytes(data)):
                        return recognizer.Result()["text"]

        # Modular "speak" function
        def speak(language: str) -> None:
            with BytesIO() as f:
                gTTS(text=language, lang="en", tld="com.au").write_to_fp(f)
                f.seek(0)

                # Use PyDub to Play Audio...
                audio = AudioSegment.from_file(f, format="mp3")
                play(audio)

        # Ideally we want a "main task running" that has the following structure
        counter = 0

        print("[*] Entering Main Control Loop...")
        while True:
            # Check for a spoken input (but don't block if there's not an input prepared...)
            # captured = listen() ?

            # If `captured` is non-empty, pass to "speak" to "echo via TTS" (in realistic setting, this blocks to handle
            #   different control flow -- e.g., calling different synchronous functions).
            # speak(captured)

            # Otherwise, continue with main compute loop...
            counter += 1
            time.sleep(1)

            # Exit...
            if counter >= 300:
                break


if __name__ == "__main__":
    asr()
