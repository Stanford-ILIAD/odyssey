"""
asr.py

Standalone script with automatic (streaming) speech recognition and text-to-speech (TTS) functionality. Speech
recognition is handled offline via Vosk, while TTS requires internet access (to ping Google's TTS API).

Note :: TTS requires `pydub and simpleaudio` installed!
"""
from io import BytesIO

from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


# Constants
MODE = "TTS"


def asr() -> None:
    if MODE == "ASR":
        raise NotImplementedError("ASR support not yet implemented!")

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


if __name__ == "__main__":
    asr()
