"""Preload the Naruto MP3 and play it without blocking video frames."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


SOUND_PATH = Path(__file__).resolve().parent / "assets" / "vanish.MP3"
TELEPORT_SOUND_PATH = SOUND_PATH.with_name("teleport.MP3")


class VanishSound:
    def __init__(self, path: Path = SOUND_PATH):
        if not path.is_file():
            raise RuntimeError(f"Som não encontrado: {path}")
        self.path = path

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        self.process = None
        self.ffplay = None
        if platform.system() == "Darwin":
            # pygame and opencv-python each bundle SDL2: loading both on macOS can crash.
            self.ffplay = shutil.which("afplay") or shutil.which("ffplay")
            if self.ffplay is None:
                raise RuntimeError("Para tocar o som é preciso o afplay ou o ffplay.")
            return
        try:
            from pygame import mixer
        except ImportError:
            self.ffplay = shutil.which("ffplay")
            if self.ffplay is None:
                raise RuntimeError("Para tocar o som instale o pygame ou o ffplay.")
            return

        self.mixer = mixer
        self.owns_mixer = mixer.get_init() is None
        try:
            if self.owns_mixer:
                mixer.init(frequency=44100, size=-16, channels=2, buffer=256)
            self.sound = mixer.Sound(str(path))  # Decode once, before opening the webcam.
        except Exception as error:
            if self.owns_mixer:
                mixer.quit()
            raise RuntimeError(f"Não foi possível abrir o som: {error}") from error

    def play(self) -> None:
        if self.ffplay is not None:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            try:
                command = [self.ffplay, str(self.path)]
                if Path(self.ffplay).name != "afplay":
                    command = [self.ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet",
                               "-i", str(self.path)]
                self.process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as error:
                raise RuntimeError(f"Não foi possível tocar o som: {error}") from error
            return
        self.sound.stop()
        if self.sound.play() is None:
            raise RuntimeError("Não foi possível tocar o som: nenhum canal de áudio livre.")

    def close(self) -> None:
        if self.ffplay is not None:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            return
        if self.owns_mixer:
            self.mixer.quit()
            self.owns_mixer = False
