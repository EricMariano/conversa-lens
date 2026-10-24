"""Gera duas ligações sintéticas (macOS ``say``) com o MESMO texto e vozes diferentes.

Só a voz do cliente muda: na versão irritada ele fala mais alto, mais rápido, com o tom
mais agudo e atropela o atendente. O texto é ambíguo de propósito ("Ótimo. Muito
obrigado, viu."): pelo texto sozinho não dá para saber se é sincero ou irônico.

É uma demonstração do pipeline, não evidência: voz sintética não tem a prosódia de uma
pessoa irritada de verdade, e são duas ligações. Gravação em estéreo, atendente no canal
esquerdo e cliente no direito, 16 kHz.

    python samples/audio/gerar_ligacoes.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16_000
OUT = Path(__file__).resolve().parent
AGENT_VOICE = "Luciana"
CUSTOMER_VOICE = "Eddy (Português (Brasil))"

SCRIPT = [
    ("atendente", "Central de atendimento, bom dia. Com quem eu falo?"),
    ("cliente", "Bom dia. É sobre o meu pedido, que ainda não chegou."),
    ("atendente", "Verifiquei aqui. A entrega foi reagendada para a próxima semana."),
    ("cliente", "Ótimo. Muito obrigado, viu."),
    ("atendente", "Posso ajudar em algo mais?"),
    ("cliente", "Não. Era só isso mesmo."),
]

# (palavras/min, ganho em dB, semitons, segundos de pausa antes; negativo = atropela)
CALM = {"rate": 170, "gain": 0.0, "pitch": 0.0, "gap": 0.9}
ANGRY = {"rate": 235, "gain": 9.0, "pitch": 3.0, "gap": -0.8}


def speak(text: str, voice: str, rate: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fala.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(path), text], check=True)
        audio, sr = sf.read(path, dtype="float32")
    return librosa.resample(audio, orig_sr=sr, target_sr=SR)


def build(customer_style: dict[str, float], angry_from: int) -> np.ndarray:
    """Monta a ligação fala a fala; ``gap`` negativo faz o cliente atropelar o atendente."""
    placed: list[tuple[str, float, np.ndarray]] = []
    cursor = 0.5
    for i, (speaker, text) in enumerate(SCRIPT):
        is_customer = speaker == "cliente"
        style = customer_style if is_customer and i >= angry_from else CALM
        audio = speak(
            text,
            CUSTOMER_VOICE if is_customer else AGENT_VOICE,
            int(style["rate"]) if is_customer else 175,
        )
        if is_customer and style["pitch"]:
            audio = librosa.effects.pitch_shift(audio, sr=SR, n_steps=style["pitch"])
        audio = audio * 10 ** ((style["gain"] if is_customer else 0) / 20) * 0.25
        start = max(0.0, cursor + (style["gap"] if is_customer else 0.7))
        placed.append((speaker, start, audio))
        cursor = start + len(audio) / SR

    length = int((cursor + 1) * SR)
    tracks = {"atendente": np.zeros(length, "float32"), "cliente": np.zeros(length, "float32")}
    for speaker, start, audio in placed:
        at = int(start * SR)
        tracks[speaker][at : at + len(audio)] += audio
    return np.clip(np.stack([tracks["atendente"], tracks["cliente"]], axis=1), -1, 1)


def main() -> None:
    for name, style in (("ligacao_calma", CALM), ("ligacao_irritada", ANGRY)):
        path = OUT / f"{name}.flac"
        sf.write(path, build(style, angry_from=3), SR)
        print(f"{path.name}: {sf.info(path).duration:.1f}s")


if __name__ == "__main__":
    main()
