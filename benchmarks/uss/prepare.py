"""Converte um subconjunto do USS para o formato ``.jsonl`` do modo lote.

USS (Sun et al., SIGIR 2021, https://github.com/sunnweiwei/user-satisfaction-simulation)
traz conversas com nota de satisfação de 1 a 5 dada por anotadores, por turno e para a
conversa inteira (linha ``OVERALL``). Usamos a nota da conversa (média dos anotadores).

- ``JDDC``: atendimento humano real do JD.com, em chinês. O mais próximo do nosso caso.
- ``MWOZ``, ``SGD``, ``ReDial``, ``CCPE``: inglês, assistentes de tarefas; quase tudo nota 3.

Atenção: a nota é de anotadores lendo a conversa, não do cliente na pesquisa. O
repositório não declara licença; os dados são baixados aqui e não são redistribuídos.

    python benchmarks/uss/prepare.py JDDC --por-nota 8 --saida benchmarks/data/jddc_piloto.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path

import certifi

URL = "https://raw.githubusercontent.com/sunnweiwei/user-satisfaction-simulation/master/dataset/{}.txt"
DATA = Path(__file__).resolve().parents[1] / "data"


def download(name: str) -> Path:
    path = DATA / f"uss_{name}.txt"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL.format(name), context=context) as response:  # noqa: S310
            path.write_bytes(response.read())
    return path


def sessions(path: Path) -> list[dict[str, object]]:
    out = []
    for n, block in enumerate(path.read_text().strip().split("\n\n")):
        messages, rating = [], None
        for line in block.strip().split("\n"):
            cols = line.split("\t")
            if cols[0] == "USER" and cols[1] == "OVERALL":
                votes = [int(v) for v in cols[3].split(",") if v.strip()]
                rating = statistics.mean(votes) if votes else None
            elif len(cols) > 1 and cols[1].strip():
                role = "customer" if cols[0] == "USER" else "agent"
                messages.append({"role": role, "content": cols[1]})
        if rating is not None and messages:
            out.append({"id": f"{path.stem}-{n}", "messages": messages, "nota": round(rating, 2)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["JDDC", "MWOZ", "SGD", "ReDial", "CCPE"])
    parser.add_argument(
        "--por-nota", type=int, default=8, help="conversas por nota arredondada (1–5)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--saida", type=Path, required=True)
    args = parser.parse_args()

    by_rating: dict[int, list[dict[str, object]]] = defaultdict(list)
    for s in sessions(download(args.dataset)):
        by_rating[round(float(s["nota"]))].append(s)  # type: ignore[arg-type]
    rng = random.Random(args.seed)
    picked = []
    for rating in sorted(by_rating):
        pool = by_rating[rating]
        picked += rng.sample(pool, min(args.por_nota, len(pool)))
        print(f"nota {rating}: {min(args.por_nota, len(pool))} de {len(pool)}")
    rng.shuffle(picked)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w") as f:
        for s in picked:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{len(picked)} conversas → {args.saida}")


if __name__ == "__main__":
    main()
