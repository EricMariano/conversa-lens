"""Traduz conversas do formato ``.jsonl`` do lote, conversa inteira de uma vez.

A tradução é a maior ameaça a uma avaliação de satisfação: um modelo tende a suavizar
reclamações e a perder ironia, e a nota foi dada sobre o original. Por isso:

- a conversa vai inteira (contexto preservado), e a saída precisa ter o mesmo número de
  mensagens, na mesma ordem e com os mesmos papéis;
- a instrução pede para manter o tom e a força de cada fala, sem suavizar nem acrescentar;
- marcadores do dataset viram marcadores neutros (``[NUMERO]``, ``[PEDIDO]``, ``[EMOJI]``).

Use junto com o original e com outra língua como controle (ver README).

    python benchmarks/uss/translate.py benchmarks/data/jddc_piloto.jsonl --para pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from conversa_lens.providers import chat_model

LANGUAGES = {
    "pt": "português do Brasil, como num chat de atendimento brasileiro",
    "en": "English, as in a US customer-support chat",
}

PROMPT = """Traduza esta conversa de atendimento ao cliente (chat de e-commerce) do chinês \
para {lang}.

Regras:
- Mantenha o tom e a força de cada fala: irritação, impaciência, ironia, frieza, gratidão. \
Não suavize reclamações, não deixe nada mais educado nem mais rude do que é.
- Não acrescente, não omita, não junte nem divida mensagens: devolva exatamente {n} \
mensagens, na mesma ordem.
- Fórmulas de tratamento do atendente (como 亲) viram o equivalente natural na língua de \
destino, no mesmo nível de cordialidade.
- Marcadores: [ORDERID_...] → [PEDIDO]; [数字x] e números mascarados → [NUMERO]; [姓名x] → \
[NOME]; códigos de emoji como #E-s[数字x] → [EMOJI]; outros [...] ficam como estão.
- Mensagens muito curtas ("嗯", "好的") viram o equivalente curto e natural.

Conversa (JSON, uma mensagem por item):
{conversation}"""


class Translation(BaseModel):
    messages: list[str]
    """O texto traduzido de cada mensagem, na mesma ordem."""


_MARKERS = [
    (re.compile(r"\[ORDERID_\d+\]"), "[PEDIDO]"),
    (re.compile(r"#E-s\[数字x\]"), "[EMOJI]"),
    (re.compile(r"\[数字x\]"), "[NUMERO]"),
    (re.compile(r"\[姓名x\]"), "[NOME]"),
]


def premask(text: str) -> str:
    for pattern, marker in _MARKERS:
        text = pattern.sub(marker, text)
    return text


def translate(model, row: dict, lang: str, attempts: int = 3) -> dict:  # type: ignore[no-untyped-def]
    msgs = row["messages"]
    payload = json.dumps(
        [{"papel": m["role"], "texto": premask(m["content"])} for m in msgs], ensure_ascii=False
    )
    prompt = PROMPT.format(lang=LANGUAGES[lang], n=len(msgs), conversation=payload)
    for _ in range(attempts):
        out: Translation = model.invoke(prompt)
        if len(out.messages) == len(msgs):
            return {
                **row,
                "messages": [
                    {"role": m["role"], "content": t.strip()}
                    for m, t in zip(msgs, out.messages, strict=True)
                ],
                "lingua": lang,
            }
    raise ValueError(f"{row['id']}: tradução com número de mensagens diferente do original")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("entrada", type=Path)
    parser.add_argument("--para", choices=list(LANGUAGES), required=True)
    parser.add_argument(
        "--modelo", default=os.environ.get("CONVERSA_CHAT_MODEL", "openai:gpt-5-nano")
    )
    parser.add_argument("--paralelo", type=int, default=8)
    args = parser.parse_args()

    out_path = args.entrada.with_name(f"{args.entrada.stem}_{args.para}.jsonl")
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines()}
    rows = [json.loads(line) for line in args.entrada.read_text().splitlines()]
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} já traduzidas, {len(todo)} a traduzir com {args.modelo}")

    model = chat_model(args.modelo).with_structured_output(Translation)
    failed = 0
    with ThreadPoolExecutor(args.paralelo) as pool, out_path.open("a") as f:
        futures = [pool.submit(translate, model, r, args.para) for r in todo]
        for n, future in enumerate(as_completed(futures), 1):  # grava assim que termina
            try:
                f.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                f.flush()
            except Exception as error:  # uma conversa ruim não para as outras
                failed += 1
                print(f"  falhou: {error}")
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print(f"→ {out_path} ({failed} falhas)")


if __name__ == "__main__":
    main()
