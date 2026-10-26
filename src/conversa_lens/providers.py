"""Escolhe classificador, resumidor e nota de satisfação a partir do ambiente."""

from __future__ import annotations

import os
from pathlib import Path

from conversa_lens.cache import ResponseCache
from conversa_lens.classify import JevTurnClassifier, OfflineHeuristicClassifier, TurnClassifier
from conversa_lens.csat import CsatEstimator, CsatModel, JevCsatEstimator
from conversa_lens.summarize import LLMSummarizer

DEFAULT_CSAT_MODEL = Path("csat_model.json")
DEFAULT_CACHE = Path.home() / ".cache" / "conversa_lens" / "jev.sqlite"
_cache: ResponseCache | None = None


def response_cache() -> ResponseCache | None:
    """Cache das respostas do Jev (só hash e probabilidades, sem texto).

    ``CONVERSA_CACHE=0`` desliga; ``CONVERSA_CACHE=/caminho.sqlite`` muda o arquivo.
    """
    global _cache
    setting = os.environ.get("CONVERSA_CACHE", "")
    if setting == "0":
        return None
    if _cache is None:
        _cache = ResponseCache(Path(setting) if setting else DEFAULT_CACHE)
    return _cache


def has_jev() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def make_classifier(offline: bool = False) -> TurnClassifier:
    if offline or not has_jev():
        return OfflineHeuristicClassifier()
    return JevTurnClassifier(cache=response_cache())


def make_summarizer(offline: bool = False) -> LLMSummarizer | None:
    """``None`` quando não há modelo configurado: o grafo usa o resumo por template."""
    model = os.environ.get("CONVERSA_CHAT_MODEL")
    if offline or not model:
        return None
    from langchain.chat_models import init_chat_model

    return LLMSummarizer(init_chat_model(model))


def make_csat_estimator(offline: bool = False) -> CsatEstimator | None:
    return None if offline or not has_jev() else JevCsatEstimator(cache=response_cache())


def load_csat_model(path: Path | None = None) -> CsatModel | None:
    """Modelo calibrado, se existir (``CONVERSA_CSAT_MODEL`` ou ``csat_model.json``)."""
    path = path or Path(os.environ.get("CONVERSA_CSAT_MODEL", DEFAULT_CSAT_MODEL))
    return CsatModel.load(path) if path.exists() else None
