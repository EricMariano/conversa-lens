# Qual LLM escreve o resumo dos momentos-chave?

Teste pequeno (30/10/2026): 6 conversas (as 4 de exemplo e 2 do JDDC traduzido), o mesmo
prompt do app, modelos pela OpenRouter com `data_collection: deny`. Custo total: US$ 0,009.

```bash
python benchmarks/llm/compare.py qwen/qwen3.7-flash mistralai/mistral-small-3.2-24b-instruct deepseek/deepseek-v4-flash
python benchmarks/llm/compare.py deepseek/deepseek-v4-flash --sem-raciocinio
```

| modelo | tempo (mediana) | tokens de saída | US$ / 1.000 resumos | leitura humana |
|---|---:|---:|---:|---|
| **`mistralai/mistral-small-3.2-24b-instruct`** | **9,3 s** | 372 | **0,12** | português claro, atribuições certas; cita turnos de forma mais vaga |
| `deepseek/deepseek-v4-flash` | 25,8 s | 2.331 | 0,49 | o melhor texto: turnos precisos, percebe o sarcasmo tomado por elogio |
| `deepseek/deepseek-v4-flash`, raciocínio desligado | 17,4 s | 1.835 | 0,40 | ainda gera milhares de tokens em algumas conversas |
| `qwen/qwen3.7-flash` | 39,3 s | 3.256 | 0,44 | prolixo, formal demais, deixou escapar "nevertheless" |

- Os três seguiram o formato e não afirmaram causa. A checagem automática acusou dois
  falsos problemas (uma menção correta a alerta do cliente; um hífen Unicode no título),
  por isso os textos foram lidos.
- Escolha: **Mistral Small 3.2** como padrão (rápido e barato). Para relatórios em lote, sem
  pressa, o DeepSeek escreve melhor.
- No app, com streaming, o primeiro pedaço do resumo do Mistral aparece em ~4 s.
- Seis conversas, uma execução: serve para escolher o padrão, não para ranquear modelos.
