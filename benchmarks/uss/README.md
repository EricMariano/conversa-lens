# Satisfação (CSAT) no JDDC: original em chinês e traduzido para português

Pergunta: dá para estimar a nota de satisfação de uma conversa de atendimento sem treinar
nada, só perguntando ao Jev? E isso se mantém em português?

## Dados

[USS](https://github.com/sunnweiwei/user-satisfaction-simulation) (Sun et al., SIGIR 2021),
subconjunto **JDDC**: atendimento humano real do JD.com, em chinês, com nota de 1 a 5 dada
por anotadores para a conversa inteira (usamos a média dos anotadores, por isso há notas
como 2,33). Não existe conjunto público em PT-BR com conversa e CSAT real; este é o proxy
mais próximo que encontramos.

Amostras estratificadas pela nota arredondada (a distribuição real do JDDC é quase toda 4, e
só existem 2 conversas com nota 1). **Insatisfeita** = nota real abaixo de 2,5.

```bash
python benchmarks/uss/prepare.py JDDC --por-nota 8 --saida benchmarks/data/jddc_piloto.jsonl
python benchmarks/uss/translate.py benchmarks/data/jddc_piloto.jsonl --para pt   # e --para en
python -m conversa_lens lote benchmarks/data/jddc_piloto_pt.jsonl --saida benchmarks/results/jddc_piloto_pt
python -m conversa_lens csat avaliar benchmarks/results/jddc_piloto_pt.jsonl
```

## 1. Piloto: 34 conversas, três línguas

Traduções com `gpt-5-nano`, cada conversa inteira de uma vez, com instrução para manter o tom
e não suavizar ([`translate.py`](translate.py)). O inglês entra como controle: se o inglês
fica bem e o português cai, o problema é o português, não a tradução.

**A mesma conversa em línguas diferentes** (estimativa de satisfação do Jev):

| par | Spearman | diferença média | maior diferença |
|---|---:|---:|---:|
| chinês × português | 0,96 | 0,11 | 0,36 |
| chinês × inglês | 0,95 | 0,17 | 0,52 |
| português × inglês | 0,94 | 0,14 | 0,71 |

**Contra a nota real** (estimativa do Jev; 10 conversas insatisfeitas):

| língua | MAE | Spearman | insatisfeitas achadas | alarmes falsos |
|---|---:|---:|---:|---:|
| chinês (original) | 0,69 | 0,80 | 6 de 10 | 2 |
| português | 0,73 | 0,75 | 6 de 10 | 2 |
| inglês | 0,74 | 0,77 | 6 de 10 | 2 |

A tradução preserva o sinal, e o português não fica atrás do inglês. Sobraram caracteres
chineses em 3% das mensagens em português (1,6% em inglês): quase tudo marcadores do dataset
(`[时间x]`, `[电话x]`), nomes de produto e endereços.

## 2. Em escala: 207 conversas em português

Amostra de 302 conversas (75 por nota, seed 1); a tradução parou em 207 quando os créditos
da API de tradução acabaram. As 207 continuam equilibradas: 2 / 50 / 52 / 54 / 49 conversas
com nota 1 / 2 / 3 / 4 / 5. 4.800 requisições ao Jev, nenhuma falha.

| método | MAE | Spearman | insatisfeitas achadas (de 52) | alarmes falsos |
|---|---:|---:|---:|---:|
| sempre a média | 0,94 | – | 0 | 0 |
| sentimento final do cliente (Jev por turno) | 0,71 | 0,62 | 32 | 13 |
| **estimativa do Jev** (1 `Score` na conversa) | 0,71 | **0,74** | **38** | 17 |
| modelo calibrado (ridge, CV 5 dobras) | **0,60** | 0,74 | 23 | 2 |

- Spearman da estimativa do Jev: IC95% [0,66; 0,80] (bootstrap).
- Estimativa média por nota real: 2,16 (nota 2), 3,11 (nota 3), 3,45 (nota 4), 3,99 (nota 5).
  Ordena bem, mas comprime a escala: quase nunca passa de 4,5.
- O modelo calibrado erra menos no valor (MAE 0,60) e quase não dá alarme falso, mas acha
  menos insatisfeitas. Na ordenação, empata com a estimativa pura (diferença de Spearman
  0,00, IC95% [−0,04; +0,04]). Os sinais que mais pesam: a estimativa do Jev e o sentimento
  médio do cliente ao longo da conversa.

## Leitura

- **Sem treino, a estimativa do Jev ordena bem as conversas em português** (Spearman 0,74) e
  acha 73% das insatisfeitas, com cerca de 1 alarme falso para cada 2 acertos.
- **Calibrar com notas reais compensa no valor, não na ordem.** Com 200 conversas, o modelo
  calibrado acerta melhor o número, mas fica conservador. Use a estimativa pura para triagem
  (achar insatisfeitos) e o calibrado quando precisar do número.
- Limites: conversas traduzidas do chinês, não PT-BR nativo; nota de anotador, não do cliente;
  amostra estratificada (na vida real há bem menos insatisfeitos, e a precisão cai).

## Correção registrada

A primeira versão desta página contava como insatisfeitas só as conversas com nota ≤ 2,0. A
nota é média de anotadores, e as conversas de 2,33 ficavam de fora, embora a amostragem as
contasse como nota 2. A definição passou a ser nota < 2,5, e o piloto foi recalculado: a
estimativa do Jev acha 6 de 10 insatisfeitas, não as 8 de 10 relatadas antes.

O piloto em chinês rodou antes de as perguntas por turno ganharem a frase sobre sinais de
`voz`; a pergunta de satisfação não mudou.

## Próximo passo

Terminar a tradução das 95 conversas restantes (`translate.py` retoma de onde parou) e repetir
com conversas reais em PT-BR com a nota da pesquisa, quando houver.
