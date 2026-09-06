# Transcripts de conversa

Cinco conversas reais com o `EmporioMusicaAgent` completo (Gemini real +
RAG sobre `data/políticas.pdf` + dados de `data/*.csv`), geradas rodando o
agente de verdade — nenhuma resposta foi escrita manualmente.

| Arquivo                                                   | Cenário                                                                    |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [01-catalogo-por-preco.md](01-catalogo-por-preco.md)         | Consulta ao catálogo com filtro de preço (tool `search_products`).           |
| [02-informacoes-da-loja.md](02-informacoes-da-loja.md)       | Informação geral da loja — endereço e telefone (RAG sobre o PDF).            |
| [03-consulta-de-preco.md](03-consulta-de-preco.md)           | Preço de um produto específico (tool `search_products`).                     |
| [04-politica-de-devolucao.md](04-politica-de-devolucao.md)   | **Cenário não trivial:** aplica a regra de prazo da política de devolução (RAG) sobre a data real de um pedido (tool `check_order_status`) — dois turnos, o agente calcula que 5 meses excede o prazo de 7 dias. |
| [05-memoria-multi-turno.md](05-memoria-multi-turno.md)       | Continuidade de contexto entre turnos: a segunda pergunta não repete o número do pedido, o agente usa o histórico da sessão para saber a qual pedido se refere. |

## Limitação resolvida na revisão 0.2

A pergunta sugerida no desafio usa "violões" (plural). Testei exatamente
essa frase antes de gerar o transcript 01 e o agente respondeu "não
consegui encontrar violões em nosso catálogo" — **um falso negativo real**.
Causa raiz (`tabular_data/pandas_service.py::search_products`): a busca é
por substring simples, e `"violões"` não aparece como substring em nenhum
texto de `name`/`description` do catálogo (que usa `"violão"`, singular).
Confirmei isolando o método:

```pycon
>>> service.search_products("violões")   # plural
0 resultados
>>> service.search_products("violão")    # singular
35 resultados
```

Troquei a pergunta do transcript 01 para o singular (que é como o dado
real está escrito) para que ele demonstrasse a capacidade pretendida —
Essa limitação foi eliminada: a busca normaliza acentos e plural leve, portanto
`"violões"` e `"violoes"` encontram o catálogo de `"violão"`. Os transcripts
permanecem como registro da versão que os gerou.
