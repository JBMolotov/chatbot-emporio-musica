# Revisão de produção

Os itens pendentes dos roadmaps foram implementados e cobertos por testes:

- tools para estoque, promoções e pedidos, com join pedido–itens–produtos;
- busca tolerante a acentos/plural, filtros de categoria/preço e retorno JSON;
- identificação opcional por contato na CLI, API `POST /chat` e bootstrap único;
- janela de histórico, validação de mensagens/sessões e gravação atômica;
- RAG com chunks sobrepostos, cosseno, reranking lexical, limiar, cache,
  reindexação idempotente por hash e persistência sem desserialização insegura;
- logs de latência, tokens e resultado das ferramentas sem registrar conteúdo
  ou argumentos potencialmente pessoais.

Antes de produção distribuída, migre o store local e histórico JSON para
serviços transacionais, use autenticação de usuário (a `API_KEY` atual protege
apenas o endpoint), defina retenção/LGPD, limite de taxa no gateway e rode uma
avaliação de recuperação e respostas em staging com dados representativos.
