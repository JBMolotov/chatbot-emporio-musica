# Roadmap de implementação — Empório da Música (chatbot)

> Histórico da versão inicial. A revisão 0.2 substitui limitações antigas de
> `pickle`, `IndexFlatL2`, reindexação duplicada, DataFrames nas tools e
> histórico ilimitado; consulte [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).

Ordem sugerida para fechar o restante do esqueleto, item a item. Cada bloco
lista o(s) arquivo(s), as funções/classes a implementar e o porquê da ordem
(dependências entre módulos).

---

## 1. RAG — ✅ concluído

- `rag/vector_store.py` — `BaseVectorStore` (ABC): `create_embedding`,
  `store_embedding`, `add_documents`, `similarity_search`, `delete`, `persist`.
- `rag/in_memory.py` — `InMemoryVectorStore(BaseVectorStore)`: índice FAISS
  (`IndexFlatL2`) + persistência em `Settings.vector_store_path`
  (`index.faiss` + `documents.pkl`, via `pickle`).
- `rag/indexer.py` — `BaseIndexer` (ABC): `_read_documents` (abstrato),
  `_chunk_document`, `index_documents`, `refresh_index`.
- `rag/pdf_indexer.py` — `PdfPolicyIndexer(BaseIndexer)`: lê
  `data/políticas.pdf` página a página com `pypdf`.
- `rag/retriever.py` — `Retriever`: `retrieve` chama
  `vector_store.similarity_search`.
- `main.py` — `chat()` já instancia `InMemoryVectorStore` → `PdfPolicyIndexer`
  → `Retriever` e injeta em `AgentDependencies`.

**Pendências resolvidas** (testadas com API key real em 2026-07-02):

- `agent/llm_client.py::create_embedding` agora chama
  `self._client.models.embed_content(model=..., contents=text, config={"output_dimensionality": ...})`
  e lê `response.embeddings[0].values` — formato correto do SDK `google-genai`.
  De quebra, corrigi também `Settings.gemini_embedding_model` (referenciado
  no código mas inexistente em `config.py` — daria `AttributeError` em
  qualquer chamada) e os IDs inválidos em `EmbeddingModel`
  (`gemini-embedding-1` não existe; confirmado via `client.models.list()`
  que os modelos reais são `gemini-embedding-001`, `gemini-embedding-2` e
  `gemini-embedding-2-preview`).
- `Settings.gemini_embedding_dim` (`768`) confirmado: o modelo
  `gemini-embedding-001` tem dimensão padrão 3072, mas aceita
  `output_dimensionality` menor (768 testado e funcionando) — o valor do
  settings agora é passado de fato na chamada, então store/config ficam
  sempre consistentes.
- `InMemoryVectorStore.delete` reconstrói o índice inteiro re-gerando
  embeddings via API — funciona, mas é custoso; aceitável para o volume de
  dados do desafio. (mantido como está, sem ação necessária)

Teste real ponta a ponta: indexação de `data/políticas.pdf` (8 chunks) +
busca por "Qual a política de troca e devolução?" retornou a página 4
(seção de trocas e devoluções) como resultado mais próximo.

---

## 2. `tabular_data.service` — ✅ concluído

Sem dependência de outros módulos pendentes — pode ser feito em paralelo ao
RAG. Usa `tabular_data/schema.py` (já pronto) como dicionário de nomes de
arquivo/coluna.

- **Arquivo novo:** `tabular_data/pandas_service.py`
  - `PandasTabularDataService(BaseTabularDataService)`
    1. `__init__(self, settings)` — carrega os 6 CSVs de `Settings.data_dir`
       com `pandas.read_csv`, guardando um `DataFrame` por tabela.
    2. `search_products(query, **filters)` — filtro por nome/descrição
       (case-insensitive) + filtros opcionais (categoria, faixa de preço).
    3. `get_product_by_id(product_id)` — lookup direto por índice.
    4. `check_stock(product_id)` — lê `stock_quantity` do produto.
    5. `get_active_promotions(product_id=None)` — filtra `promotions.csv`
       por `is_active` e, opcionalmente, `product_id`.
    6. `get_customer_orders(customer_id)` — join `orders` × `order_items` ×
       `products` filtrado por cliente.
    7. `get_order_status(order_id)` — lookup em `orders.csv` (status,
       `tracking_code`, `estimated_delivery`).
- **Arquivo:** `tabular_data/pandas_service.py` —
  `PandasTabularDataService(BaseTabularDataService)`, os 6 métodos da
  interface implementados sobre os CSVs de `Settings.data_dir`.
- **Wiring:** já conectado em `main.py::chat()`
  (`AgentDependencies(tabular_data_service=...)`).
- **Bug corrigido:** `check_stock` lia a chave `"stock"` do dict do produto,
  mas a coluna real é `stock_quantity` — sempre retornava `None`.
- **Pendência conhecida (não bloqueia):** `search_products` ainda ignora os
  `**filters` (categoria, faixa de preço) e só busca em `name`, não em
  `description`, apesar do docstring prometer os dois.
- Testes em `tests/test_tabular_data.py` (11 casos, expectativas derivadas
  dos próprios CSVs em vez de valores fixos).

---

## 3. `memory.conversation_history` — ✅ concluído

- **Arquivo:** `memory/json_history_store.py` —
  `JsonConversationHistoryStore(BaseConversationHistoryStore)`: um arquivo
  JSON por `session_id` em `storage_dir`, com carregamento sob demanda
  (`_ensure_loaded`) na primeira vez que a sessão é acessada.
- **Bugs corrigidos:**
  - `_save_to_json` serializava `message.__dict__` direto — `timestamp`
    (`datetime`) não é serializável em JSON, então **todo** `add_message`
    lançava `TypeError`. Agora serializa `role.value` (str) e
    `timestamp.isoformat()`, com `_ensure_loaded` fazendo o caminho inverso.
  - Não havia nenhum carregamento do JSON persistido — reiniciar o processo
    fazia `get_history`/`clear_session` ignorarem sessões já salvas em
    disco (o docstring prometia "salvar e carregar", só a gravação
    existia).
  - `get_history(session_id, limit=0)` retornava a lista inteira em vez de
    vazia (`lista[-0:] == lista`, não `[]`, por causa do `-0 == 0` do
    Python).
- **Pendência:** construtor recebe `storage_dir: str` solto, não
  `Settings` — ao conectar em `main.py`, passar
  `str(settings.conversation_history_path)`.
- Testes em `tests/test_memory.py` (8 casos, incluindo reabrir o store em
  uma segunda instância para simular reinício do processo).

---

## 4. `tools` — ✅ concluído

Depende dos itens 2 e 3 estarem prontos (as ferramentas envolvem os
serviços concretos, não as interfaces).

- ✅ `tools/base.py` — `BaseTool` voltou a ser `ABC` com `run` como
  `@abstractmethod` (estava sem, uma subclasse incompleta falhava
  silenciosamente em vez de dar erro).
- ✅ `tools/catalog_tools.py`
  - `SearchProductsTool(BaseTool)` — envolve
    `tabular_data_service.search_products`.
  - `CheckOrderStatusTool(BaseTool)` — envolve
    `tabular_data_service.get_order_status`.
  - **Bugs corrigidos:** faltava `from typing import Any` (só não quebrava
    por causa do `from __future__ import annotations`); `SearchProductsTool.run`
    devolvia o `DataFrame` puro (não serializável para a resposta de function
    calling) — agora converte para `list[dict]`. Isso expôs um bug em
    `PandasTabularDataService.get_product_by_id`/`get_order_status`: `NaN`
    de colunas vazias (ex.: `notes`) ia direto pro dict, e `NaN` não é JSON
    válido em modo estrito (`json.dumps(..., allow_nan=False)` rejeitava) —
    corrigido com um helper `_first_row_or_none` que troca `NaN` por `None`.
  - **Bug corrigido (efeito colateral):** `search_products` só buscava em
    `name`, nunca em `description`, apesar do docstring prometer os dois —
    na prática isso zerava buscas por categoria de produto (ex.: "violão"
    só aparece em `description`, os produtos são nomeados por marca/modelo).
    Agora busca em `name` OR `description`.
- ✅ `tools/policy_tools.py` — `SearchStorePoliciesTool(BaseTool)`, recebe
  um `Retriever` já pronto no `__init__` e chama `.retrieve(query)`.
  **Bugs corrigidos:** chamava `Retriever(...).search(query)` — método
  inexistente (`Retriever` só tem `.retrieve`; `AttributeError` garantido) —
  e reconstruía um `Retriever` novo a cada chamada de `run()` em vez de
  receber um já pronto (inconsistente com as outras tools). Também
  convertia `list[VectorDocument]` (dataclass, não serializável) em
  `list[dict]` antes de devolver, mesmo ajuste de serialização das outras
  tools.
- Testes em `tests/test_tools.py` (9 casos) e `tests/test_policy_tools.py`
  (2 casos), cobrindo ABC do `BaseTool` e serialização JSON estrita
  (`allow_nan=False`) das três tools.

---

## 5. `agent.llm_client` — ✅ concluído

- ✅ `create_embedding` corrigido (item 1).
- ✅ `generate_response` implementado: converte `messages` para
  `types.Content`, passa `system` como `system_instruction`, chama
  `client.models.generate_content` e roda o loop de function calling —
  quando o modelo devolve `function_call`, executa a tool de verdade
  (`tool.run(**call.args)`) e manda o resultado de volta como
  `FunctionResponse`, repetindo até vir uma resposta em texto (limite de 5
  rounds, `_MAX_FUNCTION_CALL_ROUNDS`, para não loopar infinito).
- **Decisão de arquitetura:** `generate_response` recebe a lista de
  `BaseTool` de verdade (não só as declarações) — ele monta as declarações
  e executa as tools internamente, então `agent.core` não precisa saber
  nada sobre o formato de function calling do Gemini.
- Validado com chamadas reais: pergunta sobre política de troca (RAG),
  status de pedido (tool `check_order_status`) e busca de produto — as três
  responderam corretamente numa conversa real ponta a ponta.

---

## 6. `agent.core` — ✅ concluído (pipeline pronto, só falta o item 5)

- `agent/core.py::EmporioMusicaAgent.handle_message` — recupera histórico,
  recupera contexto via RAG (injetado no *system prompt*, não como
  "mensagem" — são trechos de apoio, não algo que usuário/assistente
  disseram), monta `messages` no formato `{"role": ..., "content": ...}`,
  chama `llm_client.generate_response`, persiste a troca e retorna o texto.
- **Bugs corrigidos:**
  - `tools=[t... for t in tabular_data_service.get_tools()]` — método
    inexistente em `PandasTabularDataService`; com a wiring real do
    `main.py` (que já passava `tabular_data_service`), isso quebrava com
    `AttributeError` ao digitar qualquer mensagem no chat. `AgentDependencies`
    agora tem um campo `tools: list[BaseTool] | None` próprio, montado no
    `main.py` (as tools envolvem os serviços, não o contrário).
  - `history_store.add_message(session_id=..., role=..., content=...)` —
    assinatura errada; o método real recebe um `ConversationMessage`
    (`session_id`, `message`), não kwargs soltos.
  - `messages=history + retrieved_context + [user_message]` — misturava
    `ConversationMessage`, `VectorDocument` e uma `str` crua numa lista só;
    não batia com `generate_response(messages: list[dict])`.
- Testes em `tests/test_agent_core.py` (5 casos, com um `GeminiLLMClient`
  fake para exercitar o caminho onde a resposta *é* gerada, já que o real
  ainda não existe).

---

## 7. `main.py` — ✅ concluído

- `chat()` monta `vector_store` → `indexer` → `retriever`,
  `tabular_data_service`, `history_store` (`JsonConversationHistoryStore`)
  e as três `tools`, tudo injetado em `AgentDependencies`.
- **Bugs corrigidos:** `conversation_history_store=`/`settings.history_storage_dir`
  não existem (campo certo é `history_store=`, setting certo é
  `settings.conversation_history_path`) — daria `TypeError`/`AttributeError`
  na primeira execução. Também havia 3 instâncias separadas de
  `PandasTabularDataService` (uma para `dependencies`, uma por tool) —
  cada uma relendo os 6 CSVs do disco à toa; agora é uma instância só,
  reaproveitada.

---

## 8. Testes — ✅ em dia

`tests/test_rag.py`, `tests/test_tabular_data.py`, `tests/test_tools.py`,
`tests/test_policy_tools.py`, `tests/test_memory.py`,
`tests/test_agent_core.py` — 49 testes, suite inteira verde. Os testes de
`handle_message` usam um `GeminiLLMClient` fake (`_FakeLLMClient`) — não
existe teste automatizado de `generate_response` contra a API real (só a
validação manual registrada no item 5); adicionar se quiser cobertura de
regressão para o formato do SDK.

---

## Chatbot funcional de ponta a ponta

Com os itens 1–8 concluídos, `python src/main.py chat` já sustenta uma
conversa real: RAG (políticas), consulta a dados tabulares (produtos,
pedidos, promoções) e histórico persistido por sessão, tudo orquestrado
pelo `EmporioMusicaAgent` e respondido pelo Gemini com function calling.
