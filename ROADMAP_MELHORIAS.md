# Roadmap — Melhorias e expansão

> Atualização 0.2: API HTTP, identificação opcional no CLI, cross-sell de
> promoções, chunking/limiar/reranking do RAG e telemetria de tokens/tools
> foram incorporados. Streaming nativo deve ser ativado somente junto do
> transporte assíncrono do canal escolhido, para preservar o loop de tools e
> evitar respostas parciais incorretas; não é uma limitação de recuperação.

Sugestões que partem do que já existe (pipeline RAG + tools + histórico
funcionando ponta a ponta) para ampliar canal, UX, qualidade de resposta e
receita. Não depende dos fixes do `ROADMAP_BUGS.md`, mas os itens 3 e 4
abaixo ficam mais fortes depois que as tools de promoção/pedidos existirem
(`ROADMAP_BUGS.md`, item 1).

---

## 1. Canal web/API reaproveitando o mesmo agente

- **Arquivo novo:** `src/api/app.py` (FastAPI); reaproveita
  `agent.core.EmporioMusicaAgent` e `AgentDependencies` tal como montados em
  `src/main.py::chat()`.
- **Lógica a acrescentar:** um endpoint `POST /chat` que recebe
  `{session_id, message}` e chama `agent.handle_message(session_id=..., user_message=...)`;
  `session_id` vindo do cliente (cookie ou corpo da requisição) em vez de
  gerado uma vez por processo como hoje em `cli/interface.py:24`
  (`session_id = str(uuid.uuid4())` só existe enquanto o processo CLI roda).
- **Por quê:** o próprio código já sinaliza essa direção — o docstring de
  `src/cli/interface.py:3` diz literalmente "uma futura interface web/API
  pode reutilizar `agent.core.EmporioMusicaAgent`". Hoje o chatbot só existe
  como sessão de terminal; expor via API é o que permite integrar com site,
  WhatsApp Business API, etc.

---

## 2. Streaming da resposta do modelo

- **Arquivo:** `src/agent/llm_client.py:93-98` (`GeminiLLMClient.generate_response`).
- **Lógica a acrescentar:** trocar `self._client.models.generate_content(...)`
  por `self._client.models.generate_content_stream(...)` no round final (sem
  function call pendente), repassando um callback/generator para
  `agent.core.EmporioMusicaAgent.handle_message` imprimir token a token em
  `cli/interface.py::run_chat_loop`. Os rounds de function calling
  continuam não-streamados (precisam da resposta completa para decidir se
  há `function_call`).
- **Por quê:** hoje o usuário espera a resposta inteira (`gemini_max_output_tokens=1024`)
  aparecer de uma vez — em respostas mais longas (ex.: explicar uma
  política inteira) isso é perceptivelmente mais lento do que mostrar o
  texto conforme é gerado, sem custo adicional de tokens.

---

## 3. Identificação leve de cliente na sessão (personalização)

- **Arquivo:** `src/agent/core.py` (`AgentDependencies` e `handle_message`),
  `src/cli/interface.py::run_chat_loop`.
- **Lógica a acrescentar:** no início do loop, perguntar telefone/e-mail do
  cliente e resolver contra `customers_df` (novo método
  `get_customer_by_contact` em `PandasTabularDataService`); guardar o
  `customer_id` resolvido numa variável de sessão (ex.: dict `session_id → customer_id`
  dentro de `EmporioMusicaAgent` ou um novo campo em `AgentDependencies`).
  Incluir esse `customer_id` no `system_prompt` (`_build_system_prompt`)
  para que o modelo possa chamar `get_customer_orders` (depois de criada,
  ver `ROADMAP_BUGS.md` item 1) **sem pedir o ID de novo a cada pergunta**.
- **Por quê:** hoje qualquer pergunta sobre "meus pedidos" exige que o
  cliente informe um ID numérico que ele provavelmente não sabe de cor.
  Resolver por telefone/e-mail (dado que já existe em `customers.csv`) é
  uma fricção a menos e abre caminho para saudação personalizada ("Oi,
  João! Vi que seu último pedido...").

---

## 4. Upsell proativo cruzando busca de produtos com promoções ativas

- **Arquivo:** `src/tools/catalog_tools.py` (`SearchProductsTool.run`), ou
  alternativamente `src/tabular_data/pandas_service.py::search_products`.
- **Lógica a acrescentar:** depois de obter os produtos que batem na busca,
  cruzar os `product_id` resultantes com `get_active_promotions` (depois de
  criada — `ROADMAP_BUGS.md` item 1) e anexar um campo `promotion` em cada
  produto retornado quando existir uma promoção ativa para aquele
  `product_id`. Ajustar `_build_system_prompt`/instrução da tool para o
  modelo mencionar a promoção espontaneamente quando presente, em vez de só
  responder se perguntado.
- **Por quê:** a tabela `promotions.csv` já existe e hoje não influencia
  nenhuma resposta do agente — é receita deixada na mesa. Cruzar os dois
  automaticamente transforma uma busca comum ("tem baixo de 5 cordas?") em
  oportunidade de venda ("... e o modelo X está com 15% de desconto").

---

## 5. Qualidade do RAG: chunking por sentença + limiar de relevância

- **Arquivos:** `src/rag/indexer.py::BaseIndexer._chunk_document` (linhas
  27-39) e `src/rag/retriever.py::Retriever.retrieve` (linha 15-18).
- **Lógica a acrescentar:**
  - `_chunk_document`: hoje divide o texto da página só por `\n\n`; como
    `pypdf.extract_text()` nem sempre preserva parágrafos com linha dupla,
    uma página inteira pode virar um único chunk grande. Trocar por
    chunking por tamanho fixo de caracteres/tokens com overlap (ex.: 500
    caracteres, overlap de 50), mantendo o fallback por parágrafo quando
    existir.
  - `retrieve`: hoje devolve sempre os `top_k` documentos mais próximos,
    mesmo que a distância seja alta (pouco relevante). Acrescentar um
    `max_distance` (ou normalizar a distância L2 do FAISS
    — `src/rag/in_memory.py:52`, `document.score = float(distance)` — para
    uma similaridade 0-1) e filtrar chunks acima do limiar antes de
    retornar.
- **Por quê:** hoje o `system_prompt` do agente (`core.py::_build_system_prompt`)
  injeta qualquer coisa que o `similarity_search` devolver, mesmo se for
  fracamente relacionado à pergunta — isso aumenta o risco de o modelo
  "inventar" uma política citando um trecho de contexto irrelevante em vez
  de dizer que não sabe.

---

## 6. Observabilidade: uso de tokens e tools por conversa

- **Arquivo:** `src/agent/llm_client.py::generate_response` (linha 93-98) e
  `src/utils/logger.py`.
- **Lógica a acrescentar:** capturar `response.usage_metadata`
  (`prompt_token_count`, `candidates_token_count`, já disponível na
  resposta do SDK `google-genai`, hoje descartado) e logar por chamada,
  junto com quais tools foram chamadas em `_run_tool`
  (`src/agent/llm_client.py:126-135`) e quanto tempo cada uma levou.
  Estruturar como log JSON (`logger.info` com `extra={...}`) para permitir
  agregação depois (custo por sessão, tools mais usadas, taxa de erro por
  tool).
- **Por quê:** hoje não há nenhuma visibilidade de custo (tokens consumidos
  por conversa) nem de quais tools o modelo realmente usa na prática — sem
  isso é impossível saber, por exemplo, se `search_store_policies` está
  sendo chamada demais (custo) ou de menos (política respondida "no
  chute").
