# Empório da Música — Agente de Atendimento (CLI)

Agente de atendimento ao cliente da loja **Empório da Música**, com interface
via terminal (CLI). Combina **RAG** sobre o manual de políticas da loja,
**consulta a dados tabulares** (catálogo, pedidos, promoções) e **histórico de
conversas** persistente por sessão, tudo orquestrado por um LLM com function
calling.

> Status: funcional de ponta a ponta. `python src/main.py chat` sustenta uma
> conversa real — RAG, tools e histórico já respondem de verdade.

---

## Sumário

- [Arquitetura](#arquitetura)
- [Como rodar](#como-rodar)
- [Exemplos de conversa](#exemplos-de-conversa)
- [Decisões técnicas](#decisões-técnicas)
- [Limitações conhecidas e próximos passos](#limitações-conhecidas-e-próximos-passos)
- [Uso de assistente de IA](#uso-de-assistente-de-ia)

---

## Arquitetura

| Camada              | Pacote             | Responsabilidade                                                                            |
| ------------------- | ------------------ | ------------------------------------------------------------------------------------------- |
| Interface           | `cli`              | Loop de conversa no terminal. Sem lógica de negócio.                                        |
| Orquestração        | `agent`            | `EmporioMusicaAgent.handle_message`: monta o pipeline completo de atendimento.              |
| Modelo de linguagem | `agent.llm_client` | Wrapper isolado sobre o SDK `google-genai`. Nenhum outro módulo fala com a API do Gemini.   |
| RAG                 | `rag`              | Indexação do PDF de políticas, vector store FAISS em memória, recuperação por similaridade. |
| Dados tabulares     | `tabular_data`     | Consultas sobre os CSVs do desafio via `pandas`.                                            |
| Memória             | `memory`           | Histórico de conversas por sessão, persistido em JSON.                                      |
| Ferramentas         | `tools`            | Function calling — conecta o modelo a `rag` e `tabular_data`.                               |
| Configuração        | `config`           | Único ponto de leitura de variáveis de ambiente (`Settings`).                               |

```
                         ┌────────────┐
                         │    cli     │  (entrada/saída no terminal)
                         └─────┬──────┘
                               │
                         ┌─────▼──────┐
                         │   agent    │  (orquestração: EmporioMusicaAgent)
                         └──┬───┬───┬─┘
             ┌──────────────┘   │   └──────────────┐
        ┌────▼────┐      ┌──────▼──────┐     ┌──────▼───────┐
        │   rag   │      │ tabular_data│     │    memory    │
        │(políticas,│    │ (catálogo/  │     │  (histórico  │
        │ FAISS)   │     │  pedidos)   │     │  de conversa)│
        └────┬────┘      └──────┬──────┘     └──────────────┘
             │                  │
             └────────┬─────────┘
                       │ (envolvidos por)
                 ┌─────▼──────┐
                 │   tools    │  (function calling)
                 └─────┬──────┘
                       │
                 ┌─────▼──────┐
                 │llm_client  │  (Gemini via google-genai)
                 └────────────┘
```

Regra geral: **nenhum módulo lê variáveis de ambiente diretamente** (tudo
passa por `config.Settings`) e **nenhum módulo fora de `agent.llm_client`
importa o SDK do Gemini** — mantém o resto do código independente do
provedor de LLM.

### Estrutura de diretórios

```
emporio_musica_chatbot/
├── data/                          # CSVs do catálogo + políticas.pdf
├── src/
│   ├── main.py                    # Ponto de entrada da CLI (Typer)
│   ├── config.py                  # Settings (variáveis de ambiente)
│   ├── cli/interface.py           # Loop de conversa no terminal
│   ├── agent/
│   │   ├── core.py                # EmporioMusicaAgent (orquestrador)
│   │   └── llm_client.py          # Wrapper do Gemini (google-genai)
│   ├── rag/
│   │   ├── vector_store.py        # Interface BaseVectorStore (ABC)
│   │   ├── in_memory.py           # InMemoryVectorStore (FAISS)
│   │   ├── chroma.py              # Implementação alternativa (referência, não usada)
│   │   ├── indexer.py             # Interface BaseIndexer (ABC)
│   │   ├── pdf_indexer.py         # PdfPolicyIndexer (pypdf)
│   │   └── retriever.py           # Retriever
│   ├── tabular_data/
│   │   ├── schema.py              # Dicionário de dados dos CSVs
│   │   ├── service.py             # Interface BaseTabularDataService (ABC)
│   │   └── pandas_service.py      # PandasTabularDataService
│   ├── memory/
│   │   ├── conversation_history.py    # Interface BaseConversationHistoryStore (ABC)
│   │   └── json_history_store.py      # JsonConversationHistoryStore
│   └── tools/
│       ├── base.py                # Interface BaseTool (ABC)
│       ├── catalog_tools.py       # SearchProductsTool, CheckOrderStatusTool
│       └── policy_tools.py        # SearchStorePoliciesTool
├── storage/                       # Artefatos gerados em runtime (git-ignorado)
│   ├── vector_store/               # Índice FAISS + documentos persistidos
│   └── conversation_history/       # Um JSON por sessão de conversa
├── tests/                         # 49 testes (ver "Como rodar")
├── .env.example                   # Modelo de variáveis de ambiente
└── pyproject.toml
```

---

## Como rodar

### Pré-requisitos

- Python **3.10+**
- Uma chave de API do **Google AI Studio** (Gemini) — obtida em
  https://aistudio.google.com/apikey

### Instalação

```bash
# 1. Criar e ativar um ambiente virtual isolado na raiz do projeto
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Instalar o projeto em modo editável com todas as dependências
#    (rag = FAISS + pypdf; data = pandas; dev = pytest + ruff)
pip install -e ".[rag,data,dev]"
```

> Os extras `rag`/`data` existem porque nem todo módulo precisa de
> `faiss-cpu`/`pandas` — mas para rodar o chatbot completo, todos são
> necessários.

### Configuração

```bash
cp .env.example .env
```

Edite o `.env` e preencha `GOOGLE_API_KEY` com sua chave. As demais
variáveis já têm defaults sensatos:

| Variável                    | Padrão                           | Descrição                                                        |
| --------------------------- | -------------------------------- | ---------------------------------------------------------------- |
| `GOOGLE_API_KEY`            | _(vazio)_                        | Chave de API do Google AI (Gemini). **Obrigatória.**             |
| `GEMINI_MODEL`              | `gemini-2.5-flash`               | Modelo usado para gerar as respostas do agente.                  |
| `GEMINI_MAX_OUTPUT_TOKENS`  | `1024`                           | Máximo de tokens de saída por resposta.                          |
| `GEMINI_EMBEDDING_MODEL`    | `gemini-embedding-001`           | Modelo usado para gerar embeddings (RAG).                        |
| `GEMINI_EMBEDDING_DIM`      | `768`                            | Dimensão dos vetores de embedding (via `output_dimensionality`). |
| `DATA_DIR`                  | `./data`                         | Diretório com os CSVs do desafio.                                |
| `POLICIES_PDF_PATH`         | `./data/políticas.pdf`           | PDF de políticas usado pelo RAG.                                 |
| `VECTOR_STORE_PATH`         | `./storage/vector_store`         | Onde o índice FAISS é persistido.                                |
| `CONVERSATION_HISTORY_PATH` | `./storage/conversation_history` | Onde o histórico de conversas é persistido (um JSON por sessão). |

> **Nunca** commite o `.env` com a chave real — só o `.env.example` (sem
> valores) vai para o git.

### Uso

```bash
python src/main.py chat
# ou, após a instalação:
emporio-chatbot chat
```

Na primeira execução, o PDF de políticas é lido, chunkeado e indexado
(chama a API de embeddings do Gemini uma vez por chunk); nas execuções
seguintes, o índice é carregado do disco (`storage/vector_store/`) e não é
reprocessado. Digite `sair`, `exit`, `quit` ou `Ctrl+C` para encerrar.

Exemplos de perguntas para testar cada capacidade:

- _"Qual a política de troca e devolução?"_ → RAG sobre o PDF.
- _"Qual o status do pedido 1?"_ → tool `check_order_status` sobre os CSVs.
- _"Vocês têm violão?"_ → tool `search_products`.

### Testes

```bash
pytest -v
```

49 testes cobrindo `rag`, `tabular_data`, `memory`, `tools` e a orquestração
em `agent.core` (com um `GeminiLLMClient` fake, para não depender de rede
nem de API key — ver [Uso de assistente de IA](#uso-de-assistente-de-ia)
para o motivo). Nenhum teste chama a API real do Gemini.

---

## Exemplos de conversa

A pasta [`conversas/`](conversas/) tem 5 transcripts cobrindo catálogo com
filtro de preço, informação geral da loja, consulta de preço, continuidade
de contexto entre turnos e o cenário não trivial, aplicação da regra de
prazo da política de devolução sobre a data real de um pedido consultado
ao vivo.

---

## Decisões técnicas

**LLM — Google Gemini.** `gemini-2.5-flash` para geração de respostas (bom
equilíbrio custo/latência para um agente conversacional, com suporte nativo
a function calling) e `gemini-embedding-001` para embeddings, com
`output_dimensionality=768`, o modelo aceita até 3072 dimensões por
padrão, mas para o volume de dados do desafio (um único PDF pequeno) 768 já
é mais que suficiente e mantém o índice e a busca mais leves.

**RAG — FAISS em memória, não Chroma.** Cheguei a esboçar uma implementação
com ChromaDB (`rag/chroma.py`, mantida só como referência), mas optei por
`IndexFlatL2` do FAISS: para um PDF de políticas de poucas páginas, um
índice em memória com busca exaustiva (linear scan) é rápido o suficiente e
evita subir/gerenciar um serviço externo. O índice e os documentos são
persistidos em disco (`faiss.write_index` + `pickle`) para não reprocessar
o PDF a cada execução. Trade-off explícito: essa escolha não escala para um
catálogo de documentos grande, nesse cenário, Chroma (ou outro vector DB
com índice aproximado) seria a escolha certa.

**Dados tabulares — pandas em memória.** Os CSVs do desafio são pequenos
(centenas de linhas); carregar tudo em `DataFrame`s no `__init__` do
serviço evita a complexidade de um banco de dados para um volume que cabe
tranquilamente em memória.

**Histórico de conversas — JSON por sessão.** Um arquivo por
`session_id` em `storage/conversation_history/`, carregado sob demanda
(lazy load) na primeira vez que a sessão é acessada. Simples, sem
dependência extra, adequado para um CLI de uso local (sem escrita
concorrente de múltiplos processos).

**Arquitetura em camadas com interfaces (`ABC`) + implementação
concreta.** Cada módulo de domínio (`rag`, `tabular_data`, `memory`,
`tools`) define uma interface abstrata separada da implementação (ex.:
`BaseVectorStore` vs. `InMemoryVectorStore`). Isso deixa explícito o
contrato que cada peça precisa cumprir e permite trocar a implementação
(ex.: `InMemoryVectorStore` → Chroma, `JsonConversationHistoryStore` →
SQLite) sem tocar em `agent.core`. O acoplamento com o SDK do Gemini fica
isolado inteiramente em `agent/llm_client.py`.

**Injeção de dependências via `AgentDependencies`.** `EmporioMusicaAgent`
recebe um único objeto de dependências (não parâmetros soltos), o que
facilita testar `handle_message` com duplos de teste (fakes) para cada
dependência, sem precisar de API key nem rede.

**Estratégia de prompt.** O contexto recuperado via RAG entra no
`system_instruction`, não como uma "mensagem" da conversa, são trechos de
apoio (políticas relevantes), não algo que o usuário ou o assistente
disseram, então misturá-los ao histórico de turnos poluiria a conversa e
confundiria o modelo sobre quem disse o quê. As `tools` são passadas como
`FunctionDeclaration`s reais do SDK, e o loop de function calling
(pergunta → modelo pede tool → executa de verdade → devolve resultado →
nova chamada) roda inteiro dentro de `GeminiLLMClient.generate_response`,
com um limite de 5 idas-e-voltas para não loopar indefinidamente se o
modelo insistir em chamar tools.

---

## Operação e limites de escala

O índice usa cosseno normalizado, chunking por sentença com overlap, fusão
semântica/lexical, limiar de relevância, cache de embeddings, atualização por
hash da fonte e persistência atômica em JSON/NPY (sem `pickle`). O catálogo
normaliza acentos/plural e aplica filtros estruturados; todas as tools devolvem
JSON estrito. O histórico é limitado por `MAX_HISTORY_MESSAGES` (20 por padrão)
e arquivos de sessão não aceitam path traversal.

Também há uma API FastAPI em `api.app:app`: instale `.[api]` e execute
`uvicorn api.app:app`. Em produção, defina `APP_ENV=production` e `API_KEY`;
sem chave a API recusa tráfego. A API key é um controle de borda mínimo, não
substitui identidade/autorização por cliente: antes de expor pedidos ou dados
pessoais publicamente, integre o provedor de autenticação e imponha ownership
de pedido no backend.

FAISS local continua intencional para um corpus pequeno e uma única réplica.
Para alta disponibilidade, atualização concorrente ou milhões de chunks,
substitua-o por um vector DB gerenciado/pgvector por trás de `BaseVectorStore`.
Há também uma implementação pronta de Chroma em `rag.chroma.ChromaVectorStore`;
instale `pip install -e ".[chroma]"` e injete-a no bootstrap se decidir usá-la.
Os testes são determinísticos e não chamam Gemini; monitore a compatibilidade
do SDK em staging e configure alertas a partir dos logs de tokens, latência e
uso de tools.

---
