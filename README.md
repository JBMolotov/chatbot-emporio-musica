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

## Limitações conhecidas e próximos passos

- **Busca de produtos é só por substring.** `search_products` casa
  case-insensitive em `name`/`description`, mas não normaliza acento nem
  plural/singular. Confirmado na prática ao gerar os transcripts em
  [`conversas/`](conversas/): `"violão"` encontra 35 produtos, mas
  `"violões"` (plural, a forma sugerida no próprio enunciado do desafio)
  encontra 0 — um falso negativo real, não hipotético (detalhes em
  [`conversas/README.md`](conversas/README.md)). Também ignora os
  `**filters` que a interface já aceita (categoria, faixa de preço) — hoje
  quem filtra por preço é o próprio modelo, raciocinando sobre a lista
  completa que a tool devolve, o que funciona para um catálogo pequeno mas
  não escalaria. Com mais tempo, trocaria por uma busca semântica
  (embeddings sobre o catálogo, mesmo padrão do RAG de políticas) ou ao
  menos normalização de acentos/plural + filtros estruturados reais.
- **`InMemoryVectorStore.delete` reconstrói o índice inteiro**,
  re-gerando embeddings via API para todos os documentos restantes, funciona
  para um PDF pequeno, mas não escalaria para uma base de documentos maior.
- **`refresh_index` duplica chunks em vez de substituí-los** (FAISS
  `IndexFlatL2` só suporta `add`, não upsert/delete pontual), reindexar a
  mesma fonte duas vezes duplica as entradas. Precisaria de uma estratégia
  de remoção por `source` antes de reindexar.
- **Sem streaming de resposta.** O CLI espera a resposta completa do
  Gemini antes de imprimir. `generate_content_stream` resolveria isso e
  melhoraria a percepção de latência.
- **Histórico sem limite/janela de contexto.** Toda a conversa é
  recarregada e reenviada ao modelo a cada turno; para sessões longas isso
  cresce o custo e pode estourar o limite de tokens. Com mais tempo,
  adicionaria truncamento (últimas N mensagens) ou sumarização incremental.
- **Sem retries/backoff nas chamadas ao Gemini.** Uma falha transitória de
  rede propaga como exceção não tratada (a única tolerância a erro hoje é
  dentro da execução de tools, em `GeminiLLMClient._run_tool`).
- **Sem teste automatizado contra a API real do Gemini** de propósito,
  para manter a suíte rápida, determinística e sem custo/rede, mas isso
  significa que uma mudança no formato do SDK só seria pega manualmente
  (foi validado à mão durante o desenvolvimento, não há regressão automática).
- **Só CLI.** Uma API HTTP (FastAPI, por exemplo) reaproveitando
  `EmporioMusicaAgent` sem alterações seria o próximo passo natural para
  expor isso além do terminal.

---

## Uso de assistente de IA

Usei o **Claude Code** (Anthropic), rodando como extensão dentro do editor,
durante o desenvolvimento. O padrão de uso foi parecido com trabalhar com um
revisor/pair programmer teto a teto comigo:

1. **Eu escrevia o código de cada módulo primeiro**
   interfaces, classes concretas, wiring no `main.py`, e pedia pro Claude
   **verificar e testar de verdade**, não só ler o código.
2. Ele rodava o código contra os dados reais do desafio (CSVs, o PDF de
   políticas, e a API do Gemini com uma key real) em vez de só inspecionar
   visualmente, isso pegou bugs que uma leitura estática não pegaria: por
   exemplo, `check_stock` lendo a chave errada do dict (`"stock"` em vez de
   `"stock_quantity"`), `NaN` do pandas vazando pra respostas que
   precisavam ser JSON válido, um `Retriever.search()` chamando um método
   que não existia, e o formato real do SDK `google-genai` para embeddings
   e function calling (que eu tinha assumido errado de início, ele
   inspecionou o pacote instalado e testou contra a API antes de sugerir
   alterações, em vez de confiar só em memória de treinamento).
3. Ele validou o formato exato do SDK com chamadas reais passo a passo
   (texto simples → uma function call isolada → o loop completo).
   fomos mordidos duas vezes por assumir o shape errado da API, então essa
   virou a exigência: nada de assumir formato de SDK sem confirmar.
4. Usei pra escrever a suíte de testes junto de cada módulo, cada bug
   encontrado testando manualmente virou um teste de regressão.
5. Também usei pra revisar o README, sugerindo melhorias de clareza e
   completude.
