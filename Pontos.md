# Documento histórico — estado substituído

Os TODOs descritos abaixo foram implementados na revisão de produção: há
persistência JSON atômica com retenção por janela, RAG FAISS com indexação de
PDF/chunking/retrieval e tools concretas. A referência operacional atual é
[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).

main.py

Montagem das dependências do agente. `retriever`, `tabular_data_service`
e `history_store` ainda não têm implementação concreta — ver os
respectivos pacotes (`rag`, `tabular_data`, `memory`).

config.py

"""Configuração central da aplicação.

Todas as variáveis de ambiente do projeto devem ser lidas a partir daqui —
nenhum outro módulo deve chamar `os.environ` ou `os.getenv` diretamente.
Isso mantém um único ponto de verdade para configuração e facilita testes
(basta instanciar `Settings` com overrides).
"""

/agent

core.py
"""Orquestrador do agente de atendimento.

Este módulo é o ponto central que futuramente vai: 1. Recuperar contexto relevante via `rag.retriever.BaseRetriever`
(ex.: políticas da loja). 2. Consultar dados tabulares via
`tabular_data.service.BaseTabularDataService` (produtos, pedidos,
promoções etc.), possivelmente através de `tools`. 3. Recuperar/atualizar o histórico da sessão via
`memory.conversation_history.BaseConversationHistoryStore`. 4. Montar o prompt final (system prompt do Empório da Música) e chamar
`agent.llm_client.GeminiLLMClient`. 5. Persistir a nova troca de mensagens no histórico.

Nenhuma dessas etapas está implementada aqui — apenas a estrutura de
dependências e o ponto de entrada `handle_message`.
"""

/tests

text_placeholder.py
"""
Substituir/complementar com testes reais conforme cada módulo for
implementado (agent, rag, tabular_data, memory).
"""

/cli
interface.py

"""
Ponto de extensão: aqui é onde, futuramente, podem entrar comandos
especiais de CLI (ex.: "/limpar" para resetar o histórico da sessão),
indicadores de streaming da resposta, etc. A geração da resposta em si
é responsabilidade de `agent.handle_message`.
"""

/memory
init.py

"""
"""Ponto de extensão: histórico de conversas.

Este pacote deve conter a futura persistência do histórico de mensagens por
sessão/cliente, usada para dar continuidade ao atendimento entre turnos (e,
futuramente, entre sessões).
"""

conversation_history.py

"""
TODO: - Escolher o backend de persistência (ex.: SQLite, JSON em disco, Redis)
e implementar uma subclasse concreta de `BaseConversationHistoryStore`. - Persistir em `Settings.conversation_history_path`. - Definir política de retenção/tamanho máximo de histórico por sessão.
"""

/rag
init.py
Este pacote deve conter a futura indexação e recuperação de conteúdo do
documento de políticas da loja (`data/políticas_da_loja.pdf`) e de qualquer
outro material de apoio ao atendimento.

indexer.py
TODO (implementação futura): - Extrair texto de `Settings.policies_pdf_path` (ex.: com `pypdf`). - Aplicar uma estratégia de chunking (por parágrafo, por tamanho fixo etc.). - Gerar embeddings para cada chunk e enviá-los a um `BaseVectorStore`.

retriever.py
TODO (implementação futura): - Implementar a busca por similaridade sobre o `BaseVectorStore`. - Decidir o formato do contexto retornado (texto concatenado, blocos
citáveis, etc.) para alimentar o prompt do agente.

vector_store.py

TODO (implementação futura): - Escolher o backend (ex.: ChromaDB, FAISS, pgvector) e implementar uma
subclasse concreta de `BaseVectorStore`. - Definir a estratégia de geração de embeddings (modelo, dimensão). - Persistir o índice em `Settings.vector_store_path`.

/tabular_data
init.py
Ponto de extensão: consulta a dados tabulares.

Este pacote deve conter a futura lógica de consulta aos dados estruturados
do desafio (`data/*.csv`): produtos, categorias, clientes, pedidos, itens de
pedido e promoções.

schema.py
"""

Apenas referência de nomes de arquivo/colunas — nenhuma lógica de leitura ou
consulta é implementada aqui. Mantém a implementação futura (`service.py`)
desacoplada de strings mágicas espalhadas pelo código.
"""

service.py
"""

TODO (implementação futura): - Carregar os arquivos descritos em `schema.py` (ex.: com `pandas`). - Implementar os métodos abaixo com as regras de negócio reais
(join entre `products`/`categories`/`promotions`, filtros de estoque,
histórico de pedidos por cliente etc.). - Decidir se o carregamento é em memória (simples, adequado ao volume do
desafio) ou via um banco de dados leve (ex.: SQLite).
"""

/tools
init.py
"""Ponto de extensão: ferramentas (function-calling) do agente.

Cada ferramenta concreta deve envolver um serviço já existente (ex.:
`tabular_data.service.BaseTabularDataService`, `rag.retriever.BaseRetriever`)
e expor uma interface compatível com function calling da API do Gemini.
"""

base.py
"""
TODO (implementação futura): - Criar subclasses concretas (ex.: `SearchProductsTool`,
`CheckOrderStatusTool`, `SearchStorePoliciesTool`) que envolvam os
serviços de `tabular_data` e `rag`. - Implementar o registro/roteamento de chamadas de função dentro do
loop agentic em `agent.core` (ver function calling do Gemini).
"""
