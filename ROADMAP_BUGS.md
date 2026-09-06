# Roadmap — Correção de bugs e gaps funcionais

> Estado: concluído na revisão 0.2. As tools adicionais, retornos JSON, join
> de pedidos, filtros/literalidade da busca e janela de histórico foram
> implementados e têm testes de regressão. Ver
> [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).

Itens levantados lendo o código atual (não o `ROADMAP.md` original, que já
está concluído) e cruzando com os testes existentes. Ordem sugerida: do gap
que mais limita o produto (tools faltando) para os ajustes mais pontuais.

---

## 1. Tools faltando: `check_stock`, `get_active_promotions`, `get_customer_orders`

- **Arquivos:** `src/tools/catalog_tools.py` (criar 3 classes novas),
  `src/main.py:47-57` (registrar na lista `tools=[...]`).
- **O que está errado:** `BaseTabularDataService` promete 6 métodos e
  `PandasTabularDataService` implementa e testa todos os 6
  (`tests/test_tabular_data.py`), mas só 2 têm uma `BaseTool` que os expõe
  ao Gemini via function calling: `SearchProductsTool` e
  `CheckOrderStatusTool`. `check_stock`, `get_active_promotions` e
  `get_customer_orders` nunca são chamados pelo agente.
- **Lógica a acrescentar:** três classes seguindo exatamente o padrão de
  `CheckOrderStatusTool` (`name`, `description`, `parameters_schema`,
  `__init__(self, tabular_data_service)`, `run(**kwargs)` chamando o
  método correspondente do serviço):
  - `CheckStockTool` → `tabular_data_service.check_stock(product_id)`.
  - `GetActivePromotionsTool` → `tabular_data_service.get_active_promotions(product_id=None)`
    (`product_id` opcional no schema).
  - `GetCustomerOrdersTool` → `tabular_data_service.get_customer_orders(customer_id)`.
  Depois, instanciar as três em `main.py::chat()` e somá-las à lista
  `tools=[...]` já passada em `AgentDependencies`.
- **Por quê:** é o maior gap funcional do projeto hoje — um cliente
  perguntando "tem estoque desse violão?", "tem alguma promoção ativa?" ou
  "quais meus pedidos anteriores?" não recebe resposta, mesmo com o dado já
  pronto e testado na camada de serviço. A camada de dados cumpre a
  interface, mas isso nunca chega ao usuário final.

---

## 2. `get_active_promotions`/`get_customer_orders` retornam `DataFrame` cru

- **Arquivo:** `src/tabular_data/pandas_service.py:52-66`.
- **O que está errado:** mesmo bug que já foi corrigido em
  `search_products`/`get_order_status` (`NaN` não é JSON válido; `DataFrame`
  não é serializável em uma function response) ainda não foi replicado
  aqui. Assim que as tools do item 1 forem criadas, vão quebrar na primeira
  chamada real com dado ausente (ex.: uma promoção sem `description`).
- **Lógica a acrescentar:** usar o helper já existente
  `_first_row_or_none` para `get_order_status`-like, e criar uma variante
  para múltiplas linhas (ex.: `_rows_or_empty(df)`, análogo trocando `NaN`
  por `None` e devolvendo `list[dict]` via `df.where(df.notna(), None).to_dict(orient="records")`).
  Aplicar em `get_active_promotions` e `get_customer_orders`.
- **Por quê:** pré-requisito direto do item 1 — sem isso as tools novas
  herdam o mesmo bug de serialização que já foi corrigido duas vezes em
  outros métodos do mesmo arquivo (ver `ROADMAP.md`, item 2 e 4).

---

## 3. `get_customer_orders` não faz join com `order_items`/`products`

- **Arquivo:** `src/tabular_data/pandas_service.py:61-66`.
- **O que está errado:** a docstring do método diz "Retorna pedidos do
  cliente com detalhes dos itens", e o `ROADMAP.md` original (item 2) já
  prometia um join `orders × order_items × products` — mas a implementação
  atual só filtra `self.orders_df` por `customer_id`; nunca toca
  `order_items_df` nem `products_df`. Um cliente perguntando "o que eu
  comprei no pedido X" recebe só o cabeçalho do pedido (status, data,
  total), sem saber quais produtos foram comprados.
- **Lógica a acrescentar:** dentro de `get_customer_orders`, fazer
  `orders_df` filtrado por `customer_id` → `merge` com `order_items_df` em
  `order_id` → `merge` com `products_df` em `product_id`, retornando por
  pedido a lista de itens com `name`, `quantity`, `price_brl`.
- **Por quê:** sem isso a função não cumpre o que o próprio código promete
  (docstring + roadmap original), e é exatamente o dado que o desafio
  espera para responder "o que tinha no meu pedido?".

---

## 4. `search_products` ignora os filtros de categoria/faixa de preço

- **Arquivo:** `src/tabular_data/pandas_service.py:30-36`.
- **O que está errado:** o parâmetro `**filters` é recebido na assinatura
  mas nunca usado no corpo do método — a docstring promete "aplica filtros
  opcionais" e isso já está listado como pendência não resolvida no
  `ROADMAP.md` original (item 2).
- **Lógica a acrescentar:** aceitar `category` (nome, resolvido via join
  com `categories_df` por `category_id`) e `min_price`/`max_price`
  (comparando com `price_brl`), aplicando como máscaras booleanas
  adicionais (`&`) sobre o resultado da busca por texto antes de retornar.
  Atualizar também `SearchProductsTool.parameters_schema`
  (`src/tools/catalog_tools.py:18-24`) para expor esses campos ao modelo —
  hoje só existe `query`, então mesmo implementando o filtro no serviço, o
  Gemini não tem como preenchê-lo.
- **Por quê:** perguntas como "guitarras até R$2000" ou "produtos da
  categoria baterias" hoje devolvem todos os resultados que batem no texto
  e ignoram o filtro que o próprio usuário já deu — o agente responde
  informação em excesso ou irrelevante.

---

## 5. Busca de produtos quebra com caracteres de regex no termo de busca

- **Arquivo:** `src/tabular_data/pandas_service.py:34-35` (função
  `search_products`).
- **O que está errado:** `self.products_df["name"].str.contains(query, case=False, na=False)`
  interpreta `query` como **regex** por padrão (comportamento default do
  pandas). Termos de busca legítimos com parênteses, `+`, `*`, `[`/`]` etc.
  (comuns em nomes/specs de produtos, ex.: "Combo (Kit)", "50W+") levantam
  `re.error`. A exceção é capturada por `GeminiLLMClient._run_tool`
  (`src/agent/llm_client.py:131-135`), então não derruba o processo, mas a
  tool falha e o agente devolve um erro genérico para uma busca válida.
- **Lógica a acrescentar:** passar `regex=False` em ambas as chamadas de
  `str.contains` (nome e descrição), já que a intenção real é
  substring-match case-insensitive, não regex.
- **Por quê:** é a tool mais usada do agente (busca de produtos) e falha
  silenciosamente para entradas de usuário plausíveis, sem que nada no
  roadmap original tivesse coberto esse caso.

---

## 6. Histórico de conversa enviado sem limite para o Gemini

- **Arquivo:** `src/agent/core.py:66` (`EmporioMusicaAgent.handle_message`).
- **O que está errado:** `self._deps.history_store.get_history(session_id=session_id)`
  é chamado sem `limit`, o que retorna **todo** o histórico já persistido
  da sessão. Em conversas longas isso eventualmente estoura o limite de
  contexto do modelo (erro de API) ou infla custo/latência sem
  necessidade — e é o único ponto do pipeline sem nenhum teto (RAG já tem
  `top_k`, function calling já tem `_MAX_FUNCTION_CALL_ROUNDS`).
- **Lógica a acrescentar:** adicionar `max_history_messages: int = 20` em
  `Settings` (`src/config.py`) e passar
  `limit=settings.max_history_messages` para `get_history` em
  `handle_message`. `JsonConversationHistoryStore.get_history` já suporta
  `limit` corretamente (bug do `-0` já corrigido no roadmap original).
- **Por quê:** é o único componente do pipeline sem limite explícito; sem
  isso, uso real e prolongado do chatbot pode quebrar no meio de uma
  conversa com uma mensagem de erro pouco amigável para o cliente.
