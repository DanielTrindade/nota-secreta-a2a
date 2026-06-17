# Nota Secreta — solução de referência comentada

Este projeto contém uma **versão comentada e simplificada** do jogo **Nota Secreta**,
usada como base para a implementação do agente estratégico da disciplina.

> ⚡ **Rodar no Google Colab:** abra o notebook [`nota_secreta_colab.ipynb`](nota_secreta_colab.ipynb).
> Depois de publicar este repositório no GitHub, dá pra abri-lo direto no Colab pela URL
> `https://colab.research.google.com/github/DanielTrindade/nota-secreta-a2a/blob/main/nota_secreta_colab.ipynb`
> O notebook clona o repo, instala as dependências, roda os testes
> e executa uma partida — em modo mock e, opcionalmente, com um modelo GGUF real.
> Detalhes na seção [13. Rodar no Colab](#13-rodar-no-colab).

A ideia é que você possa:

- entender a arquitetura do sistema;
- rodar partidas localmente;
- testar seu agente em modo mock ou com um modelo real;
- modificar principalmente `llm_agent.py` e, se desejar, `base_agent.py`.

---

## 1. Visão geral da arquitetura

O projeto combina dois estilos de comunicação:

- **REST/FastAPI** entre os agentes e o serviço LLM centralizado (`llm_service.py`);
- **A2A / JSON-RPC** entre o Game Master e os agentes.

Em uma execução típica:

1. o `run_game.py` sobe o serviço LLM;
2. sobe o `game_master.py`;
3. sobe 1 agente estratégico e 5 agentes aleatórios;
4. registra os agentes no Game Master;
5. executa uma partida completa;
6. salva um log da partida em `logs/`.

---

## 2. Estrutura dos arquivos

Arquivos principais:

- `fasta2a.py`: mini-implementação de `A2AApp` e `@tool`
- `base_agent.py`: utilidades comuns para agentes
- `llm_service.py`: serviço LLM centralizado (real ou mock)
- `game_master.py`: coordenação da partida, votação, pontuação e logs
- `llm_agent.py`: agente estratégico a ser estudado e modificado
- `random_agent.py`: baseline aleatório
- `run_game.py`: sobe tudo e executa uma partida completa
- `brazilian_songs.csv`: base de músicas usada pelo jogo
- `tests/`: testes auxiliares

---

## 3. O que você deve modificar

Em geral, os arquivos mais importantes para o aluno são:

- `llm_agent.py`
- `base_agent.py` (opcional)

Você pode reorganizar a lógica interna do agente, desde que preserve a interface esperada
pelo restante da infraestrutura.

As ferramentas (tools) esperadas do agente são:

- `receive_hand(hand)`
- `choose_card()`
- `send_clue(lyrics, max_words=6)`
- `select_card_by_clue(clue)`
- `vote(clue, options, my_chosen_card)`

---

## 4. Instalação

Crie e ative um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

Instale as dependências:

```bash
python3 -m pip install -r requirements.txt
```

---

## 5. Execução rápida

### 5.1. Rodar em modo mock

Esse modo não usa um modelo real e é útil para validar rapidamente a arquitetura:

```bash
python3 run_game.py --force-mock
```

### 5.2. Rodar com um modelo GGUF real

```bash
python3 run_game.py --model /caminho/do/modelo.gguf
```

Exemplo:

```bash
python3 run_game.py --model ~/Documentos/LLM/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

---

## 6. Opções úteis do `run_game.py`

### Subir 6 agentes estratégicos

```bash
python3 run_game.py --all-strategic --force-mock
```

ou:

```bash
python3 run_game.py --all-strategic --model /caminho/do/modelo.gguf
```

### Alterar a base de músicas

```bash
python3 run_game.py --db outra_base.csv --force-mock
```

### Ajustar concorrência do serviço LLM

```bash
python3 run_game.py --model /caminho/do/modelo.gguf --llm-max-concurrency 1
```

---

## 7. Logs

Ao final da partida, o Game Master salva um log JSON em:

```text
logs/
```

O caminho do log também é mostrado no terminal ao fim da execução.

Esses logs ajudam a entender:

- qual agente foi narrador em cada rodada;
- qual dica foi produzida;
- quais cartas foram jogadas;
- como os votos foram distribuídos;
- como a pontuação evoluiu ao longo da partida.

---

## 8. Como ler os logs

Cada partida gera um arquivo JSON em `logs/` com a rodada, o narrador, a dica,
as cartas jogadas, os votos e a pontuação acumulada. O JSON pode ser aberto
direto em qualquer editor; o caminho exato é impresso no terminal ao fim da
execução.

---

## 9. Observações sobre a base de músicas

A base CSV deve conter, no mínimo, as colunas:

- `id`
- `title`
- `artist`
- `lyrics`

A base fornecida aqui serve para testes e desenvolvimento local.
Na avaliação, vai ser usada uma base oficial definida pelo professor.

---

## 10. Objetivo pedagógico

O foco deste trabalho não é apenas “fazer um agente funcionar”, mas construir
um **sistema multiagente baseado em LLM**.

Por isso, espera-se que o agente:

- use a LLM para decisões semânticas;
- lide com respostas imperfeitas de forma robusta;
- preserve o protocolo esperado pela infraestrutura.

Em outras palavras:

> a implementação interna pode variar, mas a interface externa do agente deve continuar compatível.

---

## 11. Resumo

Use esta versão do projeto para:

- entender a arquitetura;
- rodar testes locais;
- modificar o agente estratégico;
- experimentar diferentes prompts e estratégias.

Fluxo mínimo recomendado:

1. rodar `python3 run_game.py --force-mock`
2. rodar `python3 run_game.py --model ...`
3. inspecionar os logs
4. modificar `llm_agent.py`
5. repetir os testes

---

## 12. Estratégia implementada no `llm_agent.py`

O agente estratégico implementado usa uma abordagem híbrida, **calibrada pela
regra de pontuação do Game Master**:

- **LLM para decisões semânticas**: a LLM ranqueia cartas candidatas e gera dicas associativas.
- **Dica em banda Dixit (narrador)**: o narrador só pontua quando *alguns mas não todos* acertam. A dica é classificada como vaga / calibrada / óbvia usando as cartas da própria mão como iscas; fora da banda, há **uma única** tentativa corretiva de geração antes do fallback temático, respeitando um orçamento de tempo para não estourar o `a2a_timeout`.
- **Blefe (não-narrador)**: em `select_card_by_clue`, o desempate prefere o match semântico mais forte — a carta com maior chance de ser confundida com a do narrador — para maximizar os votos recebidos (único canal de pontos incondicional).
- **Heurísticas como apoio**: quando a LLM falha, demora ou devolve uma resposta fora do formato, o agente usa pontuação local baseada em palavras-chave, título, letra truncada e relação com a dica.
- **Fallback robusto**: todas as tools retornam respostas válidas mesmo sem modelo real.
- **Sem overfitting à base local**: a estratégia não usa ids, nomes específicos de músicas, artistas fixos ou regras dependentes do `brazilian_songs.csv`.

### 12.1. Como o agente decide

Quando é narrador, `choose_card()` combina:

- diversidade de palavras-chave;
- tamanho útil da letra truncada;
- força temática;
- ranqueamento da LLM sobre qual carta permite uma dica de dificuldade média.

Em `send_clue()`, o prompt pede uma dica de 2 a 6 palavras, sem copiar verso literal, sem usar título/artista e sem explicação. A resposta é sanitizada para:

- cortar prefixos como `Dica:` ou `Resposta:`;
- limitar a quantidade de palavras;
- rejeitar cópia literal da letra;
- remover palavras do título;
- cair para uma dica temática caso a resposta seja ruim.

Depois da sanitização, a dica passa pela **calibração de banda**: se ficar vaga ou óbvia demais (medido pela margem semântica sobre as cartas da mão), o agente faz no máximo uma regeração corretiva — mais direta se vaga, mais oblíqua se óbvia — e, persistindo fora da banda, usa o fallback temático.

Quando não é narrador, `select_card_by_clue()` escolhe a carta da mão que melhor combina semanticamente com a dica e, no empate, prefere o match mais forte para **blefar** (tornar a própria carta competitiva para receber votos).

Na votação, `vote()` pede para a LLM ranquear as 6 opções e combina esse ranking com uma heurística local. O agente sempre devolve exatamente dois votos válidos, sem votar na própria carta.

### 12.2. Prompts usados

O agente usa prompts curtos para reduzir latência no modelo local:

- ranking de carta narradora em JSON: `{"ranking":[0,1,2,3]}`;
- ranking de cartas por dica em JSON: `{"ranking":[indices em ordem]}`;
- geração de dica em texto puro, com limite de palavras.

Se a LLM não obedecer ao JSON, o agente tenta extrair índices de forma tolerante. Se ainda assim não conseguir, usa o ranking heurístico.

### 12.3. Testes recomendados

Validar a sintaxe:

```bash
python -m py_compile llm_agent.py
```

Rodar testes de pontuação:

```bash
python -m pytest tests
```

Rodar uma partida em modo mock:

```bash
python run_game.py --force-mock
```

Rodar seis agentes estratégicos em modo mock:

```bash
python run_game.py --all-strategic --force-mock
```

Com modelo real:

```bash
python run_game.py --model /caminho/do/Phi-3.5-mini-instruct-Q4_K_M.gguf --llm-max-concurrency 1
```

### 12.4. Dificuldades e soluções

- **Resposta malformada da LLM**: o agente aceita JSON, mapas de score e rankings em texto livre.
- **Dica literal demais**: a sanitização rejeita dicas que apareçam como substring da letra.
- **Latência do modelo local**: prompts curtos, `max_tokens` baixo e cache herdado de `BaseAgent`.
- **Compatibilidade com torneio**: a interface das 5 tools obrigatórias foi preservada.

---

## 13. Rodar no Colab

O arquivo [`nota_secreta_colab.ipynb`](nota_secreta_colab.ipynb) deixa o projeto pronto
para rodar no Google Colab sem precisar configurar nada localmente.

**Como usar:**

1. Abra o notebook no Colab por uma destas formas:
   - pela URL `https://colab.research.google.com/github/DanielTrindade/nota-secreta-a2a/blob/main/nota_secreta_colab.ipynb`; **ou**
   - no Colab: `Arquivo → Abrir notebook → GitHub`, cole a URL do repo.
2. Rode as células na ordem: clonar → instalar → testar → partida em mock.
   O `REPO_URL` na seção 1 já aponta para este repositório.
3. A **seção 5** (modelo real) é opcional: instala o `llama-cpp-python`, baixa um GGUF
   (`Phi-3.5-mini-instruct`, ~2,4 GB) e roda a partida com modelo real. Para acelerar,
   use um runtime com **GPU**.

> O notebook executa o projeto via `!python run_game.py ...` (subprocesso de shell), e não
> via `import`, porque o `run_game.py` usa `asyncio.run(...)`, que conflitaria com o event
> loop já ativo do Colab. Assim o código original é mantido sem nenhuma alteração.

### 13.1. Atualizar o repositório no GitHub

O repositório já está publicado em
`https://github.com/DanielTrindade/nota-secreta-a2a`. Para enviar novas alterações:

```bash
git add -A
git commit -m "mensagem"
git push
```

> Observação: o modelo GGUF (~2,4 GB) **não** faz parte do repositório (está no `.gitignore`).
> No Colab, ele é baixado novamente pela seção 5 do notebook.
