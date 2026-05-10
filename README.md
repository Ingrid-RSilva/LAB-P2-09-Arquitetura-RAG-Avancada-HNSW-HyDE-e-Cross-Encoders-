<div align="center">

# RAG Avançado — HNSW + HyDE + Cross-Encoders

**Laboratório 09 · Arquitetura RAG de Nível de Produção**

Pipeline de Retrieval-Augmented Generation para busca semântica em manuais médicos técnicos, com transformação de queries coloquiais via HyDE, indexação hierárquica via HNSW e re-ranking de precisão via Cross-Encoder.

</div>

---

## Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [O Problema Resolvido](#-o-problema-resolvido)
- [Arquitetura do Pipeline](#-arquitetura-do-pipeline)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Como Executar](#-como-executar)
- [Análise Técnica — HNSW vs KNN](#-análise-técnica--hnsw-vs-knn-exato)
- [Como o HyDE Resolve o Problema](#-como-o-hyde-resolve-o-problema)
- [Bi-Encoder vs Cross-Encoder](#-bi-encoder-vs-cross-encoder)
- [Modelos Utilizados](#-modelos-utilizados)
- [Autores](#-autores)

---

## Sobre o Projeto

Este projeto implementa um pipeline **RAG (Retrieval-Augmented Generation)** avançado voltado para busca semântica em manuais médicos técnicos. O sistema é capaz de receber perguntas em linguagem coloquial de pacientes e encontrar os fragmentos de documentação técnica mais relevantes para embasar uma resposta precisa de um LLM gerador.

> **Disciplina:** Inteligência Artificial Aplicada 
> **Instituição:** iCEV — Centro Universitário 
> **Entrega:** Laboratório 09 · Release `v1.0`

---

## O Problema Resolvido

A **Similaridade de Cosseno pura falha** quando o vocabulário do usuário é diferente do vocabulário dos documentos indexados.

```
Paciente digita: "dor de cabeça latejante e luz incomodando"
Manual técnico contém: "cefaleia pulsátil com fotofobia"
```

Apesar de semanticamente equivalentes, esses textos estão **distantes no espaço vetorial**, pois pertencem a registros linguísticos diferentes. Este pipeline resolve essa lacuna com três camadas de inteligência:

| Camada | Técnica | Problema que resolve |
|--------|---------|---------------------|
| Transformação | **HyDE** | Lacuna semântica coloquial ↔ técnico |
| Recuperação | **HNSW + Bi-Encoder** | Busca escalável em grandes corpora |
| Refinamento | **Cross-Encoder** | Imprecisão do ranking inicial |

---

## Arquitetura do Pipeline

```
 
 QUERY DO USUÁRIO 
 "dor de cabeça latejante e luz 
 incomodando" 
 
 
 
 
 PASSO 2 — HyDE 
 LLM gera um documento hipotético em 
 linguagem técnica → nova âncora vetorial 
 
 Output: "Cefaleia pulsátil unilateral, 
 fotofobia, ICHD-3, triptanos..." 
 
 vetor do documento hipotético
 
 
 PASSO 3 — Bi-Encoder + HNSW 
 Busca aproximada O(log N) no grafo 
 hierárquico → funil largo 
 
 Output: Top-10 candidatos 
 
 10 documentos candidatos
 
 
 PASSO 4 — Cross-Encoder Re-ranking 
 Atenção bidirecional completa sobre o par 
 (query original, documento) → funil fino 
 
 Output: Top-3 documentos finais 
 
 
 
 Contexto injetado no LLM gerador
 → Resposta final ao usuário
```

---

## Estrutura do Repositório

```
lab09-rag-avancado/
 rag_pipeline.py # Pipeline completo em Python puro (com Groq integrado)
 lab09_colab.ipynb # Notebook autocontido para Google Colab
 requirements.txt # Dependências do projeto
 .env.example # Modelo do arquivo de variáveis de ambiente
 .gitignore # Ignora o .env para proteger a chave de API
 README.md # Documentação (este arquivo)
```

---

## Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|-----------|--------|-----|
| [Python](https://python.org) | 3.10+ | Linguagem principal |
| [FAISS](https://github.com/facebookresearch/faiss) | ≥ 1.7.4 | Índice vetorial HNSW |
| [sentence-transformers](https://www.sbert.net/) | ≥ 2.7.0 | Bi-Encoder e Cross-Encoder |
| [NumPy](https://numpy.org/) | ≥ 1.24.0 | Operações matriciais |
| [PyTorch](https://pytorch.org/) | ≥ 2.0.0 | Backend dos modelos |
| [Groq](https://console.groq.com) | ≥ 0.9.0 | API para geração do documento hipotético (HyDE) |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | ≥ 1.0.0 | Leitura segura da chave de API (execução local) |

---

## Como Executar

### Opção 1 — Google Colab (recomendado)

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Clique em **Arquivo → Fazer upload de notebook**
3. Selecione o arquivo `lab09_colab.ipynb`
4. No menu lateral, clique no ícone de **cadeado (Secrets)**
5. Adicione um novo secret: **Name:** `GROQ_API_KEY` · **Value:** sua chave do [Groq](https://console.groq.com)
6. Ative o toggle **Notebook access** e execute as células de cima para baixo

> O notebook instala todas as dependências automaticamente na primeira célula. A chave da API nunca aparece no código — é lida com segurança via Secrets do Colab.

---

### Opção 2 — Execução Local

**Pré-requisitos:** Python 3.10+ instalado.

**1. Clone o repositório**
```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO
```

**2. Crie e ative um ambiente virtual**
```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

**3. Instale as dependências**
```bash
pip install -r requirements.txt
```

**4. Configure a chave da API**

Renomeie o arquivo `.env.example` para `.env` e preencha com sua chave do Groq:
```
GROQ_API_KEY=sua_chave_aqui
```

> Obtenha sua chave gratuita em [console.groq.com](https://console.groq.com). O arquivo `.env` está no `.gitignore` e nunca será enviado ao GitHub.

**5. Execute o pipeline**
```bash
python rag_pipeline.py
```

**Saída esperada:**
```

 PIPELINE RAG AVANÇADO — HNSW + HyDE + Cross-Encoder


 Query do usuário: "dor de cabeça latejante e luz incomodando"

...

 DOCUMENTOS FINAIS INJETADOS NO CONTEXTO DO LLM (Top-3)
 [1] Manual de Neurologia Clínica – Cap. 3 (CE Score: 8.4231)
 [2] Manual de Oftalmologia – Sintomas Associados (CE Score: 6.1872)
 [3] Farmacologia Clínica – Analgésicos (CE Score: 5.9103)
```

---

## Análise Técnica — HNSW vs KNN Exato

### O que é o HNSW?

O **Hierarchical Navigable Small World (HNSW)** é uma estrutura de dados em grafo multicamada inspirada no fenômeno *Small World* (seis graus de separação). Cada camada do grafo é progressivamente mais esparsa — as camadas superiores funcionam como "autoestradas" para busca macro, enquanto a camada base contém a vizinhança densa para precisão fina.

A busca percorre do topo para a base em complexidade **O(log N)**, enquanto o KNN exato é **O(N × d)**.

### Hiperparâmetros e Impacto na RAM

#### `M` — Conexões por Nó

Controla o número de arestas bidirecionais que cada nó mantém no grafo. Cada nó armazena até `2 × M` ponteiros nas camadas superiores e `M` ponteiros na camada base.

| Valor de `M` | RAM | Recall | Velocidade de Build |
|:---:|:---:|:---:|:---:|
| 4 – 8 | Baixa | Médio | Muito rápida |
| 16 | Moderada | Alto | Rápida |
| **32** ← *usado* | **Moderada-Alta** | **Muito Alto** | **Moderada** |
| 64+ | Alta | Muito Alto | Lenta |

**Fórmula de RAM:**
```
RAM_HNSW ≈ (N × d × 4 bytes) ← vetores float32
 + (N × M × 2 × 8 bytes) ← ponteiros do grafo
```

Exemplo com nosso corpus (N = 22, d = 384, M = 32):
```
Vetores : 22 × 384 × 4 = 33.792 bytes (~33 KB)
Ponteiros : 22 × 32 × 2 × 8 = 11.264 bytes (~11 KB)
Total HNSW ≈ 45 KB
```

#### `ef_construction` — Qualidade do Grafo

Define o tamanho da fila dinâmica de candidatos durante a fase de **indexação**. Não impacta a RAM em produção — apenas o tempo e a qualidade do grafo construído.

| `ef_construction` | Qualidade do Grafo | Tempo de Build |
|:---:|:---:|:---:|
| 40 | Baixa | Muito rápido |
| 100 | Boa | Rápido |
| **200** ← *usado* | **Muito boa** | **Moderado** |
| 500 | Excelente | Lento |

> **Regra prática:** `ef_construction ≥ M × 2`. No nosso caso: `200 ≥ 64`. 

#### `ef_search` — Recall na Busca

Controla a fila dinâmica **durante a busca**. Pode ser ajustado em runtime sem re-indexar o corpus.

```python
index.hnsw.efSearch = 50 # sempre >= k (número de resultados desejados)
```

### Comparação de Escala: HNSW vs KNN Exato

| Corpus | KNN Exato (RAM) | HNSW M=32 (RAM) | Overhead | Ganho de Velocidade |
|--------|:---------:|:----------:|:--------:|:-------------------:|
| 10K docs, d=384 | ~14,6 MB | ~17 MB | +16% | ~100× |
| 100K docs, d=384 | ~146 MB | ~170 MB | +16% | ~1.000× |
| 1M docs, d=384 | ~1,44 GB | ~1,70 GB | +18% | ~10.000× |
| 10M docs, d=768 | ~28,6 GB | ~34 GB | +19% | Inviável vs HNSW |

> **Conclusão:** O HNSW consome apenas ~15–20% a mais de RAM que os vetores brutos, mas elimina a comparação exaustiva contra todos os N documentos a cada query. Para corpora acima de 100K documentos, o KNN exato se torna impraticável em produção; o HNSW escala linearmente em memória e logaritmicamente em tempo.

---

## Como o HyDE Resolve o Problema

### O Problema da Lacuna Semântica

O espaço vetorial aprende padrões co-ocorrenciais. Termos como *"cefaleia"* e *"fotofobia"* co-ocorrem frequentemente em textos médicos, ficando próximos nesse espaço. Termos populares como *"dor de cabeça"* e *"luz incomodando"* ficam em uma região completamente diferente.

```
Espaço vetorial (representação 2D simplificada):

 Região técnica: Região coloquial:
 "cefaleia pulsátil" "dor de cabeça latejante"
 "fotofobia"
 "migrânea ICHD-3" ← grande distância de cosseno →
 "triptanos"
```

### A Solução HyDE (Hypothetical Document Embeddings)

Em vez de vetorizar a query do usuário diretamente, pedimos ao LLM para **gerar uma resposta técnica hipotética** — como se fosse um trecho do próprio manual. Esse documento falso estará geometricamente próximo dos documentos reais.

```
"dor de cabeça latejante e luz incomodando"
 
 LLM gera resposta técnica hipotética
 
"Cefaleia pulsátil unilateral, fotofobia e fonofobia,
 consistente com enxaqueca (ICHD-3). Triptanos indicados."
 
 vetor agora está próximo dos docs reais 
 
 Busca no HNSW → documentos relevantes encontrados
```

> O documento hipotético **não precisa ser factualmente correto**. Ele só precisa habitar o mesmo espaço semântico dos documentos indexados para servir como âncora geométrica.

---

## Bi-Encoder vs Cross-Encoder

| Característica | Bi-Encoder | Cross-Encoder |
|---|---|---|
| **Arquitetura** | Dois encoders independentes | Um encoder para o par completo |
| **Input** | Query e Doc separados | `[CLS] Query [SEP] Doc` |
| **Atenção cruzada** | Não há | Bidirecional completa |
| **Velocidade** | Muito rápida (vetores pré-computáveis) | Lenta (par a par, sem pré-computação) |
| **Recall** | ~80–90% | ~95–99% |
| **Uso no pipeline** | Funil largo — Top-10 | Funil fino — Top-3 |

### Estratégia de Funil em Dois Estágios

O pipeline combina o melhor dos dois mundos:

```
Bi-Encoder + HNSW
 → varredura rápida de todo o corpus
 → entrega Top-10 com alto recall (poucos falsos negativos)

 ↓

Cross-Encoder
 → analisa cada par (query, doc) com atenção total
 → re-rankeia com alta precisão
 → entrega Top-3 com mínimos falsos positivos
```

---

## Modelos Utilizados

| Componente | Modelo | Parâmetros | Uso |
|---|---|:---:|---|
| Bi-Encoder | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M | Geração de embeddings e busca |
| Cross-Encoder | [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | 22M | Re-ranking de precisão |
| Índice Vetorial | FAISS `IndexHNSWFlat` | — | Busca ANN escalável |
| LLM (HyDE) | `llama-3.3-70b-versatile` via [Groq](https://console.groq.com) | 70B | Geração do documento hipotético |

---

## Changelog

### v1.0 — Release inicial
- Corpus com 22 fragmentos de manuais médicos técnicos
- Indexação HNSW via FAISS (`M=32`, `ef_construction=200`)
- HyDE integrado com LLM real via Groq (`llama-3.3-70b-versatile`)
- Chave de API protegida via Secrets do Colab e `.env` local
- Recuperação Top-10 via Bi-Encoder
- Re-ranking Top-3 via Cross-Encoder
- Script `.py` para execução local com `python-dotenv`
- Notebook `.ipynb` autocontido para Google Colab
- `.env.example` e `.gitignore` incluídos

---

## Nota de IA

> Partes geradas/complementadas com IA, revisadas por Ingrid.
