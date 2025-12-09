# 🎯 Intelligent Ticket Response System: Agentic RAG Architecture

> **An advanced AI-powered ticket routing and response system leveraging Agentic Retrieval-Augmented Generation (RAG) for intelligent, context-aware support automation**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Traditional RAG vs Agentic RAG](#traditional-rag-vs-agentic-rag)
- [Why LangChain Framework?](#why-langchain-framework)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technical Components](#technical-components)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Cost Optimization Strategy](#cost-optimization-strategy)

---

## 🎯 Overview

This project implements an **Agentic RAG (Retrieval-Augmented Generation)** system designed to automate initial ticket responses in support environments. The system intelligently classifies incoming tickets, retrieves relevant information from domain-specific knowledge bases, and generates contextual, professional responses.

### Problem Statement

Support teams receive numerous tickets daily, many of which can be answered using existing knowledge bases (FAQs). However, manual routing and response generation is:
- **Time-consuming**: Support agents spend significant time on repetitive queries
- **Inconsistent**: Response quality varies between agents
- **Scalability bottleneck**: As ticket volume grows, response times increase

### Solution

An intelligent AI agent that:
- ✅ **Automatically classifies** tickets (Database, DevOps, or Other)
- ✅ **Retrieves** relevant information from domain-specific FAQ knowledge bases
- ✅ **Generates** professional, contextual responses
- ✅ **Falls back** gracefully when no FAQ match is found

---

## 🔄 Traditional RAG vs Agentic RAG

### Traditional RAG

**Traditional RAG** follows a linear, deterministic pipeline:

```
User Query → Vector Search → Retrieve Top-K Documents → LLM Generation → Response
```

**Characteristics:**
- **Fixed workflow**: Always retrieves, then generates
- **No decision-making**: Cannot adapt based on query type
- **Single knowledge base**: Typically searches one vector store
- **Passive retrieval**: Retrieves documents regardless of relevance
- **Limited tool usage**: Cannot use external tools or APIs

**Limitations:**
- Cannot decide when retrieval is necessary
- Cannot choose between multiple knowledge bases
- Cannot adapt strategy based on query complexity
- No ability to use specialized tools for different tasks

### Agentic RAG

**Agentic RAG** introduces **intelligent decision-making** and **tool orchestration**:

```
User Query → Agent Decision → Tool Selection → Execute Tools → Synthesize → Response
```

**Characteristics:**
- **Dynamic workflow**: Agent decides which tools to use and when
- **Intelligent routing**: Can classify and route to appropriate knowledge bases
- **Multi-source retrieval**: Can search multiple specialized knowledge bases
- **Active decision-making**: Chooses retrieval strategy based on query analysis
- **Tool orchestration**: Can use multiple specialized tools (classification, search, fallback)

**Advantages:**
- ✅ **Adaptive**: Adjusts strategy based on ticket content
- ✅ **Multi-domain**: Handles multiple knowledge domains intelligently
- ✅ **Cost-effective**: Can use lightweight classification before expensive LLM calls
- ✅ **Extensible**: Easy to add new tools and knowledge bases
- ✅ **Context-aware**: Makes decisions based on understanding of the query

### Key Differences

| Aspect | Traditional RAG | Agentic RAG |
|--------|----------------|-------------|
| **Workflow** | Linear, fixed | Dynamic, adaptive |
| **Decision-making** | None | Intelligent agent decisions |
| **Knowledge Bases** | Single | Multiple, domain-specific |
| **Tool Usage** | Limited | Rich tool ecosystem |
| **Classification** | Not built-in | Built-in classification tool |
| **Fallback Handling** | Basic | Sophisticated fallback strategies |
| **Cost Control** | Limited | Can optimize with keyword-based classification |

---

## 🛠️ Why LangChain Framework?

### What is LangChain?

**LangChain** is an open-source framework designed for building applications powered by Large Language Models (LLMs). It provides abstractions and tools to simplify the development of LLM applications.

### Why We Chose LangChain

1. **Agent Framework**: Built-in support for AI agents with tool-using capabilities, handling reasoning, tool selection, and execution orchestration
2. **Tool System**: Easy-to-use `@tool` decorator with automatic tool description generation and seamless agent integration
3. **Vector Store Integration**: Native support for FAISS, Pinecone, Chroma with standardized similarity search interfaces
4. **LLM Abstraction**: Unified interface for multiple providers (OpenAI, Groq, Anthropic) with easy model switching
5. **Document Management**: Built-in loaders for CSV, PDF, and other formats with text splitting and metadata management
6. **Production-Ready**: Well-documented, actively maintained, large community, and monitoring integrations

### Alternative Frameworks

While LangChain is our choice, here are other viable options:

1. **LlamaIndex**: Excellent for complex data ingestion and retrieval, but less emphasis on agent orchestration
2. **CrewAI**: Multi-agent collaboration framework with role-based agents, ideal for complex workflows requiring multiple specialized agents
3. **Haystack**: Strong document processing and Q&A systems, but more complex setup and steeper learning curve
4. **Semantic Kernel (Microsoft)**: Great for Microsoft-centric environments, but less mature Python ecosystem
5. **AutoGPT / BabyAGI**: Advanced autonomous agents, but less control for structured workflows
6. **Custom Implementation**: Full control and customization, but requires significant development time and maintenance

### Why LangChain for This Project?

For our ticket response system, LangChain provides the **optimal balance** of:
- ✅ Agent orchestration capabilities
- ✅ Easy tool creation and management
- ✅ Vector store integration
- ✅ Production-ready features
- ✅ Active community support
- ✅ Flexibility to extend with custom components

---

## 🏗️ System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INCOMING TICKET                              │
│                    "My database connection is timing out"            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   LangChain Agent    │
                    │   (Groq LLM)         │
                    │                      │
                    │  Decision Engine     │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │  Tool 1:     │ │  Tool 2:     │ │  Tool 3:     │
        │  Classify    │ │  Check DB    │ │  Check       │
        │  Ticket      │ │  FAQ         │ │  DevOps FAQ  │
        │              │ │              │ │              │
        │ (Keyword-    │ │ (Vector      │ │ (Vector      │
        │  based)      │ │  Search)     │ │  Search)     │
        └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
               │                │                │
               │                ▼                ▼
               │      ┌─────────────────┐ ┌─────────────────┐
               │      │  DB FAQ Vector  │ │ DevOps FAQ      │
               │      │  Store (FAISS)  │ │ Vector Store    │
               │      │                 │ │ (FAISS)         │
               │      │  Embeddings:    │ │                 │
               │      │  all-MiniLM-    │ │ Embeddings:     │
               │      │  L6-v2          │ │ all-MiniLM-     │
               │      └─────────────────┘ │ L6-v2           │
               │                          └─────────────────┘
               │
               ▼
        ┌──────────────┐
        │  Tool 4:     │
        │  Generate    │
        │  Fallback    │
        │  Message     │
        └──────┬───────┘
               │
               ▼
        ┌──────────────────────┐
        │   Agent Synthesizes  │
        │   Final Response     │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   PROFESSIONAL       │
        │   TICKET RESPONSE    │
        └──────────────────────┘
```

### Detailed Component Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         COMPONENT BREAKDOWN                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1. INPUT LAYER                                                       │
│     └─> Raw ticket text (user query)                                  │
│                                                                        │
│  2. AGENT LAYER (LangChain Agent)                                     │
│     ├─> LLM: Groq (gpt-oss-20b)                                      │
│     ├─> System Prompt: Defines agent behavior                         │
│     └─> Reasoning: Decides which tools to use                         │
│                                                                        │
│  3. TOOL LAYER                                                        │
│     ├─> classify_ticket()                                             │
│     │   └─> Keyword-based classification (DB/DevOps/Other)            │
│     │                                                                  │
│     ├─> check_db_faq()                                                │
│     │   ├─> Vector similarity search                                  │
│     │   ├─> FAISS vector store                                        │
│     │   └─> Returns top-K relevant FAQ entries                        │
│     │                                                                  │
│     ├─> check_devops_faq()                                            │
│     │   ├─> Vector similarity search                                  │
│     │   ├─> FAISS vector store                                        │
│     │   └─> Returns top-K relevant FAQ entries                        │
│     │                                                                  │
│     └─> generate_fallback_message()                                   │
│         └─> Creates helpful response when no FAQ match                │
│                                                                        │
│  4. KNOWLEDGE BASE LAYER                                              │
│     ├─> DB FAQ Vector Store                                           │
│     │   ├─> Source: db_faq.csv                                        │
│     │   ├─> Embeddings: Sentence Transformers                         │
│     │   └─> Storage: FAISS index                                      │
│     │                                                                  │
│     └─> DevOps FAQ Vector Store                                       │
│         ├─> Source: devops_faq.csv                                    │
│         ├─> Embeddings: Sentence Transformers                         │
│         └─> Storage: FAISS index                                      │
│                                                                        │
│  5. OUTPUT LAYER                                                      │
│     └─> Synthesized, professional ticket response                     │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Technical Components

### Understanding the Stack

#### 1. **LangChain Framework**
**What it is:** A Python library that simplifies building LLM applications.

**Why we use it:**
- Provides pre-built components (agents, tools, vector stores)
- Handles complex LLM interactions automatically
- Makes it easy to connect different AI components together

**Beginner analogy:** Think of LangChain as a "toolkit" that provides ready-made building blocks for AI applications, similar to how Django provides building blocks for web applications.

#### 2. **FAISS (Facebook AI Similarity Search)**
**What it is:** A library for efficient similarity search in high-dimensional spaces.

**Why we use it:**
- Fast vector similarity search (finds similar documents quickly)
- Handles large knowledge bases efficiently
- Open-source and widely used

**Beginner analogy:** Like a library's card catalog, but instead of searching by title/author, it searches by "meaning similarity" - finding documents that mean similar things even if they use different words.

#### 3. **Sentence Transformers**
**What it is:** A library that converts text into numerical vectors (embeddings).

**Why we use it:**
- Converts text to numbers that capture meaning
- Model: `all-MiniLM-L6-v2` (lightweight, fast, accurate)
- Enables semantic search (search by meaning, not just keywords)

**Beginner analogy:** Like translating text into a "language of numbers" that computers can understand and compare. Similar meanings get similar number patterns.

#### 4. **Groq LLM**
**What it is:** A fast inference engine for running large language models.

**Why we use it:**
- Very fast response times (optimized hardware)
- Cost-effective for high-volume use
- Supports various open-source models

**Beginner analogy:** Like having a very fast, efficient "brain" that can understand questions and generate intelligent responses quickly.

#### 5. **Vector Embeddings**
**What it is:** Numerical representations of text that capture semantic meaning.

**How it works:**
1. Text is converted to a vector (list of numbers)
2. Similar texts get similar vectors
3. We can measure "distance" between vectors to find similar content

**Example:**
- "database connection" → `[0.2, -0.5, 0.8, ...]`
- "DB link" → `[0.19, -0.48, 0.79, ...]` (similar numbers = similar meaning)

#### 6. **Agent System**
**What it is:** An AI system that can make decisions and use tools.

**How it works:**
1. Receives a ticket
2. **Decides** which tools to use (classification, search, etc.)
3. **Executes** tools in sequence
4. **Synthesizes** results into final response

**Beginner analogy:** Like a smart assistant who can:
- Look at your question
- Decide which reference books to check
- Search those books
- Combine the information
- Give you a complete answer

### Technical Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | LangChain | Agent orchestration and tool management |
| **LLM** | Groq (gpt-oss-20b) | Language understanding and generation |
| **Vector Store** | FAISS | Fast similarity search |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Text-to-vector conversion |
| **Language** | Python 3.8+ | Implementation language |
| **Data Format** | CSV | FAQ knowledge base storage |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/keys))
- FAQ CSV files (or use sample files provided)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Agentic-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install langchain-core langchain-community langchain-huggingface langchain-groq langchain faiss-cpu sentence-transformers python-dotenv typing-extensions
   ```

3. **Set up API key**
   ```bash
   export GROQ_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file:
   ```
   GROQ_API_KEY=your-api-key-here
   ```

4. **Prepare FAQ files**
   - Create `db_faq.csv` with columns: `question`, `answer`
   - Create `devops_faq.csv` with columns: `question`, `answer`
   - Or use the sample files generated in the notebook

5. **Run the notebook**
   - Open `Ticket_Responder_Colab.ipynb` in Google Colab or Jupyter
   - Execute cells in order

---

## 🔄 How It Works

### Step-by-Step Process

1. **Ticket Reception**
   - User submits a ticket: *"My database connection is timing out"*

2. **Agent Activation**
   - LangChain agent receives the ticket
   - Analyzes the query using the LLM

3. **Classification** (Tool 1)
   - Agent calls `classify_ticket()` tool
   - Keyword matching identifies "database" and "connection"
   - Returns: `"db"`

4. **FAQ Search** (Tool 2)
   - Agent calls `check_db_faq()` tool
   - Converts ticket to embedding vector
   - Searches DB FAQ vector store using FAISS
   - Finds similar FAQ entries above similarity threshold
   - Returns: Relevant FAQ entries with relevance scores

5. **Response Synthesis**
   - Agent receives FAQ results
   - LLM synthesizes a professional response
   - Incorporates relevant FAQ information
   - Generates final response

6. **Output**
   - Professional, contextual response delivered to user

### Example Flow

```
Input: "How do I reset a database connection pool?"

Step 1: classify_ticket() → "db"
Step 2: check_db_faq() → Finds relevant FAQ entry
Step 3: Agent synthesizes response using FAQ content
Output: "To reset a database connection pool, you can restart the 
        application server or use the connection pool management 
        interface. For most connection pools (HikariCP, C3P0, etc.), 
        you can call the close() method on the DataSource and recreate 
        it. In production, coordinate with the team to avoid service 
        disruption."
```

---

## ⚙️ Configuration

### Key Parameters

```python
# Similarity threshold for FAQ retrieval
SIMILARITY_THRESHOLD = 0.7  # Higher = more strict (0.0 to 1.0)

# Maximum FAQ results to retrieve
MAX_FAQ_RESULTS = 3  # Number of top results to return

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM configuration
LLM_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0  # Lower = more deterministic
```

### Tuning Recommendations

- **SIMILARITY_THRESHOLD**: 
  - Lower (0.5-0.6): More results, may include less relevant entries
  - Higher (0.7-0.8): Fewer results, higher quality matches
  - Adjust based on your FAQ quality and coverage

- **MAX_FAQ_RESULTS**:
  - 1-2: Faster, less context
  - 3-5: Balanced
  - 5+: More context, slower processing

---

## 💰 Cost Optimization Strategy

### Keyword-Based Classification

**Current Implementation:**
- Uses keyword matching for ticket classification
- **Avoids LLM API calls** for classification step
- Significant cost savings at scale

**How it works:**
```python
# Keyword-based classification (no LLM call)
db_keywords = ["database", "sql", "connection", ...]
devops_keywords = ["deployment", "kubernetes", "docker", ...]

# Simple counting-based classification
if db_score > devops_score:
    return "db"
```

**Cost Impact:**
- Classification: **$0** (local keyword matching)
- FAQ Search: **$0** (local vector search)
- LLM Call: **Only for final synthesis** (1 call per ticket)

### Alternative: LLM-Based Classification

**If you need more accurate classification**, you can replace keyword search with LLM classification:

```python
@tool
def classify_ticket_llm(ticket_content: str) -> str:
    """Classify ticket using LLM for better accuracy."""
    response = llm.invoke(
        f"Classify this ticket as 'db', 'devops', or 'other': {ticket_content}"
    )
    return response.content.lower().strip()
```

**Trade-offs:**
- ✅ **More accurate**: Better understanding of context and nuance
- ❌ **Higher cost**: Additional LLM API call per ticket
- ❌ **Slower**: Adds latency to classification step

### Cost Comparison

| Approach | Classification Cost | Total LLM Calls | Cost per 1000 Tickets* |
|----------|-------------------|-----------------|------------------------|
| **Keyword-based** (Current) | $0 | 1 | ~$0.10 |
| **LLM-based** | ~$0.05 | 2 | ~$0.20 |

*Estimated costs based on Groq pricing. Actual costs may vary.

### When to Use Each Approach

**Use Keyword-Based When:**
- ✅ Ticket categories have clear, distinct keywords
- ✅ Cost optimization is a priority
- ✅ Classification accuracy is acceptable with keywords
- ✅ High-volume ticket processing

**Use LLM-Based When:**
- ✅ Tickets are ambiguous or use domain-specific jargon
- ✅ Classification accuracy is critical
- ✅ Cost is less of a concern
- ✅ Lower volume, higher accuracy requirements

### Hybrid Approach (Recommended for Production)

You can implement a hybrid approach:

```python
@tool
def classify_ticket_hybrid(ticket_content: str) -> str:
    """Hybrid classification: keyword first, LLM fallback."""
    # Try keyword-based first
    keyword_result = classify_ticket_keyword(ticket_content)
    confidence = calculate_confidence(ticket_content, keyword_result)
    
    # Use LLM only if confidence is low
    if confidence < 0.7:
        return classify_ticket_llm(ticket_content)
    
    return keyword_result
```

This approach:
- ✅ Uses fast, free keyword matching for clear cases
- ✅ Falls back to LLM for ambiguous tickets
- ✅ Balances cost and accuracy
- ✅ Optimizes for both speed and quality

---

## 📊 Performance Considerations

### Vector Search Performance

- **FAISS**: Optimized for fast similarity search
- **Index size**: Grows with FAQ entries, but remains fast
- **Search time**: Typically < 10ms for thousands of entries

### LLM Inference Performance

- **Groq**: Optimized for fast inference
- **Response time**: Typically 1-3 seconds per ticket
- **Throughput**: Can handle multiple concurrent requests

### Scalability

- **Horizontal scaling**: Can run multiple agent instances
- **Vector store**: Can be shared across instances
- **Caching**: FAQ results can be cached for common queries

---



