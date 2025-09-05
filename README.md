## -------------------------

# README — Local RAG with Ollama

*A beginner-friendly and technical introduction to see AI working in practice*

Welcome. This repository brings a simple and didactic example of a local RAG system. The idea is to show, step by step and without mystery, how a text can be split, transformed into vectors (embeddings), searched by similarity, and used as context to generate intelligent answers. It is a starting point for learning and experimentation, not a production-ready solution.

---

## What this project is and why it exists

Here you will find Python code that implements a **local RAG**. RAG means *Retrieval Augmented Generation*, which is just a way of saying that a text generator (a Large Language Model) uses relevant documents found by search to provide better answers. In this project everything runs locally with a server called **Ollama**, which provides embeddings and text generation from models installed on your machine.

The focus is educational: to understand the building blocks that make up an AI application without depending on cloud services or complex frameworks.

---

## What is Ollama?

**Ollama** is a local server that lets you run LLMs (Large Language Models) and embedding models on your own computer. Instead of sending data to a remote API, you run Ollama locally, ask it to generate embeddings or answers, and it returns the results instantly.

Official site: [https://ollama.com](https://ollama.com)

---

## Key concepts explained

Before running the code, it’s worth understanding the main terms:

* **Embedding**: A numerical representation of text. The program converts each chunk of text into a vector of numbers so that it can compare similarity between queries and documents.
* **Vector search**: Once texts are represented as vectors, we can calculate which chunks are most similar to a query.
* **LLM (Large Language Model)**: A model capable of generating human-like text. In this project, it uses context (retrieved chunks) to produce more accurate answers.
* **RAG (Retrieval Augmented Generation)**: A combination of retrieval (finding the right text pieces) with generation (producing a response). This approach improves factual accuracy.
* **Indexing**: Saving embeddings and text chunks so you don’t have to recompute them every time.

---

## Types of LLMs

Not all LLMs are the same. They can serve different purposes:

* **Text-only LLMs**: These models are optimized to generate or continue text, such as GPT-2 or LLaMA. They are usually best for pure text completion or story generation.
* **Conversational LLMs**: These are adapted for dialogue, with fine-tuning to handle multi-turn conversations, instructions, and context. Examples include ChatGPT (based on GPT-3.5/4) and Claude. They are better at staying consistent in a conversation and following user intent.
* **Embedding models**: Instead of generating text, they convert text into vectors for similarity search, clustering, or classification (e.g., `nomic-embed-text` or `text-embedding-ada-002`).
* **Instruction-tuned models**: Specialized in following commands, like `gemma:it` or `mistral-instruct`.
* **Multimodal models**: Capable of working with text and other formats (images, audio, etc.), such as GPT-4 Vision or LLaVA.

👉 To explore more about LLMs and their ecosystem, check Hugging Face’s learning hub: [https://huggingface.co/learn/nlp-course/chapter1](https://huggingface.co/learn/nlp-course/chapter1)

---

## Parameters and Context Window

When evaluating or choosing an LLM, two important aspects matter:

* **Parameters**: These are the weights of the model, essentially the learned "knowledge." A model with 7B parameters (like LLaMA-7B) is usually more capable than one with 1B, but also requires more memory and compute. Parameters are often read directly from the model card or description, e.g., “Mistral-7B” → 7 billion parameters.
* **Context Window**: This defines how much text (tokens) the model can “see” at once. For example, 2k tokens ≈ a few pages of text, while 128k tokens ≈ an entire book.

  * **Advantages of a larger window**: The model can handle longer conversations and documents without forgetting earlier context.
  * **Disadvantages**: Requires more VRAM or RAM, and inference can be slower.
    In short: more parameters = more “knowledge,” larger context window = longer memory, but both increase hardware requirements.

---

## Why these models?

In this project, we use:

* **`nomic-embed-text`** → A lightweight embedding model optimized for fast and efficient text-to-vector transformation. It is ideal for building the vector database for similarity search in RAG.
* **`gemma3:1b-it-qat`** → A small instruction-tuned conversational model (1B parameters, quantized for efficiency). It is compact enough to run on modest hardware while still being able to follow instructions and generate coherent answers.

This combination makes the project accessible: embeddings are efficient, and the LLM is small but usable on consumer-level machines.

---

## Minimum requirements

* Windows 10 or 11
* Quad-core processor
* 8 GB of RAM (4 GB might work with very small models, but 8 GB is safer)
* 10 GB of free disk space for code, models, and indexes
* Python 3.10 or higher
* Ability to install Ollama and pull local models

---

## Quick Start — From zero to running

Here is a complete example flow, assuming you just cloned the repository and are on Windows.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks execution due to policy, allow it temporarily:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install and run Ollama

Follow the official instructions to install Ollama on Windows. Once installed, start the server:

```powershell
ollama serve
```

### 4. Download the models used by the code

```powershell
ollama pull nomic-embed-text
ollama pull gemma3:1b-it-qat
```

### 5. Run the program

```powershell
python rag_interaction.py
```

You will see a menu like this:

```
Local RAG - Ollama (readable version)
1) Create index from file
2) Load saved index
3) Interactive search
4) RAG chat
0) Exit
```

---

## First experiment step by step

### Step A — Ingest the example document

Choose option **1** and type:

```
document.txt
```

The program will fragment the file, generate embeddings, and save them into the folder `rag_db`.

### Step B — Load the index

Choose option **2**. The program will confirm the index is loaded.

### Step C — Do a similarity search

Choose option **3**. Example:

```
Search> Who developed the ENIAC?
```

Expected output (approximate):

```
[1] id=4 score=0.7562 chars=302
...ENIAC was developed for the U.S. government at the University of Pennsylvania by John Presper Eckert, Jr. and John W. Mauchly...

[2] id=10 score=0.8049 chars=390
...the ENIAC programmers consisted of six women, called "computers." It was in operation for ten years...
```

### Step D — Try the chat mode

Choose option **4**. Example:

```
Question> What was the Colossus computer and why was it developed?
```

Expected output (approximate):

```
Answer:
The Colossus was built in England in 1943 to help decode German messages during World War II. It supported the British Intelligence effort against the Enigma machine and is considered one of the first electronic computers.
```

---

## What the code does

The `LocalRAG` class organizes everything a RAG needs:

* **Fragmentation**: splits the text into chunks with overlap.
* **Embeddings**: generates numerical representations of the chunks.
* **Indexing**: saves chunks and embeddings for reuse.
* **Search**: compares the query embedding with the chunks’ embeddings.
* **MMR**: selects relevant and diverse chunks, avoiding repetition.
* **Generation**: sends the most relevant chunks as context to the language model for a final answer.

---

## Limitations

This project is educational. It is not designed for production and does not include layers such as authentication, sensitive data handling, complete logging, or scalability. It also simplifies parameters like chunk size and indexing strategy. Use it to learn, not to build real-world systems.

---

## Glossary

* **Embedding**: Numeric vector representing the meaning of a text.
* **LLM (Large Language Model)**: A model trained on massive amounts of text capable of generating and reasoning with natural language.
* **RAG**: Retrieval Augmented Generation, a technique to combine search with generation.
* **Vector similarity**: A mathematical measure to find how close two texts are, based on their embeddings.
* **Index**: A saved structure (files) that keeps embeddings and text for reuse.
* **Chunk**: A fragment of text split from a larger document.
* **MMR (Maximal Marginal Relevance)**: A method to select results that are both relevant and diverse.
* **Parameters**: The learned weights of an LLM that define its capability.
* **Context Window**: The maximum number of tokens the model can handle in one input.

---

## Conclusion

This repository is a lab to experiment with the concepts behind modern AI text applications. It shows the path from fragmenting a document to using relevant chunks to help a model respond better. It is a simple, clear, and useful base for beginners to understand how a local RAG works and an excellent first step to experimenting with **AI + Python locally**.

---
