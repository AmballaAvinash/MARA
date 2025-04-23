# Multi-Agent Research Assistant

A collaborative agent system that helps users find and analyze research papers using a combination of search, RAG (Retrieval-Augmented Generation), and evaluation capabilities.

## System Architecture

The system consists of multiple agents working together:

1. **Manager Agent**: Coordinates the workflow and decides which agent to use
2. **Search Agent**: Finds papers from external sources (e.g., Arxiv) and adds them to the database
3. **RAG Agent**: Answers questions based on papers in the database
4. **Evaluation Agent**: Evaluates and compares responses

## Features

- **Agentic RAG**: Uses ReAct approach for iterative reasoning and retrieval
- **LLM-powered search**: Intelligent paper discovery and indexing
- **Model Comparison**: Compare answers from different LLMs (OpenAI o1 and DeepSeek R1)
- **Evaluation**: Automatic quality assessment of responses

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Set up environment variables:
```
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

3. Run the application:

```
docker run -d -p 8080:8080 --name weaviate semitechnologies/weaviate:latest
```


```
python main.py
```

## Usage

The main function `compare_models(query)` takes a research question and returns responses from both OpenAI and DeepSeek models, along with evaluations.

Example:
```python
from main import compare_models, display_results

query = "What are the latest developments in large language model reasoning capabilities?"
results = compare_models(query)
display_results(results)
```

## Components

- `db.py`: Database connection and schema setup
- `search_agent.py`: Agent for finding and storing papers
- `rag_agent.py`: Agent for answering questions using stored papers
- `manager_agent.py`: Agent for coordinating workflow
- `evaluation.py`: Functions for evaluating responses
- `main.py`: Main application logic

## Dependencies

- LangChain
- LangGraph
- Weaviate (for vector database)
- Various LLMs (OpenAI, DeepSeek)