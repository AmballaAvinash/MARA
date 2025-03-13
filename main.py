# main.py
import os
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import faiss
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import Document
import requests

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
VECTOR_DB_PATH = "./data/vector_store"
PAPERS_PATH = "./data/papers"

# Setup directories
Path(VECTOR_DB_PATH).mkdir(exist_ok=True, parents=True)
Path(PAPERS_PATH).mkdir(exist_ok=True, parents=True)

# Data models
class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None

class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    authors: List[Author]
    full_text: Optional[str] = None
    pdf_url: Optional[str] = None
    embedding_id: Optional[str] = None
    
    @classmethod
    def create(cls, title, abstract, authors, **kwargs):
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            abstract=abstract,
            authors=[Author(**a) if isinstance(a, dict) else a for a in authors],
            **kwargs
        )

# Database Manager
class DatabaseManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.load_vector_store()
    
    def load_vector_store(self):
        try:
            self.vector_store = FAISS.load_local(VECTOR_DB_PATH, self.embeddings)
            print("Vector store loaded successfully")
        except:
            self.vector_store = FAISS.from_texts(["Initialization text"], self.embeddings)
            self.vector_store.save_local(VECTOR_DB_PATH)
            print("New vector store initialized")
    
    def add_paper(self, paper: Paper):
        # Save paper to disk
        paper_path = Path(PAPERS_PATH) / f"{paper.id}.json"
        with open(paper_path, 'w') as f:
            f.write(paper.json())
        
        # Add to vector store
        doc = Document(
            page_content=f"{paper.title}\n{paper.abstract}",
            metadata={"paper_id": paper.id}
        )
        self.vector_store.add_documents([doc])
        self.vector_store.save_local(VECTOR_DB_PATH)
        return paper.id
    
    def search_papers(self, query: str, k: int = 5):
        docs = self.vector_store.similarity_search(query, k=k)
        results = []
        for doc in docs:
            paper_id = doc.metadata.get("paper_id")
            if paper_id:
                paper = self.get_paper(paper_id)
                if paper:
                    results.append((paper, doc.metadata.get("score", 0.0)))
        return results
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        paper_path = Path(PAPERS_PATH) / f"{paper_id}.json"
        if paper_path.exists():
            with open(paper_path, 'r') as f:
                return Paper.parse_raw(f.read())
        return None

# Search Agent
class SearchAgent:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def search_arxiv(self, query: str, max_results: int = 5):
        # Simple ArXiv API implementation
        base_url = "http://export.arxiv.org/api/query?"
        search_query = f"search_query=all:{query.replace(' ', '+')}&start=0&max_results={max_results}"
        response = requests.get(base_url + search_query)
        
        # Basic parsing (in production, use a proper XML parser)
        import re
        papers = []
        entries = re.findall(r'<entry>(.*?)</entry>', response.text, re.DOTALL)
        
        for entry in entries:
            title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            abstract_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            author_matches = re.findall(r'<author>(.*?)</author>', entry, re.DOTALL)
            
            if title_match and abstract_match:
                title = title_match.group(1).strip()
                abstract = abstract_match.group(1).strip()
                
                authors = []
                for author_text in author_matches:
                    name_match = re.search(r'<name>(.*?)</name>', author_text)
                    if name_match:
                        authors.append(Author(name=name_match.group(1).strip()))
                
                paper = Paper.create(
                    title=title,
                    abstract=abstract,
                    authors=authors
                )
                
                # Add to database
                self.db_manager.add_paper(paper)
                papers.append(paper)
        
        return papers
    
    def execute_search(self, query: str):
        # First check if we already have relevant papers
        existing_results = self.db_manager.search_papers(query, k=3)
        
        # If we have enough results, return them
        if len(existing_results) >= 3:
            return [paper for paper, _ in existing_results]
        
        # Otherwise, search for new papers
        new_papers = self.search_arxiv(query)
        return new_papers

# RAG Agent
class RAGAgent:
    def __init__(self, db_manager: DatabaseManager, search_agent: SearchAgent):
        self.db_manager = db_manager
        self.search_agent = search_agent
        self.openai_model = OpenAI(temperature=0)
    
    def answer_query(self, query: str):
        # First try to find relevant papers in our database
        papers = self.db_manager.search_papers(query, k=3)
        
        # If not enough papers found, search for more
        if len(papers) < 2:
            new_papers = self.search_agent.execute_search(query)
            # Get the papers again after adding new ones
            papers = self.db_manager.search_papers(query, k=3)
        
        # Prepare context from papers
        context = ""
        for paper, _ in papers:
            context += f"Title: {paper.title}\nAbstract: {paper.abstract}\n\n"
        
        # Generate answer with OpenAI
        prompt = f"""
        Based on the following research papers, please answer the query: "{query}"
        
        {context}
        
        Provide a comprehensive answer with reasoning based only on the information in these papers.
        """
        
        return self.openai_model.predict(prompt)

# Evaluation Agent
class EvaluationAgent:
    def __init__(self):
        self.openai_model = OpenAI(temperature=0)
    
    def evaluate_response(self, query: str, response: str, papers: List[Paper]):
        # Prepare context from papers
        context = ""
        for paper in papers:
            context += f"Title: {paper.title}\nAbstract: {paper.abstract}\n\n"
        
        # Create evaluation prompt
        eval_prompt = f"""
        You are an evaluation agent assessing the quality of an AI response to a research query.
        
        Query: "{query}"
        
        Response to evaluate:
        {response}
        
        The response was generated based on these research papers:
        {context}
        
        Please evaluate the response on a scale of 1-10 for the following criteria:
        1. Relevance to the query
        2. Accuracy of information
        3. Reasoning quality
        4. Completeness
        
        Provide a brief justification for each score and an overall assessment.
        """
        
        return self.openai_model.predict(eval_prompt)

# Main application
def main():
    print("Initializing MARA - Multi-Agent Research Assistant")
    
    # Initialize components
    db_manager = DatabaseManager()
    search_agent = SearchAgent(db_manager)
    rag_agent = RAGAgent(db_manager, search_agent)
    evaluation_agent = EvaluationAgent()
    
    # Example usage
    query = "What are the latest advances in transformer architectures for NLP?"
    
    print(f"Processing query: {query}")
    answer = rag_agent.answer_query(query)
    
    print("\nAnswer:")
    print(answer)
    
    # Get papers for evaluation
    papers = [paper for paper, _ in db_manager.search_papers(query)]
    evaluation = evaluation_agent.evaluate_response(query, answer, papers)
    
    print("\nEvaluation:")
    print(evaluation)

if __name__ == "__main__":
    main()
