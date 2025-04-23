import os
import json
from typing import Dict, Any
import time
from langchain_core.messages import HumanMessage

# Import our agents and components
from db import client
from search_agent import search_agent
from rag_agent import rag_agent_openai, rag_agent_deepseek
from manager_agent import manager_agent
from evaluation import evaluate_response, compare_evaluations

# Configure environment variables
# Uncomment and set these in production


def extract_agent_decision(manager_output: str) -> str:
    """Extract the agent decision from the manager output"""
    # Try to parse JSON from the manager output
    try:
        # # Look for JSON in the string
        # start_idx = manager_output.find('{')
        # end_idx = manager_output.rfind('}') + 1
        
        # if start_idx >= 0 and end_idx > start_idx:
        #     json_str = manager_output[start_idx:end_idx]
        #     decision = json.loads(json_str)
            
        #     if 'agent' in decision:
        #         return decision['agent'].lower()
        
        # If no JSON found, look for keywords
        if "search" in manager_output.lower():
            return "search_agent"
        elif "rag" in manager_output.lower():
            return "rag_agent"
    except:
        pass
    
    # Default to RAG agent if we can't determine
    return "search_agent"

def process_query(query: str, model: str = "openai") -> Dict[str, Any]:
    """Process a research query through the agent system"""
    start_time = time.time()
    print(f"Processing query: '{query}' with {model} model")
    
    # Step 1: Consult manager agent to decide workflow
    print("\nConsulting Manager Agent...")
    state = {
    "messages": [HumanMessage(content=f"Research query: {query}")],
    "agent_scratchpad": [],
    }
    
    # manager_result = manager_agent.adapted_agent({"messages": state["messages"],"agent_scratchpad": state.get("agent_scratchpad", [])})

    manager_result = manager_agent.invoke({"input": state["messages"],"chat_history": []})

    print(manager_result)

    manager_response = manager_result["output"]
    print(f"Manager decision: {manager_response}")
    
    # Extract decision from manager
    decision = extract_agent_decision(manager_response)

    print(decision)
    
    # Step 2: If needed, use search agent to find and store papers
    if decision == "search_agent":
        print("\nUsing Search Agent to find papers...")
        search_result = search_agent.invoke({"input": [HumanMessage(content=f"Find papers for: {query}")]})

        search_response = search_result["output"]
        print(f"Search Agent result: {search_response}")
    
    # Step 3: Use RAG agent to answer the query
    print(f"\nUsing RAG Agent ({model}) to answer query...")
    if model.lower() == "openai":
        rag_agent = rag_agent_openai
        model_name = "OpenAI o1"
    else:
        rag_agent = rag_agent_deepseek
        model_name = "DeepSeek R1"
    
    rag_result = rag_agent.invoke({"input": [HumanMessage(content=query)]})
    rag_response = rag_result["output"]

    print(rag_response)
    
    # Step 4: Evaluate the response
    print("\nEvaluating response...")
    evaluation = evaluate_response(query, rag_response)
    
    # Calculate total processing time
    processing_time = time.time() - start_time
    
    return {
        "query": query,
        "model": model_name,
        "response": rag_response,
        "evaluation": evaluation,
        "processing_time": f"{processing_time:.2f} seconds"
    }

def compare_models(query: str) -> Dict[str, Any]:
    """Compare OpenAI o1 and DeepSeek R1 on the same query"""
    print(f"Comparing models on query: '{query}'")
    
    print(f"\nProcessing query with OpenAI o1...")
    openai_result = process_query(query, "openai")
    
    print(f"\nProcessing query with DeepSeek R1...")
    # deepseek_result = process_query(query, "deepseek")
    deepseek_result =  {"response": "currently not implemented due to API key"}

    
    print("\nComparing evaluations...")
    comparison = compare_evaluations(
        query, 
        openai_result["response"], 
        deepseek_result["response"]
    )
    
    return {
        "query": query,
        "openai_result": openai_result,
        "deepseek_result": deepseek_result,
        "comparison": comparison["meta_evaluation"]
    }

def display_results(results: Dict[str, Any]) -> None:
    """Display the results in a readable format"""
    print("\n" + "="*80)
    print(f"QUERY: {results['query']}")
    print("="*80)
    
    print("\nOPENAI O1 RESPONSE:")
    print("-"*80)
    print(results['openai_result']['response'])
    
    print("\nOPENAI O1 EVALUATION:")
    print("-"*80)
    print(results['openai_result']['evaluation'])
    print(f"Processing Time: {results['openai_result']['processing_time']}")
    
    print("\nDEEPSEEK R1 RESPONSE:")
    print("-"*80)
    print(results['deepseek_result']['response'])
    
    # print("\nDEEPSEEK R1 EVALUATION:")
    # print("-"*80)
    # print(results['deepseek_result']['evaluation'])
    # print(f"Processing Time: {results['deepseek_result']['processing_time']}")
    
    print("\nMODEL COMPARISON:")
    print("-"*80)
    print(results['comparison'])
    print("="*80)

# Example usage
if __name__ == "__main__":
    # Example research query
    query = "What are the latest developments in large language model reasoning capabilities?"
    
    # Run the comparison
    results = compare_models(query)
    
    # Display the results
    display_results(results)