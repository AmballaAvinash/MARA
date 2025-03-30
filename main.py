import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import json
import requests
import weaviate
from weaviate.embedded import EmbeddedOptions
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.tools.render import render_text_description
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# Configure environment - you should set these in your environment
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"


# Create a judge for evaluation (using GPT-4o)
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)


# Define Manager Agent prompt
manager_agent_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Manager Agent overseeing a collaborative research system. Your job is to:
1. Analyze the user's query
2. Decide which agent should handle it:
   - Search Agent: For finding and adding new research papers
   - RAG Agent: For answering questions based on papers in the database
3. Coordinate their activities
4. Synthesize their outputs into a final answer for the user

Use the tools available to you and think step by step."""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Define evaluation prompt
evaluation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an expert evaluator tasked with judging the quality of responses to research questions. 
    
Score each response on a scale of 1-10 based on:
1. Accuracy: How factually correct is the information?
2. Relevance: How well does it address the specific question asked?
3. Comprehensiveness: How complete is the coverage of relevant aspects?
4. Evidence: How well supported are the claims with references?
5. Clarity: How clearly is the information presented?

Provide your overall score and brief justification."""),
    HumanMessage(content="""
Question: {query}

Response to evaluate: {response}

Please evaluate this response.
""")
])

# Function to create a ReAct-style agent
def create_react_agent(llm, prompt):
    def format_tool_to_tool_schema(tool):
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.schema() if hasattr(tool, "args_schema") else {"type": "object", "properties": {}},
            },
        }

    # Format the tools for the LLM
    llm_with_tools = llm.bind_tools(tools, format_tool_to_tool_schema=format_tool_to_tool_schema)
    
    # Logic for handling the agent's decision making
    def agent_step(state):
        input_messages = state["messages"]
        
        # Get the most recent message
        most_recent_message = input_messages[-1]
        
        # If it's already an AI message, we've completed a step
        if isinstance(most_recent_message, AIMessage):
            return {"messages": input_messages}
        
        # Initialize or get agent_scratchpad
        agent_scratchpad = []
        if "agent_scratchpad" in state:
            agent_scratchpad = state["agent_scratchpad"]
        
        # Prompt the LLM
        output = llm_with_tools.invoke({
            "messages": input_messages,
            "agent_scratchpad": agent_scratchpad,
        })
        
        # Check if the LLM wants to use a tool
        if "function_call" in output.additional_kwargs:
            tool_call = output.additional_kwargs["function_call"]
            
            # Add the agent's thinking to scratchpad
            agent_scratchpad.append(output)
            
            # Extract tool name and args
            tool_name = tool_call["name"]
            tool_input = json.loads(tool_call["arguments"])
            
            # Execute the tool
            tool_result = tool_executor.invoke(
                ToolInvocation(
                    tool=tool_name,
                    tool_input=tool_input,
                )
            )
            
            # Format the result and add to scratchpad
            observation_msg = AIMessage(content=str(tool_result))
            agent_scratchpad.append(observation_msg)
            
            # Return the updated state with agent_scratchpad
            return {"messages": input_messages, "agent_scratchpad": agent_scratchpad}
        else:
            # Agent is done - return final answer message
            return {"messages": input_messages + [output]}
    
    # Build the workflow graph
    workflow = StateGraph(state_types={"messages": List, "agent_scratchpad": List})
    
    # Add the main agent node
    workflow.add_node("agent", agent_step)
    
    # Define the entry and exit
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        lambda state: "agent" if "agent_scratchpad" in state else END,
    )
    
    # Compile the workflow
    return workflow.compile()

# Create the agents
search_agent = create_react_agent(openai_o1, search_agent_prompt)
rag_agent_openai = create_react_agent(openai_o1, rag_agent_prompt)
rag_agent_deepseek = create_react_agent(deepseek_r1, rag_agent_prompt)
manager_agent = create_react_agent(openai_o1, manager_agent_prompt)

def evaluate_response(query, response):
    """Evaluate a response using the judge LLM"""
    evaluation = judge_llm.invoke(
        evaluation_prompt.format(
            query=query,
            response=response
        )
    )
    return evaluation.content

# Main interaction function
def process_query(query, model="openai"):
    """Process a research query through the agent system"""
    # Start with the manager agent
    manager_result = manager_agent.invoke({"messages": [HumanMessage(content=f"Research query: {query}")]})
    manager_response = manager_result["messages"][-1].content
    
    # Check if the manager determined we need to search for new papers
    if "search_agent" in manager_response.lower():
        search_result = search_agent.invoke({"messages": [HumanMessage(content=f"Find papers for: {query}")]})
        search_response = search_result["messages"][-1].content
        print(f"Search Agent: {search_response}")
    
    # Use the appropriate RAG agent based on model parameter
    if model.lower() == "openai":
        rag_agent = rag_agent_openai
        model_name = "OpenAI o1"
    else:
        rag_agent = rag_agent_deepseek
        model_name = "DeepSeek R1"
    
    # Get the answer from the RAG agent
    rag_result = rag_agent.invoke({"messages": [HumanMessage(content=query)]})
    rag_response = rag_result["messages"][-1].content
    
    # Evaluate the response
    evaluation = evaluate_response(query, rag_response)
    
    return {
        "query": query,
        "model": model_name,
        "response": rag_response,
        "evaluation": evaluation
    }

# Compare both models
def compare_models(query):
    """Compare OpenAI o1 and DeepSeek R1 on the same query"""
    print(f"Processing query with OpenAI o1...")
    openai_result = process_query(query, "openai")
    
    print(f"Processing query with DeepSeek R1...")
    deepseek_result = process_query(query, "deepseek")
    
    return {
        "query": query,
        "openai_result": openai_result,
        "deepseek_result": deepseek_result
    }

# Example usage
if __name__ == "__main__":
    query = "What are the latest developments in large language model reasoning capabilities?"
    results = compare_models(query)
    
    print("\n=== Results ===")
    print(f"Query: {results['query']}")
    print("\nOpenAI o1 Response:")
    print(results['openai_result']['response'])
    print("\nOpenAI o1 Evaluation:")
    print(results['openai_result']['evaluation'])
    print("\nDeepSeek R1 Response:")
    print(results['deepseek_result']['response'])
    print("\nDeepSeek R1 Evaluation:")
    print(results['deepseek_result']['evaluation'])