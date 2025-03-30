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



# Function to create a ReAct-style agent
def create_react_agent(llm, prompt, tools):
    tool_executor = ToolExecutor(tools)


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
        
        # Initialize or get agent_scratchpad (agent scratch pad stores the all the past reasoing and acting steps)
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