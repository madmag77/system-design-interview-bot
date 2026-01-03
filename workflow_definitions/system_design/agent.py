import sys
import operator
import logging
from io import StringIO
from typing import List, TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

logger = logging.getLogger(__name__)

@tool
def calculate_metrics(script: str) -> str:
    """Run python code to calculate system design metrics.
    
    Useful for calculating QPS, storage requirements, bandwidth, etc.
    The script should print the result to stdout.
    """
    try:
        logger.info(f"Running python script: {script}")
        # Create a local namespace for execution
        local_scope = {}
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        exec(script, {}, local_scope)
        
        sys.stdout = old_stdout
        return redirected_output.getvalue().strip()
    except Exception as e:
        return f"Error executing code: {e}"

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def build_agent_graph(llm):
    """Builds and compiles the LangGraph agent."""
    
    tools = [calculate_metrics]
    llm_with_tools = llm.bind_tools(tools)
    
    def call_model(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()
