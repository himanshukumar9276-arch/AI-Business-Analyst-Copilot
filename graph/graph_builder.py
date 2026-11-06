from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import State
from graph.nodes import (
    supervisor_node,
    sql_node,
    rag_node,
    web_node,
    llm_node,
)

builder = StateGraph(State)

# Nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("sql_node", sql_node)
builder.add_node("rag_node", rag_node)
builder.add_node("web_node", web_node)
builder.add_node("llm_node", llm_node)

# Start
builder.add_edge(START, "supervisor")

# Conditional Routing
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["destination"],
    {
        "sql": "sql_node",
        "rag": "rag_node",
        "web": "web_node",
        "llm": "llm_node",
    },
)

# End
builder.add_edge("sql_node", END)
builder.add_edge("rag_node", END)
builder.add_edge("web_node", END)
builder.add_edge("llm_node", END)

memory = MemorySaver()

graph = builder.compile(checkpointer = memory)