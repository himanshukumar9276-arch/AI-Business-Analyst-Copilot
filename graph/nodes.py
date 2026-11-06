from langchain_core.messages import AIMessage

from agents.model import llm
from agents.cust_sql_agent import sql_agent
from agents.rag_agent import rag_agent
from agents.web_agent import web_agent
from graph.supervisior import supervisor


def supervisor_node(state):
    question = state["messages"][-1].content

    destination = supervisor(question)

    return {
        "destination": destination
    }


def sql_node(state):
    question = state["messages"][-1].content

    answer = sql_agent(question)

    return {
        "messages": [
            AIMessage(content=answer)
        ]
    }


def rag_node(state):
    question = state["messages"][-1].content

    answer = rag_agent(question)

    return {
        "messages": [
            AIMessage(content=answer)
        ]
    }


def web_node(state):
    question = state["messages"][-1].content

    answer = web_agent(question)

    return {
        "messages": [
            AIMessage(content=answer)
        ]
    }


def llm_node(state):
    
    answer = llm.invoke(state["messages"]).content

    return {
        "messages": [
            AIMessage(content=answer)
        ]
    }