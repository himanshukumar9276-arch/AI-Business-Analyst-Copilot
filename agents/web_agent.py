from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

import config
from agents.model import llm



tool = TavilySearch(
    max_results=3,
    api_key=config.TAVILY_API_KEY
)

answer_prompt = ChatPromptTemplate.from_template("""
You are an expert business research analyst.

User Question:
{question}

Web Search Results:
{results}

Instructions:

- Answer using ONLY the search results.
- Summarize the important information.
- If multiple sources agree, combine them.
- Be concise.
""")



def validate_results(results):

    if not results:
        raise ValueError("No search results found.")

    return results


def web_agent(question: str):

    results = tool.invoke(question)

    validate_results(results)

    messages = answer_prompt.invoke(
        {
            "question": question,
            "results": results
        }
    )

    answer = llm.invoke(messages).content

    return answer