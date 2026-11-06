from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from agents.model import llm



class Route(BaseModel):
    destination: Literal["sql", "rag", "web", "llm"] = Field(
        description="The most appropriate agent to answer the user's question."
    )


# LLM that always returns a Route object
router_llm = llm.with_structured_output(Route)




supervisor_prompt = ChatPromptTemplate.from_template("""
You are an expert AI supervisor.

Your ONLY responsibility is to decide which agent should answer the user's question.

Available agents:

1. sql
- Questions about the SQLite database.
- Customers
- Employees
- Albums
- Artists
- Tracks
- Genres
- Invoices
- Sales
- Revenue
- Queries requiring SQL.

2. rag
- Questions about internal company documents.
- Refund Policy
- Privacy Policy
- Employee Handbook
- Vendor Agreement
- Marketing Strategy
- Loyalty Program
- Product Catalog
- Subscription FAQ
- Store Operations Manual

3. web
- Questions requiring internet search.
- Latest news
- Current events
- Recent AI developments
- Stock prices
- Funding news
- Weather
- Information after your knowledge cutoff

4. llm
- General knowledge.
- Greetings.
- Coding questions.
- Explanations.
- Math.
- Anything that doesn't require SQL, RAG or Web Search.

Return ONLY one of:

sql
rag
web
llm

User Question:
{question}
""")



def supervisor(question: str) -> str:

    messages = supervisor_prompt.invoke(
        {
            "question": question
        }
    )

    response = router_llm.invoke(messages)

    return response.destination