from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel, Field

from agents.model import llm




from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "Chinook.sqlite"

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

schema = db.get_table_info()




sql_prompt = ChatPromptTemplate.from_template("""
You are an expert SQLite developer.

Database Schema:
{schema}

Convert the following question into SQL.

Question:
{question}

Rules:

- Return ONLY SQL.
- Use only tables and columns present in the schema.
- Never invent table names.
- Never invent column names.
- Use valid SQLite syntax.
""")



fix_prompt = ChatPromptTemplate.from_template("""
You are an expert SQLite developer.

The previous SQL query failed.

Database Schema:
{schema}

Original Question:
{question}

Previous SQL:
{sql}

SQLite Error:
{error}

Generate a corrected SQL query.

Rules:

- Return ONLY SQL.
- Use only tables and columns present in the schema.
- Never invent table names.
- Never invent column names.
- Use valid SQLite syntax.
""")



answer_prompt = ChatPromptTemplate.from_template("""
You are a business analyst.

Question:
{question}

SQL Result:
{result}

Write a concise answer.
""")



class SQLQuery(BaseModel):
    query: str = Field(description="A valid SQLite SELECT query")


structured_llm = llm.with_structured_output(SQLQuery)




import sqlparse
from sqlparse.tokens import DML, DDL

def validate_sql(sql: str):
    sql = sql.strip()

    parsed = sqlparse.parse(sql)
    if len(parsed) != 1:
        raise ValueError("Only a single SQL statement is allowed.")

    statement = parsed[0]

    
    stmt_type = statement.get_type()
    if stmt_type != "SELECT":
        raise ValueError(f"Only SELECT statements are allowed, got: {stmt_type}")

    for token in statement.flatten():
        if token.ttype in (DDL,):
            raise ValueError("DDL statements are not allowed.")

    return sql
    



def generate_sql(question: str):

    messages = sql_prompt.invoke(
        {
            "schema": schema,
            "question": question,
        }
    )

    response = structured_llm.invoke(messages)

    return validate_sql(response.query)



def fix_sql(question: str, sql: str, error: Exception):

    messages = fix_prompt.invoke(
        {
            "schema": schema,
            "question": question,
            "sql": sql,
            "error": str(error),
        }
    )

    response = structured_llm.invoke(messages)

    return validate_sql(response.query)



def sql_agent(question: str):

    sql_query = generate_sql(question)

    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):

        try:

            # print("\n" + "=" * 60)
            # print(f"Attempt {attempt + 1}")
            # print("=" * 60)
            # print(sql_query)

            validate_sql(sql_query)

            db.run("EXPLAIN QUERY PLAN " + sql_query)

            result = db.run(sql_query)

            # print("\nSQL Executed Successfully!")

            break

        except Exception as e:

            print("\nSQL Execution Failed!")
            print(e)

            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    "Failed to generate a valid SQL query after multiple attempts."
                )

            print("\nRegenerating SQL...\n")

            sql_query = fix_sql(
                question=question,
                sql=sql_query,
                error=e,
            )

    messages = answer_prompt.invoke(
        {
            "question": question,
            "result": result,
        }
    )

    answer = llm.invoke(messages).content

    return answer