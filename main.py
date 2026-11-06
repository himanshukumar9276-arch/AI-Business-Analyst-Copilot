from langchain_core.messages import HumanMessage

from graph.graph_builder import graph


def main():

    print("=" * 60)
    print("Multi-Agent Analysis System")
    print("Type 'exit' to quit.")
    print("=" * 60)

    
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        state = {
            "messages": [
                HumanMessage(content=question)
            ]
        }

        result = graph.invoke(
            state,
            config=config
        )

        print("\nAssistant:")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
    
    
#  streamlit run app.py