import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate

# Import retriever logic (vector DB + embeddings)
from retriever import build_retriever


# 🔹 Step 1: Load environment variables (.env file)
load_dotenv()


# 🔹 Step 2: Load LLM model from OpenRouter
MODEL = os.getenv("MODEL")

# Initialize LLM
llm = ChatOpenRouter(model=MODEL)


# 🔹 Step 3: Define prompt template
# This ensures:
# - Model ONLY answers from file context
# - Avoids hallucination
prompt = ChatPromptTemplate.from_template(
    """
Answer ONLY using the context below.
If the answer is not in the context, say:
"I don't know based on the file."

Keep the answer short and clear.

Context:
{context}

Question:
{question}
"""
)

# Combine prompt + LLM into a chain
chain = prompt | llm


def main():
    print("📄 File Q&A Bot")
    print("Type 'exit' to quit\n")

    # 🔹 Step 4: Ask user for file path
    file_path = input("Enter file path: ").strip()

    # Build retriever (vector DB + embeddings)
    try:
        retriever = build_retriever(file_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 🔹 Step 5: Chat loop
    while True:
        question = input("\nYou: ").strip()

        # Exit condition
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # 🔹 Step 6: Retrieve relevant chunks from vector DB
        docs = retriever.invoke(question)

        # If nothing relevant found → avoid LLM call
        if not docs:
            print("\nAssistant: I don't know based on the file.")
            continue

        # Convert retrieved documents into a single context string
        context = "\n\n".join(doc.page_content for doc in docs)

        # 🔹 Step 7: Send context + question to LLM
        response = chain.invoke({
            "context": context,
            "question": question
        })

        # Print final answer
        print("\nAssistant:", response.content)


# Entry point of the program
if __name__ == "__main__":
    main()