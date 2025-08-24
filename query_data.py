from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from yaspin import yaspin


from get_embedding_function import get_embedding_function

WIDTH = 60
CHROMA_PATH = "chroma"
MODEL = "qwen2:7b"

RAG_PROMPT = """
You are a helpful assistant. Use the retrieved context and prior conversation
to answer the user's latest question. Be concise and cite sources if relevant.

Context:
{context}

Conversation so far:
{history}

User's question:
{question}

Answer:
"""

REWRITE_PROMPT = """
You are a query rewriter. Your job is to rewrite ONLY the user's *latest question*
into a standalone, fully self-contained query that can be used for document retrieval.

Use prior questions *only as background context*, but always make sure the 
rewritten query reflects the LATEST user question — not earlier ones.

Prior questions (for background only):
{history}

LATEST user question: {question}

FOR EXAMPLE:
Conversation: 
User: Who wrote The Hobbit?
Assistant: J.R.R. Tolkien.
Latest question: When was it published?
Rewritten query: When was The Hobbit published?

Rewritten standalone query:
"""


def build_prompt(history, context, question):
    history_str = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" for h in history])
    prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT)
    return prompt_template.format(context=context, history=history_str, question=question)


def rewrite_query(model, history, question):
    # Only use the last few user questions to alter query
    short_history = history[-2:]
    history_str = "\n".join([f"User: {h['question']}" for h in short_history])
    prompt_template = ChatPromptTemplate.from_template(REWRITE_PROMPT)
    prompt = prompt_template.format(history=history_str, question=question)

    return model.invoke(prompt).strip()


def run_chat():
    # Prepare the DB once
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = OllamaLLM(model=MODEL)

    history = []

    print()
    print('=' * WIDTH)
    print("💬 Agentic RAG Chat")
    print('=' * WIDTH)
    print()
    print("(Type 'exit' to quit.)")
    print()

    while True:
        query_text = input("You: ").strip()
        
        print()

        if query_text.lower() in ["exit", "quit", "q"]:
            print('=' * WIDTH)
            print("👋 Goodbye!")
            print('=' * WIDTH)
            break

        with yaspin(text="Optimizing query...", color="cyan") as sp:
            # Step 1: Rewrite query for retrieval
            retrieval_query = rewrite_query(model, history, query_text)

            sp.write("")
            sp.write('+' * 50)
            sp.write("Optimized Question:")
            sp.write(retrieval_query)
            sp.write('+' * 50)
            sp.write("")

            sp.text = "Retriving Information..."

            # Step 2: Retrieve context with rewritten query
            results = db.similarity_search_with_score(retrieval_query, k=5)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            sources = [doc.metadata.get("id", None) for doc, _ in results]

            # Step 3: Build final answer prompt
            prompt = build_prompt(history, context_text, query_text)

            # print("Final Prompt")
            # print(prompt)
            # print("=" * 50)

            # Step 4: Model generates answer
            response_text = model.invoke(prompt)

            # Step 5: Save turn
            history.append({"question": query_text, "answer": response_text})

            # Print result
            sp.text = "Answer:"
            sp.ok("✅")
            print(f"\n{response_text}\n")
            print('-' * WIDTH)
            print(f"RAG Sources: \n{"\n".join(sources)}")
            print('-' * WIDTH)
            print()


if __name__ == "__main__":
    run_chat()