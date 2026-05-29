
import os
from dotenv import load_dotenv
from vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv(override=True)


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile"
    ):

        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        ):

            from data_loader import load_all_documents

            docs = load_all_documents("../data")

            self.vectorstore.build_from_documents(docs)

        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    # =====================================================
    # SEARCH + SUMMARY
    # =====================================================

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
        Summarize the following context for the query:
        '{query}'

        Context:
        {context}

        Summary:
        """

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return response.content

    # =====================================================
    # CHAT WITH MEMORY
    # =====================================================

    def chat(
        self,
        query: str,
        chat_history: list,
        top_k: int = 5
    ) -> str:

        # =================================================
        # STEP 1: Convert history to text
        # =================================================

        history_text = ""

        for msg in chat_history[-6:]:

            history_text += (
                f"{msg['role']}: "
                f"{msg['content']}\n"
            )

        # =================================================
        # STEP 2: Rewrite follow-up query
        # =================================================

        rewrite_prompt = f"""
        Given the conversation history and latest user question,
        rewrite the latest question into a standalone question.

        Chat History:
        {history_text}

        Latest Question:
        {query}

        Standalone Question:
        """

        rewrite_response = self.llm.invoke(
            [HumanMessage(content=rewrite_prompt)]
        )

        rewritten_query = rewrite_response.content.strip()

        print(
            "[DEBUG] Rewritten Query:",
            rewritten_query
        )

        # =================================================
        # STEP 3: Retrieval using rewritten query
        # =================================================

        results = self.vectorstore.query(
            rewritten_query,
            top_k=top_k
        )

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant movies found for your query."

        # =================================================
        # STEP 4: Build messages
        # =================================================

        messages = []

        messages.append(
            SystemMessage(
                content="""
                You are a conversational movie recommendation assistant.

                Use the retrieved context to answer
                the user's question.

                The chat history may contain:
                - pronouns (he, she, it, they)
                - partial movie names
                - follow-up questions

                Use the conversation history
                to understand references.

                If the retrieved context
                does not contain the answer,
                say you don't know instead
                of hallucinating.
                """
            )
        )

        # Add past conversation

        for msg in chat_history[-6:]:

            if msg["role"] == "user":

                messages.append(
                    HumanMessage(
                        content=msg["content"]
                    )
                )

            else:

                messages.append(
                    AIMessage(
                        content=msg["content"]
                    )
                )

        # Add retrieved context + rewritten query

        messages.append(
            HumanMessage(
                content=f"""
                Context:
                {context}

                Question:
                {rewritten_query}
                """
            )
        )

        # =================================================
        # STEP 5: Generate response
        # =================================================

        response = self.llm.invoke(messages)

        return response.content


# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":

    rag_search = RAGSearch()

    query = "classic movies from 1990s"

    summary = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    print("Summary:", summary)

