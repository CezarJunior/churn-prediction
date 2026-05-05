"""
RAG (Retrieval-Augmented Generation) layer for churn insight retrieval.

Uses LlamaIndex with a local HuggingFace embedding model and an in-memory
FAISS vector store.  This keeps the setup fully free — no OpenAI key needed.

Each high-risk customer is converted to a LlamaIndex Document so that
analysts can ask natural-language questions such as:
  "Which high-risk customers are on month-to-month contracts?"
  "Summarise the tenure profile of customers most likely to churn."
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _row_to_text(row: pd.Series) -> str:
    """Convert a customer row to a human-readable text chunk."""
    fields = []
    interesting = [
        ("tenure", "tenure (months)"),
        ("MonthlyCharges", "monthly charges ($)"),
        ("TotalCharges", "total charges ($)"),
        ("Contract", "contract type"),
        ("InternetService", "internet service"),
        ("PaymentMethod", "payment method"),
        ("churn_probability", "churn probability"),
        ("risk_label", "risk label"),
        ("tenure_group", "tenure group"),
        ("HasMultipleServices", "number of active services"),
        ("IsLongTermContract", "long-term contract"),
        ("gender", "gender"),
        ("SeniorCitizen", "senior citizen"),
        ("Partner", "has partner"),
        ("Dependents", "has dependents"),
    ]
    for col, label in interesting:
        if col in row.index:
            val = row[col]
            if col == "churn_probability":
                fields.append(f"{label}: {val:.1%}")
            else:
                fields.append(f"{label}: {val}")
    return "Customer profile — " + " | ".join(fields)


class ChurnRAG:
    """
    Wraps LlamaIndex VectorStoreIndex over a high-risk customer dataframe.

    Usage
    -----
    >>> rag = ChurnRAG()
    >>> index = rag.build_churn_index(high_risk_df)
    >>> answer = rag.query_churn_insights(index, "Which contract types dominate?")
    >>> print(answer)
    """

    def __init__(self, embedding_model: str = _EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        self._embed_model = None  # lazy-loaded on first build

    def _get_embed_model(self):
        if self._embed_model is None:
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                logger.info("Loading embedding model %s …", self.embedding_model)
                self._embed_model = HuggingFaceEmbedding(
                    model_name=self.embedding_model,
                    embed_batch_size=32,
                )
                logger.info("Embedding model loaded")
            except ImportError as exc:
                raise ImportError(
                    "llama-index-embeddings-huggingface is required. "
                    "Run: pip install llama-index-embeddings-huggingface"
                ) from exc
        return self._embed_model

    def build_churn_index(self, df_high_risk: pd.DataFrame):
        """
        Convert each high-risk customer row to a LlamaIndex Document and
        build a FAISS-backed VectorStoreIndex.

        Parameters
        ----------
        df_high_risk : pd.DataFrame
            Output of the ``identify_high_risk_node`` graph step, or any
            dataframe containing customer profiles.

        Returns
        -------
        VectorStoreIndex
            Ready-to-query index.
        """
        try:
            import faiss
            from llama_index.core import Settings, VectorStoreIndex
            from llama_index.core.schema import Document
            from llama_index.vector_stores.faiss import FaissVectorStore
        except ImportError as exc:
            raise ImportError(
                "Required packages missing. Install with: "
                "pip install llama-index faiss-cpu llama-index-vector-stores-faiss"
            ) from exc

        embed_model = self._get_embed_model()
        Settings.embed_model = embed_model
        Settings.llm = None  # We use Groq directly; disable default OpenAI LLM

        logger.info("Building FAISS index for %d customers …", len(df_high_risk))

        documents = []
        for idx, row in df_high_risk.reset_index(drop=True).iterrows():
            text = _row_to_text(row)
            metadata = {
                "row_index": int(idx),
                "risk_label": str(row.get("risk_label", "Unknown")),
                "tenure_group": str(row.get("tenure_group", "Unknown")),
                "contract": str(row.get("Contract", "Unknown")),
                "churn_prob": float(row.get("churn_probability", 0.0)),
            }
            doc = Document(text=text, metadata=metadata)
            documents.append(doc)

        # Build dimension from a test embedding
        test_embed = embed_model.get_text_embedding("test")
        dim = len(test_embed)
        faiss_index = faiss.IndexFlatL2(dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        from llama_index.core import StorageContext

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=False,
        )

        logger.info("FAISS index built with %d documents", len(documents))
        return index

    def query_churn_insights(
        self,
        index,
        question: str,
        similarity_top_k: int = 5,
    ) -> str:
        """
        Run a natural-language query against the churn RAG index.

        Parameters
        ----------
        index : VectorStoreIndex
            Index built by :meth:`build_churn_index`.
        question : str
            Natural-language question about the customer data.
        similarity_top_k : int
            Number of similar documents to retrieve before synthesising the answer.

        Returns
        -------
        str
            Answer synthesised from the retrieved customer profiles.
        """
        groq_key = __import__("os").getenv("GROQ_API_KEY")

        if groq_key:
            try:
                from llama_index.core import Settings
                from llama_index.llms.groq import Groq

                Settings.llm = Groq(
                    model="llama-3.3-70b-versatile",
                    api_key=groq_key,
                )
            except ImportError:
                logger.warning("llama-index Groq LLM not available — using retrieval only")

        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)

        try:
            response = query_engine.query(question)
            return str(response)
        except Exception as exc:  # noqa: BLE001
            logger.error("RAG query failed: %s", exc)
            # Graceful fallback: return the raw retrieved texts
            retriever = index.as_retriever(similarity_top_k=similarity_top_k)
            nodes = retriever.retrieve(question)
            snippets = [n.get_content() for n in nodes]
            return (
                f"[LLM synthesis unavailable — raw retrieved profiles]\n\n"
                + "\n\n".join(snippets)
            )


def build_churn_index(df_high_risk: pd.DataFrame):
    """Module-level convenience wrapper around :class:`ChurnRAG`."""
    return ChurnRAG().build_churn_index(df_high_risk)


def query_churn_insights(index, question: str, similarity_top_k: int = 5) -> str:
    """Module-level convenience wrapper around :class:`ChurnRAG`."""
    return ChurnRAG().query_churn_insights(index, question, similarity_top_k)
