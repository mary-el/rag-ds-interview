import openai

from app.database import get_documents_by_ids
from app.embedder import get_embeddings
from app.faiss_index import search_index
from app.generator import generate_text
from app.logger import setup_logger
from app.reranker import rerank_documents
from app.utils import Mode, get_context
from configs import load_config

logger = setup_logger(__name__)


def rag_pipeline(question: str, context: str = None) -> dict:
    """
    Retrieval-Augmented Generation pipeline

    :param question: User question
    :param context: Optional pre-provided context (if None, retrieves from FAISS)
    :return: Model answer
    """
    try:
        if not context:
            config = load_config()
            reranker_config = config.get("reranker", {})
            reranker_enabled = reranker_config.get("enabled", False)
            
            # 1. Getting an embedding of the question
            query_embedding = get_embeddings([question])

            # 2. Searching for the relevant documents
            records_num = config["faiss"]["records_num"]
            # Retrieve records_num candidates (will be reranked if enabled, or used directly if disabled)
            doc_ids, scores = search_index(query_embedding, top_k=records_num)

            if len(doc_ids) == 0:
                return {"success": False, "error": "No relevant documents found"}
            
            # 3. Getting documents from the database by id
            documents = get_documents_by_ids(doc_ids)
            
            # 4. Rerank documents if enabled
            if reranker_enabled:
                top_k_candidates = reranker_config.get("top_k_candidates", records_num)
                documents = rerank_documents(question, documents)
                # Select top_k_candidates after reranking
                documents = documents.head(top_k_candidates)
                logger.info(f"Reranked {len(doc_ids)} candidates, selected top {len(documents)}")
            
            # 5. Building context from the found documents
            context = get_context(documents)

        # 6. Generating answer basing on the question and the context
        answer = generate_text({"question": question, "context": context}, Mode.QA)

        return {"success": True, "answer": answer}
    except openai.APITimeoutError as e:
        logger.error(f"Timeout occurred: {e}")
        return {"success": False, "error": "Timeout occurred"}
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        return {"success": False, "error": "Internal server error"}
