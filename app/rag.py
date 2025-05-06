from app.database import get_documents_by_ids
from app.embedder import get_embeddings
from app.faiss_index import search_index
from app.generator import generate_text
from app.logger import setup_logger
from app.utils import Mode, get_context

logger = setup_logger(__name__)


def rag_pipeline(question: str, context: str = None) -> str:
    """
    Retrieval-Augmented Generation pipeline

    :param question: User question
    :param top_k: number of contexts to look for in FAISS
    :return: Model answer
    """
    try:
        if not context:
            # 1. Getting an embedding of the question
            query_embedding = get_embeddings([question])

            # 2. Searching for the relevant documents
            doc_ids, scores = search_index(query_embedding)

            if len(doc_ids) == 0:
                return "No relevant documents found"
            # 3. Getting documents from the database by id
            documents = get_documents_by_ids(doc_ids)
            # 4. Building context from the found documents
            context = get_context(documents)

        # 5. Generating answer basing on the question and the context
        answer = generate_text({"question": question, "context": context}, Mode.QA)

        return answer
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        return f"Exception occurred: {e}"
