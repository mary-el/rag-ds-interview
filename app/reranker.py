import pandas as pd
from sentence_transformers import CrossEncoder

from app.logger import setup_logger
from configs import load_config

logger = setup_logger(__name__)

reranker = None
config = None


def get_reranker() -> CrossEncoder:
    """
    Get or initialize the reranker model (lazy loading)
    :return: CrossEncoder reranker instance
    """
    global reranker, config
    if config is None:
        config = load_config().get("reranker", {})
    
    if reranker is None:
        model_name = config.get("model", "cross-encoder/ms-marco-MiniLM-L6-v2")
        logger.info(f"Initializing reranker with model: {model_name}")
        reranker = CrossEncoder(model_name)
    
    return reranker


def rerank_documents(question: str, documents: pd.DataFrame) -> pd.DataFrame:
    """
    Rerank documents using cross-encoder based on relevance to the question
    
    :param question: User question
    :param documents: DataFrame with columns including 'answer' field
    :return: Reranked DataFrame sorted by relevance (most relevant first)
    """
    if documents.empty:
        return documents
    
    reranker_model = get_reranker()
    
    # Create query-document pairs using the answer field
    pairs = [(question, str(doc["answer"])) for _, doc in documents.iterrows()]
    
    # Score pairs using the cross-encoder
    rerank_scores = reranker_model.predict(pairs)
    
    # Add scores to dataframe and sort by score (descending)
    documents = documents.copy()
    documents["rerank_score"] = rerank_scores
    documents = documents.sort_values("rerank_score", ascending=False)
    
    # Reset index to maintain clean indexing
    documents = documents.reset_index(drop=True)
    
    return documents

