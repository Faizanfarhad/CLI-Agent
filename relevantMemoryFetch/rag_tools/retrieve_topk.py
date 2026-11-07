from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Any
from relevantMemoryFetch.rag_tools.chunking import CreateChunking
from relevantMemoryFetch.rag_tools.text_cleaner import clean_text
import torch

def retrieve_top_k(
    question:str,
    embedding_model: SentenceTransformer,
    faiss_index: faiss.Index,
    chunks_data: List[Dict[str, Any]],
    top_k:int=5
    )-> List[Dict[str,Any]]:
    """
        Retrieve top-k most relevant chunks for a user question.
    
        Parameters
        ----------
        question : str
            User question.
    
        embedding_model : SentenceTransformer
            The same embedding model used for chunk embeddings.
    
        faiss_index : faiss.Index
            FAISS index built from chunk embeddings.
    
        chunks_data : List[Dict[str, Any]]
            Chunk metadata list aligned with FAISS index positions.
    
        top_k : int
            Number of chunks to retrieve.
    
        Returns
        -------
        retrieved_chunks : List[Dict[str, Any]]
            List of retrieved chunks with similarity scores and ranks.
        """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    assert top_k <= len(chunks_data), "Choose top_k with the length of chunk data"  
    
    if question is None or len(question.strip()) == 0:
        raise ValueError("Question cannot be empty.")

    if faiss_index.ntotal == 0:
        raise ValueError("FAISS index is empty.")

    if len(chunks_data) == 0:
        raise ValueError("chunks_data is empty.")

    if faiss_index.ntotal != len(chunks_data):
        raise ValueError("FAISS index size and chunks_data size do not match.")
    
    cleaned_question = clean_text(question)
    
    
    question_embd = embedding_model.encode(
        [cleaned_question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        )

    # Search FAISS index
    scores, indices = faiss_index.search(question_embd, top_k)
    
    retrieved_chunks = []
    rank = 1 
    for score,index_position in zip(scores[0],indices[0]):
        if index_position == -1:
            continue
    
        original_chunk = chunks_data[index_position]
        
        retrieve_chunk = {
                    "rank": rank,
                    "retrieval_score": float(score),
                    "faiss_index": int(index_position),
                    "chunk_id": original_chunk.metadata["chunk_id"],
                    "chunk_text": original_chunk.text,
                    "word_count": original_chunk.metadata["word_count"],
                    "character_count": original_chunk.metadata["char_count"],
                    "preview": original_chunk.text[:200] if len(original_chunk.text) > 200 else original_chunk.text
                }
        
        retrieved_chunks.append(retrieve_chunk)
        rank += 1 
    
    return retrieved_chunks
