from pathlib import Path

from sentence_transformers import SentenceTransformer

from tools.file_tools import read_file
from relevantMemoryFetch.rag_tools.chunking import CreateChunking
from relevantMemoryFetch.rag_tools.create_embeddings import create_embedding
from relevantMemoryFetch.rag_tools.ranker import build_faiss_index
from relevantMemoryFetch.rag_tools.retrieve_topk import retrieve_top_k
from relevantMemoryFetch.rag_tools.text_cleaner import clean_text
            
class RagPipelineConnector:
    """
    RAG pipeline using FAISS retrieval + optional LLM generation.
    Set generate_answer=True in query() to use the local generator model.
    
    if **Doc path** is provided then dont use the **pdf_text** argument and for **pdf_text** dont use the **doc_path**
    
    
    """

    def __init__(
        self,
        doc_path: str | Path | None = None,
        pdf_text: str | None = None,
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embd_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        llm_device: str = "cpu",
    ):
        super().__init__()

        self.doc_path = doc_path
        self.doc_text = pdf_text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.device = device if device != "auto" else "cpu"
        self.embd_model_name = embd_model_name
        self.llm_device = llm_device

        self.embd_model = SentenceTransformer(self.embd_model_name, device=self.device)
        self.refresh_index()

    def refresh_index(self) -> None:
        """Re-read the memory source and rebuild its chunks/index using the loaded model."""
        source_text = (
            read_file(self.doc_path)
            if self.doc_path is not None
            else self.doc_text or ""
        )
        cleaned_text = clean_text(source_text)
        
        
        del source_text # just to make sure 
        
        
        chunks = (
            CreateChunking().create_chunk(
                doc_text=cleaned_text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            if cleaned_text
            else []
        )

        if chunks:
            embeddings = create_embedding(model=self.embd_model, chunks=chunks)
            index = build_faiss_index(chunk_embeddings=embeddings)
        else:
            embeddings = None
            index = None
        
        self.chunks = chunks
        self.chunks_embeddings = embeddings
        self.faiss_index = index

    def query(self, question: str) -> dict:
        """
        Retrieve top-k chunks from FAISS. Optionally generate an LLM answer.

        Parameters
        ----------
        question : str
            check or fetch the memory if present 
        

        Returns
        -------
        dict with keys: question, answer, source_chunks, total_sources
        """
        if not self.chunks or self.faiss_index is None:
            return {"Top_relevant_memory_chunks": []}

        retrieved = retrieve_top_k(
            question=question,
            embedding_model=self.embd_model,
            faiss_index=self.faiss_index,
            chunks_data=self.chunks,
            top_k=min(self.top_k, len(self.chunks)),
        )

        # Build source chunks list with full text
        source_chunks = [
            {
                "id": r["chunk_id"],
                "text": r["chunk_text"],
            }
            for r in retrieved
        ]

        return {
            "Top_relevant_memory_chunks": source_chunks,
        }