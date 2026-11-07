from sentence_transformers import SentenceTransformer
import numpy as np

def create_embedding(model:SentenceTransformer,chunks):
    """_summary_
        
        * chunks type is **TextNode** 
        * creates embeddings for the chunk text
        
    """
    texts = [chunk.text for chunk in chunks]
    
    
    chunk_embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,       
        show_progress_bar=True,          
    )

    return np.asarray(chunk_embeddings, dtype='float32') 
    