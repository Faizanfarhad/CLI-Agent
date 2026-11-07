from llama_index.core import Document
import os 
from llama_index.core.node_parser import TokenTextSplitter


# refrence : https://oneuptime.com/blog/post/2026-01-30-semantic-chunking/view
# not able to 
from typing import Optional

class CreateChunking:
    def __init__(self):
        super().__init__()
        """_summary_
        
        """
    
    
    def create_chunk(self,
                doc_text: str,
                file_path:Optional[str] = None,
                chunk_size: int = 512,
                chunk_overlap: int = 50
                ):
        """_summary_
            Note: for reducing the complexity i am using the whole document to create sentences
            
        """
        if file_path is not None:
            ext = os.path.splitext(file_path)[1].lower()
            
            document = Document(
                text=doc_text,
                metadata={
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": ext,
                    "file_size": os.path.getsize(file_path),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            )
        else:
        
            document = Document(
                            text=doc_text,
                            metadata={
                                "file_path": "doc path is Not provided",
                                "file_name": "doc path is Not provided",
                                "file_type": "text",
                                "file_size": "doc path is Not provided",
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap
                            }
                        )
        splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
        chunks = splitter.get_nodes_from_documents([document])
        
        for i,chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['char_count'] = len(chunk.text)
            chunk.metadata['word_count'] = len(chunk.text.split())
        return chunks
