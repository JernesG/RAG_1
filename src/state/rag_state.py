from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document


class RAGState(BaseModel):
    """Sate objcet  for RAG workflow"""

    question: str
    retrieved_docs : List[Document] = []
    answer: str = ""

    