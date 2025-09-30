"""Vector store module for document embedding and retrieval"""

from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class Vectorstore:
    """Manages vector store application"""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = None
        self.retriever = None

    def create_retriever(self,documents: List[Document]):
        """
        Create vector store from documents

        args:
            documents : list of documents to embed

        """
        self.vectorstore = Chroma.from_documents(documents,self.embeddings)
        self.retriver = self.vectorstore.as_retriever()

    def get_retriever(self):
        """
        Get the retriever instance

        returns:
        Retrive instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized, call vectore store frist")
        return self.retriever
    
    def retrive(self, query :str, k: int = 4)-> List[Document]:
        """Retrive relevent document from the query
        
        Args:
            query :Search query 
            k: Number of document to retrive

        return:
            list of relevent documents
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized, call create_vectorestore first")
        return self.retriever.invoke(query)

    