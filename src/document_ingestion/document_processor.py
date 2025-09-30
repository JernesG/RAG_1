""" Document processing module for loading and splitting"""

from typing import List, Union
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path

class Document_Processor:
    """Handles Document loading and processing"""

    def __init__(self, chunk_size: int=500, chunk_overlap: int=50):
        """
        Initialize document Processor

        Args: 
            chunk_size: Size of text chunks
            chunks_overlap : Overlap between chunks
    
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )

    def load_from_url(self,url:str)->List[Document]:
        """Load documents  from the url"""
        loader = WebBaseLoader(url)
        return loader.load()
    
    def load_from_pdf_dir(self, directory:Union[str,Path]) -> List[Document]:
        """Load document from all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()
    
    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load text from text file"""
        loader = TextLoader(str(file_path),encoding='UTF-8')
        return loader.load()

    def load_from_pdf(self,file_path : Union[str, Path]) -> List[Document]:
        """Load Document from PDF file"""
        loader = PyPDFDirectoryLoader(str("data"))
        return loader.load()
    
    def load_documents(self, sources : Union[str, Path]) -> List[Document]:
        """Load document from URLS, PDF, or TXT
        
        Args: 
            soruces : List of URLS, PDF folder paths, Txt file paths
        
        Return:
            List of Loaded Documents
        """
        docs :List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))

            path = Path("data")
            if path.is_dir():
                docs.extend(self.load_from_pdf(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}."
                    "Use URL, .txt file, or PDF dic"
                )
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks
        
        Args : 
            documents: list of document to split
        
        return : 
            List of split documents
        """  
        return self.splitter.split_documents(documents)
    
    def process_url(self, urls:List[str]) -> List[Document]:
        """Complete pipeline to load and split documents
        
            Args:
                urls: List of URLS to process
            
            returns:
                list of proceesed documents chunks
        """ 
        docs = self.load_documents(urls)
        return self.split_documents(docs)


