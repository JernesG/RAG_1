from src.state.rag_state import RAGState

class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        """initialize RAG nodes
        
            Args:
            retriever: Document retriever instance
            llm : language model instance
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state:RAGState) -> RAGState:
        """
        Retrieve relevent document node

        Args: 
            state: Current RAG state

        Returns:
            Update RAG state with Retrive document
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(question=state.question,retrieved_docs=docs)
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer from the retrieve documents node

        Args:
            state:Current RAG state with retrived documents
        Returns:
            Updated RAG state with generated answer
        """
        context = "\n\n".join([doc.page_content for doc in state.retrieve_docs])

        prompt = f"""Answer the question based on the context.
        
        Context:{context}

        Question:{state.question}
        """

        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrieved_docs= state.retrieved_docs,
            answer= response.content
        )