from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from Loader import Loader

def Chain_Setup(retriever, prompt, llm, parser):
    """
    Returns a conditional RAG-style chain for text and image queries
    using RunnableBranch.
    """

    def get_context(input_text):
        docs = retriever.invoke(input_text)
        return "\n".join([doc.page_content for doc in docs])

    # Text processing chain
    chain= (
        {
            "context": lambda x: get_context(x),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | parser
    )
    return chain