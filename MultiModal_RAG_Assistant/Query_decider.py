from langchain.schema import HumanMessage
from LLM_Image_Summarizer import GeminiMultimodal

class QueryDecider:
    """
    Decide whether a query requires an image or text response using Gemini.
    """

    def __init__(self, model_name="gpt-4o-mini", temperature=0.5):
        # Initialize the Gemini multimodal LLM
        self.llm_model = GeminiMultimodal(
            model=model_name,
            temperature=temperature,
        )
        self.llm = self.llm_model # LangChain LLM instance

    def classify(self, query: str) -> str:
        """
        Returns 'image' if the query is asking for image generation/analysis,
        'text' otherwise.
        """
        prompt = f"""
You are a query classifier. Determine if the user's query is asking for:
- IMAGE generation or analysis (respond with 'image')
- TEXT summary or information (respond with 'text')

Query: {query}
Answer:"""

        # Send prompt to Gemini
        response = self.llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()

        if 'image' in decision:
            return 'image'
        return 'text'
