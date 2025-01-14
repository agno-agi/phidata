"""Run `pip install duckduckgo-search sqlalchemy pgvector pypdf openai google.generativeai` to install dependencies."""

from agno.embedder.google import GeminiEmbedder
from agno.agent import Agent
from agno.models.google import Gemini
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        embedder=GeminiEmbedder(),
    )
)
knowledge_base.load(recreate=True)  # Comment out after first run

agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    knowledge=knowledge_base,
    use_tools=True,
    show_tool_calls=True,
)
agent.print_response("How to make Thai curry?", markdown=True)
