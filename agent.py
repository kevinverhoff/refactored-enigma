from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from tools import answer, browse_cluster, compare_years, extract_quotes, get_topics, search, summarize

load_dotenv()

# LLM SWAP: replace ChatGoogleGenerativeAI with your provider's chat model class.
# Must support tool/function calling. Set AGENT_MODEL to the corresponding model ID.
AGENT_MODEL = "gemini-2.5-flash"

_SYSTEM_PROMPT = """You are an expert research assistant for Indiana DLGF (Department of Local Government Finance) memos and guidance documents, covering 2022-2026.

You have access to tools for searching, answering questions, summarizing topics, comparing guidance across years, listing topics, browsing topic clusters, and extracting notable quotes.

Guidelines:
- For direct questions, use `answer`.
- For "find documents about X", use `search`.
- For broad synthesis or overviews, use `summarize`.
- For "what changed" or trend questions, use `compare_years`.
- For orientation or topic discovery, use `get_topics` first.
- For drilling into a specific topic cluster, use `browse_cluster` with the cluster ID from `get_topics`.
- For notable language or key statements, use `extract_quotes`.
- If a query is ambiguous, ask one clarifying question before calling a tool.
- Always include source URLs when citing documents."""

_tools = [search, answer, summarize, compare_years, get_topics, browse_cluster, extract_quotes]
llm = ChatGoogleGenerativeAI(model=AGENT_MODEL, temperature=0.1)
graph = create_react_agent(llm, _tools, prompt=_SYSTEM_PROMPT)