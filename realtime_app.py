import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from duckduckgo_search import DDGS

# 🔐 HARD-CODED Gemini API Key (for demo purposes only)
GEMINI_API_KEY = "AIzaSyAo2TKdDwcXhOoBM4NONeCEJLCuYCan-j8"  # Replace with your actual API key

# Disable logs (to avoid verbose terminal output)
import logging
logging.getLogger("langchain").setLevel(logging.CRITICAL)

# 🎯 Define DuckDuckGo search tool
def duckduckgo_search_tool(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            if results:
                return "\n\n".join(
                    f"🔹 {r['title']}\n{r['body']}\n🔗 {r['href']}" for r in results
                )
            else:
                return "No relevant results found."
    except Exception as e:
        return f"Search error: {e}"

# Define LangChain tool
search_tool = Tool(
    name="DuckDuckGo Search",
    func=duckduckgo_search_tool,
    description="Useful for answering real-world or current event questions"
)

# 🎯 Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# ⚙️ Initialize agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# 🎨 Streamlit UI
st.set_page_config(page_title="🌐 Real-Time Q&A with Gemini", layout="centered")
st.title("🌍 Real-Time Q&A App with Gemini 🤖")

st.markdown(
    "Ask anything about **current events**, **factual data**, or **live topics**. "
    "Powered by [Google Gemini](https://ai.google/) + DuckDuckGo Search 🔍"
)

user_input = st.text_input("💬 Ask your question here:")
submit = st.button("🚀 Submit")

if submit and user_input:
    with st.spinner("Thinking... 🧠"):
        try:
            response = agent.run(user_input)
            st.success("✅ Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
