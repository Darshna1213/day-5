import streamlit as st
import contextlib
import sys
from io import StringIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Hardcoded Gemini API key (replace with your actual key)
GEMINI_API_KEY = "AIzaSyAG8Ofxn8EnCitkvthCHEAXlynW96IY9Ro"

# Suppress verbose logs by redirecting stdout temporarily
@contextlib.contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = original_stdout

# Streamlit app configuration
st.set_page_config(page_title="Real-Time Q&A App", page_icon="🌐")
st.title("🌟 Real-Time Q&A with Gemini 🌟")
st.markdown("Ask any question about current events or facts, and get instant answers powered by Gemini and DuckDuckGo search! 🚀")

# Initialize session state for query and response
if "query" not in st.session_state:
    st.session_state.query = ""
if "response" not in st.session_state:
    st.session_state.response = None
if "error" not in st.session_state:
    st.session_state.error = None

# Create input form
with st.form(key="query_form"):
    user_query = st.text_input("Enter your question:", placeholder="What's happening in the world today? 📰")
    submit_button = st.form_submit_button("Ask Now! 🔍")

# Process query when button is clicked
if submit_button and user_query:
    st.session_state.query = user_query
    st.session_state.response = None
    st.session_state.error = None

    try:
        # Initialize Gemini model with hardcoded API key
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )

        # Initialize DuckDuckGo search tool
        search_tool = DuckDuckGoSearchRun()

        # Initialize agent with ZERO_SHOT_REACT_DESCRIPTION
        with suppress_stdout():
            agent = initialize_agent(
                tools=[search_tool],
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True
            )

        # Create a placeholder for streaming response
        response_placeholder = st.empty()
        callback_handler = StreamlitCallbackHandler(response_placeholder)

        # Run the agent and capture response
        with suppress_stdout():
            response = agent.run(user_query, callbacks=[callback_handler])

        # Store response in session state
        st.session_state.response = response

    except Exception as e:
        # Store error in session state
        st.session_state.error = str(e)

# Display response or error
if st.session_state.response:
    st.success("**Answer:**")
    st.markdown(st.session_state.response)
elif st.session_state.error:
    st.error(f"**Oops, something went wrong!** 😕\nError: {st.session_state.error}")
elif st.session_state.query:
    st.info("Processing your question... Please wait! ⏳")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, and Gemini. Ask away! 🌍")