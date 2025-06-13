import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()  # Loads .env file if present
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# Set page config
# -------------------------------
st.set_page_config(page_title="English to French Translator", page_icon="🌍")
st.title("🌍 English to French Translator")

# -------------------------------
# Set up Gemini model using LangChain
# -------------------------------
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set GOOGLE_API_KEY as an environment variable.")
else:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        # Prompt template using LangChain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that translates English to French."),
            ("user", "Translate this sentence to French: {input_sentence}")
        ])

        # Define the chain
        chain: Runnable = prompt | llm

        # -------------------------------
        # Streamlit Input UI
        # -------------------------------
        user_input = st.text_input("Enter an English sentence:", placeholder="e.g., How are you today?")
        if st.button("Translate"):
            if not user_input.strip():
                st.warning("Please enter a valid sentence.")
            else:
                try:
                    response = chain.invoke({"input_sentence": user_input})
                    translated = response.content
                    st.success("Translation completed successfully!")
                    st.markdown(f"**French Translation:** ✨\n\n> {translated}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

    except Exception as e:
        st.error(f"❌ Failed to initialize the Gemini model: {str(e)}")
