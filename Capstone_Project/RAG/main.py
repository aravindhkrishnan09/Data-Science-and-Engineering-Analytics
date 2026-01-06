import streamlit as st
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Tool, grounding
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EV Battery Health Prediction Project RAG System", page_icon="🧠", layout="centered")

# --- 1. LOAD SECRETS ---
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_REGION = os.getenv("MODEL_REGION")          # e.g., us-central1
DATA_STORE_LOCATION = os.getenv("DATA_STORE_LOCATION") # e.g., global
DATA_STORE_ID = os.getenv("DATA_STORE_ID")

# Safety Check
if not all([PROJECT_ID, MODEL_REGION, DATA_STORE_LOCATION, DATA_STORE_ID]):
    st.error("❌ Missing keys in .env file. Please check your configuration.")
    st.stop()

# --- 2. INITIALIZE VERTEX AI ---
try:
    # Initialize with the MODEL region (us-central1), NOT global
    vertexai.init(project=PROJECT_ID, location=MODEL_REGION)
except Exception as e:
    st.error(f"Failed to initialize Vertex AI: {e}")

# --- 3. SETUP MODEL & TOOLS ---
@st.cache_resource
def load_model():
    # Construct path using the DATA STORE location (global)
    datastore_path = f"projects/{PROJECT_ID}/locations/{DATA_STORE_LOCATION}/collections/default_collection/dataStores/{DATA_STORE_ID}"
    
    grounding_tool = Tool.from_retrieval(
        retrieval=grounding.Retrieval(
            source=grounding.VertexAISearch(
                datastore=datastore_path
            )
        )
    )
    
    model = GenerativeModel("gemini-2.5-flash")
    return model, grounding_tool

model, grounding_tool = load_model()

# --- 4. CHAT INTERFACE ---
st.title("EV Battery Health Prediction Project RAG System")
st.caption("Ask questions about EV Battery Health Prediction System")
st.caption("Example: Summarize the EV Battery Health Prediction System project in 100 words")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. HANDLE USER INPUT ---
if prompt := st.chat_input("Ask me anything about your data..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call Gemini with Grounding
                response = model.generate_content(
                    prompt,
                    tools=[grounding_tool],
                    generation_config={"temperature": 0.0,
                    "max_output_tokens": 512,
                    }
                )
                
                answer_text = response.text
                st.markdown(answer_text)
                
                # Check for citations/grounding metadata
                # Note: The structure of metadata depends on the response type
                if response.candidates[0].grounding_metadata.search_entry_point:
                     # This usually renders HTML provided by Google Search, 
                     # but for Vertex AI Search, you might prefer raw chunks.
                    sources = response.candidates[0].grounding_metadata.search_entry_point.rendered_content
                    with st.expander("📚 View Sources"):
                        st.markdown(sources, unsafe_allow_html=True)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")