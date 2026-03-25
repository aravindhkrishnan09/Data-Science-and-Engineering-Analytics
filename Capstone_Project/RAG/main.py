import streamlit as st
import os
import vertexai
import re
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
    
    system_instruction = (
        "You are an expert AI assistant for the 'EV Battery Health Prediction Project'. "
        "Your task is to answer user questions accurately based strictly on the retrieved context. "
        "If the information is not available in the context, politely state that you do not have that information. "
        "Do not hallucinate or make up facts outside of the provided documents. "
        "After providing the answer, ALWAYS suggest 2 relevant follow-up questions that the user might find useful. "
        "Format the follow-up questions under the heading '\n\n### Follow-up Questions' as bullet points."
    )
    
    model = GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=system_instruction
    )
    
    judge_model = GenerativeModel("gemini-2.5-flash")
    return model, grounding_tool, judge_model

model, grounding_tool, judge_model = load_model()

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
                    generation_config={"temperature": 0.0
                    ,"max_output_tokens": 1024,
                    }
                )
                
                answer_text = response.text
                
                # --- LLM-as-a-judge Evaluation Layer ---
                eval_prompt = f"""
                You are an evaluator for a RAG system. Check if the answer is relevant to the user query and faithful.
                
                User Query: {prompt}
                System Answer: {answer_text}
                
                Score (1-5) for:
                - Relevance (Does it directly answer the user's question?)
                - Faithfulness (Does it seem grounded without hallucinations?)
                
                Format the output exactly as:
                Relevance: [score]
                Faithfulness: [score]
                """
                
                eval_response = judge_model.generate_content(eval_prompt, generation_config={"temperature": 0.0})
                
                # Parse the scores
                rel_match = re.search(r"Relevance:\s*([1-5])", eval_response.text, re.IGNORECASE)
                faith_match = re.search(r"Faithfulness:\s*([1-5])", eval_response.text, re.IGNORECASE)
                
                relevance_score = int(rel_match.group(1)) if rel_match else 5
                faithfulness_score = int(faith_match.group(1)) if faith_match else 5
                
                threshold = 3
                if relevance_score < threshold or faithfulness_score < threshold:
                    answer_text = "I don’t have enough reliable information to answer this question."
                # ---------------------------------------
                
                st.markdown(answer_text)
                
                # Check for citations/grounding metadata
                # Note: The structure of metadata depends on the response type
                if response.candidates[0].grounding_metadata.search_entry_point:
                     # This usually renders HTML provided by Google Search, 
                     # but for Vertex AI Search, you might prefer raw chunks.
                    sources = response.candidates[0].grounding_metadata.search_entry_point.rendered_content
                    with st.expander("📚 View Sources"):
                        st.markdown(sources, unsafe_allow_html=True)
                
                # Display evaluation scores for transparency
                with st.expander("📊 Evaluation Metrics (LLM-as-a-Judge)"):
                    st.write(f"**Relevance:** {relevance_score}/5")
                    st.write(f"**Faithfulness:** {faithfulness_score}/5")
                    st.caption("Answers scoring below the threshold (3) are replaced with a safety message.")
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")