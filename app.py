import streamlit as st
import faiss
import pickle
import numpy as np
import os
from dotenv import load_dotenv

# NEW IMPORTS: Modern Google SDK & Sentence Transformers
from google import genai
from sentence_transformers import SentenceTransformer

# --- 1. SETUP & CONFIG ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Missing API Key! Please add GEMINI_API_KEY to your .env file.")
    st.stop()

# Initialize the NEW Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# --- 2. LOAD FAISS DATABASE & METADATA ---
@st.cache_resource
def load_faiss_db():
    try:
        index = faiss.read_index("dataset/employee_index.faiss")
        with open("dataset/employee_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        st.error(f"Error loading database: Make sure 'employee_index.faiss' and 'employee_metadata.pkl' are in the 'dataset/' folder. ({e})")
        st.stop()

index, metadata = load_faiss_db()

# --- 3. HELPER FUNCTIONS ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def get_query_embedding(text):
    """Converts the user's query into a 384-dimension math vector."""
    try:
        embedding = embedding_model.encode(text)
        return np.array([embedding], dtype=np.float32)
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        st.stop()

def generate_dossier(user_problem, skills_needed, matched_expert):
    prompt = f"""
    Act as a Cross-Industry Innovation Matchmaker.
    The user is trying to solve this problem: "{user_problem}"
    The core skills required are: "{skills_needed}"
    You matched them with Employee {matched_expert['id']}. 
    Here is their profile: "{matched_expert['text_for_llm']}"
    
    Write a short, professional 'Match Dossier' formatted in Markdown. Include:
    1. Why they were Matched (1 sentence bridging the user's problem with their profile).
    2. Profile Breakdown (Brief summary of their sector and certifications).
    3. Collaboration Pros (2 bullet points on how their specific background helps).
    4. Potential Friction (1 bullet point on cross-sector differences).
    """
    # UPDATED SDK CALL
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    return response.text

def draft_intro_email(user_problem, matched_expert):
    prompt = f"""
    Draft a professional email introduction from the User to Employee {matched_expert['id']}.
    The User is facing this problem: "{user_problem}".
    The User is reaching out because of the employee's background: "{matched_expert['text_for_llm']}".
    
    Acknowledge their expertise, highlight why their background solves the problem, and ask for a quick sync. Keep it under 150 words. Do not use placeholders like [Your Name].
    """
    # UPDATED SDK CALL
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    return response.text

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Nexus: AI Talent Matchmaker", layout="centered")
st.title("🤝 Nexus: FAISS Accelerated Matchmaker")
st.markdown("Enter your specific technical bottleneck. We will abstract the core skills needed and instantly perform a vector search across **10,000 employees** to find the exact match.")

user_input = st.text_area("What are you currently stuck on?", placeholder="e.g., I need to build a scalable, cloud-based financial forecasting model...")

if st.button("Find My Match"):
    if user_input:
        with st.spinner("Abstracting skills and performing millisecond FAISS search..."):
            
            # UPDATED SDK CALL
            abstract_prompt = f"Summarize the core technical skills, sectors, and certifications required to solve this problem in one short sentence: {user_input}"
            skills_response = client.models.generate_content(model='gemini-2.5-flash', contents=abstract_prompt)
            skills_needed = skills_response.text.strip()
            
            st.info(f"**Core Expertise Required:** {skills_needed}")
            
            query_vector = get_query_embedding(skills_needed)
            
            distances, indices = index.search(query_vector, k=1)
            match_index = indices[0][0]
            
            if match_index == -1:
                st.warning("Could not find a match in the database.")
                st.stop()
                
            matched_expert = metadata[match_index]
            
            st.success(f"Match found: **Employee {matched_expert['id']}**!")
            
        with st.spinner(f"Analyzing synergy between your problem and Employee {matched_expert['id']}..."):
            dossier = generate_dossier(user_input, skills_needed, matched_expert)
            
            st.markdown("---")
            st.markdown(dossier)
            
            st.markdown("### Contact Details")
            st.write(f"📧 **Email:** emp_{matched_expert['id']}@synthetic-company.com")
            
            st.session_state['matched_expert'] = matched_expert
            st.session_state['user_input'] = user_input
            st.session_state['email_draft'] = None

if 'matched_expert' in st.session_state:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Approve & Draft Intro Message"):
            with st.spinner("Drafting personalized email..."):
                st.session_state['email_draft'] = draft_intro_email(
                    st.session_state['user_input'], 
                    st.session_state['matched_expert']
                )
    with col2:
        if st.button("❌ Pass & See Next Match"):
            st.session_state.clear()
            st.rerun()

    if st.session_state.get('email_draft'):
        st.markdown("### Suggested Introduction Email")
        st.text_area("Copy and send this to the expert:", value=st.session_state['email_draft'], height=200)