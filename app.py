import streamlit as st
import faiss
import pickle
import numpy as np
import os
import time
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer

# --- 1. SETUP & PAGE CONFIG (Must be first) ---
st.set_page_config(page_title="Nexus: AI Matchmaker", layout="wide", initial_sidebar_state="expanded")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Missing API Key! Please add GEMINI_API_KEY to your .env file.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# --- 2. CUSTOM CSS INJECTION (UI Polish) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif !important;
        }
        
        .stTextArea label {
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            color: var(--text-color) !important; 
        }

        /* Animated White/Off-White Gradient Button */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%) !important;
            color: #111111 !important;
            font-weight: 600 !important;
            border: 1px solid #dcdcdc !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            transition: all 0.3s ease-in-out !important;
            width: 100%;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #f0f0f0 0%, #e8e8e8 100%) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 6px 15px rgba(255, 255, 255, 0.15) !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODELS & DB ---
@st.cache_resource
def load_faiss_db():
    index = faiss.read_index("dataset/employee_index.faiss")
    with open("dataset/employee_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

index, metadata = load_faiss_db()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def get_query_embedding(text):
    embedding = embedding_model.encode(text)
    return np.array([embedding], dtype=np.float32)

# --- 4. LLM FUNCTIONS (With Retry Logic) ---
def generate_dossier(user_problem, skills_needed, matched_expert, max_retries=3):
    prompt = f"Act as a Matchmaker. User problem: '{user_problem}'. Core skills: '{skills_needed}'. Matched profile: '{matched_expert['text_for_llm']}'. Write a 4-point Markdown Match Dossier (Why Matched, Profile, Pros, Friction)."
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
        except errors.ServerError:
            if attempt < max_retries - 1: time.sleep(2 ** attempt)
            else: return "⚠️ API Overloaded (503). Please wait and try again."
        except Exception as e: return f"⚠️ Error: {e}"

def draft_intro_email(user_problem, matched_expert, max_retries=3):
    prompt = f"Draft a <150 word email from User to Employee {matched_expert['id']} to solve: '{user_problem}' based on their profile: '{matched_expert['text_for_llm']}'."
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
        except errors.ServerError:
            if attempt < max_retries - 1: time.sleep(2 ** attempt)
            else: return "⚠️ API busy. Try again."
        except Exception as e: return f"⚠️ Error: {e}"

# --- 5. SIDEBAR (Settings & Filters) ---
with st.sidebar:
    st.title("⚙️ Nexus Settings")
    st.markdown("Streamlit natively detects your system's Dark/Light mode! You can force it via the top-right `⋮` menu -> Settings -> Theme.")
    
    st.markdown("### Search Filters")
    sector_filter = st.selectbox("Preferred Sector", ["Any", "IT", "Finance", "CA", "Digital Marketing"])
    exp_filter = st.selectbox("Experience Level", ["Any", "Fresher", "Intermediate", "Experienced"])
    
    st.markdown("*(Note: Filters will be applied to the FAISS metadata stream in v2.0)*")

# --- 6. MAIN UI ---
st.title("🤝 Nexus: Deep Semantic Matchmaker")
st.markdown("Enter your technical bottleneck below to instantly scan across **10,000 global experts**.")

st.markdown("<br>", unsafe_allow_html=True)

# Outer columns to constrain the width of the input area
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    user_input = st.text_area(
        "What are you currently stuck on?", 
        placeholder="e.g., I need to build a scalable, cloud-based financial forecasting model...", 
        height=120,
        key="main_user_input"
    )
    
    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    
    # Inner columns to center the button under the input area
    # This prevents the button from stretching to the full width of the text box
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col2:
        find_match_btn = st.button("🔍 Find My Match", key="main_search_btn")

# --- EXECUTE SEARCH ---
if find_match_btn:
    if user_input:
        with st.spinner("Abstracting skills and performing millisecond FAISS search..."):
            
            skills_response = client.models.generate_content(model='gemini-2.5-flash', contents=f"Summarize core skills for: {user_input}")
            skills_needed = skills_response.text.strip()
            
            query_vector = get_query_embedding(skills_needed)
            distances, indices = index.search(query_vector, k=10)
            
            valid_matches = [metadata[idx] for idx in indices[0] if idx != -1]
            
            if not valid_matches:
                st.warning("Could not find a match.")
                st.stop()
            
            st.session_state['matches'] = valid_matches
            st.session_state['distances'] = distances[0] # Save distances for math calculation
            st.session_state['current_index'] = 0
            st.session_state['user_input'] = user_input
            st.session_state['skills_needed'] = skills_needed
            st.session_state['email_draft'] = None

# --- 7. RESULTS DISPLAY ---
if 'matches' in st.session_state:
    matches = st.session_state['matches']
    current_idx = st.session_state['current_index']
    
    if current_idx >= len(matches):
        st.warning("⚠️ You have reviewed all highly suitable matches. Try modifying your prompt.")
    else:
        matched_expert = matches[current_idx]
        total_matches = len(matches)
        
        # Calculate a Match Percentage based on FAISS distance (lower distance = better match)
        raw_distance = st.session_state['distances'][current_idx]
        match_percentage = min(99.9, max(60.0, 100 - (raw_distance * 15))) 
        
        st.markdown("---")
        st.success(f"**Match Found: Employee {matched_expert['id']}** (Result {current_idx + 1} of {total_matches})")
        
        # --- UI LAYOUT: Split Graph and Dossier ---
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            # Create Plotly Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = match_percentage,
                number = {'suffix': "%", 'font': {'size': 40}},
                title = {'text': "Synergy Score", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00A67E"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "lightgreen"}
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Contact Details")
            st.write(f"🔗 **Social:** linkedin.com/in/employee-{matched_expert['id']}")
            st.write(f"📧 **Email:** emp_{matched_expert['id']}@synthetic.com")
            
        with res_col2:
            with st.spinner(f"Analyzing synergy..."):
                dossier = generate_dossier(st.session_state['user_input'], st.session_state['skills_needed'], matched_expert)
                st.info(f"**Core Expertise Targeted:** {st.session_state['skills_needed']}")
                st.markdown(dossier)
                
        st.markdown("---")
        act_col1, act_col2 = st.columns(2)
        
        with act_col1:
            if st.button("✅ Approve & Draft Intro Message"):
                with st.spinner("Drafting personalized email..."):
                    st.session_state['email_draft'] = draft_intro_email(st.session_state['user_input'], matched_expert)
        with act_col2:
            if st.button("❌ Pass & See Next Match"):
                st.session_state['current_index'] += 1
                st.session_state['email_draft'] = None
                st.rerun()

        if st.session_state.get('email_draft'):
            st.text_area("Suggested Introduction Email:", value=st.session_state['email_draft'], height=200)