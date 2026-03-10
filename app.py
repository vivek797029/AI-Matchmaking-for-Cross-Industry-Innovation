import streamlit as st
import chromadb
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

# --- 1. SETUP & CONFIG ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Missing API Key! Please add GEMINI_API_KEY to your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.1-pro') 

# Initialize ChromaDB (In-memory)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="employee_skills")

# --- 2. LOAD YOUR SPECIFIC DATABASE ---
@st.cache_data
def load_database():
    try:
        # Loading your specific dataset
        df = pd.read_csv('dataset/cleaned_employee_llm_data.csv')
        
        # ⚠️ Hackathon Tip: Loading 10,000 rows into a vector DB takes time. 
        # For a fast demo, we limit to 500. Change this number if you want more!
        df_demo = df.head(10000) 
        
        db_list = []
        for index, row in df_demo.iterrows():
            emp_id = str(row['id'])
            profile_text = str(row['text_for_llm'])
            
            # Since the data is condensed, we map what we can and synthesize the rest
            db_list.append({
                "id": emp_id,
                "name": f"Employee {emp_id}",
                "contact": f"emp_{emp_id}@synthetic-company.com",
                "profile": profile_text
            })
        return db_list
        
    except FileNotFoundError:
        st.error("Could not find dataset/cleaned_employee_llm_data.csv. Please ensure the folder and file exist.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading database: {e}")
        st.stop()

db = load_database()

# Seed the Vector DB on startup (only if empty)
if collection.count() == 0:
    collection.add(
        documents=[exp["profile"] for exp in db],
        metadatas=[{"id": exp["id"], "name": exp["name"]} for exp in db],
        ids=[exp["id"] for exp in db]
    )

# --- 3. LLM GENERATORS ---
def generate_dossier(user_problem, skills_needed, matched_expert):
    prompt = f"""
    Act as a Cross-Industry Innovation Matchmaker.
    
    The user is trying to solve this problem: "{user_problem}"
    The core skills/domain knowledge required are: "{skills_needed}"
    
    You matched them with {matched_expert['name']}. 
    Here is their employee profile: "{matched_expert['profile']}"
    
    Write a short, professional 'Match Dossier' formatted in Markdown. Include:
    1. Why they were Matched (1 sentence bridging the user's problem with the employee's background/certifications).
    2. Profile Breakdown (Brief summary of their sector, experience level, and certifications based strictly on their profile).
    3. Collaboration Pros (2 bullet points on how their specific background helps solve the bottleneck).
    4. Potential Friction (1 bullet point on what cross-sector differences they might need to overcome).
    """
    return model.generate_content(prompt).text

def draft_intro_email(user_problem, matched_expert):
    prompt = f"""
    Draft a professional, warm email introduction from the User to {matched_expert['name']}.
    The User is facing this problem: "{user_problem}".
    
    The User is reaching out because of the employee's background: "{matched_expert['profile']}".
    
    Acknowledge their expertise, highlight why their specific background is exactly what is needed to solve the problem, and ask for a quick sync. Keep it under 150 words. Do not use placeholders like [Your Name].
    """
    return model.generate_content(prompt).text

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Nexus: AI Talent Matchmaker", layout="centered")
st.title("🤝 Nexus: Cross-Discipline Matchmaker")
st.markdown("Enter your specific technical bottleneck. We will abstract the core skills needed and match you with a vetted employee across the company who has the right expertise.")

user_input = st.text_area("What are you currently stuck on?", placeholder="e.g., I need to build a scalable, cloud-based financial forecasting model but I am struggling with the architecture...")

if st.button("Find My Match"):
    if user_input:
        with st.spinner("Analyzing problem and searching the talent matrix..."):
            # 1. Abstract the user's input into core skills
            abstract_prompt = f"Summarize the core technical skills, sectors, and certifications required to solve this problem in one short sentence: {user_input}"
            skills_needed = model.generate_content(abstract_prompt).text.strip()
            
            st.info(f"**Core Expertise Required:** {skills_needed}")
            
            # 2. Query Vector DB against the text_for_llm profiles
            results = collection.query(
                query_texts=[skills_needed],
                n_results=1
            )
            
            if not results['ids'][0]:
                st.warning("Could not find a match in the database.")
                st.stop()
                
            # 3. Process the Match
            match_id = results['ids'][0][0]
            matched_expert = next(exp for exp in db if exp['id'] == match_id)
            
            st.success(f"Match found: **{matched_expert['name']}**!")
            
        with st.spinner(f"Analyzing synergy between your problem and {matched_expert['name']}'s profile..."):
            # 4. Generate LLM Dossier
            dossier = generate_dossier(user_input, skills_needed, matched_expert)
            
            # 5. Display Dossier
            st.markdown("---")
            st.markdown(dossier)
            
            st.markdown("### Contact Details")
            st.write(f"📧 **Email:** {matched_expert['contact']}")
            
            # Save state for the email button
            st.session_state['matched_expert'] = matched_expert
            st.session_state['user_input'] = user_input
            st.session_state['email_draft'] = None

# If a match has been found, show the email drafting buttons
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

    # Display the drafted email if generated
    if st.session_state.get('email_draft'):
        st.markdown("### Suggested Introduction Email")
        st.text_area("Copy and send this to the expert:", value=st.session_state['email_draft'], height=200)