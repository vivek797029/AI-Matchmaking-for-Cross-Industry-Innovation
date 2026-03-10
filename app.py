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
collection = chroma_client.get_or_create_collection(name="unified_experts")

# --- 2. LOAD UNIFIED DATABASE (Upgraded for Pandas/CSV) ---
@st.cache_data
# --- 2. LOAD UNIFIED DATABASE (Pointed to your dataset) ---
@st.cache_data
def load_database():
    try:
        # Pointing exactly to your filepath
        df = pd.read_csv('dataset/cleaned_employee_llm_data.csv')
        
        db_list = []
        for index, row in df.iterrows():
            # ⚠️ HACKATHON ALERT: Change the text inside the quotes below to match YOUR actual CSV column headers!
            # For example, if your CSV uses 'EmployeeName' instead of 'name', change row.get('name') to row.get('EmployeeName')
            
            db_list.append({
                "id": str(row.get('id', f"exp_{index}")),
                "name": str(row.get('name', 'Unknown')),
                "industry": str(row.get('industry', 'Unknown Domain')), 
                "contact": str(row.get('contact', 'No contact provided')),
                "credentials": {
                    "accolades": str(row.get('accolades', 'None')),
                    "past_projects": str(row.get('past_projects', 'None')),
                    "key_skills": [skill.strip() for skill in str(row.get('key_skills', '')).split(',')]
                },
                "jargon_problem": str(row.get('jargon_problem', '')), # The employee's current bottleneck
                "abstract_problem": str(row.get('abstract_problem', '')), # THIS IS REQUIRED FOR CHROMADB
                "glossary": str(row.get('glossary', 'No glossary provided'))
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
        documents=[exp["abstract_problem"] for exp in db if exp["abstract_problem"]],
        metadatas=[{"id": exp["id"], "name": exp["name"]} for exp in db if exp["abstract_problem"]],
        ids=[exp["id"] for exp in db if exp["abstract_problem"]]
    )

# --- 3. LLM GENERATORS ---
def generate_dossier(user_problem, abstract_problem, matched_expert):
    prompt = f"""
    Act as a Cross-Industry Innovation Matchmaker.
    
    User's Original Problem: "{user_problem}"
    Abstracted Core Problem: "{abstract_problem}"
    
    Matched Expert: {matched_expert['name']} ({matched_expert['industry']})
    Expert's Jargon Problem: "{matched_expert['jargon_problem']}"
    Expert's Glossary/Context: {matched_expert['glossary']}
    
    Credentials:
    - Accolades: {matched_expert['credentials']['accolades']}
    - Past Projects: {matched_expert['credentials']['past_projects']}
    - Skills: {', '.join(matched_expert['credentials']['key_skills'])}
    
    Write a short, professional 'Match Dossier' formatted in Markdown. Include:
    1. Shared Abstract Concept (1 sentence on why their problems mathematically/structurally overlap).
    2. Credential Analysis (Brief summary of why they are qualified to help).
    3. Why this is a Match (Pros) (2 bullet points).
    4. Potential Friction (Cons) (1 bullet point on what jargon they might misunderstand).
    """
    return model.generate_content(prompt).text

def draft_intro_email(user_problem, matched_expert):
    prompt = f"""
    Draft a professional, warm email introduction from the User to {matched_expert['name']}.
    The User is facing this problem: "{user_problem}".
    The Expert is solving this problem: "{matched_expert['jargon_problem']}".
    
    Acknowledge the difference in their industries ({matched_expert['industry']}), but highlight the shared abstract structural/mathematical challenge. Keep it under 150 words. Do not use placeholders like [Your Name].
    """
    return model.generate_content(prompt).text

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Nexus: Cross-Industry Matcher", layout="centered")
st.title("🤝 Nexus: Abstract Problem Matcher")
st.markdown("Enter your specific technical bottleneck. We will abstract it and match you with a vetted expert from a completely different field solving the exact same core problem.")

user_input = st.text_area("What are you currently stuck on?", placeholder="e.g., I need to fixate the distal femur without compromising the periosteal blood supply...")

if st.button("Find My Match"):
    if user_input:
        with st.spinner("Abstracting problem and searching vector space..."):
            # 1. Abstract the user's input
            abstract_prompt = f"Summarize the structural/mathematical core of this problem in one sentence, removing all industry jargon: {user_input}"
            abstracted_problem = model.generate_content(abstract_prompt).text.strip()
            
            st.info(f"**Abstracted Problem:** {abstracted_problem}")
            
            # 2. Query Vector DB
            results = collection.query(
                query_texts=[abstracted_problem],
                n_results=1
            )
            
            if not results['ids'][0]:
                st.warning("Could not find a match in the database. Try adding more experts!")
                st.stop()
                
            # 3. Process the Match
            match_id = results['ids'][0][0]
            matched_expert = next(exp for exp in db if exp['id'] == match_id)
            
            st.success(f"Match found in **{matched_expert['industry']}**!")
            
        with st.spinner(f"Analyzing credentials and synergy for {matched_expert['name']}..."):
            # 4. Generate LLM Dossier
            dossier = generate_dossier(user_input, abstracted_problem, matched_expert)
            
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
            with st.spinner("Drafting personalized email bridging your domains..."):
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