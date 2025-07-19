import streamlit as st
import requests
import json
import os

# --- Configuration ---
PERSONAL_INFO_FILE = "personal_info.json"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL_NAME = "gemma3"

# --- Load Personal Information ---
@st.cache_data(show_spinner="Loading personal data...")
def load_personal_info(file_path):
    if not os.path.exists(file_path):
        st.error(f"Error: Personal information file '{file_path}' not found. Please create it.")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error: Could not parse '{file_path}'. Please check its JSON format for errors (e.g., missing commas, unclosed brackets).")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading '{file_path}': {e}")
        return None

personal_info = load_personal_info(PERSONAL_INFO_FILE)

if personal_info is None:
    st.stop()

SECTIONS = {
    "about": "about-jash",
    "skills": "my-skills",
    "education": "my-education",
    "projects": "my-projects",
    "contact": "contact-jash"
}

# --- Ollama Interaction Function ---
def get_gemma_response(prompt, model_name=GEMMA_MODEL_NAME):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama server. Please ensure Ollama is running and Gemma 3 is pulled (`ollama serve` in terminal).")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Ollama API: {e}. Check Ollama logs for details.")
        return None

# --- Agentic Logic (Personal Knowledge Base & Prompt Engineering) ---
def create_agentic_prompt(user_query):
    if not personal_info:
        return "I apologize, but my personal information is not available at the moment."

    identity_statement = f"You are an AI chatbot assistant of {personal_info['name']}. You were created by {personal_info['name']} to provide accurate and helpful information about his professional background, skills, education, and projects. Always introduce yourself with this identity if asked 'who are you?' or similar questions."

    system_intro = f"""{identity_statement}

    Here is key information about {personal_info['name']} for you to reference:
    - **Name:** {personal_info['name']}
    - **Occupation:** {personal_info['occupation']}
    - **About Me:** {personal_info['about_me']}
    - **Skills:** {', '.join(personal_info['skills'])}
    - **Education:** {personal_info['education']}
    - **Contact Email:** {personal_info['contact_email']}
    - **LinkedIn:** {personal_info['linkedin_profile']}
    - **GitHub:** {personal_info['github_profile']}
    - **Portfolio Website:** {personal_info['portfolio_website']}

    **Projects:**
    """
    # Explicitly list projects with their types for Gemma to learn
    for project in personal_info['projects']:
        project_type = "Client Project" if "client project" in project['name'].lower() else "Personal Project"
        system_intro += f"- **{project['name']} ({project_type})**: {project['description']}\n"


    general_instructions = """
    **Primary Goal:** Answer user questions about Jash Kothari's professional profile using the provided information.

    **Specific Answering Instructions:**
    - If a user asks a factual question that can be directly answered from the provided information, *provide the answer directly and concisely*.
    - **For Projects:** If asked about "client projects", "personal projects", or "different types of projects", list the relevant projects by their name and a brief description, clearly indicating their type (Client/Personal).
    - If asked for a summary of a section (e.g., "summarize your skills"), provide a brief overview.

    **Navigation Instructions (for more detail/exploration):**
    - If the user asks to "go to", "show me", "take me to" a specific section (e.g., "show me your skills", "go to projects", "tell me about your education", "contact info"), provide a clickable Markdown link to that section. Do NOT just give text instructions.
    - Here are the available sections and their corresponding anchor links:
        - About Jash: [About Jash](#about-jash)
        - My Skills: [My Skills](#my-skills)
        - My Education: [My Education](#my-education)
        - My Projects: [My Projects](#my-projects)
        - Contact Jash: [Contact Jash](#contact-jash)
    - If the user asks for *content* of a section, answer directly first, and *then* you can offer the link for more details.

    **General Chat Behavior:**
    - Be polite, concise, and helpful.
    - If the question is a general knowledge question not related to Jash Kothari, answer it to the best of your ability using your general knowledge, but maintain your persona as Jash Kothari's assistant.
    - Do not invent information about Jash Kothari that is not explicitly provided. If you cannot find the answer in the provided information, simply state that you don't have that specific detail about Jash Kothari.
    """

    full_prompt_for_llm = f"{system_intro}\n{general_instructions}\n\nUser Query: {user_query}"
    return full_prompt_for_llm

# --- Streamlit UI ---

st.set_page_config(page_title=f"{personal_info['name']}'s Agentic Portfolio Chatbot", layout="centered")

st.title(f"Hi, I'm {personal_info['name']}'s AI Assistant!")
st.write(f"Ask me anything about {personal_info['name']}'s skills, experience, projects, or how to get in touch.")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! I am a chatbot assistant of {personal_info['name']}. I was created by {personal_info['name']} to help you learn more about his professional background. How can I assist you today? You can ask me about his **skills**, **projects**, **education**, or **how to get in touch**!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"Ask me something about {personal_info['name']} or anything else!"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        agentic_query = create_agentic_prompt(prompt)
        response = get_gemma_response(agentic_query)

    if response:
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Display Sections within Expanders for cleaner initial view ---

st.markdown("---") # Visual separator

with st.expander("About Jash", expanded=False, anchor=SECTIONS["about"]): # Default to collapsed
    st.write(personal_info['about_me'])

with st.expander("My Skills", expanded=False, anchor=SECTIONS["skills"]): # Default to collapsed
    st.markdown(f"- **Programming Languages & Technologies:** {', '.join(personal_info['skills'])}")

with st.expander("My Education", expanded=False, anchor=SECTIONS["education"]): # Default to collapsed
    st.markdown(f"- **Degree:** {personal_info['education']}")

with st.expander("My Projects", expanded=False, anchor=SECTIONS["projects"]): # Default to collapsed
    st.write("Here are some of my projects, categorized by type:")
    client_projects = [p for p in personal_info['projects'] if "client project" in p['name'].lower()]
    personal_projects = [p for p in personal_info['projects'] if "client project" not in p['name'].lower()]

    if client_projects:
        st.subheader("Client Projects:")
        for project in client_projects:
            st.markdown(f"**{project['name']}**")
            st.write(project['description'])
            st.markdown("---")
    if personal_projects:
        st.subheader("Personal Projects:")
        for project in personal_projects:
            st.markdown(f"**{project['name']}**")
            st.write(project['description'])
            st.markdown("---")


with st.expander("Contact Jash", expanded=False, anchor=SECTIONS["contact"]): # Default to collapsed
    st.markdown(f"- **Email:** [{personal_info['contact_email']}](mailto:{personal_info['contact_email']})")
    st.markdown(f"- **LinkedIn:** [{personal_info['linkedin_profile']}]({personal_info['linkedin_profile']})")
    if personal_info['github_profile']:
        st.markdown(f"- **GitHub:** [{personal_info['github_profile']}]({personal_info['github_profile']})")
    if personal_info['portfolio_website']:
        st.markdown(f"- **Portfolio:** [{personal_info['portfolio_website']}]({personal_info['portfolio_website']})")


# Option to manually trigger a reload of personal info (in sidebar)
st.sidebar.markdown("---")
st.sidebar.header("App Management")
if st.sidebar.button("Reload Personal Info"):
    st.cache_data.clear()
    st.rerun()
    st.sidebar.success("Personal information reloaded!")