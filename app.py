import streamlit as st
import requests
import json
import os # Import os module for path handling

# --- Configuration ---
PERSONAL_INFO_FILE = "personal_info.json" # Name of your JSON file
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
GEMMA_MODEL_NAME = "gemma3" # Your Gemma 3 model name (ensure you've pulled it with ollama pull gemma3)

# --- Load Personal Information ---
# @st.cache_data is important for performance. It ensures the JSON is loaded only once
# unless the file changes or the cache is cleared.
@st.cache_data(show_spinner="Loading personal data...")
def load_personal_info(file_path):
    """Loads personal information from a JSON file."""
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

# If personal_info couldn't be loaded, stop the app or show an error
if personal_info is None:
    st.stop() # Stop the app if crucial data is missing or malformed

# Define sections for navigation. The keys are user-friendly names, values are anchor IDs.
# Streamlit automatically creates anchor IDs for headings by slugifying the text.
# E.g., "My Skills" becomes "my-skills"
SECTIONS = {
    "about": "about-jash",
    "skills": "my-skills",
    "education": "my-education",
    "projects": "my-projects",
    "contact": "contact-jash"
}

# --- Ollama Interaction Function ---
def get_gemma_response(prompt, model_name=GEMMA_MODEL_NAME):
    """
    Sends a prompt to the locally running Ollama server and returns the response.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False # Set to True for streaming responses, but False for simplicity here
    }
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama server. Please ensure Ollama is running and Gemma 3 is pulled (`ollama serve` in terminal).")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Ollama API: {e}. Check Ollama logs for details.")
        return None

# --- Agentic Logic (Personal Knowledge Base & Prompt Engineering) ---
def create_agentic_prompt(user_query):
    """
    Constructs a detailed system prompt for Gemma 3, including personal information
    and instructions for identity and navigation, to guide its responses.
    """
    if not personal_info:
        return "I apologize, but my personal information is not available at the moment."

    # Chatbot's identity and purpose
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
    for project in personal_info['projects']:
        system_intro += f"- **{project['name']}**: {project['description']}\n"

    # Instructions for navigation with clickable links
    navigation_instructions = f"""
    If the user asks to go to a specific section (e.g., "show me your skills", "go to projects", "tell me about your education", "take me to contact info", "about you"), respond by providing a clickable Markdown link to that section. Do NOT just give text instructions; provide a link.

    Here are the available sections and their corresponding anchor links:
    - About Jash: [About Jash](#about-jash)
    - My Skills: [My Skills](#my-skills)
    - My Education: [My Education](#my-education)
    - My Projects: [My Projects](#my-projects)
    - Contact Jash: [Contact Jash](#contact-jash)

    Example response for skills: "You can find a detailed list of Jash's skills here: [My Skills](#my-skills)"
    Example response for projects: "Sure, check out Jash's projects here: [My Projects](#my-projects)"
    Example response for contact: "You can reach Jash via the contact section: [Contact Jash](#contact-jash)"

    Only provide an anchor link if the user explicitly asks to "go to", "show", "take me to" a section, or if their query is clearly a navigation request. If the question is about the content of a section, summarize the content first, and then offer the link for more details.
    """

    system_general_instructions = """
    When a user asks about specific details related to Jash Kothari, use the provided information directly and factually.
    If the question is a general knowledge question not related to Jash Kothari, answer it to the best of your ability using your general knowledge, but maintain your persona as Jash Kothari's assistant.
    Be polite, concise, and helpful. Do not invent information about Jash Kothari that is not explicitly provided. If you cannot find the answer in the provided information, simply state that you don't have that specific detail about Jash Kothari.
    """

    full_prompt_for_llm = f"{system_intro}\n{navigation_instructions}\n{system_general_instructions}\n\nUser Query: {user_query}"
    return full_prompt_for_llm

# --- Streamlit UI ---

# Set page config after loading personal_info
st.set_page_config(page_title=f"{personal_info['name']}'s Agentic Portfolio Chatbot", layout="centered")

st.title(f"Hi, I'm {personal_info['name']}'s AI Assistant!")
st.write(f"Ask me anything about {personal_info['name']}'s skills, experience, projects, or how to get in touch.")


# Initialize chat history
if "messages" not in st.session_state:
    # Initial message for the chatbot, introducing itself
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! I am a chatbot assistant of {personal_info['name']}. I was created by {personal_info['name']} to help you learn more about his professional background. How can I assist you today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
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

# --- Display Sections for Navigation (with Anchor Tags) ---

# About Section
st.markdown("---")
st.header("About Jash", anchor=SECTIONS["about"]) # This creates the anchor
st.write(personal_info['about_me'])

# Skills Section
st.markdown("---")
st.header("My Skills", anchor=SECTIONS["skills"]) # This creates the anchor
st.markdown(f"- **Programming Languages & Technologies:** {', '.join(personal_info['skills'])}")
# You can expand on skills here if you want more detail than just a list

# Education Section
st.markdown("---")
st.header("My Education", anchor=SECTIONS["education"]) # This creates the anchor
st.markdown(f"- **Degree:** {personal_info['education']}")
# Add more education details if needed

# Projects Section
st.markdown("---")
st.header("My Projects", anchor=SECTIONS["projects"]) # This creates the anchor
for project in personal_info['projects']:
    st.subheader(f"{project['name']}")
    st.write(project['description'])
    st.markdown("---") # Separator for projects, adjust as desired

# Contact Section
st.markdown("---")
st.header("Contact Jash", anchor=SECTIONS["contact"]) # This creates the anchor
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
    st.cache_data.clear() # Clear the cache for load_personal_info
    st.rerun() # Rerun the app to load fresh data
    st.sidebar.success("Personal information reloaded!")