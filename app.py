import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os
import torch
import pickle
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ===============================
# 1. Streamlit App Configuration
# ===============================
st.set_page_config(page_title="📥 Email Chat Application", layout="wide")
st.title("💬 Turn Emails into Conversations—Effortless Chat with Your Inbox! 📩")

# ===============================
# 2. Initialize Session State Variables
# ===============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "creds" not in st.session_state:
    st.session_state.creds = None
if "auth_url" not in st.session_state:
    st.session_state.auth_url = None
if "auth_code" not in st.session_state:
    st.session_state.auth_code = ""
if "flow" not in st.session_state:
    st.session_state.flow = None
if "data_chunks" not in st.session_state:
    st.session_state.data_chunks = []  # List to store all email chunks
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# For storing candidate context details.
if "candidate_context" not in st.session_state:
    st.session_state.candidate_context = None
if "raw_candidates" not in st.session_state:
    st.session_state.raw_candidates = None

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Flags to ensure success messages are shown only once
if "candidates_message_shown" not in st.session_state:
    st.session_state.candidates_message_shown = False
if "vector_db_message_shown" not in st.session_state:
    st.session_state.vector_db_message_shown = False

def count_tokens(text):
    return len(text.split())

# ===============================
# 3. Gmail Authentication Functions (Updated)
# ===============================
def reset_session_state():
    st.session_state.authenticated = False
    st.session_state.creds = None
    st.session_state.auth_url = None
    st.session_state.auth_code = ""
    st.session_state.flow = None
    st.session_state.data_chunks = []
    st.session_state.embeddings = None
    st.session_state.vector_store = None
    st.session_state.candidate_context = None
    st.session_state.raw_candidates = None
    st.session_state.messages = []
    st.session_state.candidates_message_shown = False
    st.session_state.vector_db_message_shown = False
    for filename in ["token.json", "data_chunks.pkl", "embeddings.pkl", "vector_store.index", "vector_database.pkl"]:
        if os.path.exists(filename):
            os.remove(filename)

def authenticate_gmail(credentials_file):
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    creds = None
    if os.path.exists('token.json'):
        try:
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            if creds and creds.valid:
                st.session_state.creds = creds
                st.session_state.authenticated = True
                if not st.session_state.candidates_message_shown:
                    st.success("✅ Authentication successful!")
                    st.session_state.candidates_message_shown = True
                return creds
        except Exception as e:
            st.error(f"❌ Invalid token.json file: {e}")
            os.remove('token.json')
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            st.session_state.creds = creds
            st.session_state.authenticated = True
            if not st.session_state.candidates_message_shown:
                st.success("✅ Authentication successful!")
                st.session_state.candidates_message_shown = True
            with open('token.json', 'w') as token_file:
                token_file.write(creds.to_json())
            return creds
        else:
            if not st.session_state.flow:
                st.session_state.flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
                st.session_state.flow.redirect_uri = 'http://localhost'
            auth_url, _ = st.session_state.flow.authorization_url(prompt='consent')
            st.session_state.auth_url = auth_url
            st.info("🔗 **Authorize the application by visiting the URL below:**")
            st.markdown(f"[Authorize]({st.session_state.auth_url})")

def submit_auth_code():
    try:
        # Attempt to fetch the token using the provided authorization code
        st.session_state.flow.fetch_token(code=st.session_state.auth_code)
        st.session_state.creds = st.session_state.flow.credentials
        
        # Attempt to write the credentials to token.json
        with open('token.json', 'w') as token_file:
            token_json = st.session_state.creds.to_json()
            token_file.write(token_json)
        
        # If writing is successful, update the session state
        st.session_state.authenticated = True
        st.success("✅ Authentication successful!")
    except Exception as e:
        # If any error occurs, ensure the authenticated flag is not set
        st.session_state.authenticated = False
        st.error(f"❌ Error during authentication: {e}")

# ===============================
# 4. Email Data Extraction, Embedding and Vector Store Functions
# ===============================
def extract_email_body(payload):
    if 'body' in payload and 'data' in payload['body'] and payload['body']['data']:
        try:
            return base64.urlsafe_b64decode(payload['body']['data'].encode('UTF-8')).decode('UTF-8')
        except Exception as e:
            st.error(f"Error decoding email body: {e}")
            return ""
    if 'parts' in payload:
        for part in payload['parts']:
            if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
                try:
                    return base64.urlsafe_b64decode(part['body']['data'].encode('UTF-8')).decode('UTF-8')
                except Exception as e:
                    st.error(f"Error decoding email part: {e}")
                    continue
        if payload['parts']:
            first_part = payload['parts'][0]
            if 'data' in first_part.get('body', {}):
                try:
                    return base64.urlsafe_b64decode(first_part['body']['data'].encode('UTF-8')).decode('UTF-8')
                except Exception as e:
                    st.error(f"Error decoding fallback email part: {e}")
                    return ""
    return ""

def combine_email_text(email):
    # Build the complete email text by joining parts with HTML line breaks.
    parts = []
    if email.get("sender"):
        parts.append("From: " + email["sender"])
    if email.get("to"):
        parts.append("To: " + email["to"])
    if email.get("date"):
        parts.append("Date: " + email["date"])
    if email.get("subject"):
        parts.append("Subject: " + email["subject"])
    if email.get("body"):
        parts.append("Body: " + email["body"])
    return "<br>".join(parts)

def create_chunks_from_gmail(service, label):
    try:
        messages = []
        result = service.users().messages().list(userId='me', labelIds=[label], maxResults=500).execute()
        messages.extend(result.get('messages', []))
        while 'nextPageToken' in result:
            token = result["nextPageToken"]
            result = service.users().messages().list(userId='me', labelIds=[label], maxResults=500, pageToken=token).execute()
            messages.extend(result.get('messages', []))

        data_chunks = []
        progress_bar = st.progress(0)
        total = len(messages)
        for idx, msg in enumerate(messages):
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            headers = msg_data.get('payload', {}).get('headers', [])
            email_dict = {"id": msg['id']}
            for header in headers:
                name = header.get('name', '').lower()
                if name == 'from':
                    email_dict['sender'] = header.get('value', '')
                elif name == 'subject':
                    email_dict['subject'] = header.get('value', '')
                elif name == 'to':
                    email_dict['to'] = header.get('value', '')
                elif name == 'date':
                    email_dict['date'] = header.get('value', '')
            email_dict['body'] = extract_email_body(msg_data.get('payload', {}))
            data_chunks.append(email_dict)
            progress_bar.progress(min((idx + 1) / total, 1.0))
        st.session_state.data_chunks.extend(data_chunks)
        if not st.session_state.vector_db_message_shown:
            st.success(f"📁 Vector database loaded successfully from upload")
            st.session_state.vector_db_message_shown = True
    except Exception as e:
        st.error(f"❌ Error creating chunks from Gmail for label '{label}': {e}")

# -------------------------------
# Cached model loaders for efficiency
# -------------------------------
@st.cache_resource
def get_embed_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

def embed_emails(email_chunks):
    st.header("🔄 Embedding Data and Creating Vector Store")
    progress_bar = st.progress(0)
    with st.spinner('🔄 Embedding data...'):
        try:
            embed_model, device = get_embed_model()
            combined_texts = [combine_email_text(email) for email in email_chunks]
            batch_size = 64
            embeddings = []
            for i in range(0, len(combined_texts), batch_size):
                batch = combined_texts[i:i+batch_size]
                batch_embeddings = embed_model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    device=device
                )
                embeddings.append(batch_embeddings)
                progress_value = min((i + batch_size) / len(combined_texts), 1.0)
                progress_bar.progress(progress_value)
            embeddings = np.vstack(embeddings)
            faiss.normalize_L2(embeddings)
            st.session_state.embeddings = embeddings
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            st.session_state.vector_store = index
            if not st.session_state.candidates_message_shown:
                st.success("✅ Data embedding and vector store created successfully!")
                st.session_state.candidates_message_shown = True
        except Exception as e:
            st.error(f"❌ Error during embedding: {e}")

# New function to save the entire vector database as a single pickle file.
def save_vector_database():
    try:
        vector_db = {
            "vector_store": st.session_state.vector_store,
            "embeddings": st.session_state.embeddings,
            "data_chunks": st.session_state.data_chunks
        }
        db_data = pickle.dumps(vector_db)
        st.download_button(
            label="💾 Download Vector Database",
            data=db_data,
            file_name="vector_database.pkl",
            mime="application/octet-stream"
        )
    except Exception as e:
        st.error(f"❌ Error saving vector database: {e}")

# ===============================
# 5. Handling User Queries (User-Controlled Threshold)
# ===============================
def preprocess_query(query):
    return query.lower().strip()

def process_candidate_emails(query, similarity_threshold):
    """
    Process the query by computing its embedding, searching the vector store,
    filtering candidates based on a similarity threshold, and building a context string.
    """
    TOP_K = 20  # Increased to allow for threshold filtering

    # Reset candidate context for each query
    st.session_state.candidate_context = None
    st.session_state.raw_candidates = None

    if st.session_state.vector_store is None:
        st.error("❌ Please process your email data or load a saved vector database first.")
        return

    try:
        embed_model, device = get_embed_model()
        processed_query = preprocess_query(query)
        query_embedding = embed_model.encode(
            [processed_query],
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device
        )
        faiss.normalize_L2(query_embedding)

        # Perform search
        distances, indices = st.session_state.vector_store.search(query_embedding, TOP_K)
        candidates = []
        for idx, sim in zip(indices[0], distances[0]):
            # Include candidate only if similarity meets the threshold
            if sim >= similarity_threshold:
                candidates.append((st.session_state.data_chunks[idx], sim))
        if not candidates:
            # Append warning message as assistant message
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ No matching embeddings found for your query with the selected threshold."})
            return

        # Build the context string by concatenating all matching email texts using HTML breaks.
        context_str = ""
        for candidate, sim in candidates:
            context_str += combine_email_text(candidate) + "<br><br>"

        # Optionally limit context size.
        MAX_CONTEXT_TOKENS = 500
        context_tokens = context_str.split()
        if len(context_tokens) > MAX_CONTEXT_TOKENS:
            context_str = " ".join(context_tokens[:MAX_CONTEXT_TOKENS])

        st.session_state.candidate_context = context_str
        st.session_state.raw_candidates = candidates
    except Exception as e:
        st.error(f"❌ An error occurred during processing: {e}")

def call_llm_api(query):
    """
    Send the user's query along with the concatenated matching email texts (context)
    to the LLM API and display the AI response.
    """
    if not st.session_state.candidate_context:
        st.error("❌ No candidate context available. Please try again.")
        return

    # Retrieve the API key from the environment variable 'GroqAPI'
    api_key = os.getenv("GroqAPI")
    if not api_key:
        st.error("❌ API key not found. Please ensure 'GroqAPI' is set in Hugging Face Secrets.")
        return

    payload = {
        "model": "llama-3.3-70b-versatile",  # Adjust model as needed.
        "messages": [
            {"role": "system", "content": f"Use the following context:\n{st.session_state.candidate_context}"},
            {"role": "user", "content": query}
        ]
    }
    url = "https://api.groq.com/openai/v1/chat/completions"  # Verify this endpoint

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred.
        response_json = response.json()
        generated_text = response_json["choices"][0]["message"]["content"]
        # Append AI response to chat messages
        st.session_state.messages.append({"role": "assistant", "content": generated_text})
    except requests.exceptions.HTTPError as http_err:
        try:
            error_info = response.json().get("error", {})
            error_message = error_info.get("message", "An unknown error occurred.")
            st.session_state.messages.append({"role": "assistant", "content": f"❌ HTTP error occurred: {error_message}"})
        except ValueError:
            st.session_state.messages.append({"role": "assistant", "content": f"❌ HTTP error occurred: {response.status_code} - {response.text}"})
    except Exception as err:
        st.session_state.messages.append({"role": "assistant", "content": f"❌ An unexpected error occurred: {err}"})

def handle_user_query():
    st.header("💬 Let's Chat with Your Emails")

    # The similarity threshold is now fixed to 0.2 (hidden from the user).
    similarity_threshold = 0.2

    # Chat input for user queries
    user_input = st.chat_input("Enter your query:")

    if user_input:
        # Append user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Process the query with the fixed similarity threshold
        process_candidate_emails(user_input, similarity_threshold)

        if st.session_state.candidate_context:
            # Send the query to the LLM API
            call_llm_api(user_input)

    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # The Matching Email Chunks expander has been removed.

# ===============================
# 6. Main Application Logic
# ===============================
def main():
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    st.sidebar.header("🔒 Gmail Authentication")
    credentials_file = st.sidebar.file_uploader("📁 Upload credentials.json", type=["json"])

    data_management_option = st.sidebar.selectbox(
        "Choose an option",
        ["Upload Pre-existing Data", "Authenticate and Create New Data"],
        index=1  # Default to "Authenticate and Create New Data"
    )

    # Show instructions only if "Authenticate and Create New Data" is selected
    # and no credentials file or vector database is present.
    if (data_management_option == "Authenticate and Create New Data") and (credentials_file is None) and (st.session_state.vector_store is None):
        with st.expander("📖 Application Guidelines and Instructions", expanded=True):
            st.markdown("## Application Guidelines and Instructions")
            st.markdown("### Creating the **credentials.json** File")
            st.markdown(
                "1. **Create or Select a Google Cloud Project:**\n"
                "   - Visit the [Google Cloud Console](https://console.cloud.google.com/) and create or select an existing project.\n\n"
                "2. **Enable the Gmail API:**\n"
                "   - Navigate to **APIs & Services > Library**, search for **Gmail API**, and click **Enable**.\n\n"
                "3. **Configure the OAuth Consent Screen:**\n"
                "   - Go to **APIs & Services > OAuth consent screen** and configure the required settings (choose External or Internal as needed).\n\n"
                "4. **Create OAuth Client ID Credentials:**\n"
                "   - Under **APIs & Services > Credentials**, click **Create Credentials** and select **OAuth client ID**.\n"
                "   - Choose **Web app** as the application type, give it a name, and create the credentials.\n\n"
                "5. **Download the Credentials:**\n"
                "   - Click **Download JSON**. Save this file as **credentials.json** and keep it secure.\n\n"
                "### Application Flow\n"
                "- **Upload Credentials:** Upload your **credentials.json** file using the sidebar uploader.\n"
                "- **Pre-existing Data:** Alternatively, select **Upload Pre-existing Data** to load an existing vector database.\n"
                "- Once credentials are provided or data is loaded, the guidelines will be hidden and you can proceed with authentication and data processing."
            )

    if data_management_option == "Upload Pre-existing Data":
        uploaded_db = st.sidebar.file_uploader("📁 Upload vector database (vector_database.pkl)", type=["pkl"])
        if uploaded_db:
            # Removed file size warning message.
            try:
                vector_db = pickle.load(uploaded_db)
                st.session_state.vector_store = vector_db.get("vector_store")
                st.session_state.embeddings = vector_db.get("embeddings")
                st.session_state.data_chunks = vector_db.get("data_chunks")
                if not st.session_state.vector_db_message_shown:
                    st.success("📁 Vector database loaded successfully from upload!")
                    st.session_state.vector_db_message_shown = True
            except Exception as e:
                st.error(f"❌ Error loading vector database: {e}")
    elif data_management_option == "Authenticate and Create New Data":
        if credentials_file and st.sidebar.button("🔓 Authenticate"):
            reset_session_state()
            with open("credentials.json", "wb") as f:
                f.write(credentials_file.getbuffer())
            authenticate_gmail("credentials.json")

        if st.session_state.auth_url:
            st.sidebar.markdown("### 🔗 **Authorization URL:**")
            st.sidebar.markdown(f"[Authorize]({st.session_state.auth_url})")
            st.sidebar.text_input("🔑 Enter the authorization code:", key="auth_code")
            if st.sidebar.button("✅ Submit Authentication Code"):
                submit_auth_code()

    if data_management_option == "Authenticate and Create New Data" and st.session_state.authenticated:
        st.sidebar.success("✅ You are authenticated!")
        st.header("📂 Data Management")
        # Multi-select widget for folders (labels)
        folders = st.multiselect("Select Labels (Folders) to Process Emails From:", 
                                 ["INBOX", "SENT", "DRAFT", "TRASH", "SPAM"], default=["INBOX"])
        if st.button("📥 Create Chunks and Embed Data"):
            service = build('gmail', 'v1', credentials=st.session_state.creds)
            all_chunks = []
            # Process each selected folder
            for folder in folders:
                # Clear temporary data_chunks so that each folder's data is separate
                st.session_state.data_chunks = []
                create_chunks_from_gmail(service, folder)
                if st.session_state.data_chunks:
                    all_chunks.extend(st.session_state.data_chunks)
            st.session_state.data_chunks = all_chunks
            if st.session_state.data_chunks:
                embed_emails(st.session_state.data_chunks)
        if st.session_state.vector_store is not None:
            with st.expander("💾 Download Data", expanded=False):
                save_vector_database()

    if st.session_state.vector_store is not None:
        handle_user_query()

if __name__ == "__main__":
    main()
