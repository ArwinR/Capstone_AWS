%%writefile app.py

import re
import time
import os
import streamlit as st
from typing import Set
from streamlit_chat import message

from ingest import ingest_doc, create_doc_obj
from utilities import (list_files,
                       save_upload,
                       create_sources_string,
                       clean_name,
                       create_or_load_summ,
                       create_or_load_checklist
                       )
from core import run_llm_summarize, run_llm_checklist, run_llm_chat


####################
# Utility functions
####################

# # Function to list files in upload directory
# def list_files():
#      # Checking if the uploads directory exists, and create it if it doesn't
#     outdir = "./backend/uploads/"
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)

#     return [f for f in os.listdir(outdir) if os.path.isfile(os.path.join(outdir, f))]

# # Saving a copy of PDF for vectorization
# def save_upload(file):
#     file_name = file.name

#     # Checking if the uploads directory exists, and create it if it doesn't
#     outdir = "./backend/uploads/"
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)

#     # Checking if the file already exists, and saving it if it doesn't
#     file_path = os.path.join(outdir, file_name)
#     if not os.path.exists(file_path):
#         # Saving the file
#         with open(os.path.join(outdir, file_name), "wb") as f:
#             f.write(file.read())

#     return file_path, file_name

# # Return response sources in formatted string
# def create_sources_string(source_urls: Set[str]) -> str:
#     if not source_urls:
#         return ""
#     sources_list = list(source_urls)
#     sources_list.sort()
#     sources_string = "Pulled from pages:\n"
#     for i, source in enumerate(sources_list):
#         sources_string += f" {source},"
#     return sources_string

# # Return file name for subheadder
# @st.cache_resource()
# def clean_name(doc_name):

#     cleaned_name = re.sub(r'.pdf', '', doc_name, flags=re.IGNORECASE)
#     cleaned_name = re.sub(r'\.', ' ', cleaned_name)
#     return cleaned_name

####################
# Global Variables
####################

# Creating list of available saved documents
saved_docs = list_files()


####################
# Streamlit interface
####################

import streamlit as st
import time

title = st.empty()
sub_header = st.empty()

title.title('Welcome to Sibyl 🤖')
sub_header.subheader('Your AI assistant for document review!')

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


# Function for selecting saved document and returning vectorized database
@st.cache_resource(show_spinner='Pulling document from database...')
def select_document_sidebar(file):
    if file_selected:
        outdir = './backend/uploads/'
        file_path = os.path.join(outdir, file)
        loading_message_container = st.empty()
        loading_message_container.info('Hangtight while I search for the document...', icon="🔎")
        vectore_store = ingest_doc(file_path, file)
        raw_docs = create_doc_obj(file_path, file)
        loading_message_container.empty()
        return True, vectore_store, raw_docs
    return False, None, None

# Function for uploading and vectorizing document
@st.cache_resource(show_spinner='Processing the document...')
def upload_document_sidebar(file):
    if file_uploaded:
        file_path, file_name = save_upload(file)
        loading_message_container = st.empty()
        loading_message_container.info("Hangtight, I'm giving the document a quick translation into a computer-friendly language. This shouldn't take more than a minute!",
                                       icon="📑")
        vectore_store = ingest_doc(file_path, file_name)
        raw_docs = create_doc_obj(file_path, file_name)
        loading_message_container.empty()
        return True, vectore_store, raw_docs
    return False, None, None



# Sidebar for selecting/uploading document
upload_placeholder = st.empty()

with upload_placeholder.info(" 👈 Select document or upload your own to start chat"):
    st.sidebar.header("Select a File or Upload New Document")
    with st.sidebar:

        sidebar_completed = False
        st.session_state.vectore_store = None

        # Adding empty line for spacing
        st.markdown("")

         # Radio button for user confirmation with agreement link
        agreement_link = "[User Agreement](https://google.com)"
        user_confirmation = st.checkbox(label=f"I confirm that I have read and understood the {agreement_link}.")

        # Adding empty line for spacing
        st.markdown("")

        if user_confirmation:  # User confirmed, allow document selection/upload

            document_selection = st.radio("Would you like to upload a document or select a saved document?",
                             ["Upload", "Select"],
                             captions = ["Load a new document", "Browse preprocessed documents"])

            # Adding empty line for spacing
            st.markdown("")

            if document_selection == "Upload":
                # Widget to upload new document
                file_uploaded = st.file_uploader("Upload your PDF file", type="pdf", key='FileUpload')
                upload_sidebar_completed, uploaded_vectore_store, raw_document_object = upload_document_sidebar(file_uploaded)

                if file_uploaded:
                    doc_name = file_uploaded.name
                    # Successful message
                    message_container = st.empty()
                    message_container.success('Document processed successfully!', icon="✅")
                    st.session_state.message_container = message_container

                    # Changing control variable to enable chatting
                    st.session_state.vectore_store = uploaded_vectore_store
                    st.session_state.document_object = raw_document_object
                    sidebar_completed = upload_sidebar_completed
                    # upload_placeholder.empty()

            elif document_selection == "Select":
                # Create a dropdown menu in the sidebar for file selection
                file_selected = st.sidebar.selectbox(label="Select a File", options=saved_docs, placeholder='Choose an option', index=None )
                select_sidebar_completed, selected_vectore_store, raw_document_object = select_document_sidebar(file_selected)

                if file_selected:
                    doc_name = file_selected
                    # Successful message
                    message_container = st.empty()
                    message_container.success('Document loaded!', icon="✅")
                    st.session_state.message_container = message_container

                    # Changing control variable to enable chatting
                    st.session_state.vectore_store = selected_vectore_store
                    st.session_state.document_object = raw_document_object
                    sidebar_completed = select_sidebar_completed
                    # upload_placeholder.empty()
            else:
                st.info("Let's pick a document to review!", icon="☝️")


# Summarize or chat selection
if sidebar_completed:
    title.title('Sibyl 🤖 is ready to go!')
    sub_header.subheader(f"Let's talk about {clean_name(doc_name)}")
    upload_placeholder.info("I've got your document ready!", icon="👇")
    st.divider()
    st.write("Get a summary or a personalized checklist of action items by simply clicking the buttons provided below.")

    col1, col2 = st.columns(2)
    with col1:
        summarize = st.button("Summarize")
    with col2:
        quick_list = st.button("Quick Guidance")



# Summarizing document
if "vectore_store" in st.session_state is not None and sidebar_completed and summarize:
    summary = create_or_load_summ(_doc_object=st.session_state.document_object, doc_name=doc_name)
    st.write(summary)

# Checklist list
if "vectore_store" in st.session_state is not None and sidebar_completed and quick_list:
    checklist = create_or_load_checklist(_doc_object=st.session_state.document_object, doc_name=doc_name)
    st.write(checklist)


# Starting chat
if "vectore_store" in st.session_state is not None and sidebar_completed:
    vectore_store = st.session_state.vectore_store

    # Adding empty line for spacing
    st.markdown("")
    st.caption("Start chatting by entering your question in the query queue at the bottom of the page!")
    st.divider()
    prompt = st.chat_input(placeholder="Enter your question here...")

    if prompt:
        with st.spinner("Searching document for the answer..."):
            generated_response, sources = run_llm_chat(vector_database=vectore_store, question=prompt)
            formatted_response = (f"{generated_response} \n\n {create_sources_string(set(sources))}")

        st.session_state.chat_history.append((prompt, generated_response))
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(formatted_response)


    # Displaying generated response with unique keys
    if st.session_state["chat_answers_history"]:
        for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        ):
            with st.chat_message("user", avatar="🤔"):
                st.write(user_query)
            with st.chat_message("ai", avatar="🤖"):
                st.write(generated_response)
