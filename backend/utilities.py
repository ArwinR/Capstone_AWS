%%writefile utilities.py

import os
import re
from typing import Set
import streamlit as st

from core import run_llm_summarize, run_llm_checklist

####################
# Utility functions
####################


# Function to list files in upload directory
def list_files():
     # Checking if the uploads directory exists, and create it if it doesn't
    outdir = "./backend/uploads/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return [f for f in os.listdir(outdir) if os.path.isfile(os.path.join(outdir, f))]


# Saving a copy of PDF for vectorization
def save_upload(file):
    file_name = file.name

    # Checking if the uploads directory exists, and create it if it doesn't
    outdir = "./backend/uploads/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Checking if the file already exists, and saving it if it doesn't
    file_path = os.path.join(outdir, file_name)
    if not os.path.exists(file_path):
        # Saving the file
        with open(os.path.join(outdir, file_name), "wb") as f:
            f.write(file.read())

    return file_path, file_name


# Return response sources in formatted string
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Pulled from pages:\n"
    for i, source in enumerate(sources_list):
        sources_string += f" {source},"
    return sources_string


# Return file name for subheadder
@st.cache_resource()
def clean_name(doc_name):

    cleaned_name = re.sub(r'.pdf', '', doc_name, flags=re.IGNORECASE)
    cleaned_name = re.sub(r'\.', ' ', cleaned_name)
    return cleaned_name


# Creating or loading summarization
@st.cache_data(show_spinner="Hey! 🤖👋 I'm diving into every page of your document to craft your summary. Depending on how many pages there are, this might take a few minutes. It's the perfect moment to grab yourself a coffee and relax for a bit!")
def create_or_load_summ(_doc_object, doc_name):

    # Checking if the uploads directory exists, and create it if it doesn't
    outdir = "./backend/summary/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    file_name = re.sub(r'.pdf', '.txt', doc_name, flags=re.IGNORECASE)
    file_path = "./backend/summary/"+file_name

    # Creating/saving summary if it doesn't exist
    if not os.path.exists(file_path):
        # Generating summary
        summary = run_llm_summarize(document_object=_doc_object)
        # Saving to file
        with open(file_path, 'w') as file:
            file.write(summary)
        return summary
    # Loading saved summary
    else:
        with open(file_path, 'r') as file:
              summary = file.read()
        return summary


# Creating or loading checklist
@st.cache_data(show_spinner="Hi there! 🤖👋 I'm currently compiling a list of suggestions based on each page in your document to create your personalized checklist. Depending on the number of pages, this process might take a little while. Feel free to take a break and grab a coffee while I work on this for you!")
def create_or_load_checklist(_doc_object, doc_name):

    # Checking if the uploads directory exists, and create it if it doesn't
    outdir = "./backend/checklist/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    file_name = re.sub(r'.pdf', '.txt', doc_name, flags=re.IGNORECASE)
    file_path = "./backend/checklist/"+file_name

    # Creating/saving checklist if it doesn't exist
    if not os.path.exists(file_path):
        # Generating checklist
        checklist = run_llm_checklist(document_object=_doc_object)
        # Saving to file
        with open(file_path, 'w') as file:
            file.write(checklist)
        return checklist
    # Loading saved checklist
    else:
        with open(file_path, 'r') as file:
              checklist = file.read()
        return checklist