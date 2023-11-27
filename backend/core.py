%%writefile core.py


from typing import Any, List, Dict

# Runpod
import runpod
#Summary and checklist packages
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
# Chat packages
import torch
import os
from langchain.llms import HuggingFaceTextGenInference
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.callbacks import streaming_stdout
from langchain.chains.question_answering import load_qa_chain
from langchain import hub


# Global variables
runpod.api_key = "O6FCUQVG8N3B58HLJZZ84P6DLNDRNQSU4KGANNG4"
huggingfacehub_api_token = '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'
zephyr_repo = 'HuggingFaceH4/zephyr-7b-beta'
rag_prompt = hub.pull("rlm/rag-prompt-mistral")

# Connecting to Runpod LLM
mistral_pod = runpod.get_pods()[0]
  # Runpod URL
inference_server_url_cloud = f"https://{mistral_pod['id']}-80.proxy.runpod.net"

# Tokenizer
embedd_model = 'BAAI/bge-reranker-large'
model_kwargs = {"device": 'cuda'}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=embedd_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


# Chat LLM
callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
chat_llm = HuggingFaceTextGenInference(
    inference_server_url=inference_server_url_cloud,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    callbacks=callbacks,
    streaming=True
)

# Summarization and checklist LLM
sum_check_llm = HuggingFaceTextGenInference(
    inference_server_url=inference_server_url_cloud,
    max_new_tokens=1000,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)


# Call LLM for summary
def run_llm_summarize(document_object: Any):

    docs = document_object

    # Map
    map_template = """<s> [INST] The following is a collection of excerpts from a compliance document:[/INST] </s>
    {docs}
    [INST] Based on the provided excerpts, summarize the main theme.
    Helpful Answer:[/INST]"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=sum_check_llm, prompt=map_prompt)

    # Reduce
    reduce_template = """<s> [INST] The following is set of summaries:[/INST] </s>
    {doc_summaries}
    [INST] Take these and distill it into a final, consolidated summary. Ensure the final output is concise and easy to read.
    Helpful Answer:[/INST]"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=sum_check_llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,)

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)

    # Calling components to generate summary
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    summary = map_reduce_chain.run(split_docs)

    return summary


# Call LLM for checklist
def run_llm_checklist(document_object: Any):

    docs = document_object

    # Map
    map_template = """<s> [INST] The following is a collection of guidance from a compliance document:[/INST] </s>
    {docs}
    [INST] Based on the provided guidance, please create a list of suggestions.
    Helpful Answer:[/INST]"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=sum_check_llm, prompt=map_prompt)

    # Reduce
    reduce_template = """<s> [INST] The following is a colection of suggestions from a compliance document:[/INST] </s>
    {doc_summaries}
    [INST] Take these and distill them into a final, consolidated list of suggestions to comply with the guidance provided in the document.
    Helpful Answer:[/INST]"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=sum_check_llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)

    # Calling components to generate checklist
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    suggestion_list = map_reduce_chain.run(split_docs)

    return suggestion_list


# Call LLM for chat
def run_llm_chat(vector_database: Any, question: str):

    # Vector DB retriever
    retriever = vector_database.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .8, 'k': 10,})
    docs = retriever.get_relevant_documents(question)

   # Chain
    chain = load_qa_chain(chat_llm, chain_type="stuff", prompt=rag_prompt)
    # Run
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=False)

    output = response['output_text']

    sources = [doc.metadata['page'] for doc in response['input_documents']]
    sources.sort()

    return output, sources

