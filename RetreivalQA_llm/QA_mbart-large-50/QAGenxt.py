#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:45:11 2024

@author: zok
"""

import os 
import csv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
import nltk

# Requirements
# pip install langchain ,faiss-cpu ,pypdf ,sentence_transformers

def text_preprocessing(text):
	# Lowercasing
	text = text.lower()

	# Whitespace normalization
	text = text.replace("\t", " ").replace("\n", " ")

	# Tokenization
	tokens = nltk.word_tokenize(text)

	# Stop word removal
	stop_words = nltk.corpus.stopwords.words("english")
	tokens = [token for token in tokens if token not in stop_words]

	# Stemming (using PorterStemmer)
	stemmer = nltk.PorterStemmer()
	stemmed_tokens = [stemmer.stem(token) for token in tokens]

	# Lemmatization (using WordNetLemmatizer)
	lemmatizer = nltk.WordNetLemmatizer()
	lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
	return stemmed_tokens, lemmatized_tokens

def model_selection():
    api_key= "" #Hugging face API
    llm = HuggingFaceHub(repo_id="facebook/mbart-large-50", model_kwargs={"temperature":1, "max_length":512}, huggingfacehub_api_token=api_key)
    return llm

def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content

    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 30
    )


    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen

def llm_pipeline(file_path):
    # File Preprocessing
    document_ques_gen, document_answer_gen = file_processing(file_path)
    # Model
    llm_ques_gen_pipeline = model_selection()

    prompt_template = """
    You are an expert at creating questions based on arxiv materials and documentation.
    Your goal is to prepare a researcher or programmer for their research or coding.
    You do this by asking questions about the text below:

    ------------
    {text}
    ------------

    Create questions that will prepare the coders or programmers for their tests.
    Make sure not to lose any important information.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = ("""
    You are an expert at creating practice questions based on arxiv material and documentation.
    Your goal is to help a researcher or programmer prepare for a coding or sumarize paper.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    # Chain_Summarize
    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline,
                                            chain_type = "refine",
                                            verbose = True,
                                            question_prompt=PROMPT_QUESTIONS,
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)
    # Emeddings
    embeddings = HuggingFaceEmbeddings()
    # Vector DB
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)
    #Model
    llm_answer_gen = model_selection()

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]
    #RetrievalQA_chain
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen,
                                                chain_type="stuff",
                                                retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list

# Saving Model into csv
def get_csv (file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'static/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain.run(question)
            print("Answer: ", answer)
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer])
    return output_file

get_csv('Attention_is_all_you_need.pdf')
