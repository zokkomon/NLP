import os 
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline 
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain

# Requirements
# pip install langchain faiss-cpu pypdf sentence_transformers

# Model Initailization
def model_selection(ques=True):
    """
    Initialize and return the tokenizer and base model for question or answer generation.

    Parameters:
    - ques (bool): If True, initializes for question generation; otherwise, initializes for answer generation.

    Returns:
    - tokenizer: Hugging Face tokenizer
    - base_model: Hugging Face base model
    """
    
    if ques:
      checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
      tokenizer = AutoTokenizer.from_pretrained(checkpoint)
      base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    else:
      checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
      tokenizer = AutoTokenizer.from_pretrained(checkpoint)
      base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return tokenizer, base_model

# Loading Model into transformer Pipeline
def load_llm(ques=True):
    """
    Load the Language Model (LLM) into a transformer pipeline for question or answer generation.
    
    Parameters:
    - ques (bool): If True, loads the model for question generation; otherwise, loads the model for answer generation.
    
    Returns:
    - llm: HuggingFacePipeline for question or answer generation
    """
    if ques:
      tokenizer, base_model = model_selection()
      pipe = pipeline(
          'text2text-generation',
          model = base_model,
          tokenizer = tokenizer,
          max_length = 256,
          do_sample = True,
          temperature = 0.3,
          top_p= 0.95
      )
      llm = HuggingFacePipeline(pipeline=pipe)

    else:
      tokenizer, base_model = model_selection(ques=False)
      pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95
        )
      llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm

def file_processing(file_path):
    """
    Process a PDF file, extracting content and splitting it into chunks for further processing.
    
    Parameters:
    - file_path (str): Path to the PDF file.
    
    Returns:
    - document_ques_gen: List of Document objects for question generation
    - document_answer_gen: List of Document objects for answer generation
    """

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
    # Chunking
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 30
    )

    # Chunking
    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen

def llm_pipeline(file_path):
    """
    Run the Language Model (LLM) pipeline for question and answer generation.
    
    Parameters:
    - file_path (str): Path to the PDF file.
    
    Returns:
    - answer_generation_chain: RetrievalQA chain for answer generation
    - filtered_ques_list: List of filtered questions
    """

    document_ques_gen, document_answer_gen = file_processing(file_path)
    # Model
    llm_ques_gen_pipeline = load_llm()

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
    Your goal is to help a researcher or programmer prepare for a coding or researching.
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
    # Chain_summarize module
    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline,
                                            chain_type = "refine",
                                            verbose = True,
                                            question_prompt=PROMPT_QUESTIONS,
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)
    #Emdedding vector
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #Vector_DB
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)
    # model
    llm_answer_gen = load_llm(ques=False)

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]
    #RetrievalAgent_QAchain
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen,
                                                chain_type="stuff",
                                                retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list

# Saving file to csv
def get_csv (file_path):
    """
    Generate questions, retrieve answers, and save the results to a CSV file.
    
    Parameters:
    - file_path (str): Path to the PDF file.
    
    Returns:
    - output_file (str): Path to the generated CSV file.
    """
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'ouput/'
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

get_csv('input/lovenia_question_answer_generation.pdf')
