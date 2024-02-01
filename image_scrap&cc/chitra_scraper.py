#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:21:59 2024

@author: zok
"""

import os
import time
import torch
from PIL import Image
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from transformers import DetrImageProcessor, DetrForObjectDetection, AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain.tools import BaseTool
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

def remove_non_jpeg_images(path):
    def is_jpeg(image_path):
        try:
            i = Image.open(image_path)
            return i.format in ['JPEG', 'PNG', 'GIF']
        except IOError:
            return False

    # get list of all files
    files = os.listdir(path)

    # remove non jpeg images
    for file in files:
        if not is_jpeg(os.path.join(path, file)):
            print(f'Removing {file}')
            os.remove(os.path.join(path, file))


def chitra_scraper(url, saving_dir):
    # Set up the webdriver
    driver = webdriver.Firefox()  # Or whichever browser you prefer
    driver.get(url)
    
    # Simulate scrolling
    for _ in range(10):  # Adjust this value based on how much you want to scroll
        body = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'body')))
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.7)  # Pause between scrolls
    
    # Get the HTML of the page after scrolling
    html = driver.page_source
    
    # Parse the content
    soup = BeautifulSoup(html, 'lxml')
    
    # Find all image tags
    img_tags = soup.find_all('img')
    
    # Create the directory if it doesn't exist
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    # Download each image
    for img in img_tags:
        img_url = urljoin(url, img['src'])
        img_data = requests.get(img_url).content
        with open(os.path.join(saving_dir, os.path.basename(img_url)), 'wb') as handler:
            handler.write(img_data)
    
    driver.quit()

class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, image_path, query):
        model_id = "vikhyatk/moondream1"
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = Tokenizer.from_pretrained(model_id)

        image = Image.open(image_path).convert('RGB')
        enc_image = model.encode_image(image)
        caption = model.answer_question(enc_image, query, tokenizer)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path, query):
        image = Image.open(img_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
        
# Model Initailization
def model_selection():
    """
    Initialize and return the tokenizer and base model for question or answer generation.

    Parameters:
    - ques (bool): If True, initializes for question generation; otherwise, initializes for answer generation.

    Returns:
    - tokenizer: Hugging Face tokenizer
    - base_model: Hugging Face base model
    """
    checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    
    return tokenizer, base_model

# Loading Model into transformer Pipeline
def load_llm():
    """
    Load the Language Model (LLM) into a transformer pipeline for question or answer generation.

    Parameters:
    - ques (bool): If True, loads the model for question generation; otherwise, loads the model for answer generation.

    Returns:
    - llm: HuggingFacePipeline for question or answer generation
    """
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

    return llm

        
#initialize the gent
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = load_llm()

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

## Scrap images
path = 'sign_up_page'
url = "https://www.behance.net/search/projects/sign%20up"
chitra_scraper(url, path)

## filtering
remove_non_jpeg_images(path)

## captioning
image_path = "sign_up"
query = "Please provide a comprehensive description of this image, including a concise overview of its content, details on color styles, font usage, font colors, and the specific location of the content within the image. Your detailed description should encompass all relevant visual elements and design aspects present in the image."
##Currently not work because need big Parameters llm
# response = agent.run(f'{query}, this is the image path: {image_path}')
ImageCaptionTool()._run(image_path, query)







