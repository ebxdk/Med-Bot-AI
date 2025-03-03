# -*- coding: utf-8 -*-
"""flashcard.ipynb (Modified for API Integration)"""

import os
import json
import openai
import faiss
import numpy as np
import fitz  # PyMuPDF
import ipywidgets as widgets
from IPython.display import display, clear_output

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-REPLACE_WITH_YOUR_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Utility Functions
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text_data = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text").strip()
            if len(text) > 50:
                text_data.append(text)
    return "\n".join(text_data)

import tiktoken
def chunk_text(text, max_tokens=300):
    encoding = tiktoken.get_encoding("cl100k_base")
    words, chunks, chunk = text.split(), [], []
    token_count = 0
    for word in words:
        word_tokens = len(encoding.encode(word))
        if token_count + word_tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, token_count = [], 0
        chunk.append(word)
        token_count += word_tokens
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def process_pdf_for_rag(pdf_path, university, course):
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    return [{"university": university, "course": course, "chunk_id": i, "text": chunk} for i, chunk in enumerate(chunks)]

# Load course data (adjust path as needed)
course_chunks = process_pdf_for_rag("collegebiologysummaryquestions.pdf", "University of Toronto", "BIOL 101")

def generate_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

# Pre-generate embeddings for course chunks
for chunk in course_chunks:
    chunk["embedding"] = generate_embedding(chunk["text"])

def store_embeddings_faiss(data):
    d = len(data[0]["embedding"])
    index = faiss.IndexFlatL2(d)
    embeddings = np.array([chunk["embedding"] for chunk in data]).astype("float32")
    index.add(embeddings)
    return index

faiss_index = store_embeddings_faiss(course_chunks)

def search_relevant_chunks(query, faiss_index, data, top_k=3):
    query_embedding = np.array([generate_embedding(query)]).astype("float32")
    _, indices = faiss_index.search(query_embedding, top_k)
    return [data[i] for i in indices[0]]

def generate_flashcards_with_context(context):
    prompt = f"""
You are an AI assistant that generates concise and high-quality flashcards for students.
Below is some course material:

{context}

Based on the above, generate 10 high-quality flashcards in JSON format.
Each flashcard should have:
- A "question" field.
- An "answer" field.

Return the flashcards as a JSON list.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return json.loads(response["choices"][0]["message"]["content"])

# -----------------------------
# API Entry Point: run()
# -----------------------------
def run(input_data):
    """
    Expects input_data to be a dictionary with keys:
      - "university": university name,
      - "course": course name,
      - "topic": the topic to search for.
    Returns the generated flashcards as a JSON string.
    """
    university = input_data.get("university", "")
    course = input_data.get("course", "")
    topic = input_data.get("topic", "")
    retrieved_chunks = search_relevant_chunks(topic, faiss_index, course_chunks)
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    flashcards = generate_flashcards_with_context(context)
    return json.dumps(flashcards)

# -----------------------------
# Interactive UI (Only if run directly)
# -----------------------------
if __name__ == '__main__':
    university_input = widgets.Text(placeholder="Enter University Name", description="University:")
    course_input = widgets.Text(placeholder="Enter Course Name", description="Course:")
    topic_input = widgets.Text(placeholder="Enter Topic (max 150 chars)", description="Topic:", max_length=150)
    generate_button = widgets.Button(description="Generate Flashcards", button_style="success")
    redo_button = widgets.Button(description="Redo Flashcards", button_style="warning")
    output_area = widgets.Output()
    redo_button.layout.display = 'none'
    
    def on_button_click(b):
        university = university_input.value.strip()
        course = course_input.value.strip()
        topic = topic_input.value.strip()
        if university and course and topic:
            clear_output(wait=True)
            display(university_input, course_input, topic_input, generate_button, redo_button, output_area)
            print(f"\nSearching for flashcards on **{topic}** in {course} at {university}...")
            global retrieved_chunks
            retrieved_chunks = search_relevant_chunks(topic, faiss_index, course_chunks)
            context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
            flashcards = generate_flashcards_with_context(context)
            with output_area:
                output_area.clear_output()
                print("\nAI-Generated Flashcards:\n")
                for i, card in enumerate(flashcards, 1):
                    print(f"{i}. Q: {card['question']}")
                    print(f"   A: {card['answer']}\n")
            redo_button.layout.display = 'inline-block'
        else:
            with output_area:
                output_area.clear_output()
                print("Please fill in all fields before generating flashcards.")
    
    def on_redo_click(b):
        context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        flashcards = generate_flashcards_with_context(context)
        with output_area:
            output_area.clear_output()
            print("\nRegenerating new AI-Generated Flashcards...\n")
            for i, card in enumerate(flashcards, 1):
                print(f"{i}. Q: {card['question']}")
                print(f"   A: {card['answer']}\n")
    
    generate_button.on_click(on_button_click)
    redo_button.on_click(on_redo_click)
    display(university_input, course_input, topic_input, generate_button, redo_button, output_area)
