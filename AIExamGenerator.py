# -*- coding: utf-8 -*-
"""AIExamGenerator.ipynb (Modified for API Integration)"""

import os
import json
import openai
import faiss
import numpy as np
import fitz  # PyMuPDF
import tiktoken
import re
import pytesseract
from PIL import Image
import io
from rank_bm25 import BM25Okapi
import ipywidgets as widgets
from IPython.display import display, clear_output

# -----------------------------
# Set and get OpenAI API Key
# -----------------------------
# Load the API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please set OPENAI_API_KEY in your environment.")

# -----------------------------
# Global Variables & Filenames
# -----------------------------
feedback_history = []   # Stores user feedback
improvement_history = []  # Stores GPT-generated improvements
last_generated_exam = ""  # Stores the last exam for evaluation
FEEDBACK_FILE = "feedback_history.json"
STUDENT_DATA_FILE = "student_data.json"
student_profiles = {}

# -----------------------------
# Data Persistence Functions
# -----------------------------
def save_feedback():
    data = {
        "feedback_history": feedback_history,
        "improvement_history": improvement_history
    }
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f)

def load_feedback():
    global feedback_history, improvement_history
    try:
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)
            feedback_history = data.get("feedback_history", [])
            improvement_history = data.get("improvement_history", [])
    except FileNotFoundError:
        feedback_history = []
        improvement_history = []

def load_student_profiles():
    global student_profiles
    try:
        with open(STUDENT_DATA_FILE, "r") as f:
            student_profiles = json.load(f)
    except FileNotFoundError:
        student_profiles = {}

def save_student_profiles():
    with open(STUDENT_DATA_FILE, "w") as f:
        json.dump(student_profiles, f)

def track_student_progress(student_id, feedback):
    if student_id not in student_profiles:
        student_profiles[student_id] = {"feedback": [], "difficulty_preference": "Medium"}
    student_profiles[student_id]["feedback"].append(feedback)
    if "too easy" in feedback.lower():
        student_profiles[student_id]["difficulty_preference"] = "Hard"
    elif "too hard" in feedback.lower():
        student_profiles[student_id]["difficulty_preference"] = "Easy"
    save_student_profiles()

def get_student_difficulty(student_id):
    if student_id in student_profiles:
        return student_profiles[student_id]["difficulty_preference"]
    return "Medium"

# Load data on startup
load_student_profiles()
load_feedback()

# -----------------------------
# Utility Functions
# -----------------------------
def extract_text_with_ocr(pdf_path):
    text_data = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text").strip()
            if not text:
                img = page.get_pixmap()
                img = Image.open(io.BytesIO(img.tobytes()))
                text = pytesseract.image_to_string(img)
            if len(text) > 50:
                text_data.append(text)
    return "\n".join(text_data)

def chunk_text(text, max_tokens=500):
    encoding = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    chunk = []
    token_count = 0
    for word in words:
        word_tokens = len(encoding.encode(word))
        if token_count + word_tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
            token_count = 0
        chunk.append(word)
        token_count += word_tokens
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

embedding_cache = {}
def generate_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    embedding_cache[text] = embedding
    return embedding

# -----------------------------
# FAISS and BM25 Functions
# -----------------------------
def store_exams_faiss(pdf_paths):
    d = 1536
    index = faiss.IndexFlatL2(d)
    exam_chunks = []
    all_embeddings = []
    for pdf_path in pdf_paths:
        pdf_text = extract_text_with_ocr(pdf_path)
        chunks = chunk_text(pdf_text)
        for chunk in chunks:
            exam_chunks.append({
                "file_name": os.path.basename(pdf_path),
                "chunk_text": chunk,
                "full_text": pdf_text
            })
    for chunk_data in exam_chunks:
        emb = generate_embedding(chunk_data["chunk_text"])
        all_embeddings.append(emb)
    index.add(np.array(all_embeddings).astype("float32"))
    return index, exam_chunks

def store_course_material_faiss(pdf_path):
    d = 1536
    index = faiss.IndexFlatL2(d)
    pdf_text = extract_text_with_ocr(pdf_path)
    chunks = chunk_text(pdf_text)
    chunk_embeddings = [generate_embedding(ch) for ch in chunks]
    index.add(np.array(chunk_embeddings).astype("float32"))
    tokenized_corpus = [ch.split() for ch in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return index, chunks, bm25

def retrieve_practice_exam(query, exam_index, exam_chunks, top_k=1):
    query_emb = np.array([generate_embedding(query)]).astype("float32")
    distances, indices = exam_index.search(query_emb, top_k)
    if len(indices[0]) == 0:
        return exam_chunks[0]["full_text"]
    top_chunk = exam_chunks[indices[0][0]]
    return top_chunk["full_text"]

def retrieve_course_material(query, course_index, course_chunks, bm25, top_k=3):
    query_emb = np.array([generate_embedding(query)]).astype("float32")
    distances, indices = course_index.search(query_emb, top_k)
    if len(indices[0]) == 0:
        top_bm25 = bm25.get_top_n(query.split(), course_chunks, n=top_k)
        return top_bm25
    results = [course_chunks[i] for i in indices[0]]
    return results

# -----------------------------
# Main Functionality: Exam Generation
# -----------------------------
def generate_practice_exam_realtime(course, difficulty):
    global last_generated_exam
    # For demonstration, use the course string as the query
    exam_text = retrieve_practice_exam(course, exam_index, practice_exams)
    course_material = retrieve_course_material(course, course_index, course_chunks, bm25, top_k=3)
    combined_improvements = "\n".join(improvement_history)
    context = (
        f"--- Past Exam Reference ---\n{exam_text}\n\n"
        f"--- Relevant Course Chunks ---\n" + "\n\n".join(course_material)
    )
    system_prompt = """\
You are a highly specialized AI in creating university-level computer science exams.
Your primary objective:
1. Closely match the tone, style, and structure of the provided past exams.
2. Incorporate relevant course content from the retrieved chunks.
3. Make the exam comprehensive and realistic.
4. Keep your chain-of-thought internal. Output only the final refined exam text.
"""
    user_prompt = f"""\
Generate a **{difficulty}**-level practice exam for the course: **{course}**.
Incorporate the following improvements:
{combined_improvements}

Use the following context for inspiration:
{context}

Now, generate a cohesive practice exam.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True,
        temperature=0.7,
        max_tokens=1500
    )
    exam_output = ""
    buffer = ""
    for chunk in response:
        if "content" in chunk["choices"][0]["delta"]:
            buffer += chunk["choices"][0]["delta"]["content"]
            if buffer.endswith("\n"):
                exam_output += buffer
                buffer = ""
    last_generated_exam = exam_output
    return exam_output

def ai_self_evaluate(last_exam):
    improvement_prompt = f"""
The user marked the last exam as 'bad'.
Here is the exam text:
-------------------
{last_exam}
-------------------
Provide 3 specific improvements to make future exams better.
Return your answer as bullet points.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI specialized in refining university exams."},
                {"role": "user", "content": improvement_prompt},
            ],
            temperature=0.7
        )
        improvements = response["choices"][0]["message"]["content"].strip()
        return improvements
    except Exception as e:
        print(f"Failed to generate improvements: {e}")
        return "Failed to get improvements."

# Build FAISS indexes (adjust paths as needed)
pdf_exam_paths = ["EXAM_COSC_2P03_JULY_2007.pdf"]
exam_index, practice_exams = store_exams_faiss(pdf_exam_paths)
course_pdf_path = "MarkAllenWeiss_DataStructures_Java.pdf"
course_index, course_chunks, bm25 = store_course_material_faiss(course_pdf_path)

# -----------------------------
# API Entry Point: run()
# -----------------------------
def run(input_data):
    """
    Expects input_data to be a dictionary with keys:
      - "course": the course name or query string,
      - "difficulty": exam difficulty level (e.g., "Easy", "Medium", "Hard")
    Returns the generated exam text.
    """
    course = input_data.get("course", "Default Course")
    difficulty = input_data.get("difficulty", "Medium")
    exam_text = generate_practice_exam_realtime(course, difficulty)
    return exam_text

# -----------------------------
# Interactive UI (Only if run directly)
# -----------------------------
if __name__ == '__main__':
    # Run interactive UI for testing if desired
    university_input = widgets.Text(placeholder="Enter University Name", description="University:")
    course_input = widgets.Text(placeholder="Enter Course Name", description="Course:")
    difficulty_dropdown = widgets.Dropdown(options=["Easy", "Medium", "Hard"], description="Difficulty:")
    generate_button = widgets.Button(description="Generate Exam", button_style="success")
    output_area = widgets.Output()

    def on_generate_exam(b):
        course = course_input.value.strip()
        difficulty = difficulty_dropdown.value
        with output_area:
            output_area.clear_output()
            print("Generating exam...\n")
            exam_text = generate_practice_exam_realtime(course, difficulty)
            print(exam_text)

    generate_button.on_click(on_generate_exam)
    display(university_input, course_input, difficulty_dropdown, generate_button, output_area)
