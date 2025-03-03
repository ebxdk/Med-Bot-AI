# -*- coding: utf-8 -*-
"""
Modified SmartStudentCalendar.py for API Integration.
This module exposes a run(input_data) function that expects a dictionary with:
  - "pdf_path": the local path to the syllabus PDF file.
It processes the PDF, extracts events and topics, generates a study plan,
and returns the study plan as a JSON string.
"""

import fitz  # PyMuPDF
import re
import json
import pandas as pd
import nltk
import openai
import torch
from datetime import datetime, timedelta
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')

# Set your OpenAI API key (ensure it is set in your environment)
openai.api_key = "sk-REPLACE_WITH_YOUR_KEY"

# ----------------------------
# Step 1: Extract Text from PDF
# ----------------------------
def extract_pdf_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return ""
    raw_text = []
    for page in doc:
        try:
            text = page.get_text("text")
            raw_text.append(text)
        except Exception as e:
            continue
    return " ".join(raw_text)

# ----------------------------
# Syllabus Detection
# ----------------------------
def is_likely_syllabus(text):
    keywords = ["syllabus", "course", "instructor", "schedule", "assignment", "exam", "reading week", "credits", "topic"]
    text_lower = text.lower()
    count = sum(1 for kw in keywords if kw in text_lower)
    return count >= 2

# ----------------------------
# Step 2: Semantic Filtering using Sentence Transformers
# ----------------------------
def filter_text_semantically(text, top_k=10):
    sentences = sent_tokenize(text)
    if not sentences:
        return text
    sem_model = SentenceTransformer('all-MiniLM-L6-v2')
    query = "important dates event lecture assignment exam"
    query_embedding = sem_model.encode(query, convert_to_tensor=True)
    sentence_embeddings = sem_model.encode(sentences, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(sentences)))
    top_sentences = [sentences[idx] for idx in top_results[1].cpu().numpy()]
    return " ".join(top_sentences)

# ----------------------------
# Utility: Clean GPT response (remove markdown fences)
# ----------------------------
def clean_response(raw_response):
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned

# ----------------------------
# Step 3: Extract Events with GPT
# ----------------------------
def extract_events_with_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts structured data from academic syllabi."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        raw_response = response["choices"][0]["message"]["content"].strip()
        raw_response = clean_response(raw_response)
        events_data = json.loads(raw_response)
        if "events" not in events_data or not isinstance(events_data["events"], list):
            return {}
        return events_data
    except Exception as e:
        return {}

# ----------------------------
# Step 4: Extract Topics with GPT
# ----------------------------
def extract_topics(text):
    topics_prompt = f"""
Extract the list of academic topics covered in the following syllabus text.
Return structured JSON in the following format:
{{
    "topics": [
        "Topic 1",
        "Topic 2",
        "Topic 3"
    ]
}}
Text:
{text}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts academic topics from course syllabi."},
                {"role": "user", "content": topics_prompt}
            ],
            temperature=0.3
        )
        raw_topic_response = response["choices"][0]["message"]["content"].strip()
        raw_topic_response = clean_response(raw_topic_response)
        topics_data = json.loads(raw_topic_response)
        if "topics" not in topics_data or not isinstance(topics_data["topics"], list):
            return []
        return topics_data["topics"]
    except Exception as e:
        return []

# ----------------------------
# Step 5: Determine Course Start Date from Events
# ----------------------------
def get_course_start_date(extracted_data):
    for event in extracted_data.get("events", []):
        if "first day of classes" in event.get("name", "").lower():
            date_str = event.get("date", "").strip()
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                continue
    return None

# ----------------------------
# Step 6: Generate Study Plan based on Extracted Events
# ----------------------------
def generate_study_plan(extracted_data):
    plan = []
    for event in extracted_data.get("events", []):
        event_date_str = event.get("date", "").strip()
        try:
            event_date = None if event_date_str.upper() == "TBA" else datetime.strptime(event_date_str, "%Y-%m-%d")
        except Exception:
            continue

        task_name = f"{event.get('name', 'Unnamed Event')} for {extracted_data.get('course_name', 'Unknown Course')}"
        if event.get("type", "").lower() == "lecture":
            if event_date:
                plan.append({
                    "date": (event_date - timedelta(days=2)).strftime("%Y-%m-%d"),
                    "task": f"Prepare for {task_name}",
                    "category": "study"
                })
                plan.append({
                    "date": (event_date + timedelta(days=2)).strftime("%Y-%m-%d"),
                    "task": f"Review {task_name}",
                    "category": "review"
                })
            else:
                plan.append({
                    "date": event_date_str,
                    "task": f"{event.get('name', 'Unnamed Event')} (TBA) for {extracted_data.get('course_name', 'Unknown Course')}",
                    "category": "major event"
                })
        else:
            if event_date:
                if event.get("type", "").lower() == "assignment":
                    for i in range(7):
                        plan.append({
                            "date": (event_date - timedelta(days=i)).strftime("%Y-%m-%d"),
                            "task": f"Work on {task_name}",
                            "category": "assignment"
                        })
                elif event.get("type", "").lower() == "lab":
                    plan.append({
                        "date": (event_date - timedelta(days=1)).strftime("%Y-%m-%d"),
                        "task": f"Prepare for {task_name}",
                        "category": "lab prep"
                    })
                elif event.get("type", "").lower() == "exam":
                    for i in range(7):
                        plan.append({
                            "date": (event_date - timedelta(days=i)).strftime("%Y-%m-%d"),
                            "task": f"Revise for {task_name}",
                            "category": "exam prep"
                        })
                else:
                    plan.append({
                        "date": event_date.strftime("%Y-%m-%d"),
                        "task": f"Review details for {task_name}",
                        "category": "general"
                    })
                plan.append({
                    "date": event_date.strftime("%Y-%m-%d"),
                    "task": f"{event.get('name', 'Unnamed Event')}",
                    "category": "major event"
                })
            else:
                plan.append({
                    "date": event_date_str,
                    "task": f"{event.get('name', 'Unnamed Event')}",
                    "category": "major event"
                })
    return plan

# ----------------------------
# Step 7: Schedule Weekly Topic Revision Tasks
# ----------------------------
def schedule_topic_revisions(plan, topics, start_date):
    topic_tasks = []
    for i, topic in enumerate(topics):
        revision_date = start_date + timedelta(weeks=i)
        topic_tasks.append({
            "date": revision_date.strftime("%Y-%m-%d"),
            "task": f"Revise Topic {i+1}: {topic}",
            "category": "topic revision"
        })
    return plan + topic_tasks

# ----------------------------
# API Entry Point: run()
# ----------------------------
def run(input_data):
    """
    Expects input_data to be a dictionary with:
      - "pdf_path": local path to the syllabus PDF.
    Returns the generated study plan as a JSON string.
    """
    pdf_path = input_data.get("pdf_path")
    if not pdf_path:
        return json.dumps({"error": "No pdf_path provided in input_data."})
    
    full_text = extract_pdf_text(pdf_path)
    if not full_text.strip():
        return json.dumps({"error": "No text extracted from PDF."})
    
    if not is_likely_syllabus(full_text):
        return json.dumps({"error": "Document does not appear to be a valid syllabus."})
    
    filtered_text = filter_text_semantically(full_text, top_k=10)
    
    # Prepare prompt for events extraction
    events_prompt = f"""
Extract all important dates and their corresponding event types from the following text.
Return structured JSON in the following format exactly:
{{
    "course_name": "Course Name",
    "events": [
        {{"date": "YYYY-MM-DD", "name": "Event Name", "type": "event_type"}}
    ]
}}
Text:
{filtered_text}
"""
    extracted_events = extract_events_with_gpt(events_prompt)
    if not extracted_events:
        return json.dumps({"error": "Failed to extract events from syllabus."})
    
    topics_list = extract_topics(full_text)
    course_start = get_course_start_date(extracted_events)
    if not course_start:
        course_start = datetime.today()
    
    study_plan_events = generate_study_plan(extracted_events)
    final_plan = study_plan_events
    if topics_list:
        final_plan = schedule_topic_revisions(final_plan, topics_list, course_start)
    
    # Sort plan by date (handle "TBA" as last)
    def sort_key(x):
        if x["date"].upper() == "TBA":
            return "9999-12-31"
        return x["date"]
    final_plan = sorted(final_plan, key=lambda x: sort_key(x))
    
    return json.dumps(final_plan, indent=2)

# For testing the module directly
if __name__ == '__main__':
    # Example: run with a sample pdf_path (adjust the path to a valid PDF on your system)
    sample_input = {"pdf_path": "syllabus.pdf"}
    result = run(sample_input)
    print("Study Plan (JSON):\n", result)
