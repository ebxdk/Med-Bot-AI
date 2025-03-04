from flask import Flask, request, jsonify
import chatbot
#import AIExamGenerator
#import flashcard
#import smart_student_calendar

app = Flask(__name__)

# Endpoint for the Chatbot
# Expects a JSON payload with a key "message"
@app.route('/chatbot', methods=['POST'])
def chatbot_route():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request body."}), 400
    result = chatbot.run(data)
    return jsonify({"response": result})

# Endpoint for the Exam Generator
# Expects a JSON payload with keys "course" and "difficulty"
#@app.route('/exam', methods=['POST'])
#def exam_route():
#    data = request.json
 #   if not data or "course" not in data or "difficulty" not in data:
  #      return jsonify({"error": "Missing 'course' or 'difficulty' in request body."}), 400
   # result = AIExamGenerator.run(data)
    #return jsonify({"response": result})

# Endpoint for the Flashcard Generator
# Expects a JSON payload with keys "university", "course", and "topic"
#@app.route('/flashcard', methods=['POST'])
#def flashcard_route():
#    data = request.json
#    if not data or "university" not in data or "course" not in data or "topic" not in data:
#        return jsonify({"error": "Missing one of the required fields: 'university', 'course', or 'topic'."}), 400
#    result = flashcard.run(data)
#    return jsonify({"response": result})

# Endpoint for the Smart Student Calendar (Study Plan Generator)
# Expects a JSON payload with a key "pdf_path" (local path to the syllabus PDF)
#@app.route('/study-plan', methods=['POST'])
#def study_plan_route():
#    data = request.json
 #   if not data or "pdf_path" not in data:
 #       return jsonify({"error": "Missing 'pdf_path' in request body."}), 400
#    result = smart_student_calendar.run(data)
#    return jsonify({"response": result})

if __name__ == '__main__':
    # Run the server on all interfaces on port 5000 with debugging enabled.
    app.run(host='0.0.0.0', port=5000, debug=True)
