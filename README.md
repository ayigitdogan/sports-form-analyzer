# Sports Form Analyzer

This project analyzes short training videos to give simple, athlete-friendly feedback on movement quality. It runs pose estimation on uploaded clips, extracts key movement metrics, and uses Gemini to generate concise coaching tips and drill suggestions. The goal is to help athletes understand what they do well, what tends to break down, and how to improve with clear, actionable cues.

# Demo
1) Prereqs: Python 3.11
2) Create a virtualenv and install deps:
   - python -m venv .venv
   - .venv\Scripts\activate
   - pip install -r requirements.txt
3) Create a .env file in the repo root, add the single line below with your Gemini API key:
   - GEMINI_API_KEY=your_gemini_api_key_here
4) Run the app in the command line:
   - py -3.11 -m streamlit run app.py
