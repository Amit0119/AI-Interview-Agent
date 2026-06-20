import streamlit as st
import os
import io
import json
import time
import asyncio
import concurrent.futures
from datetime import datetime
from typing import List, Tuple, Optional
import librosa
import torch
import PyPDF2

# Core LangChain Imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import whisper
import edge_tts
# ============================================================================
# Page Config & Premium UI
# ============================================================================
st.set_page_config(page_title="AI Interview Agent", page_icon="💼", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 30%, #0d1b2a 60%, #0a0a1a 100%);
        color: #e0e6ed;
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
    }

    /* Title */
    .main-title {
        text-align: center;
        padding: 30px 0 10px;
    }
    .main-title h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }
    .main-title p {
        color: #8892a4;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(15, 20, 40, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 20px;
        padding: 28px 32px;
        margin: 16px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transform-style: preserve-3d;
        transform: perspective(1000px) rotateX(2deg);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: perspective(1000px) rotateX(0deg) translateY(-10px) translateZ(20px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }

    /* Interviewer Question */
    .question-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.12));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 18px;
        padding: 30px 35px;
        margin: 25px 0;
        position: relative;
        overflow: hidden;
        transform-style: preserve-3d;
        transform: perspective(1000px) translateZ(10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.4), 0 5px 15px rgba(99, 102, 241, 0.2);
        transition: all 0.3s ease;
    }
    .question-card:hover {
        transform: perspective(1000px) translateZ(30px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0,0,0,0.5), 0 10px 25px rgba(99, 102, 241, 0.4);
    }
    .question-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        background: linear-gradient(180deg, #6366f1, #a855f7);
        border-radius: 4px 0 0 4px;
    }
    .question-label {
        color: #a78bfa;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    .question-text {
        color: #e8e6ff;
        font-size: 1.15rem;
        font-weight: 500;
        line-height: 1.6;
    }

    /* Progress Bar */
    .progress-container {
        background: rgba(15, 20, 40, 0.5);
        border-radius: 14px;
        padding: 18px 24px;
        margin: 16px 0;
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    .progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .progress-label {
        color: #8892a4;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .progress-count {
        color: #a78bfa;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .progress-bar-bg {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        transition: width 0.5s ease;
    }

    /* Score Card */
    .score-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(99, 102, 241, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        margin: 24px 0;
    }
    .score-number {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
        margin: 10px 0;
    }
    .score-excellent { background: linear-gradient(135deg, #10b981, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .score-good { background: linear-gradient(135deg, #6366f1, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .score-average { background: linear-gradient(135deg, #f59e0b, #fbbf24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .score-low { background: linear-gradient(135deg, #ef4444, #f87171); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .score-label {
        color: #8892a4;
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .score-grade {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 8px;
    }

    /* QA History */
    .history-item {
        background: rgba(15, 20, 40, 0.4);
        border-radius: 14px;
        padding: 18px 22px;
        margin: 10px 0;
        border-left: 3px solid;
    }
    .history-q { border-left-color: #6366f1; }
    .history-a { border-left-color: #10b981; margin-left: 20px; }
    .history-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }

    /* Form styling */
    .stTextArea textarea {
        background: rgba(15, 20, 40, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 14px !important;
        color: #e0e6ed !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 16px !important;
        transition: border-color 0.3s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.1) !important;
    }

    /* Audio Recorder Button Styles */
    [data-testid="stElementContainer"]:has(iframe[title*="audio_recorder"]),
    [data-testid="stElementContainer"]:has(iframe[title*="custom_mic"]) {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 20px 0 !important;
    }

    iframe[title*="audio_recorder"],
    iframe[title*="custom_mic"] {
        border: none !important;
        background: transparent !important;
    }

    /* Remove Streamlit default backgrounds for widgets that might wrap the iframe */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stElementContainer"] {
        background: transparent !important;
        border: none !important;
    }


    /* Form submit button */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 20px;
        padding: 50px 40px;
        text-align: center;
        margin: 40px 0;
    }
    .welcome-icon { font-size: 4rem; margin-bottom: 15px; }
    .welcome-title { color: #c4b5fd; font-size: 1.5rem; font-weight: 600; margin-bottom: 10px; }
    .welcome-desc { color: #6b7280; font-size: 1rem; line-height: 1.7; }

    /* Stat chips */
    .stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
    .stat-chip {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 10px;
        padding: 10px 18px;
        font-size: 0.85rem;
        color: #a78bfa;
        font-weight: 500;
    }

    /* Timer */
    .timer-display {
        text-align: center;
        padding: 12px;
        border-radius: 14px;
        margin: 10px 0;
    }
    .timer-normal {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .timer-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .timer-critical {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    .timer-text {
        font-size: 2rem;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
    }

    /* Hint card */
    .hint-card {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(251, 191, 36, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 14px;
        padding: 16px 20px;
        margin: 10px 0;
    }

    /* Strength/Weakness bars */
    .sw-bar-bg {
        background: rgba(99, 102, 241, 0.08);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin-top: 4px;
    }
    .sw-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }

    /* Session history card */
    .session-hist-card {
        background: rgba(15, 20, 40, 0.4);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 14px;
        padding: 16px 20px;
        margin: 8px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Audio & Divider */
    audio { border-radius: 12px !important; width: 100%; }
    hr {
        border: none; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Constants
# ============================================================================
QUESTIONS_PER_SESSION = 5

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".interview_history.json")

# ============================================================================
# Question Banks by Type & Difficulty
# ============================================================================
QUESTION_BANK = {
    "technical": {
        "easy": [
            "What programming languages are you most comfortable with?",
            "Can you explain the difference between a list and a tuple in Python?",
            "What version control system do you use and why?",
            "What is an API and how have you used one?",
            "Describe your experience with databases.",
        ],
        "medium": [
            "Explain how you would design a RESTful API for a user management system.",
            "What design patterns have you applied in your projects?",
            "How do you handle error handling and logging in production systems?",
            "Describe your experience with CI/CD pipelines.",
            "How do you optimize database queries for better performance?",
        ],
        "hard": [
            "How would you design a system that handles 10 million concurrent users?",
            "Explain eventual consistency and when you would choose it over strong consistency.",
            "Describe how you would implement a distributed caching system.",
            "How do you approach microservices vs monolithic architecture trade-offs?",
            "Explain how you would debug a memory leak in a production system.",
        ],
    },
    "behavioral": {
        "easy": [
            "Tell me about yourself and your background.",
            "Why are you interested in this kind of role?",
            "What do you enjoy most about software development?",
            "Describe your ideal work environment.",
            "How do you stay updated with the latest technologies?",
        ],
        "medium": [
            "Tell me about a time you had a conflict with a teammate. How did you resolve it?",
            "Describe a project where you had to meet a tight deadline.",
            "How do you prioritize tasks when you have multiple deadlines?",
            "Tell me about a time you received critical feedback and how you handled it.",
            "Describe a situation where you had to take initiative without being asked.",
        ],
        "hard": [
            "Tell me about a time you failed at something significant. What did you learn?",
            "Describe a situation where you had to influence a decision without direct authority.",
            "How do you handle ambiguity when requirements keep changing?",
            "Tell me about a time you had to make a difficult ethical decision at work.",
            "Describe a leadership challenge you faced and how you overcame it.",
        ],
    },
    "hr": {
        "easy": [
            "What are your salary expectations?",
            "Are you open to relocation?",
            "When can you start if selected?",
            "What is your preferred work schedule?",
            "Do you have any questions about the company?",
        ],
        "medium": [
            "Where do you see yourself in 5 years?",
            "Why are you looking to leave your current position?",
            "What makes you a good fit for this role?",
            "How do you handle work-life balance?",
            "What are your strengths and weaknesses?",
        ],
        "hard": [
            "If you had competing offers, how would you make your decision?",
            "How would you handle a situation where you disagree with company policy?",
            "Describe a time you had to adapt to a major organizational change.",
            "What would you do if you discovered a colleague was acting unethically?",
            "How do you maintain motivation during long-term projects?",
        ],
    },
}

# ============================================================================
# Helper Classes
# ============================================================================
@st.cache_resource
def get_audio_model():
    return whisper.load_model("base")

class AudioProcessor:
    def transcribe(self, audio_bytes: bytes) -> str:
        import tempfile
        import os
        model = get_audio_model()
        
        # Save bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name
            
        try:
            # Whisper's load_audio handles format conversion via ffmpeg robustly
            audio_np = whisper.load_audio(temp_path)
            result = model.transcribe(audio_np, language="en", fp16=False)
            return result.get("text", "").strip()
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

class TextToSpeechProcessor:
    async def synthesize(self, text: str, voice: str) -> bytes:
        communicate = edge_tts.Communicate(text, voice=voice)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

def run_tts_sync(text: str, voice: str) -> Optional[bytes]:
    """Run TTS in a separate thread with its own event loop."""
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(TextToSpeechProcessor().synthesize(text, voice))
        finally:
            loop.close()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            return future.result(timeout=30)
    except Exception:
        return None

# ============================================================================
# Session History (file-based persistence)
# ============================================================================
def load_session_history() -> List[dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_session_history(history: List[dict]):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception:
        pass

def add_session_to_history(score_data: dict, interview_type: str, difficulty: str):
    # Don't save sessions where no questions were answered
    if not score_data.get("evaluations"):
        return
    history = load_session_history()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%b %d, %Y %I:%M %p"),
        "percentage": score_data["percentage"],
        "grade": score_data["grade"],
        "total_score": score_data["total_score"],
        "max_score": score_data["max_score"],
        "num_questions": len(score_data["evaluations"]),
        "interview_type": interview_type,
        "difficulty": difficulty,
        "strengths": score_data.get("strengths", []),
        "weaknesses": score_data.get("weaknesses", []),
    }
    history.append(entry)
    # Keep last 50 sessions
    if len(history) > 50:
        history = history[-50:]
    save_session_history(history)

# ============================================================================
# Scoring System
# ============================================================================
def fallback_evaluate_answer(question: str, answer: str, resume_text: str) -> dict:
    """Evaluate a single answer and return score + feedback."""
    word_count = len(answer.split())
    if word_count < 10:
        return {"score": 0, "feedback": ["Your answer is too short or doesn't address the question."], "word_count": word_count, "categories": {"detail": 0, "relevance": 0, "specificity": 0}}
        
    score = 0
    feedback = []
    categories = {"detail": 0, "relevance": 0, "specificity": 0}

    # Length / Detail check
    if word_count < 15:
        categories["detail"] = 1
        feedback.append("Decent length but could be more detailed.")
    elif word_count < 50:
        categories["detail"] = 3
        feedback.append("Good detail in your response.")
    else:
        categories["detail"] = 4
        feedback.append("Comprehensive answer.")
    score += categories["detail"]

    # Relevance
    resume_words = set(resume_text.lower().split())
    answer_words = set(answer.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
                  "and", "or", "but", "with", "of", "by", "from", "as", "it", "that", "this",
                  "i", "my", "me", "we", "our", "have", "has", "had", "do", "does", "did",
                  "will", "would", "can", "could", "should", "may", "be", "been", "being", "not"}
    meaningful_overlap = (answer_words & resume_words) - stop_words
    if len(meaningful_overlap) >= 5:
        categories["relevance"] = 4
        feedback.append("Strong alignment with your resume.")
    elif len(meaningful_overlap) >= 2:
        categories["relevance"] = 2
        feedback.append("Some relevant connections to your background.")
    else:
        categories["relevance"] = 0
        feedback.append("Try to connect your answer to your experience.")
    score += categories["relevance"]

    # Specificity
    tech_list = ["python", "java", "react", "sql", "aws", "docker", "api", "ml", "ai",
                 "tensorflow", "pytorch", "node", "django", "flask", "kubernetes", "git",
                 "linux", "database", "cloud", "agile", "scrum", "javascript", "typescript",
                 "mongodb", "redis", "kafka", "microservices", "rest", "graphql", "ci/cd"]
    has_specifics = any(char.isdigit() for char in answer) or \
                    any(tech in answer.lower() for tech in tech_list)
    if has_specifics:
        categories["specificity"] = 2
        feedback.append("Good use of specific details/technologies.")
    else:
        categories["specificity"] = 0
        feedback.append("Adding specific technologies or metrics would strengthen your answer.")
    score += categories["specificity"]

    score = min(score, 10)
    return {"score": score, "feedback": feedback, "word_count": word_count, "categories": categories}


def evaluate_answer(llm, question: str, answer: str, resume_text: str) -> dict:
    """Evaluate a single answer using Gemini and strict scoring."""
    word_count = len(answer.split())
    if word_count < 3:
        return {"score": 0, "feedback": ["Answer is far too short to evaluate."], "word_count": word_count, "categories": {"detail": 0, "relevance": 0, "specificity": 0}}
        
    prompt = (
        "You are an expert technical interviewer evaluating a candidate's answer.\n"
        "Please evaluate the candidate's answer based on the question asked and their resume.\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Provide your evaluation in the following strict JSON format ONLY:\n"
        "{\n"
        '  "score": <integer from 0 to 10>,\n'
        '  "feedback": ["feedback point 1", "feedback point 2"],\n'
        '  "detail_score": <integer 0 to 5>,\n'
        '  "relevance_score": <integer 0 to 5>,\n'
        '  "specificity_score": <integer 0 to 3>\n'
        "}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- The candidate's answer MUST be factually correct and directly answer the question.\n"
        "- If the answer is factually incorrect, completely irrelevant, or clear nonsense (e.g. just introducing themselves when asked a technical question), the 'score' MUST be 0. Do NOT give points for effort if the answer is wrong.\n"
        "- If the answer is partially correct but lacks depth, give a low score (1-4).\n"
        "- Only give high scores (8-10) for detailed, relevant, and technically accurate answers."
    )
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        eval_data = json.loads(content)
        score = int(eval_data.get("score", 0))
        return {
            "score": score,
            "feedback": eval_data.get("feedback", []),
            "word_count": word_count,
            "categories": {
                "detail": int(eval_data.get("detail_score", 0)),
                "relevance": int(eval_data.get("relevance_score", 0)),
                "specificity": int(eval_data.get("specificity_score", 0))
            }
        }
    except Exception as e:
        return fallback_evaluate_answer(question, answer, resume_text)


def calculate_session_score(qa_history: List[dict], resume_text: str) -> dict:
    """Calculate overall session score with strengths/weaknesses."""
    if not qa_history:
        return {"total_score": 0, "max_score": 0, "percentage": 0, "evaluations": [],
                "grade": "N/A", "strengths": [], "weaknesses": []}

    evaluations = []
    total = 0
    cat_totals = {"detail": 0, "relevance": 0, "specificity": 0}

    for item in qa_history:
        ev = evaluate_answer(st.session_state.llm, item["q"], item["a"], resume_text)
        ev["question"] = item["q"]
        ev["answer"] = item["a"]
        evaluations.append(ev)
        total += ev["score"]
        for cat in cat_totals:
            cat_totals[cat] += ev["categories"][cat]

    max_score = len(qa_history) * 10
    percentage = round((total / max_score) * 100) if max_score > 0 else 0

    if percentage >= 80:
        grade = "Excellent"
    elif percentage >= 60:
        grade = "Good"
    elif percentage >= 40:
        grade = "Average"
    else:
        grade = "Needs Improvement"

    # Determine strengths/weaknesses
    n = len(qa_history)
    cat_max = {"detail": 5 * n, "relevance": 5 * n, "specificity": 3 * n}
    cat_pct = {}
    for cat in cat_totals:
        cat_pct[cat] = round((cat_totals[cat] / cat_max[cat]) * 100) if cat_max[cat] > 0 else 0

    strengths = []
    weaknesses = []
    labels = {"detail": "Answer Detail & Depth", "relevance": "Resume Relevance", "specificity": "Technical Specificity"}
    for cat, pct in cat_pct.items():
        if pct >= 70:
            strengths.append({"name": labels[cat], "pct": pct})
        else:
            weaknesses.append({"name": labels[cat], "pct": pct})

    return {
        "total_score": total, "max_score": max_score, "percentage": percentage,
        "evaluations": evaluations, "grade": grade,
        "strengths": strengths, "weaknesses": weaknesses, "cat_pct": cat_pct,
    }


# ============================================================================
# Question Generation
# ============================================================================
def get_question_from_bank(history: List[dict], interview_type: str, difficulty: str) -> str:
    """Get a question from the preset bank based on type and difficulty."""
    if interview_type == "mixed":
        types = ["technical", "behavioral", "hr"]
        selected_type = types[len(history) % len(types)]
    else:
        selected_type = interview_type

    questions = QUESTION_BANK.get(selected_type, {}).get(difficulty, [])
    if not questions:
        questions = QUESTION_BANK["technical"]["medium"]

    index = len(history) % len(questions)
    return questions[index]


def generate_question(llm, resume_text: str, full_history: List[dict]) -> str:
    """AI-generated question from resume."""
    if len(full_history) == 0:
        return "Please introduce yourself and give me a brief overview of your background."

    history_text = "\n".join([f"Q: {h['q']} A: {h['a']}" for h in full_history[-6:]])
    prompt = (
        "You are an elite, highly experienced Senior Engineering Manager at a top-tier FAANG company conducting a high-pressure technical and behavioral interview.\n"
        "Your task is to ask exactly ONE strict, highly specific, and challenging follow-up interview question based on the candidate's resume and their previous answers.\n\n"
        "RULES FOR 100% REALISM:\n"
        "1. DO NOT be overly polite. Be direct, professional, and probing.\n"
        "2. Analyze their previous answers. If they gave a generic answer previously, grill them on the specifics. Ask 'How exactly did you implement X?', 'What were the latency trade-offs when you used Y?', or 'Tell me about a time that failed.'\n"
        "3. Connect the question DIRECTLY to a specific project, metric, or skill listed in their resume.\n"
        "4. DO NOT ask multiple questions at once. Ask ONE deep, focused question.\n"
        "5. Never ask them to 'introduce yourself' if they have already answered previous questions.\n\n"
        f"Candidate Resume:\n{resume_text}\n\n"
        f"Interview History (Read carefully to ask contextual follow-ups):\n{history_text}\n\n"
        "Ask your next question now:"
    )
    try:
        response = llm.invoke(prompt)
        question = response.content.strip()
        if not question:
            return get_question_from_bank(full_history, st.session_state.get("interview_type", "mixed"), st.session_state.get("difficulty", "medium"))
        return question
    except Exception as e:
        return get_question_from_bank(full_history, st.session_state.get("interview_type", "mixed"), st.session_state.get("difficulty", "medium"))


# ============================================================================
# Hint System
# ============================================================================
HINT_MAP = {
    "challenge": "Think about a specific project obstacle. Use the STAR method: Situation, Task, Action, Result.",
    "technologies": "List the main tools/frameworks you used and explain WHY you chose them.",
    "role": "Describe your responsibilities, team size, and your unique contribution.",
    "achievement": "Pick something measurable — mention numbers, percentages, or impact.",
    "requirements": "Talk about how you communicated with stakeholders and adapted your plan.",
    "default": "Be specific! Use real examples from your experience. Mention tools, metrics, and outcomes.",
}

def get_hint_for_question(question: str) -> str:
    q_lower = question.lower()
    for keyword, hint in HINT_MAP.items():
        if keyword in q_lower:
            return hint
    if "tell me" in q_lower or "describe" in q_lower:
        return "Use the STAR method: Situation → Task → Action → Result. Keep it structured and concise."
    if "how do you" in q_lower or "what is your approach" in q_lower:
        return "Explain your process step by step. Give a real example from your past experience."
    if "why" in q_lower:
        return "Explain your reasoning clearly. Connect your answer to your career goals or values."
    return HINT_MAP["default"]


# ============================================================================
# PDF Report Generation
# ============================================================================
def generate_report_text(scores: dict, interview_type: str, difficulty: str) -> str:
    """Generate a downloadable text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("        AI INTERVIEW AGENT — Performance Report")
    lines.append("=" * 60)
    lines.append(f"Date: {datetime.now().strftime('%B %d, %Y %I:%M %p')}")
    lines.append(f"Interview Type: {interview_type.title()}")
    lines.append(f"Difficulty: {difficulty.title()}")
    lines.append(f"Questions Answered: {len(scores['evaluations'])}")
    lines.append("")
    lines.append("-" * 40)
    lines.append(f"  OVERALL SCORE: {scores['percentage']}% ({scores['grade']})")
    lines.append(f"  Points: {scores['total_score']} / {scores['max_score']}")
    lines.append("-" * 40)
    lines.append("")

    if scores.get("strengths"):
        lines.append("STRENGTHS:")
        for s in scores["strengths"]:
            lines.append(f"  ✅ {s['name']} ({s['pct']}%)")
        lines.append("")

    if scores.get("weaknesses"):
        lines.append("AREAS TO IMPROVE:")
        for w in scores["weaknesses"]:
            lines.append(f"  ⚠️  {w['name']} ({w['pct']}%)")
        lines.append("")

    lines.append("=" * 60)
    lines.append("DETAILED BREAKDOWN")
    lines.append("=" * 60)

    for i, ev in enumerate(scores["evaluations"], 1):
        lines.append(f"\n--- Question {i} (Score: {ev['score']}/10) ---")
        lines.append(f"Q: {ev['question']}")
        lines.append(f"A: {ev['answer']}")
        lines.append(f"Words: {ev['word_count']}")
        for fb in ev["feedback"]:
            lines.append(f"  💡 {fb}")

    lines.append("\n" + "=" * 60)
    lines.append("Generated by AI Interview Agent")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================================
# Main App
# ============================================================================
def main():
    # Title
    st.markdown("""
    <div class="main-title">
        <h1>💼 AI Interview Agent</h1>
        <p>Upload your resume • Practice interviews • Get instant feedback</p>
    </div>
    """, unsafe_allow_html=True)

    # Session state defaults
    defaults = {
        "qa_chain": None, "interview_active": False, "qa_history": [], "full_qa_history": [],
        "last_question": "", "session_complete": False, "total_questions_asked": 0,
        "interview_answer": "", "audio_answer_processed": False, "session_scores": None,
        "use_ai_questions": True, "voice_playback": True, "live_scoring": True,
        "interview_type": "mixed", "difficulty": "medium", "timer_enabled": False,
        "timer_seconds": 90, "question_start_time": None, "hint_used": False,
        "voice_model": "en-US-ChristopherNeural", "last_voice_model": "en-US-ChristopherNeural"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ---- Sidebar ----
    from dotenv import load_dotenv
    load_dotenv()
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if gemini_api_key:
        st.session_state.gemini_api_key = gemini_api_key
    else:
        st.sidebar.error("Developer: Please set GOOGLE_API_KEY in .env file")
    
    st.sidebar.markdown("### 📄 Resume Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Resume (PDF)", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded_files and st.session_state.qa_chain is None:
        if not st.session_state.get("gemini_api_key"):
            st.sidebar.error("Please enter your Gemini API Key above first.")
        else:
            with st.spinner("🔄 Processing your resume..."):
                all_text = ""
                for f in uploaded_files:
                    pdf = PyPDF2.PdfReader(io.BytesIO(f.read()))
                    all_text += "\n".join([p.extract_text() for p in pdf.pages])

                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_text(all_text)

                vectorstore = FAISS.from_texts(
                    chunks,
                    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                )

                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

                st.session_state.llm = llm
                st.session_state.resume_text = all_text
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever()
                )
                st.sidebar.success("✅ Resume processed!")

    # Sidebar controls
    if st.session_state.qa_chain:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 Interview Controls")

        if not st.session_state.interview_active and not st.session_state.session_complete:
            if st.sidebar.button("🚀 Start Interview", use_container_width=True):
                st.session_state.interview_active = True
                st.session_state.qa_history = []
                st.session_state.last_question = ""
                st.session_state.total_questions_asked = 0
                st.session_state.session_complete = False
                st.session_state.session_scores = None
                st.session_state.interview_answer = ""
                st.session_state.audio_answer_processed = False
                st.session_state.question_start_time = None
                st.session_state.hint_used = False
                st.rerun()

        if st.session_state.interview_active:
            if st.sidebar.button("⏹ Stop Interview", use_container_width=True):
                st.session_state.interview_active = False
                st.session_state.session_complete = True
                st.session_state.session_scores = calculate_session_score(
                    st.session_state.qa_history, st.session_state.resume_text
                )
                # Save to history
                add_session_to_history(
                    st.session_state.session_scores,
                    st.session_state.interview_type,
                    st.session_state.difficulty
                )
                st.rerun()

        # Session progress
        if st.session_state.interview_active:
            answered = len(st.session_state.qa_history)
            st.sidebar.markdown(f"""
            <div style="background: rgba(99,102,241,0.1); border-radius: 12px; padding: 14px; margin-top: 10px;
                        border: 1px solid rgba(99,102,241,0.2);">
                <div style="color: #a78bfa; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
                            letter-spacing: 1px;">Session Progress</div>
                <div style="color: #e0e6ed; font-size: 1.4rem; font-weight: 700; margin-top: 4px;">
                    {answered} / {QUESTIONS_PER_SESSION}
                </div>
                <div style="color: #6b7280; font-size: 0.8rem;">questions answered</div>
            </div>
            """, unsafe_allow_html=True)

        # ---- Feature Toggles ----
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Features")

        st.session_state.use_ai_questions = st.sidebar.toggle(
            "🧠 AI-Generated Questions",
            value=st.session_state.use_ai_questions,
            help="ON: AI generates questions from your resume. OFF: Uses preset question bank."
        )
        st.session_state.voice_playback = st.sidebar.toggle(
            "🔊 Voice Playback",
            value=st.session_state.voice_playback,
            help="ON: Questions are read aloud. OFF: Text only (faster)."
        )
        st.session_state.live_scoring = st.sidebar.toggle(
            "📊 Live Scoring",
            value=st.session_state.live_scoring,
            help="ON: See score after each answer. OFF: Score shown at the end."
        )
        st.session_state.timer_enabled = st.sidebar.toggle(
            "⏱️ Answer Timer",
            value=st.session_state.timer_enabled,
            help="Add time pressure to simulate real interviews."
        )

        if st.session_state.timer_enabled:
            st.session_state.timer_seconds = st.sidebar.select_slider(
                "⏱️ Time per question",
                options=[30, 45, 60, 90, 120, 180],
                value=st.session_state.timer_seconds,
                format_func=lambda x: f"{x}s"
            )

        # ---- Interview Settings ----
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🗣️ Voice Options")
        voice_sel = st.sidebar.selectbox("Agent Voice", ["Male (Christopher)", "Female (Jenny)"], index=0 if "Christopher" in st.session_state.voice_model else 1)
        st.session_state.voice_model = "en-US-ChristopherNeural" if "Male" in voice_sel else "en-US-JennyNeural"
        
        # Trigger TTS regeneration if voice changes
        if st.session_state.get("last_voice_model") != st.session_state.voice_model:
            st.session_state.tts_question_audio = None
            st.session_state.tts_attempted = False
            st.session_state.last_voice_model = st.session_state.voice_model
            st.rerun()
        
        st.sidebar.markdown("### ⚙️ Interview Settings")

        st.session_state.interview_type = st.sidebar.selectbox(
            "Interview Type",
            ["mixed", "technical", "behavioral", "hr"],
            format_func=lambda x: {"mixed": "🎯 Mixed", "technical": "💻 Technical",
                                   "behavioral": "🤝 Behavioral", "hr": "📋 HR"}[x],
            index=["mixed", "technical", "behavioral", "hr"].index(st.session_state.interview_type)
        )
        st.session_state.difficulty = st.sidebar.selectbox(
            "Difficulty Level",
            ["easy", "medium", "hard"],
            format_func=lambda x: {"easy": "🟢 Easy", "medium": "🟡 Medium", "hard": "🔴 Hard"}[x],
            index=["easy", "medium", "hard"].index(st.session_state.difficulty)
        )

        # ---- Session History ----
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏆 Session History")
        history = load_session_history()
        if history:
            for entry in reversed(history[-5:]):
                grade_emoji = {"Excellent": "🏆", "Good": "👏", "Average": "💪", "Needs Improvement": "📚"}.get(entry["grade"], "📊")
                st.sidebar.markdown(f"""
                <div style="background: rgba(15,20,40,0.4); border-radius: 10px; padding: 10px 14px; margin: 6px 0;
                            border: 1px solid rgba(99,102,241,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #a78bfa; font-size: 0.75rem;">{entry.get('date', 'N/A')}</span>
                        <span style="color: #e0e6ed; font-weight: 700;">{grade_emoji} {entry['percentage']}%</span>
                    </div>
                    <div style="color: #6b7280; font-size: 0.7rem; margin-top: 2px;">
                        {entry.get('interview_type', 'mixed').title()} • {entry.get('difficulty', 'medium').title()}
                        • {entry.get('num_questions', '?')} Qs
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.sidebar.caption("No past sessions yet.")

    # ======================================================================
    # Main Content Area
    # ======================================================================
    if not st.session_state.qa_chain:
        # Welcome screen (no resume uploaded)
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">📋</div>
            <div class="welcome-title">Welcome to AI Interview Agent</div>
            <div class="welcome-desc">
                Upload your resume in the sidebar to get started.<br>
                I'll conduct a mock interview based on your experience and provide detailed feedback.
            </div>
            <div class="stat-row" style="justify-content: center; margin-top: 24px;">
                <div class="stat-chip">📝 5 Questions per Session</div>
                <div class="stat-chip">📊 Instant Scoring</div>
                <div class="stat-chip">🔄 Continue Option</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ---- Session Complete: Show Score ----
    if st.session_state.session_complete and st.session_state.session_scores:
        scores = st.session_state.session_scores
        percentage = scores["percentage"]
        grade = scores["grade"]

        if percentage >= 80:
            score_class, grade_color, emoji = "score-excellent", "#10b981", "🏆"
        elif percentage >= 60:
            score_class, grade_color, emoji = "score-good", "#6366f1", "👏"
        elif percentage >= 40:
            score_class, grade_color, emoji = "score-average", "#f59e0b", "💪"
        else:
            score_class, grade_color, emoji = "score-low", "#ef4444", "📚"

        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Interview Performance</div>
            <div class="score-number {score_class}">{percentage}%</div>
            <div class="score-grade" style="color: {grade_color};">{emoji} {grade}</div>
            <div style="color: #6b7280; margin-top: 8px; font-size: 0.9rem;">
                {scores['total_score']} / {scores['max_score']} points across {len(scores['evaluations'])} questions
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Strengths & Weaknesses ----
        if scores.get("cat_pct"):
            st.markdown("### 📈 Strengths & Weaknesses")
            cols = st.columns(3)
            labels = {"detail": "📝 Detail & Depth", "relevance": "🎯 Resume Relevance", "specificity": "🔧 Technical Specificity"}
            for idx, (cat, pct) in enumerate(scores["cat_pct"].items()):
                with cols[idx]:
                    bar_color = "#10b981" if pct >= 70 else "#f59e0b" if pct >= 40 else "#ef4444"
                    status = "💪 Strong" if pct >= 70 else "⚡ Improve" if pct >= 40 else "⚠️ Weak"
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center; padding: 20px;">
                        <div style="color: #a78bfa; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
                                    letter-spacing: 1px; margin-bottom: 8px;">{labels[cat]}</div>
                        <div style="color: {bar_color}; font-size: 2rem; font-weight: 800;">{pct}%</div>
                        <div style="color: {bar_color}; font-size: 0.85rem; margin-top: 4px;">{status}</div>
                        <div class="sw-bar-bg" style="margin-top: 10px;">
                            <div class="sw-bar-fill" style="width: {pct}%; background: {bar_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ---- Detailed Breakdown ----
        st.markdown("### 📋 Detailed Breakdown")
        for i, ev in enumerate(scores["evaluations"], 1):
            score_color = "#10b981" if ev["score"] >= 7 else "#f59e0b" if ev["score"] >= 4 else "#ef4444"
            st.markdown(f"""
            <div class="glass-card" style="margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="color: #a78bfa; font-weight: 600;">Question {i}</span>
                    <span style="color: {score_color}; font-weight: 700; font-size: 1.1rem;">{ev['score']}/10</span>
                </div>
                <div class="history-item history-q" style="margin: 8px 0;">
                    <div class="history-label" style="color: #6366f1;">Question</div>
                    <div style="color: #c4b5fd;">{ev['question']}</div>
                </div>
                <div class="history-item history-a" style="margin: 8px 0;">
                    <div class="history-label" style="color: #10b981;">Your Answer</div>
                    <div style="color: #a7f3d0;">{ev['answer']}</div>
                </div>
                <div style="margin-top: 10px;">
                    {''.join([f'<div style="color: #8892a4; font-size: 0.85rem; margin: 3px 0;">💡 {fb}</div>' for fb in ev['feedback']])}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ---- Download Report ----
        st.markdown("---")
        report_text = generate_report_text(scores, st.session_state.interview_type, st.session_state.difficulty)
        st.download_button(
            label="📥 Download Interview Report",
            data=report_text,
            file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown("---")

        # Continue prompt
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 1.3rem; font-weight: 600; color: #c4b5fd; margin-bottom: 8px;">
                Would you like to continue practicing?
            </div>
            <div style="color: #6b7280; font-size: 0.9rem;">
                Start a new session of 5 questions to keep improving!
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Continue!", use_container_width=True):
                st.session_state.interview_active = True
                st.session_state.qa_history = []
                st.session_state.last_question = ""
                st.session_state.total_questions_asked = 0
                st.session_state.session_complete = False
                st.session_state.session_scores = None
                st.session_state.interview_answer = ""
                st.session_state.audio_answer_processed = False
                st.session_state.tts_question_audio = None
                st.session_state.tts_attempted = False
                st.session_state.question_start_time = None
                st.session_state.hint_used = False
                st.rerun()
        with col2:
            if st.button("🏁 End Session", use_container_width=True):
                st.session_state.session_complete = False
                st.session_state.session_scores = None
                st.rerun()
        return

    # ======================================================================
    # Active Interview
    # ======================================================================
    if st.session_state.interview_active:
        answered = len(st.session_state.qa_history)
        remaining = QUESTIONS_PER_SESSION - answered

        # Check if session should end
        if answered >= QUESTIONS_PER_SESSION:
            st.session_state.interview_active = False
            st.session_state.session_complete = True
            st.session_state.session_scores = calculate_session_score(
                st.session_state.qa_history, st.session_state.resume_text
            )
            add_session_to_history(
                st.session_state.session_scores,
                st.session_state.interview_type,
                st.session_state.difficulty
            )
            st.rerun()
            return

        # Progress bar
        progress_pct = (answered / QUESTIONS_PER_SESSION) * 100
        type_label = {"mixed": "🎯 Mixed", "technical": "💻 Technical",
                      "behavioral": "🤝 Behavioral", "hr": "📋 HR"}.get(st.session_state.interview_type, "Mixed")
        diff_label = {"easy": "🟢 Easy", "medium": "🟡 Medium", "hard": "🔴 Hard"}.get(st.session_state.difficulty, "Medium")

        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-header">
                <span class="progress-label">{type_label} • {diff_label}</span>
                <span class="progress-count">{answered} / {QUESTIONS_PER_SESSION} answered</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: {progress_pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Generate question if needed
        if not st.session_state.last_question:
            if st.session_state.use_ai_questions:
                q = generate_question(st.session_state.llm, st.session_state.resume_text, st.session_state.full_qa_history)
                st.session_state.last_question = q
            else:
                st.session_state.last_question = get_question_from_bank(
                    st.session_state.qa_history, st.session_state.interview_type, st.session_state.difficulty
                )
            st.session_state.tts_question_audio = None
            st.session_state.tts_attempted = False
            st.session_state.question_start_time = time.time()
            st.session_state.hint_used = False

        # TTS
        if st.session_state.voice_playback:
            if not st.session_state.get("tts_question_audio") and not st.session_state.get("tts_attempted"):
                st.session_state.tts_question_audio = run_tts_sync(st.session_state.last_question, st.session_state.voice_model)
                st.session_state.tts_attempted = True

        # Display question card
        q_mode = "🧠 AI" if st.session_state.use_ai_questions else "📝 Preset"
        st.markdown(f"""
        <div class="question-card">
            <div class="question-label">🎤 Question {answered + 1} of {QUESTIONS_PER_SESSION}  •  {q_mode}</div>
            <div class="question-text">{st.session_state.last_question}</div>
        </div>
        """, unsafe_allow_html=True)

        # 3D Agent Avatar & Audio (Manual Play Button embedded in Avatar)
        if st.session_state.voice_playback and st.session_state.get("tts_question_audio"):
            import base64
            import streamlit.components.v1 as components
            audio_base64 = base64.b64encode(st.session_state.tts_question_audio).decode('utf-8')
            
            avatar_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <style>
                body {{
                    margin: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background-color: transparent;
                }}
                .avatar-container {{
                    border-radius: 50%;
                    overflow: hidden;
                    border: 3px solid rgba(99, 102, 241, 0.5);
                    box-shadow: 0 5px 20px rgba(99, 102, 241, 0.4);
                    background: #0f172a;
                    width: 100px;
                    height: 100px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    transition: transform 0.2s, box-shadow 0.2s;
                    position: relative;
                }}
                .avatar-container:hover {{
                    transform: scale(1.05);
                    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6);
                }}
                .speaking {{
                    animation: pulse 1.5s infinite;
                    border-color: #ec4899 !important;
                    box-shadow: 0 0 25px rgba(236, 72, 153, 0.8) !important;
                }}
                @keyframes pulse {{
                    0% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.1); }}
                    100% {{ transform: scale(1); }}
                }}
                .play-icon {{
                    position: absolute;
                    font-size: 30px;
                    color: white;
                    opacity: 0.8;
                    background: rgba(0,0,0,0.4);
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    pointer-events: none;
                }}
                .avatar-container:hover .play-icon {{
                    opacity: 1;
                    background: rgba(0,0,0,0.6);
                }}
                .label {{
                    color: #a78bfa;
                    font-family: sans-serif;
                    font-size: 14px;
                    margin-bottom: 10px;
                    text-align: center;
                    font-weight: bold;
                }}
            </style>
            </head>
            <body>
                <div style="display: flex; flex-direction: column; align-items: center; transform: perspective(1000px) translateZ(20px);">
                    <div class="label">🔊 Click the agent to listen</div>
                    <div class="avatar-container" id="avatarBtn" onclick="playAudio()" title="Click to hear the agent">
                        <img src="https://media.tenor.com/NqKNFHSmbw0AAAAi/bot-bot-3d.gif" width="100" height="100" style="object-fit: cover; opacity: 0.8;">
                        <div class="play-icon" id="playIcon">▶</div>
                    </div>
                    <audio id="agentAudio">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                </div>
                
                <script>
                    var audio = document.getElementById("agentAudio");
                    var btn = document.getElementById("avatarBtn");
                    var icon = document.getElementById("playIcon");
                    
                    function playAudio() {{
                        if (audio.paused) {{
                            audio.play();
                            btn.classList.add("speaking");
                            icon.style.display = "none";
                        }} else {{
                            audio.pause();
                            audio.currentTime = 0;
                            btn.classList.remove("speaking");
                            icon.style.display = "flex";
                        }}
                    }}
                    
                    audio.onended = function() {{
                        btn.classList.remove("speaking");
                        icon.style.display = "flex";
                    }};
                </script>
            </body>
            </html>
            """
            components.html(avatar_html, height=160)
        elif not st.session_state.voice_playback:
            st.caption("🔇 Voice playback is off. Enable it from the sidebar.")

        # ---- Timer ----
        if st.session_state.timer_enabled and st.session_state.question_start_time:
            elapsed = time.time() - st.session_state.question_start_time
            remaining_time = max(0, st.session_state.timer_seconds - int(elapsed))
            mins, secs = divmod(remaining_time, 60)

            if remaining_time <= 0:
                timer_class = "timer-critical"
                timer_color = "#ef4444"
            elif remaining_time <= 15:
                timer_class = "timer-critical"
                timer_color = "#ef4444"
            elif remaining_time <= 30:
                timer_class = "timer-warning"
                timer_color = "#f59e0b"
            else:
                timer_class = "timer-normal"
                timer_color = "#a78bfa"

            st.markdown(f"""
            <div class="timer-display {timer_class}">
                <div class="timer-text" style="color: {timer_color};">{mins:02d}:{secs:02d}</div>
                <div style="color: #6b7280; font-size: 0.8rem; margin-top: 2px;">Time Remaining</div>
            </div>
            """, unsafe_allow_html=True)

        # ---- Hint Button ----
        if not st.session_state.hint_used:
            if st.button("💡 Need a Hint?", use_container_width=False):
                st.session_state.hint_used = True
                st.rerun()

        if st.session_state.hint_used:
            hint = get_hint_for_question(st.session_state.last_question)
            st.markdown(f"""
            <div class="hint-card">
                <div style="color: #f59e0b; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
                            letter-spacing: 1px; margin-bottom: 6px;">💡 Hint</div>
                <div style="color: #fde68a; font-size: 0.95rem;">{hint}</div>
            </div>
            """, unsafe_allow_html=True)

        # Voice capture
        try:
            from audio_recorder_streamlit import audio_recorder
            
            st.markdown('<div style="text-align: center; color: #a78bfa; font-weight: 600; font-size: 1rem; margin-bottom: 10px;">🎤 Click Mic to Record Voice Answer</div>', unsafe_allow_html=True)
            
            # Using icon_size="2x" to prevent the icon from overflowing and being pushed to the top left
            audio_bytes = audio_recorder(
                text="",
                recording_color="#ef4444",
                neutral_color="#a855f7",
                icon_name="microphone",
                icon_size="2x"
            )
            
            if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
                st.session_state.last_audio_bytes = audio_bytes
                
                with st.spinner(f"Transcribing {len(audio_bytes)} bytes of audio..."):
                    user_text = AudioProcessor().transcribe(audio_bytes)
                
                if user_text:
                    # Append or set the answer
                    if st.session_state.interview_answer:
                        st.session_state.interview_answer += " " + user_text
                    else:
                        st.session_state.interview_answer = user_text
                    
                    st.success(f"🗣️ Added to answer: {user_text}")
                    time.sleep(1.5) # brief pause to show the info message
                    st.rerun()
                else:
                    st.warning(f"⚠️ Audio processed ({len(audio_bytes)} bytes) but no speech was detected. Please check your mic and try again.")
        except Exception as e:
            st.error(f"Error loading audio recorder: {e}")

        # Text answer form
        with st.form("answer_form", clear_on_submit=True):
            default_val = st.session_state.interview_answer if st.session_state.interview_answer else ""
            text_input = st.text_area(
                "✍️ Type your answer below",
                value=default_val,
                height=120,
                placeholder="Share your experience, be specific about technologies, challenges, and outcomes..."
            )
            submitted = st.form_submit_button(f"Submit Answer ({remaining} remaining)")

            if submitted:
                answer = text_input.strip()
                if answer.lower() in ("stop", "quit", "exit"):
                    st.session_state.interview_active = False
                    st.session_state.session_complete = True
                    st.session_state.session_scores = calculate_session_score(
                        st.session_state.qa_history, st.session_state.resume_text
                    )
                    add_session_to_history(
                        st.session_state.session_scores,
                        st.session_state.interview_type,
                        st.session_state.difficulty
                    )
                    st.session_state.interview_answer = ""
                    st.session_state.audio_answer_processed = False
                    st.rerun()
                elif answer:
                    st.session_state.qa_history.append({"q": st.session_state.last_question, "a": answer})
                    st.session_state.full_qa_history.append({"q": st.session_state.last_question, "a": answer})

                    # Live score
                    if st.session_state.live_scoring:
                        ev = evaluate_answer(
                            st.session_state.llm, st.session_state.last_question, answer, st.session_state.resume_text
                        )
                        score_color = "#10b981" if ev['score'] >= 7 else "#f59e0b" if ev['score'] >= 4 else "#ef4444"
                        
                        # Add polite feedback for very low scores
                        polite_message = ""
                        if ev['score'] <= 3:
                            polite_message = '<div style="color: #ef4444; font-size: 0.95rem; font-weight: 600; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(239, 68, 68, 0.2);">Agent: "I appreciate your effort, but could you please try elaborating with a slightly more accurate or relevant answer?"</div>'

                        feedback_html = ''.join([f'<div style="color: #8892a4; font-size: 0.85rem; margin: 3px 0;">💡 {fb}</div>' for fb in ev['feedback']])
                        html_str = f'<div class="glass-card" style="border-left: 4px solid {score_color}; margin: 12px 0;">'
                        if polite_message:
                            html_str += polite_message
                        html_str += f'<div style="display: flex; justify-content: space-between; align-items: center;"><span style="color: #a78bfa; font-weight: 600;">Answer Score</span><span style="color: {score_color}; font-weight: 700; font-size: 1.3rem;">{ev["score"]}/10</span></div><div style="margin-top: 8px;">{feedback_html}</div></div>'
                        st.markdown(html_str, unsafe_allow_html=True)
                        time.sleep(2)

                    st.session_state.last_question = ""
                    st.session_state.tts_question_audio = None
                    st.session_state.tts_attempted = False
                    st.session_state.interview_answer = ""
                    st.session_state.audio_answer_processed = False
                    st.session_state.question_start_time = None
                    st.session_state.hint_used = False
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter an answer before submitting.")

        # Previous answers
        if st.session_state.qa_history:
            with st.expander(f"📜 Previous Answers ({len(st.session_state.qa_history)})", expanded=False):
                for i, item in enumerate(st.session_state.qa_history, 1):
                    item_q = item['q'].replace('"', '&quot;')
                    item_a = item['a'].replace('"', '&quot;')
                    st.markdown(f'<div class="history-item history-q"><div class="history-label" style="color: #6366f1;">Q{i}</div><div style="color: #c4b5fd; font-size: 0.9rem;">{item_q}</div></div><div class="history-item history-a"><div class="history-label" style="color: #10b981;">Your Answer</div><div style="color: #a7f3d0; font-size: 0.9rem;">{item_a}</div></div>', unsafe_allow_html=True)

    else:
        # Ready state — resume processed, not in interview
        if not st.session_state.session_complete:
            ai_status = "✅ ON" if st.session_state.use_ai_questions else "❌ OFF"
            voice_status = "✅ ON" if st.session_state.voice_playback else "❌ OFF"
            score_status = "✅ ON" if st.session_state.live_scoring else "❌ OFF"
            timer_status = f"✅ {st.session_state.timer_seconds}s" if st.session_state.timer_enabled else "❌ OFF"
            type_label = st.session_state.interview_type.title()
            diff_label = st.session_state.difficulty.title()

            st.markdown(f"""
            <div class="welcome-card">
                <div class="welcome-icon">🎯</div>
                <div class="welcome-title">Ready to Practice!</div>
                <div class="welcome-desc">
                    Your resume has been processed. Click <strong>"🚀 Start Interview"</strong> in the sidebar to begin.<br>
                    Each session consists of {QUESTIONS_PER_SESSION} tailored questions based on your resume.
                </div>
                <div class="stat-row" style="justify-content: center; margin-top: 24px;">
                    <div class="stat-chip">🧠 AI Questions {ai_status}</div>
                    <div class="stat-chip">🔊 Voice {voice_status}</div>
                    <div class="stat-chip">📊 Live Score {score_status}</div>
                    <div class="stat-chip">⏱️ Timer {timer_status}</div>
                </div>
                <div class="stat-row" style="justify-content: center;">
                    <div class="stat-chip">📋 Type: {type_label}</div>
                    <div class="stat-chip">🎯 Difficulty: {diff_label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()