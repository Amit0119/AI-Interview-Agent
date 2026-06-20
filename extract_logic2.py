import re

with open(r"c:\Users\R K\OneDrive\Desktop\Agentic_Ai+AWS_internship\AI-interview-agent\app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

def print_context(pattern_str):
    print(f"\n--- Searching for {pattern_str} ---")
    pattern = re.compile(pattern_str, re.IGNORECASE)
    for i, line in enumerate(lines):
        if pattern.search(line):
            print(f"L{i+1}: {line.strip()}")

print_context(r"st\.session_state\.questions")
print_context(r"def generate_questions")
print_context(r"run_tts_sync")
