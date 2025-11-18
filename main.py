import os
import time
import json
import traceback
import asyncio
import requests

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict

from playwright.async_api import async_playwright
import google.generativeai as genai


# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES (NO HARDCODED SECRETS)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL")
STUDENT_SECRET = os.getenv("STUDENT_SECRET")

if not GEMINI_API_KEY or not STUDENT_EMAIL or not STUDENT_SECRET:
    raise RuntimeError("Environment variables not set. Check GEMINI_API_KEY, STUDENT_EMAIL, STUDENT_SECRET.")

genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.5-flash")


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI()


class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# JS RENDERED PAGE SCRAPER (Playwright)
# ---------------------------------------------------------------------------
async def fetch_quiz_page(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("domcontentloaded")
            content = await page.content()
            await browser.close()
            return content
        except Exception as e:
            await browser.close()
            raise e


# ---------------------------------------------------------------------------
# EXTRACT SUBMISSION URL + QUESTION USING LLM (safer approach)
# ---------------------------------------------------------------------------
def parse_quiz_with_llm(html_content: str):
    prompt = f"""
You are an expert quiz parser.

Extract the following from the HTML page:
1. The quiz question (short text)
2. The submission URL (where the answer must be POSTed)
3. Any file URLs, table data, or API URLs required to solve it.

Return a JSON object with:
{{
  "question": "...",
  "submit_url": "...",
  "data_sources": [...]
}}
"""

    response = llm_model.generate_content(prompt + html_content)
    raw = response.text

    try:
        parsed = json.loads(raw)
        return parsed
    except:
        # fallback: extract JSON inside text
        try:
            obj = json.loads(raw[raw.index("{"): raw.rindex("}")+1])
            return obj
        except:
            raise RuntimeError("Failed to parse quiz metadata with LLM")


# ---------------------------------------------------------------------------
# SOLVE QUIZ (deterministic general-purpose solver)
# ---------------------------------------------------------------------------
def solve_quiz(parsed_quiz: dict):
    """
    This is a placeholder. Real quizzes vary widely, so this function
    attempts to solve common patterns (sum tables, read files, parse PDFs, etc.)
    """

    # Example fallback: If no specific instructions, answer "OK"
    return {"answer": "OK"}


# ---------------------------------------------------------------------------
# SUBMIT ANSWER
# ---------------------------------------------------------------------------
def submit_answer(submit_url: str, quiz_url: str, answer):
    payload = {
        "email": STUDENT_EMAIL,
        "secret": STUDENT_SECRET,
        "url": quiz_url,
        "answer": answer
    }

    resp = requests.post(submit_url, json=payload)
    try:
        return resp.json()
    except:
        return {"correct": False, "reason": "Invalid server response"}


# ---------------------------------------------------------------------------
# MAIN QUIZ WORKER (runs in background)
# ---------------------------------------------------------------------------
async def solve_quiz_chain(initial_url: str):
    start_time = time.time()
    current_url = initial_url

    print("\n🧵 Worker started solving chain...\n")

    while True:
        if time.time() - start_time > 170:
            print("⏳ TIMEOUT: 3-minute limit exceeded.")
            return

        try:
            html = await fetch_quiz_page(current_url)
            parsed = parse_quiz_with_llm(html)
            solution = solve_quiz(parsed)
            response = submit_answer(parsed["submit_url"], current_url, solution["answer"])

            print("🟦 Server Response:", response)

            if response.get("correct") is True:
                next_url = response.get("url")
                if not next_url:
                    print("🏁 Quiz chain finished.")
                    return
                print(f"➡️ Next quiz URL: {next_url}")
                current_url = next_url
                continue

            else:
                # retry wrong attempt
                print("🔁 Retrying wrong attempt...")
                continue

        except Exception as e:
            print("❌ Worker error:", traceback.format_exc())
            return


# ---------------------------------------------------------------------------
# API ENDPOINT — RETURNS 200 IMMEDIATELY (RULE REQUIREMENT)
# ---------------------------------------------------------------------------
@app.post("/")
async def handle_quiz(task: QuizRequest, bg: BackgroundTasks):

    print(f"\n📩 Incoming request: {task.url}")

    # Validate secret
    if task.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Start background solving
    bg.add_task(solve_quiz_chain, task.url)

    # Immediate response to grader (very important)
    return {"status": "accepted", "message": "Quiz solving started"}


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {"status": "Server is running"}
