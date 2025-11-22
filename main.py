import os
import time
import json
import traceback
import asyncio
import requests
import mimetypes
from io import BytesIO

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict

from playwright.async_api import async_playwright
import google.generativeai as genai

# Optional data tools (already in your requirements)
import pandas as pd
import pdfplumber

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/render/project/.playwright"

# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL")
STUDENT_SECRET = os.getenv("STUDENT_SECRET")

if not GEMINI_API_KEY or not STUDENT_EMAIL or not STUDENT_SECRET:
    raise RuntimeError(
        "Environment variables not set. Check GEMINI_API_KEY, STUDENT_EMAIL, STUDENT_SECRET."
    )

genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI()

# ---------------------------------------------------------------------------
# SPEC REQUIREMENT: INVALID JSON MUST RETURN 400 (not 422)
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid JSON or missing required fields."},
    )

# ---------------------------------------------------------------------------
# QUIZ ANALYSIS (SINGLE) - for your project/viva
# ---------------------------------------------------------------------------
class QuizEvalRequest(BaseModel):
    question: str
    correct_answer: str
    student_answer: str
    model_config = ConfigDict(extra="ignore")


@app.post("/quiz/analyze")
def analyze_quiz(request: QuizEvalRequest):
    prompt = (
        "You are an expert evaluator.\n\n"
        f"Question: {request.question}\n"
        f"Correct Answer: {request.correct_answer}\n"
        f"Student's Answer: {request.student_answer}\n\n"
        "Evaluate the student's answer and respond in JSON with:\n"
        "- correctness: true/false\n"
        "- score: number from 0 to 1\n"
        "- explanation: short reasoning.\n\n"
        "Return ONLY valid JSON. No markdown."
    )

    analysis_text = ""
    try:
        response = llm_model.generate_content(prompt)
        analysis_text = response.text.strip()

        if "{" in analysis_text and "}" in analysis_text:
            json_str = analysis_text[
                analysis_text.index("{") : analysis_text.rindex("}") + 1
            ]
        else:
            json_str = analysis_text

        return json.loads(json_str)

    except Exception as e:
        return {"error": str(e), "raw": analysis_text}


# ---------------------------------------------------------------------------
# QUIZ ANALYSIS (BATCH) - for your project/viva
# ---------------------------------------------------------------------------
class BatchQuizEvalRequest(BaseModel):
    items: list[QuizEvalRequest]
    model_config = ConfigDict(extra="ignore")


@app.post("/quiz/analyze_batch")
def analyze_quiz_batch(request: BatchQuizEvalRequest):
    results = []
    correct_count = 0

    for item in request.items:
        single = analyze_quiz(item)
        results.append(single)
        if isinstance(single, dict) and single.get("correctness") is True:
            correct_count += 1

    total = len(request.items)
    accuracy = correct_count / total if total else 0

    return {
        "total": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "results": results
    }


# ---------------------------------------------------------------------------
# MAIN QUIZ API REQUEST MODEL (SPEC)
# ---------------------------------------------------------------------------
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
        except Exception:
            await browser.close()
            raise


# ---------------------------------------------------------------------------
# LLM JSON EXTRACTOR (handles ```json fences)
# ---------------------------------------------------------------------------
def safe_json_loads(text: str):
    t = text.strip()
    if "{" in t and "}" in t:
        t = t[t.index("{"):t.rindex("}") + 1]
    return json.loads(t)


# ---------------------------------------------------------------------------
# EXTRACT QUIZ METADATA USING LLM
# Must return: question, submit_url, data_sources
# ---------------------------------------------------------------------------
def parse_quiz_with_llm(html_content: str):
    prompt = (
        "You are an expert quiz parser.\n\n"
        "From the HTML below, extract:\n"
        "1. The quiz question (short text)\n"
        "2. The submission URL (where answer must be POSTed)\n"
        "3. Any data/file/API URLs needed to solve.\n\n"
        "Return ONLY valid JSON in this format:\n"
        '{ "question": "...", "submit_url": "...", "data_sources": ["..."] }\n\n'
        "No markdown, no extra text.\n"
    )

    response = llm_model.generate_content(prompt + html_content)
    raw = response.text
    try:
        return safe_json_loads(raw)
    except Exception:
        raise RuntimeError("Failed to parse quiz metadata with LLM")


# ---------------------------------------------------------------------------
# DOWNLOAD HELPER
# ---------------------------------------------------------------------------
def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def guess_ext(url: str, content_type: str | None):
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext
    # fallback from URL
    return os.path.splitext(url)[1].lower()


# ---------------------------------------------------------------------------
# SIMPLE DATA PREVIEW HELPERS
# ---------------------------------------------------------------------------
def preview_pdf_text(pdf_bytes: bytes, max_pages=2) -> str:
    out = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
            out.append(page.extract_text() or "")
    return "\n".join(out)


def preview_table_bytes(data_bytes: bytes, ext: str) -> str:
    try:
        if ext in [".csv"]:
            df = pd.read_csv(BytesIO(data_bytes))
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(BytesIO(data_bytes))
        elif ext in [".json"]:
            df = pd.read_json(BytesIO(data_bytes))
        else:
            return ""
        return df.head(8).to_string(index=False)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# SOLVE QUIZ (LLM + previews)
# This is general-purpose because real quizzes vary.
# ---------------------------------------------------------------------------
def solve_quiz(parsed_quiz: dict):
    question = parsed_quiz.get("question", "")
    sources = parsed_quiz.get("data_sources", []) or []

    previews = []
    for u in sources:
        try:
            content = download_bytes(u)
            ctype = requests.head(u).headers.get("content-type")
            ext = guess_ext(u, ctype)

            if ext == ".pdf":
                previews.append(f"\n--- PDF PREVIEW ({u}) ---\n{preview_pdf_text(content)}\n")
            elif ext in [".csv", ".xlsx", ".xls", ".json"]:
                previews.append(f"\n--- TABLE PREVIEW ({u}) ---\n{preview_table_bytes(content, ext)}\n")
            else:
                # for text / html / unknown: just note URL
                previews.append(f"\n--- DATA SOURCE ({u}) ---\n(Downloaded {len(content)} bytes)\n")
        except Exception as e:
            previews.append(f"\n--- DATA SOURCE ({u}) FAILED ---\n{e}\n")

    prompt = (
        "You are solving a data quiz.\n\n"
        f"Question:\n{question}\n\n"
        "Here are previews of any data sources:\n"
        + "\n".join(previews)
        + "\n\n"
        "Compute the final answer.\n"
        "Return ONLY valid JSON like:\n"
        '{ "answer": <number|string|boolean|object> }\n'
        "No markdown.\n"
    )

    resp = llm_model.generate_content(prompt).text
    try:
        return safe_json_loads(resp)
    except Exception:
        # fallback: return raw text
        return {"answer": resp.strip()}


# ---------------------------------------------------------------------------
# SUBMIT ANSWER (NO HARDCODED URL)
# ---------------------------------------------------------------------------
def submit_answer(submit_url: str, quiz_url: str, answer):
    payload = {
        "email": STUDENT_EMAIL,
        "secret": STUDENT_SECRET,
        "url": quiz_url,
        "answer": answer
    }
    resp = requests.post(submit_url, json=payload, timeout=30)
    try:
        return resp.json()
    except Exception:
        return {"correct": False, "reason": "Invalid server response"}


# ---------------------------------------------------------------------------
# MAIN QUIZ WORKER (BACKGROUND) - 3 minute limit
# ---------------------------------------------------------------------------
async def solve_quiz_chain(initial_url: str):
    start_time = time.time()
    current_url = initial_url

    print("\n🧵 Worker started solving chain...\n")

    while True:
        if time.time() - start_time > 175:
            print("⏳ TIMEOUT: 3-minute limit exceeded.")
            return

        try:
            html = await fetch_quiz_page(current_url)
            parsed = parse_quiz_with_llm(html)
            solution = solve_quiz(parsed)

            submit_url = parsed.get("submit_url")
            if not submit_url:
                print("❌ No submit_url found.")
                return

            response = submit_answer(submit_url, current_url, solution.get("answer"))

            print("🟦 Server Response:", response)

            # If correct → go next if present, else finish
            if response.get("correct") is True:
                next_url = response.get("url")
                if not next_url:
                    print("🏁 Quiz chain finished.")
                    return
                current_url = next_url
                continue

            # If wrong but new url exists → skip to next quiz (spec allows)
            next_url = response.get("url")
            if next_url:
                print(f"➡️ Skipping to next quiz URL: {next_url}")
                current_url = next_url
                continue

            # Else retry same quiz within time
            print("🔁 Retrying same quiz...")
            continue

        except Exception:
            print("❌ Worker error:", traceback.format_exc())
            return


# ---------------------------------------------------------------------------
# API ENDPOINT — SPEC COMPLIANT
# ---------------------------------------------------------------------------
@app.post("/")
async def handle_quiz(task: QuizRequest, bg: BackgroundTasks):

    # Validate secret (spec)
    if task.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Start background solving
    bg.add_task(solve_quiz_chain, task.url)

    # Immediate 200 response (spec)
    return {"status": "accepted", "message": "Quiz solving started"}


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {"status": "Server is running"}


# ---------------------------------------------------------------------------
# START UVICORN SERVER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
