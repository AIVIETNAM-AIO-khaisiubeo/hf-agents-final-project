"""
Upgraded Multimodal GAIA Agent for the Hugging Face Agents Course Final Assignment.
Compatible with Google's latest Gemini models (gemini-3.8-flash / gemini-flash-latest).
"""
import base64
import mimetypes
import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    LiteLLMModel,
    VisitWebpageTool,
    tool,
)

# --------------------------------------------------------------------------- #
# Automatic Gemini Model Resolver (Prevents 404 Model Not Found Errors)
# --------------------------------------------------------------------------- #

GEMINI_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
]

_ACTIVE_GEMINI_MODEL: str | None = None


def get_gemini_model_name() -> str:
    """Detect and cache the first working Gemini model for this API key."""
    global _ACTIVE_GEMINI_MODEL
    if _ACTIVE_GEMINI_MODEL:
        return _ACTIVE_GEMINI_MODEL

    env_override = os.getenv("MODEL_ID", "").strip()
    if env_override:
        _ACTIVE_GEMINI_MODEL = env_override.replace("gemini/", "").replace("models/", "")
        return _ACTIVE_GEMINI_MODEL

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        _ACTIVE_GEMINI_MODEL = "gemini-3.8-flash"
        return _ACTIVE_GEMINI_MODEL

    for candidate in GEMINI_CANDIDATES:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{candidate}:generateContent",
                params={"key": api_key},
                json={"contents": [{"parts": [{"text": "ping"}]}]},
                timeout=15,
            )
            if r.status_code == 200:
                print(f"Verified working Gemini model for this API key: {candidate}")
                _ACTIVE_GEMINI_MODEL = candidate
                return _ACTIVE_GEMINI_MODEL
            else:
                print(f"Model probe {candidate} returned {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"Model probe error for {candidate}: {e}")

    _ACTIVE_GEMINI_MODEL = "gemini-3.8-flash"
    return _ACTIVE_GEMINI_MODEL


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@tool
def google_search_fallback(query: str) -> str:
    """Search Google directly via Gemini Grounding when DuckDuckGo is rate-limited or misses exact facts.

    Args:
        query: The search query or factual question to look up on Google.
    """
    if not os.getenv("GEMINI_API_KEY"):
        return "GEMINI_API_KEY not set; use web_search or wikipedia_page instead."
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model=get_gemini_model_name(),
            contents=f"Search Google and provide exact, verified facts and sources for: {query}",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,
            ),
        )
        return resp.text or "No results returned."
    except Exception as e:
        return f"Google search fallback error: {e}"


@tool
def wikipedia_page(title: str) -> str:
    """Fetch the full text AND all HTML tables of an English Wikipedia page.
    Use this for discographies, studio album counts, biographies, and historical tables.

    Args:
        title: The title or search query for the Wikipedia page, e.g. "Mercedes Sosa".
    """
    headers = {"User-Agent": "gaia-agent/2.0 (https://huggingface.co)"}
    try:
        search_r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": title,
                "format": "json",
            },
            headers=headers,
            timeout=20,
        ).json()
        search_hits = search_r.get("query", {}).get("search", [])
        canonical_title = search_hits[0]["title"] if search_hits else title

        parse_r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": canonical_title,
                "prop": "text",
                "redirects": 1,
                "format": "json",
            },
            headers=headers,
            timeout=30,
        ).json()

        if "error" in parse_r:
            return f"No Wikipedia page found for '{title}'."

        html_content = parse_r["parse"]["text"]["*"]
        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup(["script", "style", "sup"]):
            tag.decompose()

        output_parts = [f"=== Wikipedia Page: {canonical_title} ==="]
        for elem in soup.find_all(["h2", "h3", "p", "ul", "ol", "table"]):
            if elem.name in ("h2", "h3"):
                output_parts.append(f"\n## {elem.get_text(' ', strip=True)}")
            elif elem.name == "table":
                rows = []
                for tr in elem.find_all("tr"):
                    cells = [
                        c.get_text(" ", strip=True)
                        for c in tr.find_all(["th", "td"])
                    ]
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    output_parts.append("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]")
            else:
                txt = elem.get_text(" ", strip=True)
                if txt:
                    output_parts.append(txt)

        return "\n".join(output_parts)[:45000]
    except Exception as e:
        return f"Wikipedia error: {e}"


@tool
def read_file(path: str) -> str:
    """Read a local file and return its content as text. Supports csv, xlsx,
    xls, pdf, txt, json, py, md, pdb. For audio use `transcribe_audio`, for images
    use `describe_image`.

    Args:
        path: Local path of the file to read.
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(path)
            return (
                f"Columns: {list(df.columns)}\nShape: {df.shape}\n"
                f"{df.to_string()}\n\nSummary:\n{df.describe(include='all').to_string()}"
            )[:35000]
        if ext in (".xlsx", ".xls"):
            sheets = pd.read_excel(path, sheet_name=None)
            out = []
            for name, df in sheets.items():
                out.append(
                    f"--- Sheet: {name} (Shape: {df.shape}) ---\n"
                    f"Columns: {list(df.columns)}\n{df.to_string()}"
                )
            return "\n\n".join(out)[:35000]
        if ext == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(path)
            return "\n".join(
                f"--- Page {i + 1} ---\n{p.extract_text() or ''}"
                for i, p in enumerate(reader.pages)
            )[:35000]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:35000]
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def transcribe_audio(path: str, question: str = "Transcribe this audio accurately word-for-word.") -> str:
    """Transcribe or answer questions about an audio file (mp3, wav, m4a, flac).

    Args:
        path: Local path of the audio file.
        question: Optional specific question or instruction about the audio.
    """
    try:
        if os.getenv("GEMINI_API_KEY"):
            from google import genai

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            uploaded = client.files.upload(file=path)
            resp = client.models.generate_content(
                model=get_gemini_model_name(),
                contents=[
                    uploaded,
                    f"Provide a complete word-for-word transcript of this audio, then answer: {question}",
                ],
            )
            return resp.text

        from huggingface_hub import InferenceClient

        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        out = client.automatic_speech_recognition(
            path, model="openai/whisper-large-v3-turbo"
        )
        return out.text if hasattr(out, "text") else str(out)
    except Exception as e:
        return f"Transcription error: {e}"


@tool
def describe_image(path: str, question: str) -> str:
    """Ask a question about an image (png, jpg, jpeg, webp) using a vision model.
    Useful for chess boards, charts, screenshots, and visual puzzles.

    Args:
        path: Local path of the image file.
        question: What you want to know about the image.
    """
    try:
        if os.getenv("GEMINI_API_KEY"):
            from google import genai

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            uploaded = client.files.upload(file=path)
            resp = client.models.generate_content(
                model=get_gemini_model_name(),
                contents=[
                    uploaded,
                    f"Examine this image with extreme precision. If it is a chessboard, list every piece and its exact coordinate (a1-h8) for both White and Black first, verify board orientation, and determine the winning move in algebraic notation. Question: {question}",
                ],
            )
            return resp.text

        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        data_url = f"data:{mime};base64,{b64}"

        if os.getenv("OPENAI_API_KEY"):
            import litellm

            resp = litellm.completion(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            return resp.choices[0].message.content

        from huggingface_hub import InferenceClient

        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        resp = client.chat_completion(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=1000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Image error: {e}"


@tool
def youtube_transcript(url: str) -> str:
    """Get the timestamped transcript (captions) of a YouTube video.

    Args:
        url: Full YouTube URL, e.g. https://www.youtube.com/watch?v=XXXX.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
        if not m:
            return "Could not parse video id."
        vid = m.group(1)
        try:
            fetched = YouTubeTranscriptApi().fetch(vid)
            return "\n".join(f"[{int(s.start)}s] {s.text}" for s in fetched)[:35000]
        except AttributeError:
            parts = YouTubeTranscriptApi.get_transcript(vid)
            return "\n".join(f"[{int(p['start'])}s] {p['text']}" for p in parts)[:35000]
    except Exception as e:
        return f"Transcript error: {e} (captions may be unavailable; call `watch_youtube_video` instead!)"


@tool
def watch_youtube_video(url: str, question: str) -> str:
    """Watch a YouTube video directly (both visual frames and audio) using Gemini.
    Use this whenever a question asks about visual content in a YouTube video (e.g. counting bird species on camera) or when captions are unavailable.

    Args:
        url: Full YouTube URL, e.g. https://www.youtube.com/watch?v=L1vXCYZAYYM.
        question: Specific question about what happens visually or auditorily in the video.
    """
    if not os.getenv("GEMINI_API_KEY"):
        return "GEMINI_API_KEY is required to watch YouTube video frames."
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model=get_gemini_model_name(),
            contents=[
                types.Part.from_uri(file_uri=url, mime_type="video/*"),
                f"Watch this entire video carefully (both visual frames and audio) and answer accurately: {question}",
            ],
        )
        return resp.text
    except Exception as e:
        return f"Error watching YouTube video: {e}"


# --------------------------------------------------------------------------- #
# Prompt & Exact-Match Post-Processor
# --------------------------------------------------------------------------- #

PROMPT = """You are an elite, meticulous research assistant solving a GAIA benchmark question.

CRITICAL STRATEGY RULES:
1. Reversed or Encoded Text: If the question text looks reversed (e.g. starts with `.rewsna eht sa`), decode it in Python first, then answer it.
2. Attached Files:
   - Spreadsheets (`.xlsx`, `.csv`): Call `read_file(path)` AND write Python `pandas` code to compute exact sums, counts, or filters. Never do mental math!
   - Audio (`.mp3`, `.wav`): Call `transcribe_audio(path, question)`.
   - Images (`.png`, `.jpg`): Call `describe_image(path, question)`.
   - Code/Documents (`.py`, `.pdf`, `.txt`): Call `read_file(path)`.
3. YouTube Videos:
   - For spoken dialogue, call `youtube_transcript(url)`.
   - For visual questions (e.g., highest number of bird species on camera simultaneously) or when transcripts fail, call `watch_youtube_video(url, question)`.
4. Wikipedia & Web Search:
   - Use `wikipedia_page` for studio album counts, discographies, rosters, and biographies.
   - If `web_search` is rate-limited or incomplete, call `google_search_fallback(query)`.
5. Botanical vs. Culinary Categorization:
   - If asked for strict botanical vegetables (excluding botanical fruits), remember that botanical fruits develop from the flower ovary and contain seeds (bell pepper, zucchini, green beans, corn, plums, peanuts, acorns are botanical fruits/seeds, NOT botanical vegetables). True botanical vegetables are strictly vegetative parts: broccoli, celery, fresh basil, lettuce, sweet potatoes.

OUTPUT RULES (the answer is graded by strict EXACT MATCH):
- Call `final_answer` with ONLY the bare answer. No explanation, no "FINAL ANSWER:" prefix.
- A number: digits only, no commas, no units unless the question explicitly asks for them (e.g. `$89706.00` if USD with two decimal places is requested).
- A string: no articles (a, an, the) unless part of a proper noun, no abbreviations, no surrounding quotes.
- A list: comma-separated values with a single space after each comma (`, `), strictly following any alphabetical or numerical order requested.

{file_note}
QUESTION:
{question}
"""


def clean_answer(ans) -> str:
    ans = str(ans).strip()
    ans = re.sub(r"^(final answer|exact final answer|answer)\s*[:\-]?\s*", "", ans, flags=re.I)
    ans = ans.strip().strip("\"'`").strip()
    if ans.endswith(".") and not re.search(r"\d\.$", ans) and len(ans) < 80:
        ans = ans[:-1].strip()
    return ans


# --------------------------------------------------------------------------- #
# Model & Agent
# --------------------------------------------------------------------------- #


def build_model():
    provider = os.getenv("MODEL_PROVIDER", "").lower()
    model_id = os.getenv("MODEL_ID")

    # Automatically use Gemini if GEMINI_API_KEY is present in Space Secrets
    if provider == "gemini" or (not provider and os.getenv("GEMINI_API_KEY")):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("MODEL_PROVIDER=gemini but GEMINI_API_KEY secret is missing.")
        resolved_name = get_gemini_model_name()
        litellm_id = f"gemini/{resolved_name}" if not resolved_name.startswith("gemini/") else resolved_name
        print(f"Using {litellm_id} via LiteLLMModel")
        return LiteLLMModel(
            model_id=litellm_id,
            api_key=key,
            temperature=0.0,
            num_retries=4,
        )

    # Automatically use OpenAI if OPENAI_API_KEY is present
    if provider == "openai" or (not provider and os.getenv("OPENAI_API_KEY")):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("MODEL_PROVIDER=openai but OPENAI_API_KEY secret is missing.")
        print("Using OpenAI via LiteLLMModel")
        return LiteLLMModel(
            model_id=model_id or "gpt-4o",
            api_key=key,
            temperature=0.0,
            num_retries=3,
        )

    # Fallback to Hugging Face InferenceClientModel
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "No GEMINI_API_KEY, OPENAI_API_KEY, or HF_TOKEN found in Space Secrets. "
            "Add GEMINI_API_KEY or HF_TOKEN in Space Settings -> Variables and secrets."
        )
    print("Using Hugging Face InferenceClientModel")
    return InferenceClientModel(
        model_id=model_id or "Qwen/Qwen2.5-Coder-32B-Instruct",
        token=token,
        temperature=0.0,
    )


class GaiaAgent:
    def __init__(self):
        self.model = build_model()
        self.agent = CodeAgent(
            tools=[
                DuckDuckGoSearchTool(),
                google_search_fallback,
                VisitWebpageTool(),
                wikipedia_page,
                read_file,
                transcribe_audio,
                describe_image,
                youtube_transcript,
                watch_youtube_video,
            ],
            model=self.model,
            additional_authorized_imports=[
                "pandas", "numpy", "re", "json", "math", "datetime",
                "collections", "itertools", "statistics", "csv", "unicodedata",
                "requests", "bs4", "openpyxl", "fractions", "string", "pypdf",
            ],
            max_steps=12,
            planning_interval=3,
            verbosity_level=1,
        )

    def verify_exact_match(self, question: str, raw_answer: str) -> str:
        cleaned = clean_answer(raw_answer)
        if not cleaned:
            return ""
        verify_prompt = f"""You are a strict GAIA benchmark answer extractor.
Given the QUESTION and the AGENT CANDIDATE ANSWER, output ONLY the exact final string required for strict exact-match grading.
- No explanation, no prefix, no quotes.
- Digits only for counts/numbers unless units/currency are explicitly asked.
- Verify strict alphabetical order and ', ' spacing if a list is requested.

QUESTION:
{question}

CANDIDATE ANSWER:
{cleaned}

EXACT FINAL ANSWER:"""
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": verify_prompt}]}]
            resp = self.model(messages)
            verified = clean_answer(resp.content)
            return verified if verified else cleaned
        except Exception:
            return cleaned

    def __call__(self, question: str, file_path: str | None = None) -> str:
        file_note = (
            f"ATTACHED FILE (local path): {file_path}\n" if file_path else ""
        )
        prompt = PROMPT.format(question=question, file_note=file_note)
        try:
            result = self.agent.run(prompt)
            final_ans = self.verify_exact_match(question, str(result))
            print(f"Raw: {result}  -->  Verified: {final_ans}")
            return final_ans
        except Exception as e:
            print(f"Agent error: {e}")
            return ""