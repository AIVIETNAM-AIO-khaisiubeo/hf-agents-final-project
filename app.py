import json
import os
import time

import gradio as gr
import pandas as pd
import requests

from agent import GaiaAgent

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
FILES_DIR = "task_files"
CACHE_FILE = "cached_answers.json"
os.makedirs(FILES_DIR, exist_ok=True)


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def download_task_file(api_url: str, task_id: str, file_name: str) -> str | None:
    """Download the file attached to a task (if any) and return its local path.

    1) try the course scoring API  2) fall back to the GAIA dataset on the Hub
    (needs access to the gated dataset gaia-benchmark/GAIA + HF_TOKEN).
    """
    if not file_name:
        return None
    path = os.path.join(FILES_DIR, f"{task_id}_{file_name}")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    try:
        r = requests.get(f"{api_url}/files/{task_id}", timeout=60)
        r.raise_for_status()
        if len(r.content) > 0:
            with open(path, "wb") as f:
                f.write(r.content)
            return path
    except Exception as e:
        print(f"API file download failed for {task_id}: {e}")

    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            filename=f"2023/validation/{file_name}",
            token=os.getenv("HF_TOKEN"),
        )
        print(f"Got {file_name} from the GAIA dataset.")
        return local
    except Exception as e:
        print(f"GAIA dataset fallback failed for {task_id}: {e}")
        return None


def run_and_submit_all(use_cache: bool = True, profile: gr.OAuthProfile | None = None):
    """Fetch all questions, run the agent on them, submit, and show the score."""
    space_id = os.getenv("SPACE_ID", "")

    if profile and getattr(profile, "username", None):
        username = f"{profile.username}"
        print(f"User logged in via OAuth: {username}")
    elif os.getenv("HF_USERNAME"):
        username = os.getenv("HF_USERNAME")
        print(f"Using HF_USERNAME env var: {username}")
    elif space_id and "/" in space_id:
        username = space_id.split("/")[0]
        print(f"Using Space owner username fallback: {username}")
    else:
        return "Please Login to Hugging Face with the button above.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate the agent
    try:
        agent = GaiaAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    agent_code = (
        f"https://huggingface.co/spaces/{space_id}/tree/main"
        if space_id
        else f"https://huggingface.co/spaces/{username}/Final_Assignment_Template/tree/main"
    )
    print(f"Agent code link: {agent_code}")

    # 2. Fetch questions (check local questions.json first if present, otherwise live API)
    questions_data = None
    if os.path.exists("questions.json"):
        try:
            with open("questions.json", "r", encoding="utf-8") as f:
                questions_data = json.load(f)
            print(f"Loaded {len(questions_data)} questions from local questions.json.")
        except Exception as e:
            print(f"Warning reading local questions.json: {e}")

    if not questions_data:
        try:
            response = requests.get(questions_url, timeout=30)
            response.raise_for_status()
            questions_data = response.json()
            if not questions_data:
                return "Fetched questions list is empty or invalid format.", None
            print(f"Fetched {len(questions_data)} questions from API.")
        except Exception as e:
            return f"Error fetching questions: {e}", None

    # 3. Run the agent with caching and retry logic
    cache = load_cache() if use_cache else {}
    results_log = []
    answers_payload = []

    for i, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name") or ""
        if not task_id or question_text is None:
            continue

        print(f"\n=== [{i}/{len(questions_data)}] task {task_id} ===")

        if use_cache and task_id in cache and cache[task_id] not in ("", "Error", None):
            submitted_answer = cache[task_id]
            print(f"Loaded from cache: {submitted_answer}")
        else:
            file_path = download_task_file(api_url, task_id, file_name)
            submitted_answer = ""
            for attempt in range(2):  # one retry on empty answer / transient error
                submitted_answer = agent(question_text, file_path)
                if submitted_answer and submitted_answer != "Error":
                    break
                time.sleep(5)

            if submitted_answer and submitted_answer != "Error":
                cache[task_id] = submitted_answer
                save_cache(cache)
            time.sleep(2)  # be gentle with rate limits

        answers_payload.append(
            {"task_id": task_id, "submitted_answer": submitted_answer}
        )
        results_log.append(
            {
                "Task ID": task_id,
                "Question": question_text,
                "File": file_name,
                "Submitted Answer": submitted_answer,
            }
        )

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Submit
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/"
            f"{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions**
        1. Add `GEMINI_API_KEY` (and optionally `HF_TOKEN`) in **Space Settings -> Variables and secrets**.
        2. Log in with Hugging Face below.
        3. Click **Run Evaluation & Submit All Answers** (takes ~5–10 minutes).

        Keep your Space **public** so the code link can be verified.
        """
    )
    gr.LoginButton()
    use_cache_checkbox = gr.Checkbox(
        value=True,
        label="Use cached answers from previous runs (uncheck to re-run all 20 questions from scratch)",
    )
    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        inputs=[use_cache_checkbox],
        outputs=[status_output, results_table],
    )

if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    demo.launch(debug=True, share=False, ssr_mode=False)