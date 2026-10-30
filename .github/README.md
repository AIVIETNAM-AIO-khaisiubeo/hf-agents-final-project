# GAIA Multimodal Research Agent

A tool-using agent for the Hugging Face Agents Course final assignment. It uses **smolagents** to answer GAIA benchmark questions, work with attached files, and submit answers to the course evaluator.

## How it works

1. Fetch tasks from a local \`questions.json\` when present, or from the course API.
2. Download optional task attachments; if the course API fails, try the gated GAIA dataset.
3. Run a \`smolagents.CodeAgent\` that selects research, file-processing, or multimodal tools and can use Python for calculations.
4. Clean the candidate answer and ask the model to verify its exact-match formatting.
5. Cache successful answers by task ID and submit the answers to the evaluator.

## Methods worth noting

- **Tool use:** DuckDuckGo search, webpage retrieval, Wikipedia, and an optional Gemini-powered Google Search fallback.
- **Structured-data reasoning:** pandas for CSV and Excel; the agent prompt encourages code-based calculations instead of mental arithmetic.
- **Multimodal tools:** PDF text extraction, image analysis, audio transcription, YouTube captions, and optional Gemini video analysis.
- **Answer formatting:** prompt constraints, answer cleanup, and a second model call. This improves format consistency but does not prove that an answer is factually correct.
- **Reliability:** one retry for empty/error answers, short delays between requests, and an optional local cache.

## Code map

| File | Responsibility |
|---|---|
| [\`agent.py\`](../agent.py) | Model selection, tools, agent prompt, \`CodeAgent\`, answer cleanup and verification. |
| [\`app.py\`](../app.py) | Gradio/OAuth UI, task and attachment retrieval, cache, submission and score display. |
| [\`requirements.txt\`](../requirements.txt) | Runtime dependencies. |
| [Space configuration](../README.md) | Hugging Face Spaces metadata, including Gradio and OAuth settings. |

## Run locally

Use Python 3.10+ and a virtual environment. On Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
~~~

For the intended course submission flow, deploy to a Hugging Face **Gradio Space** with OAuth enabled. Local execution may not reproduce hosted OAuth and submission attribution.

## Configuration

Add credentials in the Space's **Settings → Variables and secrets**. Never commit keys to Git.

| Variable | Purpose |
|---|---|
| \`GEMINI_API_KEY\` | Gemini, Google Search grounding, and video analysis. |
| \`OPENAI_API_KEY\` | OpenAI model path and image-analysis fallback. |
| \`HF_TOKEN\` | Hugging Face inference and, if authorized, access to the gated GAIA dataset. |
| \`MODEL_PROVIDER\` | Optional primary provider: \`gemini\` or \`openai\`. |
| \`MODEL_ID\` | Optional primary model ID override. |
| \`SPACE_ID\` | Space identifier used in the submitted code URL. |

Without an explicit provider, the code selects Gemini or OpenAI when the corresponding key is present; otherwise it uses Hugging Face Inference, which requires \`HF_TOKEN\` or \`HUGGINGFACEHUB_API_TOKEN\`.

## Evaluation and cache

The app uses the course API at [agents-course-unit4-scoring.hf.space](https://agents-course-unit4-scoring.hf.space):

- \`GET /questions\` retrieves tasks.
- \`GET /files/{task_id}\` attempts to retrieve attachments.
- \`POST /submit\` sends the username, agent-code URL, and answers for scoring.

The UI can reuse \`cached_answers.json\`. Because the cache is keyed by task ID rather than model or prompt version, disable caching or remove the file when you want to rerun tasks after changing the agent.

## Limitations

- Output quality depends on the selected model, tool results, credentials, quotas, and network availability.
- PDF support extracts text only; OCR for scanned pages is not implemented.
- YouTube transcript retrieval depends on available captions; video analysis requires Gemini.
- \`CodeAgent\` executes generated code. Run the project only in an environment whose permissions and isolation you understand.
- A successful submission does not establish correctness; use the score returned by the evaluator as the benchmark result.

## References

- [smolagents documentation](https://huggingface.co/docs/smolagents)
- [Hugging Face Spaces configuration](https://huggingface.co/docs/hub/spaces-config-reference)
