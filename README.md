---
title: GAIA Multimodal Research Agent
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# GAIA Multimodal Research Agent

A tool-using research agent built with **smolagents** for the Hugging Face Agents Course final assignment. The app retrieves GAIA questions, optionally downloads their attached files, reasons with a configurable language model, calls specialist tools, formats answers for strict exact-match grading, caches successful answers, and submits the result to the course evaluation API.

The implementation is designed for benchmark problem-solving and experimentation. A successful API submission or a plausible answer does not guarantee correctness; the scoring service is the source of the reported benchmark score.

## What the agent does

For each task, the app can:

1. Read the question and identify whether it requires web research, structured calculation, file interpretation, image understanding, audio transcription, or video inspection.
2. Call tools to gather evidence or process the supplied attachment.
3. Use Python-based reasoning and calculations when needed, rather than relying on mental arithmetic.
4. Produce a concise answer in the exact format requested by the question.
5. Clean and verify the candidate answer with a second model call.
6. Reuse a cached answer for a task ID when caching is enabled.
7. Submit the collected answers and display the score returned by the course evaluator.

## Architecture and code map

The main agent implementation and the Gradio evaluation interface are separated so that task orchestration is distinct from reasoning and tool implementation.

| File | Responsibility |
|---|---|
| **app.py** | Gradio UI, Hugging Face login, question retrieval, attachment downloads, retry logic, local answer cache, submission payload and score display. |
| **agent.py** | Model selection, custom tools, agent prompt, CodeAgent configuration, answer cleaning and exact-match verification. |
| **requirements.txt** | Python packages needed for the app and tool integrations. |
| **README.md** | Hugging Face Spaces configuration and project documentation. |

### Request and submission flow

~~~text
User opens the Gradio Space and signs in
                 |
                 v
app.py: choose username and fetch questions
                 |
                 v
For each task: read task_id, question, optional file_name
                 |
                 +------> download attachment if present
                 |          1. Course API /files/{task_id}
                 |          2. GAIA dataset fallback (requires dataset access)
                 v
agent.py: build model -> run CodeAgent
                 |
                 +------> search / webpages / Wikipedia
                 +------> CSV, Excel, PDF and text-file reading
                 +------> image analysis / audio transcription
                 +------> YouTube transcript or video analysis
                 +------> Python calculations and transformations
                 |
                 v
Clean candidate answer -> verify exact formatting
                 |
                 v
app.py: save successful answer in cached_answers.json
                 |
                 v
POST answers to course /submit endpoint
                 |
                 v
Display score and per-question submitted answers
~~~

## Methods and design choices

### 1. Tool-using agent with smolagents

The GaiaAgent class constructs a smolagents CodeAgent. The agent is given a language model, a set of tools, a maximum of 12 reasoning steps, a planning interval of 3 steps, and a restricted list of additional authorized Python imports.

The code agent can choose tools and generate Python code to perform operations such as filtering a table or computing an exact total. This is useful for questions where the answer depends on structured data rather than a general-language estimate.

The main tools registered in GaiaAgent are:

- **DuckDuckGoSearchTool** — general web search.
- **VisitWebpageTool** — visit and read a webpage.
- **google_search_fallback** — query Google Search grounding through Gemini when a Gemini API key is available.
- **wikipedia_page** — search for an English Wikipedia page and extract its headings, paragraphs, lists and HTML tables.
- **read_file** — read tabular, PDF or text-like attachments.
- **transcribe_audio** — transcribe or answer a question about an audio attachment.
- **describe_image** — ask a vision model about a local image.
- **youtube_transcript** — retrieve timestamped captions when available.
- **watch_youtube_video** — use Gemini to answer questions about video content when configured.

The agent prompt tells it to use evidence-gathering tools and computation when the task calls for them. Tool output can still be incomplete or incorrect, so answers that require verification should be checked against reliable source material.

### 2. Multimodal and file-processing strategy

The implementation routes tasks to different tools according to file type and question intent:

| Input | Method in the code | Notes |
|---|---|---|
| CSV | pandas.read_csv() | Returns columns, shape, content and a descriptive summary. |
| Excel (.xlsx, .xls) | pandas.read_excel() | Reads all sheets and reports their shapes, columns and values. |
| PDF | pypdf.PdfReader | Extracts page text; scanned pages may need OCR, which is not implemented here. |
| Text and code files | UTF-8 text reading | Used for formats such as .txt, .json, .py and .md. |
| Images | describe_image() | Uses Gemini, OpenAI-compatible vision, or a Hugging Face vision endpoint depending on available keys. |
| Audio | transcribe_audio() | Uses Gemini when configured, otherwise the Hugging Face automatic speech-recognition endpoint. |
| YouTube spoken content | youtube_transcript() | Fetches captions and includes timestamps when captions are available. |
| YouTube visual/audio content | watch_youtube_video() | Requires GEMINI_API_KEY; intended for questions that cannot be answered from captions alone. |
| Research questions | Search, webpage and Wikipedia tools | Use multiple sources for facts when needed. |

For data questions, the prompt explicitly asks the agent to inspect the file and use Python/pandas for exact counts, sums and filters instead of mental arithmetic. For reversed or encoded-looking question text, it is instructed to decode the text before answering.

### 3. Model provider selection

The model used by the primary CodeAgent is chosen in the build_model() function:

1. **Gemini** — selected when MODEL_PROVIDER=gemini, or by default when GEMINI_API_KEY is set and no provider is specified. It is accessed through LiteLLMModel.
2. **OpenAI** — selected when MODEL_PROVIDER=openai, or by default when OPENAI_API_KEY is set and no provider is specified. It is accessed through LiteLLMModel.
3. **Hugging Face Inference** — fallback when neither of the previous providers is selected. The code requires HF_TOKEN or HUGGINGFACEHUB_API_TOKEN and uses InferenceClientModel.

MODEL_ID overrides the default model identifier for the selected provider. For Gemini, the code also contains a model-name resolver that can test a list of candidate model IDs and cache the first one that responds successfully.

Image and audio helper tools select their own model/API path based on the available credentials. In particular, direct Google Search grounding and video analysis require GEMINI_API_KEY; the OpenAI image path requires OPENAI_API_KEY; and Hugging Face inference requires a valid HF_TOKEN.

Provider availability, account permissions, model names, quotas and API behavior can change. Set an explicit, currently supported model ID if automatic selection fails.

### 4. Exact-match answer formatting

GAIA evaluation can require an answer to match a target string exactly. The implementation therefore applies multiple formatting steps:

- PROMPT requests only the bare answer, without an explanation or prefix.
- It gives format rules for numbers, strings and comma-separated lists.
- clean_answer() removes common answer prefixes, surrounding quotes/backticks and certain trailing punctuation.
- verify_exact_match() asks the selected model to extract the final answer from the question and candidate response.
- GaiaAgent.__call__() returns the verified candidate when available, falling back to the cleaned candidate if verification fails.

This is a **formatting and answer-extraction strategy**, not a proof that the answer is factually correct. The verification step can also make mistakes, so exact-match performance still needs to be measured by the evaluator.

### 5. Retries and answer cache

app.py supports a local cached_answers.json file, keyed by task_id.

- When **Use cached answers** is enabled, a non-empty cached response can be reused without running the model again.
- New answers are written to the cache after a successful, non-empty result.
- Empty answers and the literal Error result are retried once, so a task can be attempted up to twice during that run.
- The app adds short delays between requests to reduce pressure on rate-limited services.
- Uncheck the cache option to rerun tasks instead of reusing prior answers.

The cache is keyed by task ID, not by model name or configuration. If you change the prompt, provider, model ID or tools and want fresh answers, disable caching or remove the local cache file deliberately. Treat cached answers and downloaded benchmark attachments as generated local files, and do not commit private or restricted task data.

## Setup and configuration

### Run locally for development

Use Python in a virtual environment if possible:

~~~bash
python -m venv .venv
~~~

On Windows PowerShell:

~~~powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
~~~

On macOS/Linux:

~~~bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
~~~

The app uses Hugging Face OAuth for the intended Space evaluation flow. For official submission, deploy the project as a Hugging Face Space with OAuth enabled and configure the actual Space ID. A local Gradio launch may not reproduce the full hosted OAuth/attribution behavior.

### Configure a Hugging Face Space

1. Create or open a Gradio Space and upload the repository files.
2. In **Settings → Variables and secrets**, add the credentials required by your chosen provider.
3. Keep the Space public if the course evaluator needs to inspect the submitted agent_code URL.
4. Confirm that SPACE_ID identifies the actual Space; app.py uses it to construct the code URL sent to the evaluator.
5. Sign in through the Gradio Hugging Face login button, then run the evaluation and submission.

Do not hard-code credentials in source code or commit them to Git.

### Environment variables

| Variable | Purpose |
|---|---|
| GEMINI_API_KEY | Enables Gemini model access; also required by Google Search grounding and video analysis. |
| OPENAI_API_KEY | Enables the OpenAI-compatible model path and image-analysis fallback. |
| HF_TOKEN | Hugging Face authentication for model inference and, when needed, gated GAIA dataset access. |
| HUGGINGFACEHUB_API_TOKEN | Alternative token variable accepted by the primary Hugging Face model selection path. |
| MODEL_PROVIDER | Optional explicit provider: gemini or openai; otherwise credentials determine the default provider. |
| MODEL_ID | Optional model ID override for the primary agent. |
| HF_USERNAME | Optional username fallback when a Hugging Face OAuth profile is not provided. |
| SPACE_ID | Space identifier used to generate the agent-code URL sent with the submission. |

Avoid setting MODEL_PROVIDER to a provider whose API key is missing; the app deliberately raises a configuration error in that case.

## Evaluation flow

The app uses the course scoring API configured in DEFAULT_API_URL:

- GET /questions — fetch the current task list, unless a valid local questions.json is present.
- GET /files/{task_id} — try to download an attachment for a task.
- POST /submit — send the username, code URL and answer list to the evaluator.

If the course API cannot provide a task attachment, the app attempts to download it from the gated gaia-benchmark/GAIA dataset at 2023/validation/{file_name}. This fallback requires dataset access and a suitable Hugging Face token. The app reports failures in its logs if both routes fail.

The evaluation interface displays the score, correct count, total attempted count and a table of submitted answers when the service returns them. The score is calculated by the remote course evaluator, not by a local metric implementation in this repository.

## Troubleshooting

| Symptom | Checks |
|---|---|
| Agent fails to initialize | Verify that the selected provider's key/token is present and MODEL_ID is supported. |
| Model returns 401/403 | Check key validity, account access, Space secrets and dataset permissions. Never print or share the full key. |
| Model returns 404 | Check the configured model ID and whether the provider still exposes it. |
| Rate limits or intermittent network failures | Check provider quota and network availability; the app has limited retries, not unlimited recovery. |
| Attachment cannot be downloaded | Check the task API first, then confirm access to the gated GAIA dataset if the fallback is needed. |
| Audio or image analysis fails | Check the relevant provider credentials and whether the selected model supports the input type. |
| No score is returned | Check the Hugging Face login, SPACE_ID, the public code URL and the course scoring API response. |
| Answers are repeated after changing models | Disable caching or remove cached_answers.json so cached results are not reused. |

## Limitations and responsible use

- The code is optimized for a benchmark assignment, not a general-purpose production agent.
- Generated code from CodeAgent is executed as part of the agent workflow. Run the project in an environment you trust and understand the permissions and isolation of the execution environment before processing untrusted tasks.
- Web pages, Wikipedia, model outputs, transcriptions and tool responses can contain errors. Verify high-impact factual claims.
- PDF handling extracts text only; it does not implement OCR for scanned documents.
- YouTube visual analysis depends on Gemini availability; caption retrieval depends on captions being available.
- Rate limits, costs, privacy requirements, model changes and gated dataset permissions can affect reproducibility.
- Never place API keys, private data, or restricted benchmark attachments in a public repository.

## Dependencies

Runtime dependencies are listed in requirements.txt, including Gradio with OAuth support, smolagents, LiteLLM, Hugging Face Hub clients, pandas/openpyxl, pypdf, BeautifulSoup, requests and YouTube transcript support.

## Further reading

- [smolagents documentation](https://huggingface.co/docs/smolagents)
- [Hugging Face Spaces configuration reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [Hugging Face Hub documentation](https://huggingface.co/docs/huggingface_hub)
