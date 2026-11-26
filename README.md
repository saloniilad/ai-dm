# AI Dungeon Master: RAG, Tool Calling, and Fine-Tuning

This project is a multiplayer tabletop-style Dungeon Master application. It combines deterministic game rules with an LLM narrator. It remembers campaign events, retrieves relevant world lore, rolls dice, tracks characters and inventory, controls geographic travel, and prevents players from seeing secret or flag-gated information.

The system separates responsibilities:

- The LLM generates narration and decides which tools to call.
- The game engine rolls dice and updates HP, gold, inventory, flags, locations, and other state.
- Retrieval searches campaign memory and world lore.
- Fine-tuning can adapt a smaller open model to the project's response style and tool format.

## Requirements

- Python 3.11 or later
- Windows, macOS, or Linux
- An Anthropic API key for the Anthropic backend
- A local or remote OpenAI-compatible server for the OpenAI backend
- A GPU and additional dependencies for local LoRA inference or training

## Installation

Create and activate a virtual environment, then install the development dependencies.

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

macOS or Linux:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Configuration

Copy `.env.example` to `.env` for the settings you want to customize:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS or Linux:

```bash
cp .env.example .env
```

Environment variables are read automatically from the project `.env` file and from the shell or Docker command. Direct process environment variables take precedence over values in `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `DM_BACKEND` | `mock` unless an Anthropic key is present | Selects `mock`, `anthropic`, `openai`, or `local`. |
| `ANTHROPIC_API_KEY` | Empty | API key for the Anthropic backend. |
| `DM_MODEL` | `claude-sonnet-5-5` | Model name for Anthropic or an OpenAI-compatible server. |
| `DM_OPENAI_BASE_URL` | `http://localhost:8000/v1` | Base URL of an OpenAI-compatible server. |
| `DM_OPENAI_API_KEY` | `none` | API key required by some OpenAI-compatible servers. |
| `DM_ADAPTER_PATH` | `models/dm-lora` | Directory containing the local LoRA adapter. |
| `DM_BASE_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model loaded by the local backend. |
| `DM_DENSE` | `0` | Set to `1` to enable dense embedding retrieval. Requires the dense dependencies. |
| `DM_RERANK` | `0` | Set to `1` to enable cross-encoder reranking. Requires the dense dependencies. |
| `DM_DATA_DIR` | `data` | Directory used for sessions and persistent campaign state. |
| `DM_LOG_DATASET` | Empty | Optional JSONL file that receives every turn for fine-tuning. |
| `DISCORD_TOKEN` | Empty | Bot token used by the Discord integration. |

Backend selection rules:

- `DM_BACKEND=mock` runs without API credentials and is suitable for local testing.
- `DM_BACKEND=anthropic` uses Anthropic and requires `ANTHROPIC_API_KEY`.
- `DM_BACKEND=openai` connects to a server that implements the OpenAI chat-completions API.
- `DM_BACKEND=local` loads the LoRA adapter directly in the Python process.

## Run the CLI

Use the offline mock backend:

```bash
python -m app.cli --name Asha
```

Use Anthropic:

Windows PowerShell:

```powershell
$env:DM_BACKEND="anthropic"
$env:ANTHROPIC_API_KEY="your-api-key"
python -m app.cli --name Asha
```

macOS or Linux:

```bash
export DM_BACKEND=anthropic
export ANTHROPIC_API_KEY=your-api-key
python -m app.cli --name Asha
```

CLI commands include `/state`, `/recap`, and `/quit`.

## Run the Web Interface

```bash
uvicorn app.api:app --reload
```

Open `http://localhost:8000` in a browser.

To use a different backend, set the relevant environment variables before starting Uvicorn.

## Run the Fine-Tuned Local LoRA Model

Install the local inference dependencies:

```bash
pip install -r requirements-local.txt
```

Windows:

```powershell
.\run_local.bat
```

macOS or Linux:

```bash
chmod +x run_local.sh
./run_local.sh
```

You can also start the local backend from the CLI:

```bash
python -m app.cli --name Asha
```

Set `DM_BACKEND=local` in the environment before running the CLI.

The first run downloads the base model, approximately 3 GB, and caches it locally. GPU execution is substantially faster than CPU execution. To use another adapter, replace `models/dm-lora/` or set `DM_ADAPTER_PATH`.

## Run the Discord Bot

Install the Discord dependency set:

```bash
pip install -r requirements-discord.txt
```

Set `DISCORD_TOKEN`, enable the Message Content Intent in the Discord developer portal, and run:

```bash
python -m app.discord_bot
```

Bot commands:

- `!join` joins or starts a campaign in the current channel.
- `!do <action>` performs a player action.
- `!state` displays campaign state.
- `!recap` summarizes recent events.

## Run with Docker

```bash
make docker
```

The Docker image exposes port 8000 and stores persistent data in the `/data` volume.

## Project Structure

| Path | Purpose |
|---|---|
| `dm/engine.py` | Game rules, dice, inventory, HP, flags, geography, and tool schemas. |
| `dm/retrieval.py` | BM25, optional dense retrieval, optional reranking, and geographic ranking. |
| `dm/memory.py` | Lore, campaign events, secret gating, flag gating, and summary compaction. |
| `dm/llm.py` | Anthropic, OpenAI-compatible, local LoRA, and mock backends. |
| `dm/master.py` | Prompt assembly, retrieval, tool calls, narration, memory updates, and dataset logging. |
| `dm/session.py` | Multi-player session creation and persistent state. |
| `app/` | CLI, FastAPI application, web interface, and Discord bot. |
| `eval/` | Retrieval, behaviour, and ablation evaluations. |
| `train/` | Self-play data, dataset review, SFT preparation, and LoRA training. |
| `world/lore.json` | World lore. Entries may contain `coords`, `secret`, `reveal_flag`, or `requires_flag`. |

## Tests and Evaluation

Run all tests:

```bash
python -m pytest -q
```

Run the offline retrieval benchmark:

```bash
python -m eval.retrieval_eval
```

Run prompt and retrieval ablations:

```bash
python -m eval.ablation
```

Run behaviour checks:

```bash
python -m eval.behavior_eval
```

Dense retrieval and reranking require additional dependencies:

```bash
pip install -r requirements-dense.txt
set DM_DENSE=1
set DM_RERANK=1
```

Use `export` instead of `set` on macOS or Linux.

## Fine-Tuning Workflow

1. Generate teacher responses:

```bash
python -m train.self_play --sessions 50 --turns 8 --out data/dataset/teacher.jsonl
```

You can also log turns from live play:

```bash
set DM_LOG_DATASET=data/dataset/live.jsonl
python -m app.cli --name Asha
```

2. Review and edit the generated examples:

```bash
python -m train.review data/dataset/teacher.jsonl
```

3. Build training and validation data:

```bash
python -m train.build_sft data/dataset/*.jsonl --out data/sft
```

4. Install training dependencies and train the LoRA adapter:

```bash
pip install -r requirements-train.txt
python -m train.train_lora --data data/sft --out out/dm-lora
```

5. Serve the fine-tuned model with a compatible OpenAI server, then run evaluations:

```bash
python -m eval.ablation --ft-model dm --ft-base-url http://localhost:8000/v1
```

Include adversarial examples such as cheat attempts, secret probes, and impossible actions. Review the terms of each model provider before training on generated provider output.

## Verification Notes

Verified components include game-engine behavior, memory and secret gating, geographic retrieval, state persistence, the REST API, offline evaluations, and dataset export.

Real-provider responses, dense embeddings, reranking, LoRA training, Discord connectivity, and browser rendering require external credentials, dependencies, hardware, or network access and were not fully executed in this environment.
