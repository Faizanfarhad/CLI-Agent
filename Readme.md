
# CLI Agent 

A local Ollama CLI agent with streaming responses, thinking output, tool calling,
Docker sandboxed execution, session-based chat history, SQLite long-term memory,
and RAG-based retrieval over Markdown notes.

> **Warning:** This agent can read, write, move, or delete files and execute shell
> commands inside its Docker playground. Review tool requests and generated commands
> carefully. Docker access is powerful; membership in the `docker` group is
> effectively root-level access to the host.

## Features

- Streams Ollama responses and thinking output.
- Dispatches structured Ollama tool calls to file, OS, Docker, memory, and text-to-speech tools.
- Runs one reusable Docker playground container for each agent process.
- Creates a SQLite session for the program run and lets users list or resume previous sessions.
- Stores only the final user/assistant answer for each completed chat turn; tool calls and model thinking are not added to session history.
- Searches SQLite and a FAISS RAG index before adding relevant context to a request.
- Automatically reviews completed answers for durable facts and saves them to SQLite with validation and duplicate checks.
- Uses `testing/` as the default destination for new files and test artifacts.

## Requirements
- Linux.
- Python 3.10 or later.
- Ollama installed and running, with a local model matching `MODEL_NAME` in
  `config.py`.
- Docker Engine running, with permission for your user to access the Docker
  daemon.
- Python dependencies listed in [`requirements.txt`](requirements.txt). Install
  them in a virtual environment using the commands in [Run the agent](#run-the-agent).
- The `agent-sandbox` Docker image. See [Build the Docker playground image](#build-the-docker-playground-image).

## Configuration

Edit `config.py`:

```python
models_availabel = ["gpt20b", "gemma27b-4bit", "qwen3.8-27b"]
MODEL_NAME = models_availabel[0]
```

Add names of models already available in your Ollama installation to
`models_availabel`. Switch models while the agent is running with `/models <id>`
or change `MODEL_NAME` before startup. Check available local models with:

```bash
ollama list
```

Start and test Ollama independently before launching the agent:

```bash
ollama serve
ollama run gemma27b-4bit
```

If Ollama is already managed by systemd, it may already be running. Do not start a
second server on the same address.

The current chat request options are set in `Agent.py`, including `num_ctx=5120`,
`keep_alive="20m"`, and `num_thread=9`. The context setting limits the combined
prompt, tools, retrieved memory, and conversation messages. The maximum number
of tool rounds and memory retrieval limits are configured in `config.py`.

## Build the Docker playground image

Run this from the repository root:

```bash
sudo docker build -f DOCKERFILE -t agent-sandbox .
```

The Dockerfile creates an image based on `python:3.11-slim`, installs common
command-line/build utilities, sets `/playground` as the working directory, and
keeps the container alive with `sleep infinity`.

The agent starts a container when `Agent.py` launches and stops it when the
interactive loop exits with `/exit`. The playground tool is intended to reuse
that running container for commands during the session.

### Docker permissions

The agent's Python Docker client connects to the Docker daemon socket (commonly
`/var/run/docker.sock`). Making the project files executable with `chmod +x` does
not grant access to this socket. If access is denied, check:

```bash
id
ls -l /var/run/docker.sock
```

On a personal Linux machine, an administrator can add the user to the socket's
Docker group, then the user must log out and back in(if logout still didnt worked then reboot the device):

```bash
sudo usermod -aG docker "$USER"
```

Docker group membership grants highly privileged control of the host. Only do
this on a machine where that level of access is appropriate. Do not make the
Docker socket world-writable.

## Run the agent

Use Python 3.10 or later on Linux. From the repository root, create and activate
a virtual environment, then install the packages listed in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`sentence-transformers` uses PyTorch; `torch` is also listed explicitly because
the retrieval code imports it directly. If pip cannot find a compatible PyTorch
wheel for your platform, install the appropriate build following the official
PyTorch installation instructions.

After installing dependencies, make sure Ollama is running and build the Docker
playground image as described above. Run the agent from the repository root so
imports such as `from config import ...` resolve correctly:

```bash
python Agent.py
```

Type a request at the prompt. Use `/help` to print the CLI commands; use `/exit`
to close the active session and stop the playground container.

### CLI commands

| Command | Purpose |
| --- | --- |
| `/help` | Show available commands. |
| `/models` | List model names configured in `config.py`. |
| `/models <id>` | Switch to a listed model ID. |
| `/current_model` | Show the active model name. |
| `/mic` | Record one utterance from the microphone and transcribe it. |
| `/sessions` | List previous sessions that contain saved chats. |
| `/sessions <id>` | Resume a listed session and close the active session. |
| `/delete_session <id>` | Permanently delete a session and its chat history after confirmation. |
| `/new` | Close the active session and start a new one. |
| `/exit` | Close the session and stop the playground container. |

## Available tools

The model is given these tools by `Agent.py`:

| Tool | Purpose |
| --- | --- |
| `read_file(path)` | Read a UTF-8 text file. |
| `read_pdf(path)` | Extract text from a PDF. |
| `write(path, content)` | Write UTF-8 text and create missing parent directories. |
| `ls(command)` | Run `ls` with the supplied argument list. |
| `pwd(command)` | Run `pwd` with the supplied argument list. |
| `cd(command)` | Change the agent process's working directory. |
| `find(command)` | Run `find` with the supplied argument list. |
| `get_full_path(suffix_path)` | Construct a path from the current directory and a suffix. |
| `playground(...)` | Run shell commands or Python inside the session's Docker container. |
| `persist_long_term_memory(content, category, importance)` | Save one validated durable SQLite memory entry for the active session. |
| `text_to_voice(text, own_voice=None)` | Generate speech and play it through the host audio player. |

The source also defines other OS helpers, but a helper is only available to the
model if it appears in the `tools` list and has a matching entry in
`available_functions`. Check both places before adding a tool.

## Playground behavior and limits

- Shell commands and Python code run in a Docker container, not directly on the
  host.
- The repository is mounted read-only at `/workspace`.
- `testing/` is mounted read-write at `/workspace/testing`.
- Shell and Python execution currently use `/playground` as the container working
  directory, while the repository mount remains available under `/workspace`.
- Commands run with `/playground` as their working directory.
- The container is started with Docker bridge networking. Outbound network
  access depends on Docker and host network configuration; it is not guaranteed.
- The container has a 512 MB memory limit and one CPU core.
- Use the playground for testing generated code and commands when practical.
  Verify the returned exit code and output before reporting success.

Files written through `/workspace/testing` can be saved to the host testing
directory. Files created elsewhere in the container need to be copied back using
the tool loop or recreated on the host.

## Text-to-speech

The optional `text_to_voice` tool uses `pocket_tts`, `scipy`, and `aplay` to
generate `testing/output.wav` and play it.

- Set `voice_path` in `config.py` to a valid local audio prompt, or leave it as
  `None`/a missing path to use Pocket TTS's default voice.
- Pass `own_voice` only as a valid audio-file path or omit it.
- Do not pass a human-language code or a built-in voice label such as `eve` as
  `own_voice`; that argument is a file-path argument, not a voice-name lookup.
- Run the agent on a Linux desktop with an available ALSA audio device because
  `aplay` is used to play the output.

### Microphone input

Use `/mic` to record a single utterance. Silero VAD detects speech and
Faster-Whisper's `small` model transcribes it on CPU. The model may be downloaded
the first time it is initialized. Microphone capture requires a working audio
input device and the PortAudio system library used by `sounddevice`; package
names vary by Linux distribution.

`voice_chat.py` also loads an `AutoProcessor` from `ASR_MODEL_DIR`. If startup
fails while loading it, verify that this setting in `config.py` points to a
compatible local model directory.

## Persistent memory and RAG

The default memory paths are defined in `config.py`:

- `savedMem/memory.md` is the source document.
- `savedMem/agent_memory.db` stores durable SQLite memories.
- `savedMem/` is created by the agent if it does not exist.

Each program run starts a new conversation session. The CLI accepts:

- `/sessions` to list sessions that contain saved chats.
- `/sessions <id>` to resume a listed session.
- `/new` to start another session.
- `/exit` to close the active session and exit.

The SQLite tables have separate roles: `sessions` identifies conversations,
`messages` stores completed user/assistant turns for each session, and
`memories` stores durable facts that can optionally point back to their source
session. Deleting a session cascades to its messages; linked long-term memories
are preserved and detached from the deleted session.

Completed turns save only the user's request and the assistant's final answer
to that session; intermediate tool exchanges and model thinking are not stored.
When a session is resumed, the most recent 12 user/assistant messages are added
to the next request as conversational context. Sessions and long-term memories
are separate; durable memory entries can reference the session they came from.

At startup, `Agent.py` creates a `RagPipelineConnector` over `memory.md`. The
pipeline chunks and embeds the document, builds an in-memory FAISS index, and
retrieves up to three relevant chunks for a user request. Each retrieved chunk
is limited to 900 characters before it is added to the model's system message.
Tool results included in subsequent model messages are limited to 1,600
characters, and the maximum tool rounds are configured in `config.py`.

Before each user request, the agent also searches the SQLite database for up to
three matching active, unexpired long-term memories and includes them in the
system message, limited to 900 characters per entry. The `persist_long_term_memory`
tool writes durable entries to this database; the agent does not load the entire
database into the prompt.

When the model calls `write` for `savedMem/memory.md`, the wrapper refreshes the
RAG index after the write. Other file writes do not refresh it. The RAG index is
in memory and is rebuilt from the complete memory file at startup and after a
memory-file update.

After a completed response, the application makes a separate structured memory
review call. It saves only concise durable preferences, decisions, project
facts, or tool-confirmed outcomes to SQLite; routine requests are not promoted
to long-term memory and full tool transcripts are not stored. Invalid extraction
or database errors are reported instead of being presented as a successful
save. The model can also use
`persist_long_term_memory` during a task when a durable fact must be saved
before the final response.

## Project layout

```text
Agent_using_MCP/
├── Agent.py                         # Interactive Ollama agent and tool loop
├── config.py                        # Model, paths, and system prompt
├── DOCKERFILE                       # Image used by the playground
├── Readme.md                        # This guide
├── tools/
│   ├── file_tools.py                # Read, PDF extraction, and write tools
│   ├── os_tools.py                  # Local operating-system helpers
│   ├── agent_playground.py          # Docker playground integration
│   ├── save_memory.py               # SQLite sessions and long-term memory
│   └── voice_chat.py                # Pocket TTS speech generation
├── relevantMemoryFetch/
│   ├── rag_pipeline.py              # Memory chunking, embedding, retrieval
│   └── rag_tools/                   # Chunking, embeddings, and FAISS helpers
├── savedMem/
│   └── memory.md                    # Persistent memory source
└── testing/                         # Default location for new test artifacts
```

## Troubleshooting

### `PermissionError(13, 'Permission denied')` connecting to Docker

The Python Docker client cannot access the Docker socket. Check group membership
and socket ownership with the commands in [Docker permissions](#docker-permissions).
Changing permissions on the project directory does not fix socket permissions.

### `Container is not running`

The playground execution tool expects the environment created at agent startup
to have a live container attached. Confirm that `playground_env.start()` ran
successfully and that the tool dispatcher passes that same environment to the
playground helper.

### Ollama tool-call JSON is incomplete

An error such as `unexpected end of JSON input` means the server received
incomplete tool arguments. Check the Ollama logs for truncation indicators such
as `truncated = 1`. Keep prompts, retrieved memory, tool schemas, tool output, and
requested tool arguments concise. Increasing output limits does not increase
`num_ctx`; increasing context may require additional model/system memory.

### Text-to-speech cannot open the voice file

If Pocket TTS reports `LibsndfileError` or `Error opening ...`, verify that
`own_voice` is a real audio file path and that the file exists. Human-readable
voice labels are not valid file paths for the `own_voice` argument. If no custom
voice is needed, omit `own_voice` or pass `None`. Set `voice_path` in `config.py`
only to an existing audio prompt file.

### CUDA illegal memory access

This is an error from Ollama's model runner during GPU inference. A normal
`ollama run` succeeding does not guarantee every tool-enabled or longer-context
request will succeed. Compare the failing request with a plain request and
inspect Ollama service logs for details.

## Development notes

- Run a syntax check after Python changes:

  ```bash
  python -m py_compile Agent.py config.py tools/*.py relevantMemoryFetch/*.py relevantMemoryFetch/rag_tools/*.py
  ```

- Do not claim a playground or model test passed unless its returned output
  confirms that it did.
- Keep secrets and credentials out of prompts, memory files, logs, and tool
  output sent to external services.

### The Local Model that i used 
- [GPT-oss-20b Uncensored](https://huggingface.co/DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf)
- [Gemma3-27b-4bit-Uncensored](https://huggingface.co/llmfan46/gemma-3-27b-it-ultra-uncensored-heretic-GGUF)