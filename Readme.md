
# CLI Agent [v1]

A command-line AI agent that uses a locally running Ollama model, streams responses
and thinking output, and can call local file-system tools and a Docker playground.
The project also includes a retrieval-augmented generation (RAG) pipeline for
searching saved project memory.

> **Warning:** This agent can read, write, move, or delete files and execute shell
> commands inside its Docker playground. Review tool requests and generated commands
> carefully. Docker access is powerful; membership in the `docker` group is
> effectively root-level access to the host.

## Features

- Streams model responses and thinking output from Ollama.
- Uses Ollama tool calls to read and write files, inspect directories, and run
  shell commands or Python in a Docker container.
- Retrieves relevant memory chunks for each user request.
- Limits model context (`num_ctx`), generated tokens (`num_predict`), retrieved
  memory, tool output, and tool-call rounds in `Agent.py`.
- Uses `testing/` as the default destination for newly generated files and test
  artifacts when no destination is specified. A user-specified path takes priority.

## Requirements

- Linux (the Docker playground uses a Linux image).
- Python 3.10 or later. Python 3.11 is used in the playground image.
- A running Ollama service and a locally available model matching `MODEL_NAME` in
  `config.py`.
- Docker Engine, permission to access the Docker daemon, and the `agent-sandbox`
  image built as described below.
- Enough host RAM and disk for Python dependencies, the embedding model, and the
  Ollama model. The playground container itself is limited to 512 MB RAM and one
  CPU core.

There is currently no `requirements.txt` or `pyproject.toml`. The Python
environment needs these project dependencies:

```bash
python -m pip install ollama docker pymupdf sentence-transformers \
  faiss-cpu llama-index-core
```

`sentence-transformers` installs PyTorch as a dependency in typical environments;
the RAG retrieval code also imports `torch` directly. If dependency installation
does not provide a compatible PyTorch build for your platform, install the
appropriate PyTorch build following the official PyTorch instructions.

## Configuration

Edit `config.py`:

```python
MODEL_NAME = "gemma27b-4bit"
```

Set this to a model name that is already available in your Ollama installation.
Check available local models with:

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

The current request options are set in `Agent.py`, including `num_ctx=5120`,
`num_predict=4096`, `keep_alive="20m"`, and `num_thread=9`. The context setting
limits the combined prompt, tools, retrieved memory, and conversation messages.
`num_predict` limits generated output; raising it does **not** enlarge the context
window. These values may need adjustment for the selected model and available
memory.

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

Activate the Python environment that has the dependencies installed, change to
the repository root, and run:

```bash
python Agent.py
```

Type a request at the prompt. Enter `/exit` to leave the loop and stop the
playground container.

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

The source also defines other OS helpers, but a helper is only available to the
model if it appears in the `tools` list and has a matching entry in
`available_functions`. Check both places before adding a tool.

## Playground behavior and limits

- Shell commands and Python code run in a Docker container, not directly on the
  host.
- Commands run with `/playground` as their working directory.
- The container is started with Docker bridge networking. Outbound network
  access depends on Docker and host network configuration; it is not guaranteed.
- The container has a 512 MB memory limit and one CPU core.
- The container does not mount the repository's `testing/` directory. Files
  written inside `/playground` are not automatically saved to the host or
  repository. Copy or recreate desired deliverables in the host workspace using
  the file tools.
- Use the playground for testing generated code and commands when practical.
  Verify the returned exit code and output before reporting success.

## Persistent memory and RAG

The default memory paths are defined in `config.py`:

- `savedMem/memory.md` is the source document.
- `savedMem/` is created by the agent if it does not exist.

At startup, `Agent.py` creates a `RagPipelineConnector` over `memory.md`. The
pipeline chunks and embeds the document, builds an in-memory FAISS index, and
retrieves up to three relevant chunks for a user request. Each retrieved chunk
is limited to 900 characters before it is added to the model's system message.
Tool results included in subsequent model messages are limited to 1,600
characters, and the agent stops after six tool rounds(can be configure inside the config.py).

When the model calls `write` for `savedMem/memory.md`, the wrapper refreshes the
RAG index after the write. Other file writes do not refresh it. The RAG index is
in memory and is rebuilt from the complete memory file at startup and after a
memory-file update.

Memory retrieval does not automatically summarize or update memory after every
chat. The system prompt instructs the model to update durable notes, but this is
an instruction to the model, not an application-enforced automatic save.

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
│   └── agent_playground.py          # Docker playground integration
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
requested tool arguments concise. Increasing `num_predict` does not increase
`num_ctx`; increasing context may require additional model/system memory.

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