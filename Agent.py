 #hf_repo : https://huggingface.co/DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf

import json
from pathlib import Path

import ollama
from config import *
from tools.file_tools import (read_file,
                              read_pdf,
                              write)
from tools.os_tools import (
    ls,
    pwd,
    cd ,
    get_full_path,
    grep,
    find,
    mkdir,
    touch,
    cat,
    rm,
    mv,
    stat,
    )

from relevantMemoryFetch.rag_pipeline import RagPipelineConnector

from tools.agent_playground import AgentLinuxEnv,playground

#load the docker env for the agent playground
playground_env = AgentLinuxEnv()
playground_env.start()

# Rag pipeline for agent to load memory efficiently if needed
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE.touch(exist_ok=True)
pipeline = RagPipelineConnector(doc_path=MEMORY_FILE,device="cpu",chunk_overlap=30)



def command_tool(name, description):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command arguments, without the command name.",
                    }
                },
                "required": [],
            },
        },
    }


def playground_tool():
    return {
        "type": "function",
        "function": {
            "name": "playground",
            "description": (
                "Run shell commands or Python code in the isolated Docker playground. "
                "Returns exit codes and output. Network access may depend on Docker configuration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "isshell_cmd": {"type": "boolean"},
                    "shell_command": {
                        "type": "string",
                        "description": "Shell command to run when isshell_cmd is true.",
                    },
                    "ispython_cmd": {"type": "boolean"},
                    "python_command": {
                        "type": "string",
                        "description": "Python code to run when ispython_cmd is true.",
                    },
                },
                "required": ["isshell_cmd", "ispython_cmd"],
            },
        },
    }


def write_and_refresh_memory(path: str, content: str) -> str:
    result = write(path, content)
    if Path(path).resolve() == MEMORY_FILE.resolve():
        pipeline.refresh_index()
    return result


def run_in_playground(
    isshell_cmd: bool = False,
    shell_command: str | None = None,
    ispython_cmd: bool = False,
    python_command: str | None = None,
):
    return playground(
        isshell_cmd=isshell_cmd,
        shell_command=shell_command,
        ispython_cmd=ispython_cmd,
        python_command=python_command,
        env=playground_env,
    )


tools = [
    read_file,
    read_pdf,
    write,
    command_tool("ls", "List files and directories."),
    command_tool("pwd", "Show the current working directory."),
    command_tool("cd", "Change the current working directory."),
    command_tool("find", "find the directory."),
    playground_tool(),
    get_full_path,
]
available_functions = {
    #file tools
    "read_file": read_file,
    "read_pdf": read_pdf,
    "write": write_and_refresh_memory,

    #os tools
    "grep":grep,
    "find":find,
    "mkdir":mkdir,
    "touch":touch,
    "cat":cat,
    "rm":rm,
    "mv":mv,
    "stat":stat,
    "ls": ls,
    "pwd": pwd,
    "cd": cd,
    "get_full_path": get_full_path,

    #playground
    "playground":run_in_playground,

}

def tool_call(query):
    retrieved = pipeline.query(query).get("Top_relevant_memory_chunks", [])
    relevant_memory = [
        {
            "id": chunk["id"],
            "text": chunk["text"][:MAX_MEMORY_CHARS_PER_CHUNK],
        }
        for chunk in retrieved[:MAX_MEMORY_CHUNKS]
    ]
    messages = [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                "Relevant saved memory (may be empty):\n"
                f"{json.dumps(relevant_memory, ensure_ascii=False)}"
            ),
        },
        {"role": "user", "content": query},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            think=True,
            keep_alive="20m",
            stream=True,
            options={
                "num_ctx": 5120,     # Locks the total memory allocation safely inside your 6GB  VRAM (defined to take ~2GB)
                "num_predict": 4096,  # Bound output generation; this does not increase num_ctx.
                "num_thread": 9,
            },

        )

        assistant_message = None
        thinking_started = False
        answer_started = False

        for chunk in stream:
            if chunk.message.thinking:
                if not thinking_started:
                    print(f"\n{"=" * 40}THINKING{"=" * 40}\n")
                    thinking_started = True
                print(chunk.message.thinking, flush=True, end="")

            if chunk.message.content:
                if not answer_started:
                    print(f"\n{"=" * 40}Final Answer{"=" * 40}\n")
                    answer_started = True
                print(chunk.message.content, flush=True, end="")

            if chunk.message.tool_calls:
                assistant_message = chunk.message

        if assistant_message is None:
            return

        print(f"\n{"=" * 40}TOOL CALLING{"=" * 40}\n")
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            function = available_functions.get(tool_call.function.name)
            if function is None:
                raise ValueError(f"Unknown tool: {tool_call.function.name}")
            print(f"\ntool '{tool_call.function.name}' called\n")

            result = function(**tool_call.function.arguments)
            result_text = str(result)
            if len(result_text) > MAX_TOOL_RESULT_CHARS:
                result_text = (
                    result_text[:MAX_TOOL_RESULT_CHARS]
                    + "\n[Tool output truncated to limit context size.]"
                )
            messages.append({
                "role": "tool",
                "content": result_text,
            })
            print(f"\n'tool call result : {result_text}' \n")

    print(
        f"\nStopped after {MAX_TOOL_ROUNDS} tool rounds. "
        "Status: INCOMPLETE (no verified final result).\n",
        flush=True,
    )

def chat(query):
    tool_call(query)


# making it work as a chatbot 
want_to_talk = True
print(f"{'=' * 50}Ask Something{'=' * 50}")
while want_to_talk:
    query = input("\nEnter you query(type '/exit' to quit):")
    if query.strip().lower() == "/exit":
        want_to_talk = False
        print("\nUnderstood. Closing session. Have a great day!")
        playground_env.stop()
    else:
        chat(query)