import docker
from typing import Optional


class AgentLinuxEnv:
    def __init__(self, image_name="agent-sandbox"):
        self.client = docker.from_env()
        self.image_name = image_name
        self.container = None

    def start(self):
        """Starts the isolated Linux container."""
        self.container = self.client.containers.run(
            self.image_name,
            detach=True,
            tty=True,
            network_mode="bridge", # Limits network if needed
            mem_limit="512m",      # Prevents agent from freezing host RAM
            nano_cpus=1000000000,   # Limits agent to 1 CPU core
            volumes={
    "/mnt/storage/Agent_using_MCP": {
        "bind": "/workspace",
        "mode": "ro",
    },
    "/mnt/storage/Agent_using_MCP/testing": {
        "bind": "/workspace/testing",
        "mode": "rw",
    },
}
        )
        return self.container.id

    def execute_shell(self, command: str):
        """Executes any Linux shell command inside the container."""
        if not self.container:
            raise RuntimeError("Container is not running.")
        
        # Runs command inside the container as a bash shell string
        exit_code, output = self.container.exec_run(
            cmd=["/bin/bash", "-c", command],
            workdir="/playground"
        )
        return {
            "exit_code": exit_code,
            "output": output.decode("utf-8", errors="ignore")
        }

    def execute_python(self, python_code: str):
        """Executes inline Python code inside the container."""
        if not self.container:
            raise RuntimeError("Container is not running.")

        exit_code, output = self.container.exec_run(
            cmd=["python3", "-c", python_code],
            workdir="/playground",
        )
        return {
            "exit_code": exit_code,
            "output": output.decode("utf-8", errors="ignore"),
        }

    def stop(self):
        """Cleans up and removes the container."""
        if self.container:
            self.container.stop()
            self.container.remove()

# --- Example Usage for your Agent Loop ---
def playground(
    isshell_cmd: bool = False,
    shell_command: Optional[str] = None,
    ispython_cmd: bool = False,
    python_command: Optional[str] = None,
    env: Optional[AgentLinuxEnv] = None,
):
    if not isshell_cmd and not ispython_cmd:
        raise ValueError("Select either a shell command or Python code to execute.")

    if env is None:
        raise ValueError("Pass the running playground environment to playground().")

    results = {}
    if isshell_cmd:
        results["shell"] = env.execute_shell(shell_command or "")

    if ispython_cmd:
        results["python"] = env.execute_python(python_command or "")

    return results
