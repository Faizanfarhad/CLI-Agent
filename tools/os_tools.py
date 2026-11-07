import subprocess 
import os
from pathlib import Path
from typing import Optional

def grep(command:list):
    """ produce the command in list dont produce the keyword of command ex: "grep" because it is already inbuild ex:- for list it should be ["--help"]  """
    full_command = []
    full_command.append("grep")
    if isinstance(command,list):
        for cm in command:
            full_command.append(cm)
    return subprocess.run(full_command)

def find(command:list):
    """ produce the command in list dont produce the keyword of command ex:- for finding the memory.md  just produce ["/mnt/storage","-name","memory.md"]"""
    
    full_command = []
    full_command.append("find")
    if isinstance(command,list):
        for cm in command:
            full_command.append(cm)
        
    subprocess.run(full_command)
    
    
def ls(command: list[str] | None = None):
    """  produce the command in list dont produce the keyword of command if only need to execute ls command then just called it otherwise produce the arguments like ["-la"] """
    full_command = []
    full_command.append("ls")
    if isinstance(command,list):
        for cm in command:
            full_command.append(cm)
    result = subprocess.run(full_command, capture_output=True, text=True)
    return result.stdout or result.stderr
    

def pwd(command: list[str] | None = None):
    """ produce the command in list dont produce the keyword of command if only need to execute pwd command then just called it otherwise produce the arguments like ["-L"] """
    full_command = []
    full_command.append("pwd")
    if isinstance(command,list):
        for cm in command:
            full_command.append(cm)
    result = subprocess.run(full_command, capture_output=True, text=True)
    return result.stdout or result.stderr
    

def rm(command:Optional[list | None]):
    """ produce the command in list dont produce the keyword of command produce the arguments like ["-r"] or ["-rf"] """
    full_command = []
    full_command.append("rm")
    if isinstance(command,list):
        for cm in command:
            full_command.append(cm)
    subprocess.run(full_command)

def mv(command:Optional[list | None]):
    """ produce the command in list dont produce the keyword of command produce the arguments like ["from_here"] to ["there"] """
    full_command = []
    full_command.append("mv")
    if isinstance(command,list):
        for cm in command:
            full_command.append(cm)
    subprocess.run(full_command)

def stat(command:Optional[list | None]):
    """ produce the command in list dont produce the keyword of command produce the arguments  """
    full_command = []
    full_command.append("stat")
    if isinstance(command,list):
        for cm in command:
            full_command.append(cm)
    subprocess.run(full_command)


def mkdir(command:Optional[list | None]):
    """ produce the command in list dont produce the keyword of command produce the arguments  """
    full_command = []
    full_command.append("mkdir")
    if isinstance(command,list):
        for cm in command:
            full_command.append(cm)
    subprocess.run(full_command)

def cat(command:Optional[list | None]):
    """ produce the command in list dont produce the keyword of command produce the arguments  """
    full_command = []
    full_command.append("cat")
    if isinstance(command,list):
        for cm in command:
            full_command.append(cm)
    subprocess.run(full_command)

def touch(command:Optional[list | None]):
    """ produce the command in list dont produce the keyword of command produce the arguments  """
    full_command = []
    full_command.append("touch")
    if isinstance(command,list):
        for cm in command:
            full_command.append(cm)
    subprocess.run(full_command)


def cd(command: list[str] | None = None):
    """ produce the command in list dont produce the keyword of command if only need to execute cd command then just called it otherwise produce the arguments like [".."] """
    if command:
        os.chdir(command[0])
    return str(Path.cwd())

def get_full_path(suffix_path):
    """ use this func when you need to find full path of the file by providing the suffix path """
    path = Path(suffix_path)
    posfix_path = path.cwd()
    full_path = str(posfix_path)+"/"+suffix_path if suffix_path[0] != "/" else  str(posfix_path) + suffix_path
    return full_path
