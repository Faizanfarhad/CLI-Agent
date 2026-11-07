from pathlib import Path

import pymupdf



def read_file(path: str) -> str:
    """ provide the path for the file that you need to read """
    return Path(path).read_text(encoding="utf-8")


def read_pdf(path: str) -> str:
    """ provide the path of the pdf to read it """
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def write(path,content: str) -> str:
    """ provide the path and the content that should be written in the target file  """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Saved content to {path}"
    