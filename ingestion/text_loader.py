import os
from langchain_community.document_loaders import TextLoader
from .pdf_loader import load_pdf

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
        return loader.load()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")