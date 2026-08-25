from langchain_community.document_loaders import WebBaseLoader
from utils.security import is_safe_url

def load_web_url(url: str):
    if not is_safe_url(url):
        raise ValueError(
            "This URL can't be fetched (internal, private, or link-local addresses are blocked)."
        )
    loader = WebBaseLoader(url)
    return loader.load()