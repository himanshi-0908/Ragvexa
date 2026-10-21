from bs4 import BeautifulSoup
from langchain_core.documents import Document

from utils.security import fetch_url_safely


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def load_web_url(url: str):
    """
    Fetch a public web page and return a list with one LangChain Document.

    The fetch goes through utils.security.fetch_url_safely, which validates
    the URL and every redirect hop against the SSRF policy and caps the
    response size.
    """
    final_url, html = fetch_url_safely(url)
    text = _html_to_text(html)
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": final_url})]
