"""Security helpers: SSRF guard for URL ingestion, safe filenames for uploads,
private upload storage, and upload validation."""

import ipaddress
import os
import socket
import uuid
from urllib.parse import urlparse

import requests

from config import (
    ALLOWED_UPLOAD_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    MAX_URL_FETCH_BYTES,
    MAX_URL_REDIRECTS,
    UPLOAD_DIR,
    URL_FETCH_TIMEOUT_SECONDS,
)

_BLOCKED_HOSTNAMES = {"localhost", "0.0.0.0"}


class UploadValidationError(ValueError):
    """Raised when an upload fails validation (type, size, content)."""


def _ip_is_public(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def is_safe_url(url: str) -> bool:
    """
    Block requests to internal/private/link-local/loopback addresses.
    Prevents SSRF via the URL-ingestion feature (server-side fetch of
    user-supplied URLs, e.g. cloud metadata endpoints or internal services).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    if hostname.lower() in _BLOCKED_HOSTNAMES or hostname.lower().endswith(".localhost"):
        return False

    # Literal IP: judge it directly (no DNS involved).
    try:
        ipaddress.ip_address(hostname)
        return _ip_is_public(hostname)
    except ValueError:
        pass

    try:
        resolved = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    if not resolved:
        return False

    for _family, _type, _proto, _canon, sockaddr in resolved:
        if not _ip_is_public(sockaddr[0]):
            return False

    return True


def fetch_url_safely(url: str) -> tuple[str, str]:
    """
    Fetch a user-supplied URL with SSRF protection on EVERY hop.

    Redirects are followed manually so each Location is re-validated with
    is_safe_url (a public URL that 302s to http://169.254.169.254/ is blocked).
    The response body is capped at MAX_URL_FETCH_BYTES and only text/HTML
    content types are accepted.

    Returns (final_url, body_text). Raises ValueError on any policy violation.
    """
    current = url
    for _hop in range(MAX_URL_REDIRECTS + 1):
        if not is_safe_url(current):
            raise ValueError(
                "This URL can't be fetched (internal, private, or link-local addresses are blocked)."
            )
        resp = requests.get(
            current,
            allow_redirects=False,
            timeout=URL_FETCH_TIMEOUT_SECONDS,
            stream=True,
            headers={"User-Agent": "Ragvexa/1.0 (+document-ingestion)"},
        )
        try:
            if resp.is_redirect or resp.is_permanent_redirect:
                location = resp.headers.get("Location")
                if not location:
                    raise ValueError("Redirect without a Location header.")
                current = requests.compat.urljoin(current, location)
                continue

            if resp.status_code >= 400:
                raise ValueError(f"URL returned HTTP {resp.status_code}.")

            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype and not (ctype.startswith("text/") or ctype in {"application/xhtml+xml", "application/xml"}):
                raise ValueError(f"Unsupported content type '{ctype}'. Only HTML/text pages can be ingested.")

            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_URL_FETCH_BYTES:
                raise ValueError("Page is larger than the allowed fetch size.")

            body = bytearray()
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                body.extend(chunk)
                if len(body) > MAX_URL_FETCH_BYTES:
                    raise ValueError("Page is larger than the allowed fetch size.")
            encoding = resp.encoding or "utf-8"
            return current, body.decode(encoding, errors="replace")
        finally:
            resp.close()

    raise ValueError("Too many redirects.")


def safe_filename(original_name: str) -> str:
    """
    Strip any directory components from an uploaded filename and return
    a unique, filesystem-safe name. Prevents path traversal
    (e.g. "../../etc/something") via crafted upload filenames.
    """
    base = os.path.basename(original_name or "").replace("\\", "_")
    ext = os.path.splitext(base)[1][:10].lower()
    ext = "".join(ch for ch in ext if ch.isalnum() or ch == ".")
    return f"{uuid.uuid4().hex}{ext}"


def validate_upload(filename: str, size_bytes: int, head: bytes = b"") -> str:
    """
    Validate an upload before it is parsed. Returns the normalised extension.

    Checks: allowed extension, size cap, and a cheap magic-byte check so a
    renamed binary is not handed to the PDF/text parser.
    """
    ext = os.path.splitext(os.path.basename(filename or ""))[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}"
        )
    if size_bytes <= 0:
        raise UploadValidationError("File is empty.")
    if size_bytes > MAX_UPLOAD_SIZE_BYTES:
        raise UploadValidationError(f"File exceeds the {MAX_UPLOAD_SIZE_MB}MB upload limit")
    if head:
        if ext == ".pdf" and not head.lstrip().startswith(b"%PDF"):
            raise UploadValidationError("File does not look like a valid PDF.")
        if ext == ".txt" and b"\x00" in head:
            raise UploadValidationError("File does not look like a text file.")
    return ext


def private_upload_path(user_id: int, original_name: str) -> str:
    """
    Return a path inside a per-user, owner-only directory for a temporary
    upload. The directory is created with mode 0o700; the caller must delete
    the file after parsing.
    """
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("A valid user_id is required for private storage.")
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, mode=0o700, exist_ok=True)
    try:
        os.chmod(UPLOAD_DIR, 0o700)
        os.chmod(user_dir, 0o700)
    except OSError:
        pass
    return os.path.join(user_dir, safe_filename(original_name))


def write_private_file(path: str, data: bytes) -> None:
    """Write bytes to `path` with owner-only permissions."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def remove_quietly(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
