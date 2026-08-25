"""Security helpers: SSRF guard for URL ingestion, safe filenames for uploads."""

import ipaddress
import os
import socket
import uuid
from urllib.parse import urlparse

_BLOCKED_HOSTNAMES = {"localhost", "0.0.0.0"}


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

    if hostname.lower() in _BLOCKED_HOSTNAMES:
        return False

    try:
        resolved = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for _family, _type, _proto, _canon, sockaddr in resolved:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False

    return True


def safe_filename(original_name: str) -> str:
    """
    Strip any directory components from an uploaded filename and return
    a unique, filesystem-safe name. Prevents path traversal
    (e.g. "../../etc/something") via crafted upload filenames.
    """
    base = os.path.basename(original_name or "")
    ext = os.path.splitext(base)[1][:10]  # keep extension, cap its length
    return f"{uuid.uuid4().hex}{ext}"
