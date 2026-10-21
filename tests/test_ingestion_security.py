"""
test_ingestion_security.py — SSRF guard (including redirects), upload
validation, safe filenames and private storage.
"""

import os
import stat
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.security import (
    UploadValidationError,
    fetch_url_safely,
    is_safe_url,
    private_upload_path,
    safe_filename,
    validate_upload,
    write_private_file,
)


class TestIsSafeUrl:
    @pytest.mark.parametrize("url", [
        "http://localhost/x", "http://127.0.0.1/x", "http://0.0.0.0/", "http://[::1]/",
        "http://10.0.0.5/", "http://192.168.1.1/", "http://172.16.0.1/", "http://169.254.169.254/latest",
        "ftp://example.com/", "file:///etc/passwd", "not a url", "http://foo.localhost/",
    ])
    def test_blocked(self, url):
        assert is_safe_url(url) is False

    def test_public_allowed(self):
        with patch("utils.security.socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 0))]):
            assert is_safe_url("https://example.com/page") is True

    def test_dns_to_private_blocked(self):
        with patch("utils.security.socket.getaddrinfo", return_value=[(None, None, None, None, ("10.1.2.3", 0))]):
            assert is_safe_url("https://evil.example.com/") is False

    def test_unresolvable_blocked(self):
        import socket
        with patch("utils.security.socket.getaddrinfo", side_effect=socket.gaierror):
            assert is_safe_url("https://nope.invalid/") is False


def _resp(status=200, headers=None, body=b"<html><body>hi</body></html>", redirect=False):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {"Content-Type": "text/html"}
    r.is_redirect = redirect
    r.is_permanent_redirect = False
    r.encoding = "utf-8"
    r.iter_content = lambda chunk_size: iter([body])
    return r


class TestFetchUrlSafely:
    def _public(self):
        return patch("utils.security.socket.getaddrinfo",
                     side_effect=lambda host, *_: [(None, None, None, None,
                                                   ("10.0.0.1" if "internal" in host else "93.184.216.34", 0))])

    def test_redirect_to_private_blocked(self):
        with self._public(), patch("utils.security.requests.get") as get:
            get.return_value = _resp(302, {"Location": "http://internal.corp/secret"}, redirect=True)
            with pytest.raises(ValueError, match="blocked"):
                fetch_url_safely("https://public.example.com/")

    def test_redirect_to_loopback_ip_blocked(self):
        with self._public(), patch("utils.security.requests.get") as get:
            get.return_value = _resp(302, {"Location": "http://127.0.0.1:9000/"}, redirect=True)
            with pytest.raises(ValueError, match="blocked"):
                fetch_url_safely("https://public.example.com/")

    def test_public_redirect_followed_once(self):
        with self._public(), patch("utils.security.requests.get") as get:
            get.side_effect = [
                _resp(301, {"Location": "https://public.example.com/final"}, redirect=True),
                _resp(200),
            ]
            final, body = fetch_url_safely("https://public.example.com/")
            assert final == "https://public.example.com/final" and "hi" in body
            assert all(c.kwargs["allow_redirects"] is False for c in get.call_args_list)

    def test_too_many_redirects(self):
        with self._public(), patch("utils.security.requests.get") as get:
            get.return_value = _resp(302, {"Location": "https://public.example.com/loop"}, redirect=True)
            with pytest.raises(ValueError, match="redirects"):
                fetch_url_safely("https://public.example.com/")

    def test_binary_content_rejected(self):
        with self._public(), patch("utils.security.requests.get") as get:
            get.return_value = _resp(200, {"Content-Type": "application/zip"})
            with pytest.raises(ValueError, match="content type"):
                fetch_url_safely("https://public.example.com/a.zip")

    def test_size_cap_enforced(self):
        from config import MAX_URL_FETCH_BYTES
        with self._public(), patch("utils.security.requests.get") as get:
            get.return_value = _resp(200, body=b"x" * (MAX_URL_FETCH_BYTES + 10))
            with pytest.raises(ValueError, match="larger"):
                fetch_url_safely("https://public.example.com/")

    def test_http_error_surfaces(self):
        with self._public(), patch("utils.security.requests.get") as get:
            get.return_value = _resp(404)
            with pytest.raises(ValueError, match="404"):
                fetch_url_safely("https://public.example.com/missing")

    def test_web_loader_strips_scripts(self):
        from ingestion.web_loader import load_web_url
        html = b"<html><head><script>evil()</script><style>x{}</style></head><body><p>Real content</p></body></html>"
        with self._public(), patch("utils.security.requests.get") as get:
            get.return_value = _resp(200, body=html)
            docs = load_web_url("https://public.example.com/")
        assert docs[0].page_content == "Real content"
        assert "evil" not in docs[0].page_content


class TestUploadValidation:
    def test_extension_whitelist(self):
        with pytest.raises(UploadValidationError):
            validate_upload("a.exe", 10, b"MZ")
        with pytest.raises(UploadValidationError):
            validate_upload("noext", 10, b"abc")

    def test_size_limits(self):
        from config import MAX_UPLOAD_SIZE_BYTES
        with pytest.raises(UploadValidationError):
            validate_upload("a.txt", 0)
        with pytest.raises(UploadValidationError):
            validate_upload("a.txt", MAX_UPLOAD_SIZE_BYTES + 1)
        assert validate_upload("a.txt", 10, b"hello") == ".txt"

    def test_magic_bytes(self):
        with pytest.raises(UploadValidationError):
            validate_upload("a.pdf", 10, b"GIF89a....")
        assert validate_upload("a.pdf", 10, b"%PDF-1.7") == ".pdf"
        with pytest.raises(UploadValidationError):
            validate_upload("a.txt", 10, b"ab\x00cd")


class TestFilenamesAndStorage:
    def test_safe_filename_strips_paths(self):
        for evil in ["../../etc/passwd.txt", "..\\..\\win.txt", "/abs/path/x.pdf", "a/b/c.txt"]:
            name = safe_filename(evil)
            assert "/" not in name and "\\" not in name and ".." not in name
            assert name.endswith(".txt") or name.endswith(".pdf")

    def test_safe_filename_unique(self):
        assert safe_filename("a.txt") != safe_filename("a.txt")

    def test_private_upload_path_is_per_user_and_owner_only(self):
        from config import UPLOAD_DIR
        p = private_upload_path(5, "../../x.pdf")
        assert os.path.abspath(p).startswith(os.path.abspath(os.path.join(UPLOAD_DIR, "5")))
        mode = stat.S_IMODE(os.stat(os.path.dirname(p)).st_mode)
        assert mode == 0o700
        write_private_file(p, b"%PDF-1.4 test")
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
        os.remove(p)

    def test_private_upload_path_rejects_bad_user(self):
        with pytest.raises(ValueError):
            private_upload_path(None, "a.txt")
