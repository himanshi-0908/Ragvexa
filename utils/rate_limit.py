from slowapi import Limiter
from slowapi.util import get_remote_address

# Per-client-IP rate limiting. Swap get_remote_address for a
# per-user-id key func later if this sits behind a shared proxy/NAT
# where many real users share one IP.
limiter = Limiter(key_func=get_remote_address)
