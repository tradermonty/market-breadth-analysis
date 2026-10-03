"""Shared helpers to redact credentials and sensitive values from logs, CLI output, and exceptions.

Use ``register_secret`` to register known secret values once (at init or import) and
``redact`` to scrub a string before it reaches stdout, stderr, or a log. ``redact`` is
defense-in-depth: it removes registered values wherever they appear and additionally masks
sensitive URL query parameters and Authorization header values, so even unregistered or
temporary dummy secrets are scrubbed.

No real credential should ever be committed. Tests must use dummy values only.
"""

import re

REDACTED = '***REDACTED***'

# Deliberate placeholders / too-short values that, if registered, would over-redact normal
# text (e.g. "demo", "test"). Actual keys and tokens are long enough that a 8-char floor is
# safe while filtering out the common placeholders.
_MIN_SECRET_LENGTH = 8
_PLACEHOLDERS = {'demo', 'test', 'password', 'secret', 'token', 'key', 'dummy'}

_registry: set[str] = set()

# Masks the value of sensitive query parameters inside any URL-like substring. Applies to
# both lower- and upper-case names. The guarded names avoid a too-broad bare "key" match
# so ordinary text is not scrubbed.
_URL_QUERY = re.compile(
    r'(?i)([?&](?:apikey|api_key|token|access_token|secret|client_id|client_secret|'
    r'private_key|signature|password|passwd|auth)=)([^&\s"\']+)'
)

# Masks Authorization header values (Bearer / Basic / Token). Requests exceptions do not
# generally embed request headers, but the registered-secret substitution above is the real
# protection for the GitHub/Bearer path; this regex is an additional safety layer for any
# header-like text that reaches a log.
_AUTH = re.compile(r'(?i)(Authorization:\s*(?:Bearer|Basic|Token)\s+)[^\s,"\']+')


def register_secret(value: str | None) -> None:
    """Register a secret value so ``redact`` replaces it wherever it appears.

    Values shorter than 8 characters or known placeholders such as ``demo``/``test`` are
    ignored to avoid over-redacting ordinary text.
    """
    if not value:
        return
    value = str(value)
    if len(value) < _MIN_SECRET_LENGTH or value.lower() in _PLACEHOLDERS:
        return
    _registry.add(value)


def redact(text: str | None) -> str:
    """Return ``text`` with registered secrets and sensitive URL/header values masked.

    Never raises; ``None`` becomes an empty string.
    """
    if not isinstance(text, str):
        return ''
    out = text
    for secret in _registry:
        out = out.replace(secret, REDACTED)
    out = _URL_QUERY.sub(r'\1' + REDACTED, out)
    out = _AUTH.sub(r'\1' + REDACTED, out)
    return out
