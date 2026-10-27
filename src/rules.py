"""
Rules that decide whether a piece of on-screen text is private.

Stage I stand-in for "understanding". Two layers:
  1. PATTERNS  - structured identifiers that have a fixed shape (email, phone...)
  2. DEMO_SECRETS - exact strings with NO shape (names, codenames, addresses you
     will show on camera). Regex can never catch these, so we list them.

Matching is done on a *normalized* copy of the text (lowercased, punctuation and
extra spaces removed) so small OCR differences (a missing space, a trailing
comma, "no.7" vs "no. 7") do not cause a miss.

Stage II replaces this whole file with a small context classifier on the NPU so
free-form private text (like a name) is caught without a hardcoded list.
"""

import re

# ---- structured identifiers (matched on the raw text) ----
PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?\d[\d\-\s]{7,}\d)"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "tw_id": re.compile(r"\b[A-Z][12]\d{8}\b"),
}

# Street-type tokens: seeing one of these is a strong address signal on its own.
STRONG_ADDRESS = {"rd", "st", "ave", "blvd", "ln", "dr", "road", "street",
                  "avenue", "boulevard", "lane", "drive"}
STRONG_ADDRESS_CJK = ["路", "街", "號", "巷", "弄"]
# Weaker hints: only count as an address if a digit is also present.
WEAK_ADDRESS = {"apt", "floor", "district", "city", "no", "區", "市", "樓"}

# Exact private strings to guarantee-catch in the demo. Keep them FAKE.
# Edit this list for your own recording.
DEMO_SECRETS = [
    "1234 maple street",
    "no. 7 roosevelt rd",
    "cheng han li",          # a name has no pattern, so it must live here
#    "See the Dentist",
]


def _normalize(text: str) -> str:
    """lowercase, turn punctuation into spaces, collapse spaces. Keeps CJK."""
    low = text.lower()
    cleaned = re.sub(r"[^0-9a-z一-鿿]+", " ", low)
    return re.sub(r"\s+", " ", cleaned).strip()


def classify_text(text: str):
    """Return a short label if `text` looks private, else None."""
    if not text or len(text.strip()) < 2:
        return None

    norm = _normalize(text)

    # 1. exact (but normalized) demo secrets
    for secret in DEMO_SECRETS:
        s = _normalize(secret)
        if s and s in norm:
            return "secret"

    # 2. structured identifiers on the raw text
    for label, pattern in PATTERNS.items():
        if pattern.search(text):
            return label

    # 3. addresses
    tokens = norm.split()
    if any(t in STRONG_ADDRESS for t in tokens) or \
       any(c in norm for c in STRONG_ADDRESS_CJK):
        return "address"
    if any(t in WEAK_ADDRESS for t in tokens) and any(c.isdigit() for c in text):
        return "address"

    return None
