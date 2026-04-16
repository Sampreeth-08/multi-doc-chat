"""Preliminary API tests — run while the server is up:
    python test_api.py
"""

import sys
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
PDF  = Path("data/2025 - Chaudhary et al. - Automated metrology for additively manufactured parts using deep learning and co.pdf")


def sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


# ── 1. Health check ───────────────────────────────────────────────────────────
sep("1 / GET /  (page load)")
r = requests.get(f"{BASE}/")
print(f"Status : {r.status_code}")
print(f"Content: {r.text[:80].strip()}…")
assert r.status_code == 200, "Expected 200"


# ── 2. Start session with file upload ─────────────────────────────────────────
sep("2 / POST /start  (upload PDF)")
assert PDF.exists(), f"File not found: {PDF}"

with PDF.open("rb") as f:
    r = requests.post(
        f"{BASE}/start",
        files=[("files", (PDF.name, f, "application/pdf"))],
    )

print(f"Status : {r.status_code}")
if r.status_code != 201:
    print(f"Error  : {r.text}")
    sys.exit(1)

data = r.json()
session_id = data["session_id"]
print(f"Session ID    : {session_id}")
print(f"Files count   : {data['files_count']}")
print(f"Docs loaded   : {data['files_loaded']}")
print(f"Chunks created: {data['chunks_created']}")
print(f"Started at    : {data['started_at']}")


# ── 3. Chat ───────────────────────────────────────────────────────────────────
sep("3 / POST /chat/{session_id}  (first question)")
question = "What is this paper about? Give a one-sentence summary."
r = requests.post(
    f"{BASE}/chat/{session_id}",
    json={"question": question},
)
print(f"Status   : {r.status_code}")
if r.status_code != 200:
    print(f"Error    : {r.text}")
    sys.exit(1)

resp = r.json()
print(f"Question : {resp['question']}")
print(f"\nReasoning:\n{resp['reasoning'][:300]}…")
print(f"\nAnswer:\n{resp['answer']}")


# ── 4. Follow-up question ─────────────────────────────────────────────────────
sep("4 / POST /chat/{session_id}  (follow-up)")
question2 = "What deep learning model did they use?"
r = requests.post(
    f"{BASE}/chat/{session_id}",
    json={"question": question2},
)
print(f"Status   : {r.status_code}")
resp2 = r.json()
print(f"Question : {resp2['question']}")
print(f"\nAnswer:\n{resp2['answer']}")


# ── 5. Delete session ─────────────────────────────────────────────────────────
sep("5 / DELETE /sessions/{session_id}")
r = requests.delete(f"{BASE}/sessions/{session_id}")
print(f"Status : {r.status_code}")
assert r.status_code == 204, f"Expected 204, got {r.status_code}"
print("Session deleted successfully.")

# ── 6. Verify session is gone ─────────────────────────────────────────────────
sep("6 / POST /chat/{session_id}  (should be 404 after delete)")
r = requests.post(f"{BASE}/chat/{session_id}", json={"question": "hello"})
print(f"Status : {r.status_code}")
assert r.status_code == 404, f"Expected 404, got {r.status_code}"
print("Confirmed: session no longer exists.")

sep("ALL TESTS PASSED")
