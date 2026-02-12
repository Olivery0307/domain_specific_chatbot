# eval.py (Simple automated test harness)
import requests

URL = "YOUR_LIVE_CLOUDRUN_URL/chat"
GOLDEN_DATASET = [
    {"q": "Valid in-domain question?", "expected_keyword": "keyword"},
    {"q": "Who is the president?", "expected_keyword": "sorry"} # Escape hatch test
]

for case in GOLDEN_DATASET:
    resp = requests.post(URL, json={"message": case["q"]}).json()
    print(f"Q: {case['q']} | A: {resp['response']}")