import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

TEST_SERVER_URL = os.getenv("TEST_SERVER_URL")
if not TEST_SERVER_URL:
    print("❌ TEST_SERVER_URL is not set in environment variables.")
    print("Add TEST_SERVER_URL to your .env file.")
    exit()

student_email = os.getenv("STUDENT_EMAIL", "example@student.com")
student_secret = os.getenv("STUDENT_SECRET", "placeholder_secret")

payload = {
    "email": student_email,
    "secret": student_secret,
    "url": "https://tds-llm-analysis.s-anand.net/demo"
}

print(f"\nSending POST request to: {TEST_SERVER_URL} (root path)\n")
print("Payload:")
print(json.dumps(payload, indent=2))

try:
    # POST to root (/) because main.py expects POST at "/"
    response = requests.post(TEST_SERVER_URL, json=payload, timeout=30)
    print("\nResponse Status Code:", response.status_code)
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except:
        print("Raw Response:")
        print(response.text)

    if response.status_code == 200:
        print("\n✅ TEST PASSED: Server accepted the request.\n")
    else:
        print("\n❌ TEST FAILED: Server rejected the request.\n")

except Exception as e:
    print("\n❌ CONNECTION ERROR:", e)