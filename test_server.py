import os
import json
import requests

def test_connection():
    # The URL of your deployed FastAPI backend
    url = os.getenv("TEST_SERVER_URL")

    if not url:
        print("❌ TEST_SERVER_URL is not set in environment variables.")
        print("Add TEST_SERVER_URL to your .env file.")
        return

    student_email = os.getenv("STUDENT_EMAIL", "example@student.com")
    student_secret = os.getenv("STUDENT_SECRET", "placeholder_secret")

    payload = {
        "email": student_email,
        "secret": student_secret,
        "url": "https://tds-llm-analysis.s-anand.net/demo"   # demo quiz endpoint
    }

    print(f"\nSending test POST request to: {url}\n")
    print("Payload:")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(url, json=payload)
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


if __name__ == "__main__":
    test_connection()
