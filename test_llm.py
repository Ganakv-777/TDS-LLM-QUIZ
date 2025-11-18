import os
import google.generativeai as genai

def test_gemini():
    print("1. Loading GEMINI_API_KEY from environment...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable is not set.")
        print("Set it in your .env file before running this script.")
        return

    genai.configure(api_key=api_key)

    try:
        print("2. Sending a test request to Gemini (gemini-2.5-flash)...")
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            "Write a Python print statement that says 'Hello from Gemini'"
        )

        print("\n3. Gemini Response:")
        print(response.text)
        print("\n4. ✅ Test Successful!")

    except Exception as e:
        print("❌ Error while testing Gemini:")
        print(e)


if __name__ == "__main__":
    test_gemini()
