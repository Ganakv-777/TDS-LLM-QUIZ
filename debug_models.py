import os
import google.generativeai as genai

def list_available_models():
    print("Checking available Gemini models...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set.")
        print("Add it to your .env file.")
        return

    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()

        print("\nAvailable models supporting text generation:\n")

        found = False
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                print(f" - {m.name}")
                found = True

        if not found:
            print("❌ No text-generation models accessible with your API key.")

    except Exception as e:
        print("❌ Error retrieving model list:")
        print(e)


if __name__ == "__main__":
    list_available_models()
