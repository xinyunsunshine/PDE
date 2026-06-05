import os
#!/usr/bin/env python3
"""Simple health check script for the LiteLLM API."""

from openai import OpenAI

LITELLM_BASE_URL = "https://ete-litellm.bx.cloud9.ibm.com"
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")


def test_api():
    """Send a simple text request to verify API health."""
    print("Testing LiteLLM API...")
    print(f"Endpoint: {LITELLM_BASE_URL}")
    print("-" * 40)

    try:
        client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

        response = client.chat.completions.create(
            model="claude-opus-4-5-20251101",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Say 'API is healthy' and nothing else."}
            ],
        )

        result = response.choices[0].message.content
        print(f"Response: {result}")
        print("-" * 40)
        print("Status: OK")
        return True

    except Exception as e:
        print(f"Error: {e}")
        print("-" * 40)
        print("Status: FAILED")
        return False


if __name__ == "__main__":
    test_api()
