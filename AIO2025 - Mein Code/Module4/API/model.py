import os
from cerebras.cloud.sdk import Cerebras

# Check if API key is set
api_key = os.environ.get("CEREBRAS_API_KEY")
if not api_key:
    print("Error: CEREBRAS_API_KEY environment variable is not set.")
    print("Please set it with: $env:CEREBRAS_API_KEY = 'your-api-key-here'")
    exit(1)

client = Cerebras(
    api_key=api_key
)

def post_chat(system, user):
    # Example chat completion
    try:
        stream = client.chat.completions.create(

            messages=[system, user],
            model="qwen-3-235b-a22b-instruct-2507",
            stream=True,
            max_completion_tokens=20000,
            temperature=0.7,
            top_p=0.8
        )
        out = ""
        print("Response: ", end="")
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                out += f"{content}"

        return out

    except Exception as e:
        print(f"Error: {e}")

def custom_chat(context):
    user = {
        "role": "user",
        "content": "Help me to summarize the paragraph in a pirate way"
    }
    system = {
        "role": "system",
        "content": "You are a helpful Summarizer AI assistant. Your job is to summarize the given context within 100-150 characters" + f"Base on the {context}"
    }

    out  = post_chat(system, user)

    return out

if __name__ == '__main__':
    context = "Aliens are hypothetical extraterrestrial beings from other planets. They are often depicted in science fiction as having advanced technology, unique appearances, and the potential to visit or influence Earth. Some theories suggest they could be exploring the universe or seeking resources."
    out  = custom_chat(context)
    print(out)