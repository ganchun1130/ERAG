from xinference.client import Client
import openai


def xinference_client(model_uid):
    client = Client("http://localhost:9997")
    model = client.get_model(model_uid)
    messages = [
        {
            "role": "user",
            "content": "What is the largest animal?"
        }
    ]

    response = model.chat(
        messages,
        generate_config={"max_tokens": 1024}
    )
    answer = response['choices'][0]['message']['content']
    print(answer)

def openai_client(model_uid):
    client = openai.Client(api_key="not empty", base_url="http://localhost:9997/v1")
    response = client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "content": "who are you?",
                "role": "user",
            }
        ],
        max_tokens=1024
    )
    answer = response.choices[0].message.content
    print(answer)

if __name__ == "__main__":
    xinference_client("qwen2.5-instruct")
    openai_client("qwen2.5-instruct")