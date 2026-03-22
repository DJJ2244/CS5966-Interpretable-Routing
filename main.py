from routellm.controller import Controller

WEAK_MODEL = "ollama/llama3.2:1b"
STRONG_MODEL   = "ollama/llama3"
ROUTER       = "bert"
THRESHOLD    = 0.11593

client = Controller(
    routers=[ROUTER],
    strong_model=STRONG_MODEL,
    weak_model=WEAK_MODEL,
)

prompt = "Write a Python function that checks if a number is prime."

response = client.chat.completions.create(
    model=f"router-{ROUTER}-{THRESHOLD}",
    messages=[{"role": "user", "content": prompt}],
)

print(f"Model used : {response.model}")
print(f"Response   :\n{response.choices[0].message.content}")
