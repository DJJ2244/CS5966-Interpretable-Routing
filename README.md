# CS5966-Interpretable-Routing

## Setup

```bash
python -m venv .venv
# Mac/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Make sure [Ollama](https://ollama.com) is installed and running, then pull the models:

```bash
ollama pull llama3.2:1b
ollama pull llama3
```

## Run

```bash
python main.py
```

## Approach

The baseline router is RouteLLM's built-in `bert` router. The plan is to replace it with a custom router that runs the same bert method with our own MLP classification head, which is modified for our work. RouteLLM's router interface only requires one method — `calculate_strong_win_rate(prompt) -> float` — so the internals can be swapped freely without touching the rest of the framework.

## Evaluation

Evaluation approach TBD.

RouteLLM has eval but we might want to use our own. They might be able to work together.
