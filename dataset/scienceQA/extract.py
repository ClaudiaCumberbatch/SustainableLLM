import json

with open('problems.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

result = {}
for key, value in data.items():
    question = value.get('question', '')
    choices = value.get('choices', [])
    prompt = f"{question} Choices: {', '.join(choices)}"
    result[key] = {'prompt': prompt}

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)