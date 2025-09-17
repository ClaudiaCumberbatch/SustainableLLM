import json

# 读取原始数据
with open('alpaca_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 只保留 instruction 和 input 字段
filtered = [
    {'instruction': item['instruction'], 'input': item['input']}
    for item in data
]

# 写入新文件
with open('filtered.json', 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)