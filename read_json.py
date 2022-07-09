import json

filename = "test-1200-best-params.json"
with open(filename, 'r') as f:
    data = json.load(f)
print(data)
print(data.__str__())
