import json

path = r'g:\Code\Deep Learning\ETTForecasting\35_Report_Model.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

sources = [ ''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' ]

with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(sources, f, indent=2)
