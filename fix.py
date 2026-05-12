import json

path = r'g:\Code\Deep Learning\ETTForecasting\35_Report_Model.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'cfg.DEC_IN_DIM' in line:
                source[i] = line.replace('cfg.DEC_IN_DIM', 'cfg.S2S_DEC_IN_DIM')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

