import json

with open('Report_Model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open('cells_info.txt', 'w', encoding='utf-8') as f:
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            source = c['source']
            first_line = source[0].strip() if source else ''
            f.write(f"Cell {i}: {first_line[:100]}\n")
