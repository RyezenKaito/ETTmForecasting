import json

with open('Report_Model_Backup.ipynb', encoding='utf-8') as f:
    d = json.load(f)

code = ''
for c in d['cells']:
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        code += source + '\n'
        if 'scaler       = StandardScaler()' in source:
            break

code += '''
print("COLS:", train_df.columns.tolist())
print("N_FEATURES:", n_features)
print("TARGET_INDEX:", target_index)
'''

with open('debug_pipeline2.py', 'w', encoding='utf-8') as f:
    f.write(code)
