import json
import os

path = r'g:\Code\Deep Learning\ETTForecasting\35_Report_Model.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if 's2s_history = train_s2s' in ''.join(source):
            cell['source'] = [
                'import os\n',
                'from train_seq2seq import train as train_s2s\n',
                '\n',
                's2s_optimizer = torch.optim.AdamW(seq2seq_model.parameters(), lr=cfg.S2S_LR, weight_decay=cfg.S2S_WEIGHT_DECAY)\n',
                's2s_criterion = nn.MSELoss()\n',
                's2s_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(s2s_optimizer, mode="min", patience=3, factor=0.5)\n',
                '\n',
                'os.makedirs(cfg.CKPT_DIR, exist_ok=True)\n',
                '\n',
                'if not os.path.exists(cfg.S2S_CKPT):\n',
                '    s2s_history = train_s2s(\n',
                '        seq2seq_model, train_loader_seq, val_loader_seq,\n',
                '        s2s_optimizer, s2s_criterion, s2s_scheduler,\n',
                '        device, scaler, target_idx,\n',
                '        cfg.S2S_EPOCHS, cfg.S2S_PATIENCE, cfg.S2S_CKPT,\n',
                '    )\n',
                'else:\n',
                '    print("Skipping S2S training, checkpoint exists.")\n',
                '    s2s_history = {"train": [], "val": []}\n'
            ]
        elif 'inf_history = train_inf' in ''.join(source):
            cell['source'] = [
                'import os\n',
                'from train_informer import train as train_inf\n',
                '\n',
                'inf_optimizer = torch.optim.Adam(informer_model.parameters(), lr=cfg.INF_LR)\n',
                'inf_criterion = nn.MSELoss()\n',
                '\n',
                'if not os.path.exists(cfg.INF_CKPT):\n',
                '    inf_history = train_inf(\n',
                '        informer_model, train_loader_inf, val_loader_inf,\n',
                '        inf_optimizer, inf_criterion,\n',
                '        cfg.PRED_LEN, cfg.LABEL_LEN, target_idx, device,\n',
                '        cfg.INF_EPOCHS, cfg.INF_PATIENCE, cfg.INF_CKPT,\n',
                '    )\n',
                'else:\n',
                '    print("Skipping Informer training, checkpoint exists.")\n',
                '    inf_history = {"train": [], "val": []}\n'
            ]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
