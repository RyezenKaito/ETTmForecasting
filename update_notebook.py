import nbformat as nbf

file_path = 'Report_Model.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

markdown_cell = nbf.v4.new_markdown_cell('## 8. Real Data Prediction & Comparison')

code_cell_1 = nbf.v4.new_code_cell('''# =============================================
# PREDICT ON REAL DATA (test_data.xlsx)
# =============================================
import pandas as pd
import numpy as np

# 1. Load real data
df_real = pd.read_excel('data/test_data.xlsx')
df_real['date'] = pd.to_datetime(df_real['date'])
df_real = df_real.set_index('date')

# Drop unused columns if exist
for col in ['MUFL', 'MULL']:
    if col in df_real.columns:
        df_real.drop(col, axis=1, inplace=True)

# Generate time features for real input
df_real = add_time_features(df_real)
df_real['trend'] = apply_trend(df_real, window=period)
df_real['seasonal'] = apply_seasonal(df_real, seasonal_pattern)
df_real['residual'] = (df_real['OT'] - df_real['trend'] - df_real['seasonal']).values
df_real = df_real[train_df.columns] # ensure order

# Scale input sequence
real_scaled = scaler.transform(df_real.values)
Xb_real = torch.tensor(real_scaled, dtype=torch.float32).unsqueeze(0).to(device)

# 2. Load label to compare
df_label = pd.read_excel('data/label.xlsx')
y_true_actual = df_label['OT'].values

# 3. Construct future covariates for the 24 steps
last_date = df_real.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=15), periods=pred_len, freq='15min')
df_future = pd.DataFrame(index=future_dates)
df_future['OT'] = y_true_actual  # using actual to align with training distribution
for c in ['HUFL', 'HULL', 'LUFL', 'LULL']:
    df_future[c] = df_real[c].iloc[-1]
df_future = add_time_features(df_future)
combined_df = pd.concat([df_real, df_future])
df_future['trend'] = apply_trend(combined_df, window=period)[-pred_len:]
df_future['seasonal'] = apply_seasonal(df_future, seasonal_pattern)
df_future['residual'] = (df_future['OT'] - df_future['trend'] - df_future['seasonal']).values
df_future = df_future[train_df.columns] # ensure order

future_scaled = scaler.transform(df_future.values)
f_cov_real = torch.tensor(future_scaled[:, -N_COVARIATE:], dtype=torch.float32).unsqueeze(0).to(device)

# 4. Predictions
# --- TCN_v2 ---
tcn_model.eval()
with torch.no_grad():
    y_pred_tcn_scaled = tcn_model(Xb_real, future_features=f_cov_real)
y_pred_tcn = inverse_target(y_pred_tcn_scaled.cpu().numpy()[0], scaler, target_index)

# --- Seq2Seq ---
s2s_model.eval()
with torch.no_grad():
    # Provide dummy y filled with zeros for autoregressive target placeholder
    y_dummy = torch.zeros((1, pred_len)).to(device)
    y_pred_s2s_scaled = s2s_model(Xb_real, y=y_dummy, future_cov=f_cov_real, tf_ratio=0.0)
y_pred_s2s = inverse_target(y_pred_s2s_scaled.cpu().numpy()[0], scaler, target_index)

print('Predictions generated successfully.')
''')

code_cell_2 = nbf.v4.new_code_cell('''# =============================================
# COMPARISON ON ACTUAL DATA
# =============================================
metrics_tcn_actual = calc_metrics(y_pred_tcn, y_true_actual)
metrics_s2s_actual = calc_metrics(y_pred_s2s, y_true_actual)

df_comp_actual = pd.DataFrame([metrics_tcn_actual, metrics_s2s_actual], index=['TCN_v2', 'Seq2SeqLSTM'])
print("=== PERFORMANCE ON ACTUAL DATA (24 STEPS) ===")
display(df_comp_actual)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_true_actual, label='Ground Truth (label.xlsx)', color='black', linewidth=2)
plt.plot(y_pred_tcn, label='TCN_v2 Prediction', color='blue', linestyle='--', marker='o')
plt.plot(y_pred_s2s, label='Seq2Seq Prediction', color='red', linestyle='--', marker='s')
plt.title('Multi-Step Forecasting on Actual Data (24 steps)')
plt.xlabel('Steps')
plt.ylabel('Oil Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('actual_prediction_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
''')

nb.cells.extend([markdown_cell, code_cell_1, code_cell_2])

with open(file_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Updated Report_Model.ipynb')
