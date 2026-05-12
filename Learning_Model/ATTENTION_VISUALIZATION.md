# 👁️ ATTENTION VISUALIZATION & INTERPRETABILITY

## 1. WHAT ARE ATTENTION WEIGHTS?

### Definition

```
At each decoder step t, Attention computes:

weights[t] = softmax(decoder_state @ encoder_states^T / √d)
           = (336,) vector of values in [0, 1]
           
Each weight[t, i] = probability that decoder looks at encoder step i

Example:
  weights[t] = [0.02, 0.34, 0.05, 0.01, ..., 0.01]
               [enc0, enc1, enc2, enc3, ..., enc335]
               
Interpretation:
  "When predicting hour t, model focuses 34% on encoder step 1"
```

### Why Important?

```
✓ Interpretable: See what model attends to
✓ Debuggable: Is model learning sensible patterns?
✓ Explainable: Justify predictions to users
✓ Trustworthy: Model decision-making is visible

Example Application:
  "Temperature spike at hour t is because model attended to
   similar pattern 24 hours ago (hour t-24)"
```

---

## 2. EXTRACTING ATTENTION WEIGHTS

### Modify Model to Save Weights

```python
class Seq2SeqLSTM_WithAttentionHooks(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = []  # Store weights
    
    def forward(self, x, y=None, future_cov=None, tf_ratio=0.5):
        enc_out, h, c = self.encoder(x)
        
        prev_out = x[:, -1, self.target_index].unsqueeze(-1)
        outputs = []
        self.attention_weights = []  # Reset
        
        for t in range(self.pred_len):
            # Compute attention
            q = h[-1].unsqueeze(1)  # (B, 1, H)
            scores = torch.bmm(q, enc_out.transpose(1, 2)) * (self.hidden_size * 2) ** -0.5
            weights = torch.softmax(scores, dim=-1)  # (B, 1, T)
            
            # Save weights for visualization
            self.attention_weights.append(weights.squeeze(1).detach().cpu().numpy())
            
            ctx = torch.bmm(weights, enc_out).squeeze(1)  # (B, H)
            
            # Rest of forward pass...
            cov_t = future_cov[:, t, :] if future_cov is not None else None
            dec_in = torch.cat([prev_out, cov_t], dim=-1).unsqueeze(1) \
                     if cov_t is not None else prev_out.unsqueeze(1)
            
            pred, h, c = self.decoder(dec_in, h, c, enc_out)
            outputs.append(pred)
            
            use_tf = self.training and y is not None and random.random() < tf_ratio
            prev_out = y[:, t].unsqueeze(-1) if use_tf else pred
        
        return torch.cat(outputs, dim=1)


# Usage
model = Seq2SeqLSTM_WithAttentionHooks(...)
predictions = model(X_test)

# attention_weights = list of 24 arrays, each (B, T=336)
attention_weights = model.attention_weights
```

### Extract as Dictionary

```python
def get_attention_weights(model, X_test, future_cov_test):
    """Extract attention weights for analysis."""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test, y=None, future_cov=future_cov_test, tf_ratio=0.0)
    
    # Convert to numpy
    attention_dict = {
        't': list(range(24)),
        'weights': [w for w in model.attention_weights],  # 24 × (B, 336)
        'predictions': predictions.cpu().numpy(),
    }
    
    return attention_dict

# Get weights for first batch element
attention = get_attention_weights(model, X_test[:1], future_cov_test[:1])
weights_seq1 = [w[0] for w in attention['weights']]  # 24 × (336,)
```

---

## 3. VISUALIZE ATTENTION HEATMAP

### Heatmap: Time Steps vs Decoder Steps

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_attention_heatmap(attention_weights, save_path='attention_heatmap.png'):
    """
    Plot attention weights as heatmap.
    
    X-axis: Encoder timesteps (0-335, past 7 days)
    Y-axis: Decoder timesteps (0-23, next 24 hours)
    Color: Attention weight (0-1)
    """
    # Stack all weights: (pred_len=24, seq_len=336)
    attention_matrix = np.array([w[0] for w in attention_weights])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Heatmap
    im = ax.imshow(attention_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    
    # Labels
    ax.set_xlabel('Encoder Timestep (Past 7 days)', fontsize=12)
    ax.set_ylabel('Decoder Timestep (Next 24 hours)', fontsize=12)
    ax.set_title('Attention Weights: Which Past Hours Attend to Which Future Hours?', fontsize=14)
    
    # X-axis: days
    day_labels = [f'Day {i//48}' if i % 48 == 0 else '' for i in range(336)]
    ax.set_xticks(range(0, 336, 48))
    ax.set_xticklabels(['Day 0', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6'])
    
    # Y-axis: hours
    ax.set_yticks(range(24))
    ax.set_yticklabels([f'Hour {i}' for i in range(24)])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

# Usage
attention = get_attention_weights(model, X_test[:1], future_cov_test[:1])
plot_attention_heatmap(attention['weights'])
```

### What the Heatmap Shows

```
Example Pattern (typical temperature):

            |Day 0 |Day 1 |Day 2 |Day 3 |Day 4 |Day 5 |Day 6 |
            └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
Hour 0      │ . . │ . . │ . . │ . . │ . . │ . . │★★★★★│  ← Attends to Day 6
Hour 1      │ . . │ . . │ . . │ . . │ . . │ . . │★★★★★│
Hour 2      │ . . │ . . │ . . │ . . │ . . │ . . │★★★★★│
...
Hour 12     │ . . │ . . │ . . │★★★★★│ . . │ . . │ . . │  ← 24h cycle
Hour 13     │ . . │ . . │ . . │★★★★★│ . . │ . . │ . . │
...

Interpretation:
  ★ = high attention (>0.3)
  . = low attention (<0.1)
  
Pattern: Each hour looks at:
  1. Recent hours (right side = Day 6)
  2. Same hour 24h ago (Day 3, same position)
  
This makes sense! Temperature has 24-hour cycle + recent trend.
```

---

## 4. ANALYZE ATTENTION PATTERNS

### Most Attended Timesteps

```python
def get_top_attended_steps(attention_weights, top_k=5):
    """For each decoder step, find top-k most attended encoder steps."""
    results = {}
    
    for t, weights_t in enumerate(attention_weights):
        weights_batch1 = weights_t[0]  # First batch element (336,)
        top_indices = np.argsort(weights_batch1)[-top_k:][::-1]
        top_weights = weights_batch1[top_indices]
        
        results[f'hour_{t}'] = {
            'indices': top_indices,
            'weights': top_weights,
            'encoder_hours_ago': [335 - idx for idx in top_indices],  # How many hours ago
        }
    
    return results

# Usage
top_attention = get_top_attended_steps(attention['weights'], top_k=5)

# Print results
for pred_hour, info in top_attention.items():
    print(f"\n{pred_hour}:")
    for idx, weight, hours_ago in zip(
        info['indices'], 
        info['weights'], 
        info['encoder_hours_ago']
    ):
        print(f"  Encoder step {idx} ({hours_ago}h ago): {weight:.3f}")

# Output example:
# hour_0:
#   Encoder step 335 (0h ago): 0.342
#   Encoder step 334 (1h ago): 0.298
#   Encoder step 311 (24h ago): 0.156
#   Encoder step 287 (48h ago): 0.089
#   Encoder step 263 (72h ago): 0.045
```

### Attention Distribution Statistics

```python
def analyze_attention_distribution(attention_weights):
    """Analyze how spread out attention is."""
    results = {
        'mean_entropy': [],
        'max_weight': [],
        'concentration': [],  # % weight in top-5
    }
    
    for weights_t in attention_weights:
        w = weights_t[0]  # First batch element
        
        # Entropy (how spread out)
        entropy = -np.sum(w * np.log(w + 1e-9))
        results['mean_entropy'].append(entropy)
        
        # Maximum weight
        results['max_weight'].append(np.max(w))
        
        # Concentration in top-5
        top_5_weight = np.sum(np.sort(w)[-5:])
        results['concentration'].append(top_5_weight)
    
    return results

# Usage
stats = analyze_attention_distribution(attention['weights'])

print("Attention Distribution Statistics:")
print(f"  Mean entropy: {np.mean(stats['mean_entropy']):.3f}")
print(f"    (0 = focused, 5.8 = uniform)")
print(f"  Max weight: {np.mean(stats['max_weight']):.3f}")
print(f"    (higher = more focused)")
print(f"  Top-5 concentration: {np.mean(stats['concentration']):.3f}")
print(f"    (higher = more peaked)")

# Interpretation:
# - High entropy: attention spread over many timesteps
# - Low entropy: attention focused on few timesteps
# - High max_weight: strong peak
# - High concentration: weight concentrated in top-5 steps
```

### Temporal Attention Patterns

```python
def plot_attention_over_time(attention_weights, save_path='attention_time.png'):
    """
    Plot how attention evolves over prediction horizon.
    
    Shows: For each decoder timestep, where does it attend?
    """
    fig, axes = plt.subplots(4, 6, figsize=(16, 10))
    axes = axes.flatten()
    
    for t, weights_t in enumerate(attention_weights):
        w = weights_t[0]
        ax = axes[t]
        
        # Smoothed line plot
        ax.plot(w, linewidth=1.5, color='steelblue', alpha=0.7)
        ax.fill_between(range(len(w)), w, alpha=0.3, color='steelblue')
        
        # Highlight where attention is highest
        top_idx = np.argmax(w)
        ax.axvline(top_idx, color='red', linestyle='--', alpha=0.5)
        
        ax.set_ylim([0, max(w) * 1.2])
        ax.set_title(f'Decoder Hour {t} (max @ step {top_idx})')
        ax.set_xlabel('Encoder Step')
        ax.set_ylabel('Attention Weight')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Attention Weight Distribution at Each Decoder Step', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Usage
plot_attention_over_time(attention['weights'])
```

---

## 5. INTERPRETABILITY: WHAT IS THE MODEL LEARNING?

### Case Study: Temperature Prediction

```python
def interpret_prediction(
    model, 
    X_test,
    y_true,
    future_cov,
    scaler,
    sample_idx=0
):
    """Interpret a single prediction using attention weights."""
    
    # Get prediction and attention
    with torch.no_grad():
        predictions = model(X_test, y=None, future_cov=future_cov, tf_ratio=0.0)
    
    attention_weights = model.attention_weights
    pred_scaled = predictions[sample_idx].cpu().numpy()
    true_scaled = y_true[sample_idx, -24:, 0].numpy()
    
    # Inverse scale
    pred_celsius = pred_scaled * scaler.scale_[0] + scaler.mean_[0]
    true_celsius = true_scaled * scaler.scale_[0] + scaler.mean_[0]
    
    # Analyze each prediction
    print("="*70)
    print("PREDICTION INTERPRETATION")
    print("="*70)
    
    for t in range(24):
        print(f"\n📊 Hour {t}: Predicted {pred_celsius[t]:.1f}°C (True: {true_celsius[t]:.1f}°C)")
        
        # Top attended timesteps
        w = attention_weights[t][sample_idx]
        top_5_idx = np.argsort(w)[-5:][::-1]
        
        print("  Top 5 attended encoder steps:")
        for rank, idx in enumerate(top_5_idx, 1):
            hours_ago = 335 - idx
            weight = w[idx]
            hour_of_day = (hours_ago // 2) % 24  # Assuming 30-min resolution
            day_offset = hours_ago // 48
            
            print(f"    {rank}. Step {idx} ({hours_ago}h ago = "
                  f"Day-{day_offset} Hour-{hour_of_day:02d}): "
                  f"weight={weight:.3f}")
        
        # Error
        error = pred_celsius[t] - true_celsius[t]
        error_direction = "↑ overpredict" if error > 0 else "↓ underpredict"
        print(f"  Error: {error:+.2f}°C {error_direction}")

# Usage
interpret_prediction(model, X_test[:1], y_test[:1], future_cov_test[:1], scaler)
```

### Output Example

```
Hour 0: Predicted 21.5°C (True: 21.4°C)
  Top 5 attended encoder steps:
    1. Step 335 (0h ago = Day-6 Hour-23): weight=0.342
    2. Step 334 (1h ago = Day-6 Hour-22): weight=0.298
    3. Step 311 (24h ago = Day-5 Hour-23): weight=0.156
    4. Step 287 (48h ago = Day-4 Hour-23): weight=0.089
    5. Step 263 (72h ago = Day-3 Hour-23): weight=0.045
  Error: +0.10°C ↑ overpredict
  
  → Model looks at: recent hours (most important) + same hour 24h ago
  → This makes sense! Temperature has 24h cycle + recent trend

Hour 12: Predicted 22.3°C (True: 22.4°C)
  Top 5 attended encoder steps:
    1. Step 323 (12h ago = Day-6 Hour-11): weight=0.287
    2. Step 324 (11h ago = Day-6 Hour-12): weight=0.265
    3. Step 299 (36h ago = Day-5 Hour-12): weight=0.198
    4. Step 275 (60h ago = Day-4 Hour-12): weight=0.142
    5. Step 251 (84h ago = Day-3 Hour-12): weight=0.076
  Error: -0.10°C ↓ underpredict
  
  → Model attends to: similar hour from today (recent) + same hour previous days
  → Captures diurnal (daily) cycle!
```

### Error Analysis with Attention

```python
def analyze_errors_with_attention(
    model,
    X_test,
    y_true,
    future_cov_test,
    scaler,
    threshold=0.2  # MAE > 0.2°C is "error"
):
    """Identify if large errors correlate with unusual attention."""
    
    with torch.no_grad():
        predictions = model(X_test, y=None, future_cov=future_cov_test, tf_ratio=0.0)
    
    pred_celsius = predictions.cpu().numpy() * scaler.scale_[0] + scaler.mean_[0]
    true_celsius = y_true[:, -24:, 0].numpy() * scaler.scale_[0] + scaler.mean_[0]
    
    errors = np.abs(pred_celsius - true_celsius)  # (N, 24)
    
    # Find large errors
    large_error_mask = errors > threshold
    
    print("Large Error Timesteps (MAE > 0.2°C):")
    print("="*80)
    
    for sample_idx in range(X_test.shape[0]):
        error_steps = np.where(large_error_mask[sample_idx])[0]
        
        if len(error_steps) == 0:
            continue
        
        print(f"\nSample {sample_idx}: {len(error_steps)} large errors")
        
        for t in error_steps[:3]:  # Show first 3
            pred = pred_celsius[sample_idx, t]
            true = true_celsius[sample_idx, t]
            error = pred - true
            
            # Check attention concentration
            w = model.attention_weights[t][sample_idx]
            entropy = -np.sum(w * np.log(w + 1e-9))
            max_weight = np.max(w)
            
            print(f"  Hour {t}: {pred:.1f}°C vs {true:.1f}°C "
                  f"(error={error:+.2f}°C), "
                  f"attention entropy={entropy:.2f}, "
                  f"max_weight={max_weight:.3f}")
            
            # Hypothesis: Was attention unusual?
            if entropy > 4.0:  # Very spread out
                print(f"    ⚠️  Unusual: Attention is very spread (confused)")
            elif max_weight > 0.5:  # Very focused
                print(f"    ⚠️  Unusual: Attention is very focused (over-reliant)")
```

---

## 6. COMPARE ATTENTION ACROSS SAMPLES

### Attention Similarity

```python
from scipy.spatial.distance import cosine

def compare_attention_between_samples(
    model,
    X_test_batch,
    future_cov_batch
):
    """Compare attention patterns between different test samples."""
    
    with torch.no_grad():
        predictions = model(X_test_batch, y=None, future_cov=future_cov_batch, tf_ratio=0.0)
    
    attention_list = model.attention_weights  # 24 × (B, 336)
    B = X_test_batch.shape[0]
    
    # For each decoder timestep, compute similarity between samples
    similarities = []
    
    for t in range(24):
        weights_t = attention_list[t]  # (B, 336)
        
        # Pairwise cosine similarity
        pairwise_sim = np.zeros((B, B))
        for i in range(B):
            for j in range(i, B):
                sim = 1 - cosine(weights_t[i], weights_t[j])
                pairwise_sim[i, j] = sim
                pairwise_sim[j, i] = sim
        
        similarities.append(pairwise_sim)
    
    # Average similarity
    avg_similarity = np.mean([s[np.triu_indices(B, k=1)] 
                               for s in similarities])
    
    print(f"Average attention similarity between samples: {avg_similarity:.3f}")
    print(f"  (1.0 = identical patterns, 0.0 = completely different)")
    
    return similarities

# Usage
similarities = compare_attention_between_samples(model, X_test[:4], future_cov_test[:4])
```

---

## 7. SAVE & LOAD ATTENTION DATA

### Export for External Analysis

```python
import json

def save_attention_analysis(attention_weights, predictions, output_path='attention_analysis.json'):
    """Save attention data for external visualization or analysis."""
    
    data = {
        'metadata': {
            'num_steps': len(attention_weights),
            'seq_len': attention_weights[0].shape[1],
            'batch_size': attention_weights[0].shape[0],
        },
        'attention_weights': [
            w[0].tolist() for w in attention_weights  # First batch element
        ],
        'predictions': predictions[0].tolist() if isinstance(predictions, np.ndarray) 
                       else predictions[0].cpu().numpy().tolist(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved to {output_path}")

# Usage
save_attention_analysis(attention['weights'], attention['predictions'])
```

### Load and Visualize in Another Tool

```python
import json
import matplotlib.pyplot as plt

def load_and_visualize_attention(json_path):
    """Load saved attention and create visualizations."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    weights = np.array(data['attention_weights'])  # (24, 336)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(weights, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    
    ax.set_xlabel('Encoder Step')
    ax.set_ylabel('Decoder Step')
    ax.set_title('Attention Weights')
    plt.colorbar(im, ax=ax)
    plt.savefig('attention_viz.png', dpi=150, bbox_inches='tight')
    plt.show()

# Usage
load_and_visualize_attention('attention_analysis.json')
```

---

## 8. SUMMARY: WHAT ATTENTION TELLS US

```
✓ Model attends to recent hours: Learning recent trend
✓ Model attends to same hour 24h ago: Capturing daily cycle
✓ Model attends to weekends vs weekdays: Handling weekly pattern
✓ Model ignores distant past: Not needed for short-term forecast
✓ Attention is interpretable: Explains predictions to humans
✓ Attention reveals bugs: Unusual patterns = potential issues

Example Insights:
━━━━━━━━━━━━━━━━━
Temperature Prediction:
  Hour 0: Attends to recent hours (trend)
  Hour 12: Attends to same hour yesterday (daily cycle)
  Hour 18: Attends to evening peak (diurnal pattern)
  
  → Model learns physically meaningful patterns!

Energy Consumption:
  Peak hours (9-17): Attend to recent hours (demand changes fast)
  Off-peak (21-6): Attend to same hour yesterday (consistent baseline)
  
  → Model adapts attention to forecast horizon!

Wind Speed:
  All hours: Attend broadly (chaotic, less predictable)
  
  → Model is uncertain, hedges bets across time steps!
```

