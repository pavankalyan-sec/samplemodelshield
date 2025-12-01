# Model Extraction Attack Demo

A hands-on demonstration of how machine learning models can be "stolen" through knowledge distillation attacks. This project accompanies a security blog post on ML model theft.

## What is Model Extraction?

Model extraction (or model stealing) is an attack where an adversary creates a copy of a target ML model by only querying its API. The attacker:

1. **Has no access** to the victim's training data
2. **Has no knowledge** of the victim's architecture
3. **Can only observe** input-output behavior (black-box access)

Despite these limitations, the attacker can train a "stolen" model that closely mimics the victim's predictions.

## This Demo

| Component | Description |
|-----------|-------------|
| **Victim Model** | A CNN trained on Fashion-MNIST (simulates a proprietary model) |
| **Stolen Model** | A simpler CNN with different architecture |
| **Attack Budget** | Only 1,000 queries to the victim |
| **Method** | Knowledge distillation using soft labels |

## Quick Start

### 1. Set up the environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the victim model

This creates the "proprietary" model that will be targeted:

```bash
python train_victim.py
```

Output:
```
Training on: cpu
Epoch 1/10 | Loss: 0.4821 | Train Acc: 82.35% | Val Acc: 87.12%
...
Epoch 10/10 | Loss: 0.1923 | Train Acc: 92.84% | Val Acc: 91.45%

Victim model saved. Final accuracy: 91.45%
```

### 3. Run the extraction attack

This queries the victim model and trains a substitute:

```bash
python extract_model.py
```

Output:
```
Extracting model using cpu
Attack budget: 1000 queries to victim model

[Phase 1] Querying victim model...
Collected 1000 query-response pairs

[Phase 2] Training stolen model via knowledge distillation...
Epoch 5/20 | Distillation Loss: 0.1842
...

[Phase 3] Evaluating extraction success...

==================================================
EXTRACTION RESULTS
==================================================
Victim Model Accuracy:  91.45%
Stolen Model Accuracy:  85.23%
Agreement Rate:         81.5%
Fidelity (stolen/victim): 93.2%
==================================================
```

### 4. Generate visualizations

```bash
python visualizations.py
```

### 5. Launch the interactive demo

```bash
python app.py
```

Then open http://127.0.0.1:7860 in your browser.

---

## Understanding the Results

### Prediction Comparison

The comparison grid shows how the stolen model's predictions align with the victim:

![Comparison Grid](static/comparison_grid.png)

Each row shows:
- **Left**: Original Fashion-MNIST image with true label
- **Center**: Victim model's probability distribution (green = predicted class)
- **Right**: Stolen model's probability distribution (red = predicted class)
- **Far right**: Whether predictions match

Note how the stolen model often produces **similar confidence patterns**, not just matching predictions.

### Per-Category Agreement

Not all categories are equally easy to steal:

![Agreement Chart](static/agreement_chart.png)

Key observations:
- **Trousers** (94.2%) and **Bags** (93.6%) are easiest to extract
- **Shirts** (39.8%) are hardest due to visual similarity with T-shirts and Coats
- Overall agreement: **81.5%** with just 1,000 queries

### Confusion Matrix Analysis

Comparing error patterns reveals successful extraction:

![Confusion Matrices](static/confusion_matrices.png)

The stolen model learned the victim's **mistakes**, not just its correct predictions. Both models confuse:
- Shirts ↔ T-shirts ↔ Coats (upper body garments)
- Pullovers ↔ Coats (similar silhouettes)

This error correlation is a strong indicator of successful model extraction.

---

## How the Attack Works

### Phase 1: Query Collection

```python
# Attacker sends images to victim API and records soft predictions
for images in query_loader:
    probs = victim_model.predict_proba(images)  # Soft labels!
    soft_labels.append(probs)
```

The attacker uses **any available images** (not the victim's training data) and records the victim's probability outputs, not just hard labels.

### Phase 2: Knowledge Distillation

```python
# Train stolen model to match victim's probability distributions
def knowledge_distillation_loss(student_logits, teacher_probs, temperature=3.0):
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(soft_student, teacher_probs, reduction='batchmean')
```

The temperature parameter "softens" the probability distributions, transferring more information per query than hard labels alone.

### Phase 3: Deployment

The stolen model can now be used without querying the victim, potentially:
- Avoiding API costs
- Bypassing rate limits
- Operating offline
- Serving as a base for further attacks

---

## Defense Considerations

This demo highlights why ML model providers should consider:

1. **Output perturbation**: Add noise to prediction probabilities
2. **Query monitoring**: Detect systematic query patterns
3. **Rate limiting**: Restrict query volume per user
4. **Watermarking**: Embed detectable signatures in model behavior
5. **Prediction rounding**: Return top-k labels instead of full distributions

---

## Project Structure

```
model-extraction-demo/
├── models.py           # CNN architectures (VictimCNN, StolenCNN)
├── train_victim.py     # Train the target model
├── extract_model.py    # Knowledge distillation attack
├── visualizations.py   # Generate comparison images
├── app.py              # Gradio interactive demo
├── requirements.txt    # Python dependencies
├── static/             # Generated visualization images
│   ├── comparison_grid.png
│   ├── agreement_chart.png
│   └── confusion_matrices.png
├── victim_model.pt     # Trained victim weights
└── stolen_model.pt     # Extracted model weights
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

The Fashion-MNIST dataset (~30MB) is automatically downloaded on first run.

## License

MIT License. Fashion-MNIST dataset is also MIT licensed (Zalando Research).

## References

- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Tramèr et al., "Stealing Machine Learning Models via Prediction APIs" (2016)
