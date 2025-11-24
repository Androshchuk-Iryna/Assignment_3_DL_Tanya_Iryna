# Assignment_3_DL

##  Results Summary

| Model | Validation F1 |
|-------|---------------|
| 1-layer GRU | 0.9065 | 
| 1-layer LSTM | 0.9070 | 
| BPE GRU | 0.8932 | 
| 2-layer GRU | 0.9093 |
| 3-layer LSTM | 0.9135 | 
| **FastText GRU** | 0.9143 |
| **Voting Ensemble** | 0.9230 | 
| LR Stacking | 0.9214 |

---

##  Data Insights

**Dataset:**
- Training samples: 1,010,000
- Test samples: 477
- Samples with locations: 23.1%
- Average text length: 14.6 words

**Examples:**
```
Text: "У Львові 34-річний мешканець Яворівського району..."
Locations: ['Львові', 'Яворівського району']

Text: "Нагадаємо, президент України Володимир Зеленський..."
Locations: ['України']
```

**Key Observations:**
- Class imbalance: 76.9% samples have no locations
- Short texts (good for RNN processing)
- Multiple locations per sample common
- High-quality Ukrainian annotations

---

## Metric Analysis
The competition uses **entity-level F1-score**, where each entity is a text span (start, end).
The model receives a score of 1 only when it has completely correctly restored the location boundaries.
Partial matches (for example, finding “Львів” instead of “місто Львів”) are counted as errors.

**Advantages:**

* A classic approach to NER, consistent with standards (CoNLL, spaCy).
* Clearly penalizes incorrect boundaries, which encourages high-quality sequence labeling.
* Stable and understandable metric for comparing models.

**Disadvantages:**

* Complete dependence on exact span boundaries: even a 1-character offset = 0 points.
* Does not take partial information into account - a model that “almost guessed right” gets the same score as a model that missed completely.
* Very sensitive to tokenization (Byte Pair Encoding can cut names into components -> F1 drop).

**Edge cases:**

* Complex toponyms such as “Яворівського району” may have different segmentations -> risk of incorrect boundaries.
* Locations with declension (Львів/Львові/Львова) - the model may consider them different tokens.
* Several locations in a sentence next to each other -> risk of confusing intervals.
* Phrases with quotation marks or punctuation (“в місті Києві,”) - often errors on commas.
---

## Validation Strategy

The task already provides for an official split via the is_valid field, where:

is_valid = 0 -> training data

is_valid = 1 -> validation data

This is not a random split - it was formed by the authors of the dataset in order to:

* maintain the same distribution of location types in train and validation;

* divide sentences from one document into different parts, which prevents data leakage;

* simulate the expected statistics on the test, which ensures correct correlation with the leaderboard.

Therefore, using this particular split (is_valid) is the most reliable strategy, consistent with the official Kaggle evaluation.

---

##  Feature Engineering

### Vocabulary Construction
```python
# Build vocab with frequency filtering
vocab = ["<pad>", "<unk>"] + [word for word, count in Counter(tokens) 
                                if count >= 2]
```
---

## Model Architecture

### Base: RNNTagger

```python
class RNNTagger(pl.LightningModule):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, 
                 num_labels=2, rnn_type="gru", num_layers=1):
        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # 2. Bidirectional RNN
        self.rnn = nn.GRU/LSTM(
            emb_dim, hidden_dim, 
            bidirectional=True,
            num_layers=num_layers
        )
        
        # 3. Classification head
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
```

##  Experiments

### 1. Baseline Models (1-layer)
- **GRU:** Val F1 = 0.9065, Train F1 = 0.9373
- **LSTM:** Val F1 = 0.9070, Train F1 = 0.9382
- **Finding:** GRU and LSTM perform similarly

### 2. Tokenization
- **Word-level:** 0.9065 
- **BPE:** 0.8932 

### 3. Multi-Layer Architectures
- **2-layer GRU:** Val F1 = 0.9093 (+0.28% vs 1-layer)
- **3-layer LSTM:** Val F1 = 0.9135 (+0.70% vs 1-layer)

### 4. Pre-trained Embeddings
- **FastText GRU:** Val F1 = 0.9143 (+0.78% vs baseline)
- Train F1 = 0.9633 (higher overfitting)

### 5. Threshold Optimization
Tested thresholds: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8

**Results:**
- Threshold 0.4: F1 = 0.8847 (best)
- Default 0.5: F1 = 0.8843

### 6. Entity Merging
Tested max_gap: 0, 1, 2, 3

**Results:**
- Gap 0: F1 = 0.8843 (no merging)
- Gap 1: F1 = 0.7651 (worse!)

---

##  Ensemble Methods

### Voting Ensemble

**Models combined:**
1. 1-layer GRU
2. 1-layer LSTM  
3. 2-layer GRU
4. 3-layer LSTM
5. FastText GRU


**Results:**
- **Ensemble F1: 0.9230**
- Best individual: 0.9143
- **Improvement: +0.87%** 

### Stacking with Logistic Regression

**Method:**
- Use predictions from 5 models as features
- Train Logistic Regression as meta-learner
- Predict final labels

**Results:**
- **Stacking F1: 0.9214**
- vs Voting: -0.16%

---

## Hyperparameters

### Final Configuration

```python
# Best Single Model (FastText GRU)
model = RNNTagger(
    vocab_size=len(vocab),
    emb_dim=300,              # FastText dimension
    hidden_dim=256,
    num_labels=2,
    rnn_type="gru",
    num_layers=1,
    dropout=0.0,              # 1 layer, no dropout needed
    pretrained_embeddings=fasttext_embeddings
)

# Training
optimizer = Adam(lr=1e-3)
epochs = 3
batch_size = 256
```

