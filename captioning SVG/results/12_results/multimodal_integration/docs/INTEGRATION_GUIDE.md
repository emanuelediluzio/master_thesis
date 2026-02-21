# 🔗 GUIDA INTEGRAZIONE MULTIMODALE

## 📋 OVERVIEW

Questa guida spiega come integrare l'encoder per immagini con i modelli LLM trained per creare modelli multimodali.

## 🎯 OBIETTIVO

Trasformare i modelli da **text-to-text** a **image-to-text**:

```
PRIMA:  XML text → LLM → Caption
DOPO:   Image → Encoder → Embedding → Adapter → LLM → Caption
```

## 🔧 ARCHITETTURA

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Image     │    │   Encoder   │    │   Adapter   │    │     LLM     │
│   Input     │───▶│  (Leonardo) │───▶│ Projection  │───▶│  (Trained)  │
│             │    │             │    │   Layer     │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                    │                    │
                    encoder_dim          llm_hidden_dim      output_text
```

## 📊 DIMENSIONALITÀ

### **LLM Hidden Dimensions:**
- **Gemma-2-9B**: 3584 dim
- **Llama-3.1-8B**: 4096 dim

### **Encoder Dimensions:** (da Leonardo)
- **Output Dimension**: TBD
- **Sequence Length**: TBD
- **Format**: TBD

### **Adapter Layer:**
```python
adapter = nn.Linear(encoder_dim, llm_hidden_dim)
```

## 🚀 SETUP INIZIALE

### 1. **Ricevi Materiali da Leonardo**

```bash
multimodal_integration/
├── encoder_weights/
│   ├── image_encoder.pth      # ← DA LEONARDO
│   ├── projection_layer.pth   # ← DA LEONARDO  
│   └── config.json           # ← DA LEONARDO
├── embeddings/
│   ├── train_embeddings.pkl  # ← DA LEONARDO
│   ├── test_embeddings.pkl   # ← DA LEONARDO
│   └── embedding_metadata.json # ← DA LEONARDO
└── integration_code/
    └── encoder_model.py       # ← DA LEONARDO
```

### 2. **Analizza Dimensionalità**

```bash
cd /work/tesi_ediluzio
python multimodal_integration/scripts/analyze_dimensions.py
```

Questo script:
- ✅ Analizza dimensioni LLM
- ✅ Carica e analizza encoder weights
- ✅ Analizza embedding format
- ✅ Crea piano integrazione

### 3. **Aggiorna Configurazioni**

Modifica i config files con le dimensioni reali:

```yaml
# configs/gemma_multimodal_config.yaml
encoder:
  output_dim: XXX  # Da Leonardo
  sequence_length: YYY  # Da Leonardo

adapter:
  input_dim: XXX  # = encoder.output_dim
```

## 🔧 IMPLEMENTAZIONE

### 1. **Carica Modello Multimodale**

```python
from multimodal_integration.integration_code.multimodal_model import create_multimodal_model

# Crea modello
model = create_multimodal_model(
    llm_model_name="google/gemma-2-9b-it",
    encoder_dim=encoder_output_dim,  # Da Leonardo
    adapter_config={
        'dropout': 0.1
    }
)
```

### 2. **Carica Encoder Weights**

```python
# Carica encoder da Leonardo
encoder = load_image_encoder("multimodal_integration/encoder_weights/image_encoder.pth")

# Carica embedding pre-calcolati
with open("multimodal_integration/embeddings/train_embeddings.pkl", 'rb') as f:
    train_embeddings = pickle.load(f)
```

### 3. **Training Adapter**

```python
# Freeze LLM, train solo adapter
for param in model.llm.parameters():
    param.requires_grad = False

# Train adapter
optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-4)

for batch in dataloader:
    image_embeddings = batch['image_embeddings']
    target_text = batch['caption']
    
    # Forward pass
    outputs = model(
        image_embeddings=image_embeddings,
        labels=target_text
    )
    
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

### 4. **Inference**

```python
# Generate caption da immagine
with torch.no_grad():
    generated_ids = model.generate(
        image_embeddings=image_embedding,
        max_new_tokens=256,
        temperature=0.7
    )
    
    caption = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

## 📊 STRATEGIE INTEGRAZIONE

### **Opzione 1: Prepend Visual Tokens (CONSIGLIATA)**
```
[visual_token_1] [visual_token_2] ... [visual_token_N] [text_tokens...]
```

**Pro:**
- ✅ Semplice da implementare
- ✅ Compatibile con LLM esistenti
- ✅ Buone performance

**Contro:**
- ❌ Aumenta lunghezza sequenza

### **Opzione 2: Cross-Attention**
```
Text Tokens ──┐
              ├─ Cross-Attention ─→ Output
Visual Tokens ┘
```

**Pro:**
- ✅ Non aumenta lunghezza sequenza
- ✅ Interazione più sofisticata

**Contro:**
- ❌ Più complesso da implementare
- ❌ Richiede modifiche architettura LLM

### **Opzione 3: Fusion Layer**
```
Text Embedding + Visual Embedding → Fused Embedding → LLM
```

**Pro:**
- ✅ Integrazione diretta
- ✅ Controllo fine su fusion

**Contro:**
- ❌ Richiede training più complesso
- ❌ Possibili problemi dimensionalità

## 🧪 TESTING

### 1. **Test Dimensionalità**
```python
# Test adapter
image_emb = torch.randn(1, seq_len, encoder_dim)
projected = model.adapter(image_emb)
assert projected.shape == (1, seq_len, llm_dim)
```

### 2. **Test Integrazione**
```python
# Test forward pass
outputs = model(image_embeddings=image_emb)
assert outputs.logits.shape[-1] == model.llm.config.vocab_size
```

### 3. **Test Generation**
```python
# Test generation
generated = model.generate(image_embeddings=image_emb, max_new_tokens=50)
assert generated.shape[1] > image_emb.shape[1]  # Generated tokens
```

## 📈 EVALUATION

### **Metriche:**
- **BLEU-1,2,3,4**: N-gram precision
- **ROUGE-L**: Longest common subsequence
- **METEOR**: Semantic similarity
- **CIDEr**: Consensus-based evaluation
- **CLIP Score**: Visual-semantic alignment

### **Confronti:**
1. **Text-based vs Multimodal**: Stesso modello, input diverso
2. **Baseline vs Trained**: Confronto con modelli baseline
3. **Gemma vs Llama**: Confronto architetture

## 🚨 TROUBLESHOOTING

### **Problema: Dimensioni non compatibili**
```python
# Soluzione: Verifica dimensioni
print(f"Encoder dim: {encoder_output.shape}")
print(f"LLM dim: {model.llm.config.hidden_size}")
print(f"Adapter input: {model.adapter.encoder_dim}")
print(f"Adapter output: {model.adapter.llm_dim}")
```

### **Problema: Out of Memory**
```python
# Soluzioni:
# 1. Riduci batch size
# 2. Usa gradient checkpointing
# 3. Usa mixed precision
model.llm.gradient_checkpointing_enable()
```

### **Problema: Poor Performance**
```python
# Soluzioni:
# 1. Verifica learning rate
# 2. Aumenta training epochs
# 3. Usa warmup
# 4. Verifica data quality
```

## 📞 COORDINAMENTO CON LEONARDO

### **Checklist Materiali:**
- [ ] **image_encoder.pth** - Pesi encoder
- [ ] **projection_layer.pth** - Layer proiezione (se presente)
- [ ] **config.json** - Configurazione encoder
- [ ] **train_embeddings.pkl** - Embedding training set
- [ ] **test_embeddings.pkl** - Embedding test set
- [ ] **embedding_metadata.json** - Metadati formato
- [ ] **encoder_model.py** - Codice encoder
- [ ] **usage_example.py** - Esempio utilizzo

### **Informazioni da Chiarire:**
- [ ] **Dimensione output encoder**
- [ ] **Formato embedding** (shape, dtype)
- [ ] **Preprocessing necessario**
- [ ] **Strategia integrazione preferita**
- [ ] **Performance target**

---

**🎯 READY FOR MULTIMODAL INTEGRATION!** 🤖✨
