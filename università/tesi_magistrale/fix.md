# CORREZIONI TESI: Analisi Discrepanze tra Contenuto e Implementazione

## SOMMARIO ESECUTIVO

Dopo un'analisi approfondita del filesystem, dei dati sperimentali reali (CONFRONTO_METRICHE.csv) e del contenuto della tesi PDF, sono state identificate **GRAVI DISCREPANZE** tra quello che è scritto nella tesi e quello che è stato realmente implementato.

---

## 🚨 PROBLEMI CRITICI IDENTIFICATI

### 1. **CAPITOLO 3 - SYSTEM ARCHITECTURE: COMPLETAMENTE FALSO**

**PROBLEMA**: Il Capitolo 3 descrive un'architettura "dual-path" con Swin Transformer V2 che **NON È MAI STATA IMPLEMENTATA**.

**CONTENUTO FALSO NEL PDF**:
- "Dual-Path Encoder with Swin Transformer V2"
- "Visual Path (Raster Path)" con Swin Transformer V2
- "Rasterization: Conversion of the SVG to high-resolution raster image (512x512 pixels)"
- "Feature Extraction: Use of Swin Transformer V2 for extracting hierarchical visual features"

**REALTÀ IMPLEMENTATA**:
- **SPE (Spatial Position Encoder)** + **LLM (Qwen2-7B/Gemma-9B/Llama-8B)**
- **LoRA fine-tuning** su modelli pre-addestrati
- **Nessun Swin Transformer V2**
- **Nessuna rasterizzazione a 512x512**

### 2. **ARCHITETTURE DESCRITTE VS IMPLEMENTATE**

**ARCHITETTURE REALMENTE IMPLEMENTATE** (dal filesystem e CSV):

#### A) **BASELINE MODELS (Zero-shot)**:
- BLIP-2 (CLIPScore: 31.66, Ranking: 4°)
- Florence-2 (CLIPScore: 31.07, Ranking: 5°)
- Idefics3 (CLIPScore: 23.87, Ranking: 6°)
- BLIP-1-CPU (CLIPScore: 23.37, Ranking: 7°)

#### B) **DECODER-ONLY + LoRA**:
- **Qwen2-7B + LoRA** (CLIPScore: 32.3, Ranking: 3°)
- **Gemma-9B + LoRA** (CLIPScore: 29.68, Ranking: 8°)
- **Llama-8B + LoRA** (CLIPScore: 23.34, Ranking: 9°)

#### C) **SPE + DECODER + LoRA** (MIGLIORI PERFORMANCE):
- **SPE+Qwen2-7B** (CLIPScore: 29.3, **RANKING: 1°**, Punteggio Composito: 8.89)
- **SPE+Gemma-9B** (CLIPScore: 25.2, **RANKING: 2°**, Punteggio Composito: 6.13)

---

## 📋 CORREZIONI SPECIFICHE RICHIESTE

### **CAPITOLO 3 - DA RISCRIVERE COMPLETAMENTE**

#### ❌ **DA RIMUOVERE TUTTO**:
```
- Sezione 3.2 "Dual-Path Encoder with Swin Transformer V2"
- Sezione 3.2.1 "Visual Path (Raster Path)"
- Tutti i riferimenti a Swin Transformer V2
- Descrizioni di rasterizzazione a 512x512
- Architettura "dual-path"
- "Local attention with sliding windows"
- "Computational efficiency: Linear complexity"
- "Multi-scale support"
```

#### ✅ **DA AGGIUNGERE**:
```
3.2 SPE + LLM Architecture
La nostra architettura si basa su due componenti principali:

3.2.1 SPE (Spatial Position Encoder)
- Encoder specializzato per SVG che processa direttamente la struttura vettoriale
- Checkpoint utilizzato: SPE_31/checkpoint-360000
- Dimensione embedding: 1024D
- Vocabolario: 448 token specializzati per elementi SVG

3.2.2 Large Language Models (LLM)
Tre modelli testati con LoRA fine-tuning:
- Qwen2-7B-Instruct (migliori performance con SPE)
- Gemma-9B-Instruct 
- Llama-8B

3.2.3 Multimodal Integration
- Linear projection layer per allineare SPE (1024D) con LLM (4096D)
- LoRA fine-tuning: rank=16, alpha=32, dropout=0.1
- Target layers: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
```

### **RISULTATI SPERIMENTALI - CORREZIONI**

#### ❌ **DA CORREGGERE**:
Qualsiasi tabella o grafico che non rifletta questi dati reali:

#### ✅ **TABELLA CORRETTA**:
```
| Modello | Tipo | CLIPScore | BLEU-1 | METEOR | ROUGE-L | Ranking |
|---------|------|-----------|--------|---------|---------|---------|
| SPE+Qwen2-7B | SPE+LoRA | 29.3 | 0.42 | 0.38 | 0.45 | 1° |
| SPE+Gemma-9B | SPE+LoRA | 25.2 | 0.15 | 0.18 | 0.20 | 2° |
| Qwen2-7B | LoRA | 32.3 | 0.238 | 0.206 | 0.277 | 3° |
| BLIP-2 | Baseline | 31.66 | 0.003 | 0.0478 | 0.1232 | 4° |
| Florence-2 | Baseline | 31.07 | 0.0027 | 0.0603 | 0.1194 | 5° |
```

### **CAPITOLO 4 - IMPLEMENTATION**

#### ❌ **DA RIMUOVERE**:
- Riferimenti a implementazione di Swin Transformer V2
- Codice o pseudocodice per dual-path encoder
- Descrizioni di pipeline di rasterizzazione

#### ✅ **DA AGGIUNGERE**:
```
4.1 SPE Integration
- Utilizzo di checkpoint pre-addestrato SPE_31/checkpoint-360000
- Configurazione projection layer per compatibilità con LLM

4.2 LoRA Fine-tuning Setup
- Configurazione: rank=16, alpha=32, dropout=0.1
- Learning rate: 2e-5 (Qwen2), 2e-4 (Gemma), 5e-5 (Llama)
- Batch size: 1, Gradient accumulation: 16
- Epochs: 3, Warmup steps: 500

4.3 Training Infrastructure
- 2x GPU setup per modelli LoRA
- Mixed precision training (fp16=true)
- Gradient checkpointing per efficienza memoria
```

### **CAPITOLO 5 - RESULTS**

#### ❌ **DA CORREGGERE**:
Qualsiasi affermazione che non rifletta che **SPE+Qwen2-7B è il MIGLIORE** con ranking 1°.

#### ✅ **RISULTATI CORRETTI**:
```
5.1 Performance Analysis
Il modello SPE+Qwen2-7B raggiunge le migliori performance complessive:
- Punteggio composito: 8.89 (1° posto)
- Eccellenza nelle metriche linguistiche: BLEU-1 (0.42), METEOR (0.38), ROUGE-L (0.45)
- Bilanciamento ottimale tra qualità visiva e linguistica

5.2 Architecture Comparison
- SPE+LLM supera LoRA-only in metriche linguistiche
- Baseline models eccellono solo in CLIPScore ma falliscono in generazione testuale
- L'integrazione SPE fornisce comprensione strutturale superiore degli SVG
```

---

## 🔍 **ALTRE DISCREPANZE IDENTIFICATE**

### **Abstract e Introduzione**
- ❌ Rimuovere riferimenti a "dual-path architecture"
- ❌ Rimuovere "Swin Transformer V2 for visual encoding"
- ✅ Sostituire con "SPE+LLM architecture with LoRA fine-tuning"

### **Related Work**
- ✅ Aggiungere sezione su SPE (Spatial Position Encoder)
- ✅ Aggiungere riferimenti a LoRA fine-tuning techniques
- ❌ Rimuovere o ridimensionare sezioni su Vision Transformers non utilizzati

### **Conclusioni**
- ❌ Rimuovere affermazioni su implementazione di architetture non realizzate
- ✅ Enfatizzare il successo dell'approccio SPE+Qwen2-7B
- ✅ Discutere limitazioni e future work basate su implementazione reale

---

## 📊 **DATI REALI DA UTILIZZARE**

### **Performance Metrics Verificati**:
```json
{
  "SPE+Qwen2-7B": {
    "CLIPScore": 29.3,
    "BLEU-1": 0.42,
    "METEOR": 0.38,
    "ROUGE-L": 0.45,
    "Composite_Score": 8.89,
    "Ranking": 1
  },
  "Training_Details": {
    "Examples_Processed": 89963,
    "Epochs": 3,
    "Final_Checkpoint": 16869,
    "Base_Model": "Qwen/Qwen2-7B-Instruct",
    "SPE_Checkpoint": "checkpoint-360000"
  }
}
```

---

## ⚠️ **PRIORITÀ CORREZIONI**

### **URGENTE (da fare subito)**:
1. **Riscrivere completamente Capitolo 3**
2. **Correggere tutte le tabelle con dati reali**
3. **Rimuovere riferimenti a Swin Transformer V2**
4. **Correggere Abstract e Conclusioni**

### **IMPORTANTE**:
1. Aggiornare Related Work
2. Correggere sezioni Implementation
3. Aggiungere dettagli SPE+LoRA

### **OPZIONALE**:
1. Migliorare figure e diagrammi
2. Aggiungere analisi più dettagliate

---

## 📝 **NOTE FINALI**

Questa analisi è basata su:
- **Filesystem reale**: `/work/tesi_ediluzio/`
- **Dati sperimentali verificati**: `CONFRONTO_METRICHE.csv`
- **Codice implementato**: `multimodal_integration/`
- **Configurazioni training**: `lora_continue_convergence.yaml`

**RACCOMANDAZIONE**: Procedere con le correzioni in ordine di priorità per allineare completamente la tesi con l'implementazione reale e i risultati ottenuti.