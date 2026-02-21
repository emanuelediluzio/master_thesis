# 📊 Report Progresso Training - 4 Modelli

*Generato il: 8 Gennaio 2025*

## 🎯 Panoramica Generale

Questo report documenta il progresso del training per i 4 modelli principali del progetto di SVG captioning:

1. **Gemma-T9 2GPU** (Non quantizzato)
2. **Llama-T8 2GPU** (Non quantizzato) 
3. **Gemma-T9 Quantizzato** (4-bit)
4. **Llama-T8 Quantizzato** (4-bit)

---

## 📈 Dettagli per Modello

### 1. 🔥 **Gemma-T9 2GPU** (Non Quantizzato)

**📍 Status**: ✅ **COMPLETATO**

| Parametro | Valore |
|-----------|--------|
| **Step Finale** | 30,000 |
| **Training Loss Finale** | 0.3976 |
| **Eval Loss Finale** | N/A (non registrata) |
| **Epoche Completate** | 2.67 |
| **Learning Rate Finale** | 6.63e-11 |
| **Gradient Norm Finale** | 0.476 |

**⚙️ Configurazione Training**:
- `num_train_epochs`: 3
- `save_steps`: 250
- `eval_steps`: 500
- `max_steps`: 30,000
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 1
- `learning_rate`: 2e-4
- `lr_scheduler_type`: "cosine"

**🔧 Configurazione LoRA**:
- `r`: 16
- `lora_alpha`: 32
- `lora_dropout`: 0.1
- `target_modules`: ["q_proj", "k_proj", "v_proj", "o_proj"]

**📁 Checkpoint Path**: `/work/tesi_ediluzio/experiments/xml_direct_input/outputs/gemma_t9_2gpu_continue_30k_fixed/checkpoint-30000`

---

### 2. 🚀 **Llama-T8 2GPU** (Non Quantizzato)

**📍 Status**: ✅ **COMPLETATO**

| Parametro | Valore |
|-----------|--------|
| **Step Finale** | 30,000 |
| **Training Loss Finale** | 0.6582 |
| **Eval Loss Finale** | N/A (non registrata) |
| **Epoche Completate** | 2.67 |
| **Learning Rate Finale** | 4.10e-11 |
| **Gradient Norm Finale** | 0.707 |

**⚙️ Configurazione Training**:
- `num_train_epochs`: 3
- `save_steps`: 250
- `eval_steps`: 500
- `max_steps`: 30,000
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 1
- `learning_rate`: 2e-4
- `lr_scheduler_type`: "cosine"

**🔧 Configurazione LoRA**:
- `r`: 16
- `lora_alpha`: 32
- `lora_dropout`: 0.1
- `target_modules`: ["q_proj", "k_proj", "v_proj", "o_proj"]

**📁 Checkpoint Path**: `/work/tesi_ediluzio/experiments/xml_direct_input/outputs/llama_t8_2gpu_continue_30k_fixed/checkpoint-30000`

---

### 3. ⚡ **Gemma-T9 Quantizzato** (4-bit)

**📍 Status**: ⚠️ **INTERROTTO** (Training non completato)

| Parametro | Valore |
|-----------|--------|
| **Step Raggiunto** | 5,000 / 30,000 |
| **Training Loss** | 0.4505 |
| **Eval Loss** | N/A (non registrata) |
| **Epoche Completate** | 0.44 |
| **Learning Rate** | 9.87e-5 |
| **Gradient Norm** | 0.436 |
| **Progresso** | 16.7% |

**⚙️ Configurazione Training**:
- `num_train_epochs`: 3
- `save_steps`: 250
- `eval_steps`: 500
- `max_steps`: 30,000
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 8
- `learning_rate`: 1e-4

**🔧 Configurazione LoRA**:
- `r`: 8
- `lora_alpha`: 16
- `lora_dropout`: 0.05
- `target_modules`: ["q_proj", "k_proj", "v_proj", "o_proj"]

**🔧 Quantizzazione**:
- `load_in_4bit`: true
- `bnb_4bit_compute_dtype`: "float16"
- `bnb_4bit_quant_type`: "nf4"
- `bnb_4bit_use_double_quant`: true

**📁 Checkpoint Path**: `/work/tesi_ediluzio/experiments/xml_direct_input/outputs/gemma_t9_quantized_continue_30k_fixed/checkpoint-5000`

---

### 4. ⚡ **Llama-T8 Quantizzato** (4-bit)

**📍 Status**: ⚠️ **INTERROTTO** (Training non completato)

| Parametro | Valore |
|-----------|--------|
| **Step Raggiunto** | 3,500 / 30,000 |
| **Training Loss** | 0.7556 |
| **Eval Loss** | N/A (non registrata) |
| **Epoche Completate** | 0.31 |
| **Learning Rate** | 9.99e-5 |
| **Gradient Norm** | 0.397 |
| **Progresso** | 11.7% |

**⚙️ Configurazione Training**:
- `num_train_epochs`: 3
- `save_steps`: 250
- `eval_steps`: 500
- `max_steps`: 30,000
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 8
- `learning_rate`: 1e-4

**🔧 Configurazione LoRA**:
- `r`: 8
- `lora_alpha`: 16
- `lora_dropout`: 0.05
- `target_modules`: ["q_proj", "k_proj", "v_proj", "o_proj"]

**🔧 Quantizzazione**:
- `load_in_4bit`: true
- `bnb_4bit_compute_dtype`: "float16"
- `bnb_4bit_quant_type`: "nf4"
- `bnb_4bit_use_double_quant`: true

**📁 Checkpoint Path**: `/work/tesi_ediluzio/experiments/xml_direct_input/outputs/llama_t8_quantized_continue_30k_fixed/checkpoint-3500`

---

## 📊 Riepilogo Comparativo

| Modello | Status | Steps | Training Loss | Epoche | Progresso |
|---------|--------|-------|---------------|--------|-----------|
| **Gemma-T9 2GPU** | ✅ Completato | 30,000 | 0.3976 | 2.67 | 100% |
| **Llama-T8 2GPU** | ✅ Completato | 30,000 | 0.6582 | 2.67 | 100% |
| **Gemma-T9 Quantizzato** | ⚠️ Interrotto | 5,000 | 0.4505 | 0.44 | 16.7% |
| **Llama-T8 Quantizzato** | ⚠️ Interrotto | 3,500 | 0.7556 | 0.31 | 11.7% |

## 🎯 Osservazioni Chiave

### ✅ **Modelli Completati (2GPU)**
- **Gemma-T9**: Ottima convergenza con loss finale di 0.3976
- **Llama-T8**: Convergenza buona con loss finale di 0.6582
- Entrambi hanno completato 2.67 epoche su 3 pianificate
- Learning rate ridotto a valori molto bassi (cosine decay)

### ⚠️ **Modelli Quantizzati (Interrotti)**
- **Gemma-T9 Quantizzato**: Interrotto al 16.7% (5,000/30,000 steps)
- **Llama-T8 Quantizzato**: Interrotto al 11.7% (3,500/30,000 steps)
- Loss ancora in fase di convergenza al momento dell'interruzione
- Configurazione LoRA più conservativa (r=8 vs r=16)

### 📈 **Differenze di Configurazione**
- **Non Quantizzati**: LoRA r=16, alpha=32, dropout=0.1, LR=2e-4
- **Quantizzati**: LoRA r=8, alpha=16, dropout=0.05, LR=1e-4
- **Batch Size**: Tutti usano batch_size=1
- **Gradient Accumulation**: 1 per 2GPU, 8 per quantizzati

## 🔄 Raccomandazioni

1. **Completare i modelli quantizzati** se necessario per il confronto
2. **Analizzare le performance** dei modelli completati
3. **Valutare l'efficacia** della quantizzazione vs performance
4. **Considerare eval_loss** per future sessioni di training

---

*Report generato automaticamente dal sistema di monitoraggio training*