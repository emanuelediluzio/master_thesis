# Mappa dei modelli elencati in `CONFRONTO_METRICHE.csv`

Questo documento collega ogni modello riportato nel file `CONFRONTO_METRICHE.csv` ai relativi script di training, inferenza, configurazioni e output principali presenti nel repository. Le sezioni sono raggruppate per categoria (Baseline, LoRA, LoRA quantizzati, SPE). Tutti i percorsi sono relativi alla root `/work/tesi_ediluzio`.

---

## Risorse comuni

- Configurazione globale dell’inferenza: `inferenza/configs/inference_config.yaml`
- Piano dei modelli per i benchmark SVG (30 campioni): `inferenza/visual_benchmark_30samples/model_plan.json`
- Pipeline completa inferenza + report: `inferenza/scripts/master_pipeline.py`
- Report HTML aggregato: `multi_model_comparison_root.html`
- Dataset principali citati nei config:
  - Train Qwen/Gemma SPE: `data/jsonl/qwen2_svg_train.jsonl`
  - Train Gemma SPE normalizzato: `data/processed/spe_gemma_training_data_90k.jsonl`

---

## Baseline (Zero-shot)

### BLIP-2
- **Configurazione**: `inferenza/models/model_config.json` (`blip2_opt`)
- **Script inferenza automatizzata**: `inferenza/scripts/automated_inference.py` (branch `load_baseline_model`)
- **Simulatore zero-shot dedicato**: `scripts/blip2_zero_shot_inference.py`
- **Job SLURM**: `scripts/slurm/run_model_inference_blip2.slurm`
- **Output benchmark**: `inferenza/visual_benchmark_30samples/json/BLIP-2_results.json`

### Florence-2
- **Configurazione**: `inferenza/models/model_config.json` (`florence2`)
- **Inferenza baseline**: `inferenza/scripts/automated_inference.py`
- **Script helper**: `scripts/florence2_zero_shot_inference.py`
- **Job SLURM**: `scripts/slurm/run_model_inference_Florence2.slurm`
- **Output benchmark**: `inferenza/visual_benchmark_30samples/json/Florence-2_results.json`

### Idefics3
- **Configurazione**: `inferenza/models/model_config.json` (`idefics3`)
- **Inferenza (caricamento quantizzato)**: `inferenza/scripts/automated_inference.py`, blocco dedicato a Idefics3
- **Job SLURM**: `scripts/slurm/run_model_inference_Idefics3.slurm`
- **Output benchmark**: `inferenza/visual_benchmark_30samples/json/Idefics3_results.json`

### BLIP-1-CPU
- **Configurazione**: `inferenza/models/model_config.json` (`blip1_cpu`)
- **Inferenza CPU**: `inferenza/scripts/automated_inference.py`, sezione `BLIP-1-CPU`
- **Esecuzione batch**: `scripts/inference/run_full_comparison.sh` (funzione `run_model "BLIP-1-CPU"`)
- **Output benchmark**: `inferenza/visual_benchmark_30samples/json/BLIP-1-CPU_results.json`

---

## Modelli LoRA (decoder-only)

### Qwen2-7b
- **Training**
  - Job SLURM: `scripts/train_qwen_lora_spe.sh`
  - Launcher generico: `scripts/training/robust_training_launcher.py`
  - Config principale: `configs/qwen_lora_spe_config.json` (dataset `data/jsonl/qwen2_svg_train.jsonl`)
- **Inferenza / Valutazione**
  - Runner LoRA+SPE: `scripts/inference/spe_qwen2_inference_final.py`
  - Script completo con aggiornamento CSV: `spe_qwen2_v2_complete_evaluation.py`
  - Pipeline automatizzata: `inferenza/scripts/automated_inference.py` (`load_custom_model`)
  - Job SLURM per scenari SPE: `slurm_spe_qwen_missing.sh`, `slurm_spe_qwen_cherry_only.sh`
- **Output principali**
  - Result JSONL finale: `outputs/inference/spe_qwen2_final_results.jsonl`
  - Benchmark 30 campioni: `inferenza/visual_benchmark_30samples/json/SPE_Qwen2-7b_v2_results.json`

### Gemma 9b instruct
- **Training**
  - Job SLURM: `scripts/train_gemma_lora_spe.sh`
  - Launcher: `scripts/training/robust_training_launcher.py`
  - Config LoRA (2 GPU): `configs/gemma_lora_spe_2gpu_config.json`
  - Dataset SPE: `data/processed/spe_gemma_training_data_90k.jsonl`
- **Inferenza / Valutazione**
  - Runner dedicato: `scripts/inference/spe_gemma_inference_final.py`
  - Integrazione pipeline: `comprehensive_inference_system.py` (chiave `spe_gemma_9b_v2`)
  - Job SLURM: `slurm_spe_gemma_inference.sh`
- **Output principali**
  - Result JSONL: `outputs/inference/spe_gemma_final_results.jsonl`
  - Benchmark: `inferenza/visual_benchmark_30samples/json/SPE_Gemma-9b_v2_results.json`

### Llama-8b
- **Training**
  - Config base: `configs/training/llama_config.json` (+ versione ottimizzata `configs/training/llama_config_memory_optimized.json`)
  - Lancio robusto multi-modello: `scripts/slurm/launch_robust_training.sh` (parametri `llama`)
  - Quantizzazione opzionale: `configs/quantization/models_config.json` (entry Llama)
- **Inferenza / Valutazione**
  - Pipeline generica quant/notquant: `scripts/evaluation/COMPLETE_INFERENCE.py`
  - Job benchmark: `slurm_visual_benchmark_boost_lora.sh` (include `Llama-8b`)
- **Output principali**
  - Risultati in `evaluation_results/` (es. `llama_quantized_inference_results.json`)
  - Benchmark 30 campioni: `inferenza/visual_benchmark_30samples/json/Llama-8b_results.json`

---

## Modelli LoRA quantizzati

### Llama-9b-Quantized
- **Config quantizzazione**: `configs/quantization/models_config.json` (voce `meta-llama/Llama-3.1-8B-Instruct`)
- **Setup inferenza quantizzata**: `scripts/evaluation/COMPLETE_INFERENCE.py` (`--quantized`)
- **Tool dedicati**: `scripts/evaluation/evaluate_quantized_models.py`
- **Batch/benchmark**: `slurm_visual_benchmark_boost_lora.sh`
- **Output tipici**: `evaluation_results/llama_quantized_inference_results.json`, benchmark JSON `inferenza/visual_benchmark_30samples/json/Llama-9b-Quantized_results.json`

### Gemma-9b-Quantized
- **Config quantizzazione**: `configs/quantization/models_config.json` (voce `google/gemma-2-9b-it`)
- **Setup inferenza**: `scripts/evaluation/GEMMA_ROBUST_INFERENCE.py` & `scripts/evaluation/evaluate_quantized_models.py`
- **Batch/benchmark**: `slurm_visual_benchmark_boost_lora.sh`
- **Output tipici**: `evaluation_results/gemma_quantized_inference_results.json`, benchmark `inferenza/visual_benchmark_30samples/json/Gemma-9b-Quantized_results.json`

---

## Modelli SPE (LoRA + struttura propria)

### SPE+Qwen2-7b (v1 e v2)
- **Training base**: identico a Qwen2-7b LoRA (vedi sopra) con dataset SPE
- **Stack SPE**: `SPE/src/models/spe_model.py`, `SPE/src/spe_inference.py`, `SPE/src/utils/`
- **Valutazione completa**: `spe_qwen2_v2_complete_evaluation.py` (aggiorna `CONFRONTO_METRICHE.csv`)
- **Monitoraggio checkpoint SPE**: `spe_status_report.json`
- **Outputs**: `outputs/inference/spe_qwen2_final_results.jsonl`, `inferenza/visual_benchmark_30samples/json/SPE_Qwen2-7b_v2_results.json`

### SPE+Gemma 9b instruct / v2
- **Training base**: Gemma LoRA (config SPE sotto `configs/gemma_9b_spe_corrected_config.json`, `configs/gemma_9b_spe_memory_optimized.json`)
- **Stack SPE**: file nella cartella `SPE/src/` (come per Qwen)
- **Inferenza finale**: `scripts/inference/spe_gemma_inference_final.py`
- **Pipeline e benchmark**: `comprehensive_inference_system.py` (chiave `spe_gemma_9b_v2`), `slurm_spe_gemma_inference.sh`
- **Outputs**: `outputs/inference/spe_gemma_final_results.jsonl`, `inferenza/visual_benchmark_30samples/json/SPE_Gemma-9b_v2_results.json`

---

## Log e report di supporto

- Log cluster SLURM (tutti i job): `logs/slurm/` (es. `gemma_cherry_2638134.err`, `spe_qwen_missing_2638095.err`)
- Analisi metriche dettagliate: `detailed_analysis.json`, `CLIP_SCORE_REPORT_INFERENCE.md`, `spe_status_report.html/json`
- Report comparativi finali: `inferenza/html_reports/` e `comprehensive_inference_system.py` (generazione automatica)

---

## Come estendere o ri-eseguire

1. **Verificare il modello nel CSV** e identificare la categoria.
2. **Consultare la sezione corrispondente** in questo documento per trovare:
   - Config di training/inferenza
   - Script Python (locale) o SLURM (cluster) da eseguire
   - Percorsi di output per validare i risultati
3. **Aggiornare eventuali nuovi risultati** attraverso gli script “complete evaluation” (`spe_qwen2_v2_complete_evaluation.py`, ecc.) che si occupano di riscrivere `CONFRONTO_METRICHE.csv`.

Seguendo questa mappa è possibile capire in modo rapido dove intervenire per ripetere training, rilanciare inferenze o analizzare i log associati a ciascun modello monitorato nella tesi.
