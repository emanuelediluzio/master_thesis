import subprocess
import json
import re

# Comprehensive Mapping of Text Sections to Server Verification Commands
VERIFICATION_MAP = {
    # --- CHAPTER 4 ---
    "4.1 System Architecture": {
        "cmd": "ls -F /work/tesi_ediluzio",
        "expect": "SPE/",
        "desc": "Verify modular directory structure"
    },
    "4.1.1 Dataset Construction": {
        "cmd": "ls /work/tesi_ediluzio/convert_90k_dataset.py",
        "expect": "convert_90k_dataset.py",
        "desc": "Verify 90k dataset script existence"
    },
    "4.2 SVG Parsing": {
        "cmd": "grep 'path_pattern' /work/tesi_ediluzio/svg_multipath_preprocessor.py",
        "expect": "path_pattern",
        "desc": "Verify Regex tokenization logic"
    },
    "4.3 Dynamic Padding": {
        "cmd": "grep -r 'collate' /work/tesi_ediluzio/SPE/src/utils",
        "expect": "collate",
        "desc": "Verify collation/padding logic"
    },
    "4.4 SPE Module": {
        "cmd": "head -n 20 /work/tesi_ediluzio/SPE/src/models/spe.py",
        "expect": "class SPE",
        "desc": "Verify SPE class definition"
    },
    "4.5.1 Loading LLM": {
        "cmd": "grep 'AutoModelForCausalLM' /work/tesi_ediluzio/tesi_magistrale/captioning\\ SVG/IPOTESI/nuova\\ versione/codice/utils/advanced_training.py",
        "expect": "AutoModelForCausalLM",
        "desc": "Verify HuggingFace model loading"
    },
    "4.5.2 Injecting LoRA": {
        "cmd": "cat /work/tesi_ediluzio/configs/qwen2_lora_spe_local_config.json",
        "expect": "lora_config",
        "desc": "Verify LoRA configuration block"
    },
    "4.5.5 Visual Metric (CLIP)": {
        "cmd": "grep 'CLIPModel' /work/tesi_ediluzio/calculate_clip_scores_all_models.py",
        "expect": "CLIPModel",
        "desc": "Verify CLIP implementation"
    },
    "4.6 Challenges (Gradient)": {
        "cmd": "grep 'clip_grad_norm' /work/tesi_ediluzio/tesi_magistrale/captioning\\ SVG/IPOTESI/nuova\\ versione/codice/utils/advanced_training.py",
        "expect": "clip_grad_norm",
        "desc": "Verify gradient clipping"
    },
    "4.6.4 Hyperparameters": {
        "cmd": "cat /work/tesi_ediluzio/configs/qwen2_lora_spe_local_config.json",
        "expect": "\"r\": 16",
        "desc": "Verify Rank 16 (Corrected from 8)"
    },

    # --- CHAPTER 5 ---
    "5.1.1 Dataset Curation": {
        "cmd": "cat /work/tesi_ediluzio/configs/qwen2_lora_spe_local_config.json",
        "expect": "qwen2_svg_train.jsonl",
        "desc": "Verify dataset path in config"
    },
    "5.1.3 Evaluation Metrics": {
        "cmd": "head -n 1 /work/tesi_ediluzio/CONFRONTO_METRICHE.csv",
        "expect": "CLIPScore,BLEU-1",
        "desc": "Verify CSV headers match metrics"
    },
    "5.3 Quantitative Results": {
        "cmd": "grep 'SPE+Qwen2' /work/tesi_ediluzio/CONFRONTO_METRICHE.csv",
        "expect": "8.89",
        "desc": "Verify SPE+Qwen2 score data"
    },
    "5.5 Qualitative Analysis": {
        "cmd": "grep -E 'sample_id\": 38|sample_id\": 22' /work/tesi_ediluzio/tesi_magistrale/visual_benchmark_samples.json",
        "expect": "sample_id",
        "desc": "Verify existence of analyzed samples"
    },

    # --- CHAPTER 6 ---
    "6.3 Normalization Findings": {
        "cmd": "grep 'mean' /work/tesi_ediluzio/SPE/src/models/spe.py",
        "expect": "cls_vec - mean",
        "desc": "Verify Z-score normalization logic"
    },
    "6.4 Limitations (OCR)": {
        "cmd": "ls /work/tesi_ediluzio/SPE/src/models/",
        "expect": "spe.py", 
        "not_expect": "ocr.py",
        "desc": "Verify absence of OCR module"
    },
    "6.4 Limitations (Style)": {
        "cmd": "cat /work/tesi_ediluzio/SPE/src/models/base.py",
        "expect": "nn.Embedding",
        "not_expect": "ColorHistogram",
        "desc": "Verify simple discrete embedding (no style encoder)"
    }
}

def run_verification():
    results = []
    print("Starting Deep Verification Suite...")
    
    for section, check in VERIFICATION_MAP.items():
        cmd = f"sshpass -p 'Fulmine88!' ssh -o StrictHostKeyChecking=no ediluzio@ailb-login-03.ing.unimore.it \"{check['cmd']}\""
        
        try:
            # Run SSH command
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            output = result.stdout + result.stderr
            
            # Check expectations
            passed = True
            if check['expect'] not in output:
                passed = False
            if 'not_expect' in check and check['not_expect'] in output:
                passed = False
                
            status = "✅ PASS" if passed else "❌ FAIL"
            results.append(f"| {section} | {check['desc']} | `{check['cmd'][:40]}...` | {status} |")
            
            if not passed:
                print(f"FAILED: {section}")
                print(f"Output: {output[:200]}...")
            else:
                print(f"Verified: {section}")
                
        except Exception as e:
            results.append(f"| {section} | {check['desc']} | ERROR | ❌ FAIL |")
            print(f"Error checking {section}: {e}")

    # Generate Report
    report = "# Final Section-by-Section Verification Report\n\n"
    report += "| Section | Claim/Check | Command | Status |\n"
    report += "|---|---|---|---|\n"
    report += "\n".join(results)
    
    with open("final_verification_result.md", "w") as f:
        f.write(report)
        
    print("\nVerification Complete. Report generated.")

if __name__ == "__main__":
    run_verification()
