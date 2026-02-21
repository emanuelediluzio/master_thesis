#!/usr/bin/env python3
"""
Script semplificato per calcolare metriche testuali
BLEU, METEOR, ROUGE-L usando solo NLTK
"""

import json
import sys
import os
import argparse
from datetime import datetime
import numpy as np
from typing import List, Dict, Any

# NLTK imports
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class SimpleMetricsCalculator:
    def __init__(self):
        self.smoothing = SmoothingFunction().method1
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        print("✅ Metriche inizializzate")
    
    def calculate_bleu_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calcola BLEU-1, BLEU-2, BLEU-3, BLEU-4"""
        if not prediction.strip() or not reference.strip():
            return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        
        if len(pred_tokens) == 0:
            return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        
        try:
            bleu_1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
            bleu_2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
            bleu_3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing)
            bleu_4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
            
            return {
                'bleu_1': float(bleu_1),
                'bleu_2': float(bleu_2),
                'bleu_3': float(bleu_3),
                'bleu_4': float(bleu_4)
            }
        except Exception as e:
            print(f"⚠️ Errore BLEU: {e}")
            return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
    
    def calculate_meteor(self, prediction: str, reference: str) -> float:
        """Calcola METEOR score"""
        if not prediction.strip() or not reference.strip():
            return 0.0
        
        try:
            return meteor_score([reference.lower()], prediction.lower())
        except Exception as e:
            print(f"⚠️ Errore METEOR: {e}")
            return 0.0
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calcola ROUGE scores"""
        if not prediction.strip() or not reference.strip():
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                'rouge1': float(scores['rouge1'].fmeasure),
                'rouge2': float(scores['rouge2'].fmeasure),
                'rougeL': float(scores['rougeL'].fmeasure)
            }
        except Exception as e:
            print(f"⚠️ Errore ROUGE: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_metrics_for_file(self, input_file: str, output_file: str):
        """Calcola metriche per un file JSON"""
        print(f"📊 Calcolo metriche per: {input_file}")
        
        # Carica dati
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data['results']
        print(f"📄 Processando {len(results)} esempi...")
        
        # Liste per raccogliere tutte le metriche
        all_bleu_1, all_bleu_2, all_bleu_3, all_bleu_4 = [], [], [], []
        all_meteor = []
        all_rouge1, all_rouge2, all_rougeL = [], [], []
        
        valid_examples = 0
        
        for i, result in enumerate(results):
            prediction = result.get('generated_caption', '').strip()
            reference = result.get('ground_truth', '').strip()
            
            if not prediction or not reference:
                print(f"⚠️ Esempio {i}: caption o riferimento vuoto")
                continue
            
            valid_examples += 1
            
            # BLEU scores
            bleu_scores = self.calculate_bleu_scores(prediction, reference)
            all_bleu_1.append(bleu_scores['bleu_1'])
            all_bleu_2.append(bleu_scores['bleu_2'])
            all_bleu_3.append(bleu_scores['bleu_3'])
            all_bleu_4.append(bleu_scores['bleu_4'])
            
            # METEOR
            meteor = self.calculate_meteor(prediction, reference)
            all_meteor.append(meteor)
            
            # ROUGE
            rouge_scores = self.calculate_rouge(prediction, reference)
            all_rouge1.append(rouge_scores['rouge1'])
            all_rouge2.append(rouge_scores['rouge2'])
            all_rougeL.append(rouge_scores['rougeL'])
            
            if (i + 1) % 10 == 0:
                print(f"📈 Processati {i + 1}/{len(results)} esempi")
        
        # Calcola statistiche
        def calc_stats(scores):
            if not scores:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            return {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
        
        # Risultati finali
        metrics = {
            'model_info': {
                'model': data.get('model', 'Unknown'),
                'checkpoint': data.get('checkpoint', 'Unknown'),
                'total_examples': len(results),
                'valid_examples': valid_examples,
                'quantized': data.get('quantized', False)
            },
            'text_metrics': {
                'bleu_1': calc_stats(all_bleu_1),
                'bleu_2': calc_stats(all_bleu_2),
                'bleu_3': calc_stats(all_bleu_3),
                'bleu_4': calc_stats(all_bleu_4),
                'meteor': calc_stats(all_meteor),
                'rouge1': calc_stats(all_rouge1),
                'rouge2': calc_stats(all_rouge2),
                'rougeL': calc_stats(all_rougeL)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Salva risultati
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Log riassunto
        print(f"\n📈 RIASSUNTO METRICHE ({data.get('model', 'Unknown')}):")
        print(f"   Esempi validi: {valid_examples}/{len(results)}")
        print(f"   BLEU-1: {metrics['text_metrics']['bleu_1']['mean']:.4f} (±{metrics['text_metrics']['bleu_1']['std']:.4f})")
        print(f"   BLEU-4: {metrics['text_metrics']['bleu_4']['mean']:.4f} (±{metrics['text_metrics']['bleu_4']['std']:.4f})")
        print(f"   METEOR: {metrics['text_metrics']['meteor']['mean']:.4f} (±{metrics['text_metrics']['meteor']['std']:.4f})")
        print(f"   ROUGE-L: {metrics['text_metrics']['rougeL']['mean']:.4f} (±{metrics['text_metrics']['rougeL']['std']:.4f})")
        print(f"✅ Metriche salvate in: {output_file}")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Calcola metriche testuali semplici')
    parser.add_argument('--input', required=True, help='File JSON di input')
    parser.add_argument('--output', required=True, help='File JSON di output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ File non trovato: {args.input}")
        sys.exit(1)
    
    try:
        calculator = SimpleMetricsCalculator()
        calculator.calculate_metrics_for_file(args.input, args.output)
        print("🎉 Calcolo completato!")
    except Exception as e:
        print(f"❌ Errore: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()