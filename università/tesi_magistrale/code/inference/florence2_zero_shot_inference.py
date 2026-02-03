#!/usr/bin/env python3
"""
Script per inferenza zero-shot con Florence-2-base
Calcola metriche BLEU-1/4, METEOR, ROUGE-L, CLIPScore
"""

import os
import sys
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import numpy as np

# Transformers e Florence-2
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
from io import BytesIO

# Metriche
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# CLIPScore
from transformers import CLIPProcessor, CLIPModel
import cairosvg
import gc

# Download NLTK data
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Florence2ZeroShotEvaluator:
    """Valutatore zero-shot per Florence-2-base."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Dispositivo: {self.device}")
        
        # Inizializza modelli
        self.florence_model = None
        self.florence_processor = None
        self.clip_model = None
        self.clip_processor = None
        
        # Metriche testuali
        self.smoothing = SmoothingFunction().method1
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Statistiche
        self.stats = {
            'total_examples': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'inference_times': [],
            'caption_lengths': []
        }
    
    def load_florence2_model(self):
        """Carica Florence-2-base."""
        logger.info("📥 Caricamento Florence-2-base...")
        
        try:
            model_name = "microsoft/Florence-2-base"
            
            self.florence_processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(self.device)
            
            logger.info("✅ Florence-2-base caricato con successo")
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento Florence-2: {e}")
            raise
    
    def load_clip_model(self):
        """Carica CLIP per CLIPScore."""
        logger.info("📥 Caricamento CLIP...")
        
        try:
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            logger.info("✅ CLIP caricato con successo")
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento CLIP: {e}")
            raise
    
    def svg_to_image(self, svg_content: str) -> Image.Image:
        """Converte SVG in immagine PIL."""
        try:
            # Renderizza SVG come PNG
            png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
            image = Image.open(BytesIO(png_data)).convert('RGB')
            return image
        except Exception as e:
            logger.warning(f"⚠️  Errore conversione SVG: {e}")
            # Fallback: immagine bianca
            return Image.new('RGB', (224, 224), 'white')
    
    def generate_caption(self, svg_content: str) -> Tuple[str, float]:
        """Genera caption per SVG con Florence-2."""
        try:
            start_time = datetime.now()
            
            # Converte SVG in immagine
            image = self.svg_to_image(svg_content)
            
            # Prepara input per Florence-2
            prompt = "<CAPTION>"
            inputs = self.florence_processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            # Genera caption
            with torch.no_grad():
                generated_ids = self.florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=100,
                    num_beams=3,
                    do_sample=False,
                    temperature=1.0
                )
            
            # Decodifica risultato
            generated_text = self.florence_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # Post-processing per estrarre caption
            parsed_answer = self.florence_processor.post_process_generation(
                generated_text, 
                task="<CAPTION>", 
                image_size=(image.width, image.height)
            )
            
            caption = parsed_answer.get('<CAPTION>', 'No caption generated')
            
            # Calcola tempo inferenza
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Cleanup
            del inputs, generated_ids
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return caption, inference_time
            
        except Exception as e:
            logger.error(f"❌ Errore generazione caption: {e}")
            return "Error generating caption", 0.0
    
    def calculate_text_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calcola metriche testuali."""
        metrics = {}
        
        try:
            # BLEU scores
            ref_tokens = reference.lower().split()
            pred_tokens = prediction.lower().split()
            
            if len(pred_tokens) > 0:
                # BLEU-1
                metrics['bleu_1'] = sentence_bleu(
                    [ref_tokens], pred_tokens, 
                    weights=(1, 0, 0, 0), 
                    smoothing_function=self.smoothing
                )
                
                # BLEU-4
                metrics['bleu_4'] = sentence_bleu(
                    [ref_tokens], pred_tokens, 
                    weights=(0.25, 0.25, 0.25, 0.25), 
                    smoothing_function=self.smoothing
                )
            else:
                metrics['bleu_1'] = 0.0
                metrics['bleu_4'] = 0.0
            
            # METEOR
            try:
                metrics['meteor'] = meteor_score([reference.lower()], prediction.lower())
            except:
                metrics['meteor'] = 0.0
            
            # ROUGE-L
            try:
                rouge_scores = self.rouge_scorer.score(reference, prediction)
                metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
            except:
                metrics['rouge_l'] = 0.0
                
        except Exception as e:
            logger.error(f"❌ Errore calcolo metriche testuali: {e}")
            metrics = {'bleu_1': 0.0, 'bleu_4': 0.0, 'meteor': 0.0, 'rouge_l': 0.0}
        
        return metrics
    
    def calculate_clip_score(self, svg_content: str, caption: str) -> float:
        """Calcola CLIPScore."""
        try:
            # Converte SVG in immagine
            image = self.svg_to_image(svg_content)
            
            # Prepara input per CLIP
            inputs = self.clip_processor(
                text=[caption], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Calcola embeddings
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                clip_score = logits_per_image.item()
            
            # Cleanup
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return clip_score
            
        except Exception as e:
            logger.error(f"❌ Errore calcolo CLIPScore: {e}")
            return 0.0
    
    def load_test_dataset(self, dataset_path: str = None) -> List[Dict]:
        """Carica dataset di test."""
        if dataset_path is None:
            # Usa dataset di default
            dataset_path = '/work/tesi_ediluzio/data/processed/FINAL_CORRECT_RGB/baseline_set_400_RGB.json'
        
        logger.info(f"📖 Caricamento dataset: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Estrai esempi
            if isinstance(data, dict) and 'examples' in data:
                examples = data['examples']
            elif isinstance(data, list):
                examples = data
            else:
                raise ValueError("Formato dataset non riconosciuto")
            
            logger.info(f"✅ Caricati {len(examples)} esempi")
            return examples
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento dataset: {e}")
            return []
    
    def evaluate_zero_shot(self, dataset_path: str = None, max_examples: int = None) -> Dict[str, Any]:
        """Esegue valutazione zero-shot completa."""
        logger.info("🚀 Avvio valutazione zero-shot Florence-2-base")
        
        # Carica modelli
        self.load_florence2_model()
        self.load_clip_model()
        
        # Carica dataset
        examples = self.load_test_dataset(dataset_path)
        if not examples:
            raise ValueError("Nessun esempio nel dataset")
        
        # Limita esempi se richiesto
        if max_examples:
            examples = examples[:max_examples]
            logger.info(f"🔢 Limitato a {max_examples} esempi")
        
        # Inizializza risultati
        results = []
        all_metrics = {
            'bleu_1': [], 'bleu_4': [], 'meteor': [], 'rouge_l': [], 'clip_score': []
        }
        
        # Elabora esempi
        logger.info(f"🔄 Elaborazione {len(examples)} esempi...")
        
        for i, example in enumerate(tqdm(examples, desc="Inferenza zero-shot")):
            try:
                # Verifica che l'esempio abbia tutti i campi necessari
                if 'xml' in example and 'caption' in example:
                    svg_content = example['xml']
                    ground_truth = example['caption']
                elif 'svg_content' in example and 'caption' in example:
                    svg_content = example['svg_content']
                    ground_truth = example['caption']
                else:
                    logger.warning(f"⚠️  Esempio {i} incompleto, saltato")
                    continue
                
                if not svg_content or not ground_truth:
                    logger.warning(f"⚠️  Esempio {i} con contenuto vuoto, saltato")
                    continue
                
                # Genera caption
                generated_caption, inference_time = self.generate_caption(svg_content)
                
                if "Error" in generated_caption:
                    self.stats['failed_generations'] += 1
                    continue
                
                # Calcola metriche testuali
                text_metrics = self.calculate_text_metrics(generated_caption, ground_truth)
                
                # Calcola CLIPScore
                clip_score = self.calculate_clip_score(svg_content, generated_caption)
                
                # Salva risultato
                result = {
                    'example_id': i,
                    'svg_content': svg_content,
                    'ground_truth': ground_truth,
                    'generated_caption': generated_caption,
                    'inference_time': inference_time,
                    'text_metrics': text_metrics,
                    'clip_score': clip_score
                }
                
                results.append(result)
                
                # Aggiungi alle statistiche
                for metric in all_metrics:
                    if metric == 'clip_score':
                        all_metrics[metric].append(clip_score)
                    else:
                        all_metrics[metric].append(text_metrics[metric])
                
                self.stats['successful_generations'] += 1
                self.stats['inference_times'].append(inference_time)
                self.stats['caption_lengths'].append(len(generated_caption.split()))
                
                # Log progresso ogni 10 esempi
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Elaborati {i + 1}/{len(examples)} esempi")
                
            except Exception as e:
                logger.error(f"❌ Errore esempio {i}: {e}")
                self.stats['failed_generations'] += 1
                continue
        
        # Calcola statistiche finali
        final_metrics = self.calculate_final_metrics(all_metrics)
        
        # Prepara risultato finale
        evaluation_result = {
            'model': 'Florence-2-Base-Zero-Shot',
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_examples': len(examples),
                'successful_examples': self.stats['successful_generations'],
                'failed_examples': self.stats['failed_generations'],
                'success_rate': self.stats['successful_generations'] / len(examples) * 100
            },
            'metrics': final_metrics,
            'performance': {
                'avg_inference_time': np.mean(self.stats['inference_times']) if self.stats['inference_times'] else 0,
                'avg_caption_length': np.mean(self.stats['caption_lengths']) if self.stats['caption_lengths'] else 0
            },
            'results': results
        }
        
        # Log riassunto
        self.log_summary(evaluation_result)
        
        return evaluation_result
    
    def calculate_final_metrics(self, all_metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Calcola statistiche finali delle metriche."""
        final_metrics = {}
        
        for metric_name, values in all_metrics.items():
            if values:
                final_metrics[f'{metric_name}_mean'] = float(np.mean(values))
                final_metrics[f'{metric_name}_std'] = float(np.std(values))
                final_metrics[f'{metric_name}_min'] = float(np.min(values))
                final_metrics[f'{metric_name}_max'] = float(np.max(values))
            else:
                final_metrics[f'{metric_name}_mean'] = 0.0
                final_metrics[f'{metric_name}_std'] = 0.0
                final_metrics[f'{metric_name}_min'] = 0.0
                final_metrics[f'{metric_name}_max'] = 0.0
        
        return final_metrics
    
    def log_summary(self, results: Dict[str, Any]):
        """Stampa riassunto dei risultati."""
        logger.info("\n" + "="*80)
        logger.info("🎯 RISULTATI FLORENCE-2-BASE ZERO-SHOT")
        logger.info("="*80)
        
        dataset_info = results['dataset_info']
        metrics = results['metrics']
        performance = results['performance']
        
        logger.info(f"📊 Esempi totali: {dataset_info['total_examples']}")
        logger.info(f"✅ Esempi riusciti: {dataset_info['successful_examples']}")
        logger.info(f"❌ Esempi falliti: {dataset_info['failed_examples']}")
        logger.info(f"📈 Tasso successo: {dataset_info['success_rate']:.1f}%")
        
        logger.info("\n📊 METRICHE:")
        logger.info(f"   BLEU-1: {metrics['bleu_1_mean']:.4f} (±{metrics['bleu_1_std']:.4f})")
        logger.info(f"   BLEU-4: {metrics['bleu_4_mean']:.4f} (±{metrics['bleu_4_std']:.4f})")
        logger.info(f"   METEOR: {metrics['meteor_mean']:.4f} (±{metrics['meteor_std']:.4f})")
        logger.info(f"   ROUGE-L: {metrics['rouge_l_mean']:.4f} (±{metrics['rouge_l_std']:.4f})")
        logger.info(f"   CLIPScore: {metrics['clip_score_mean']:.4f} (±{metrics['clip_score_std']:.4f})")
        
        logger.info("\n⚡ PERFORMANCE:")
        logger.info(f"   Tempo medio inferenza: {performance['avg_inference_time']:.3f}s")
        logger.info(f"   Lunghezza media caption: {performance['avg_caption_length']:.1f} parole")
        
        logger.info("="*80)
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> str:
        """Salva risultati su file."""
        if output_dir is None:
            output_dir = '/work/tesi_ediluzio/evaluation_results'
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'florence2_zero_shot_results_{timestamp}.json')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Risultati salvati in: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Errore salvataggio: {e}")
            raise


def main():
    """Funzione principale."""
    logger.info("🚀 Avvio inferenza zero-shot Florence-2-base")
    
    try:
        # Inizializza valutatore
        evaluator = Florence2ZeroShotEvaluator()
        
        # Esegui valutazione (limita a 50 esempi per test)
        results = evaluator.evaluate_zero_shot(max_examples=50)
        
        # Salva risultati
        output_file = evaluator.save_results(results)
        
        logger.info(f"🎉 Valutazione completata! Risultati in: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Errore durante valutazione: {e}")
        raise


if __name__ == '__main__':
    main()