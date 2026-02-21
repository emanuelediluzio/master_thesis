"""
Modulo di logging condiviso per il progetto di captioning SVG.

Questo modulo fornisce funzionalità di logging comuni utilizzabili
sia dall'implementazione decoder-only che da quella encoder-decoder.
"""

import os
import json
import logging
import csv
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# Configurazione del logger principale
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class BaseLogger:
    """
    Classe base per il logging nel progetto di captioning SVG.
    
    Questa classe fornisce metodi comuni per:
    - Configurare il logging su file e console
    - Registrare metriche di addestramento
    - Esportare metriche in vari formati
    - Integrare con TensorBoard
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        config_path: Optional[str] = None
    ):
        """
        Inizializza il logger.
        
        Args:
            log_dir: Directory per i file di log
            experiment_name: Nome dell'esperimento (opzionale)
            use_tensorboard: Se True, utilizza TensorBoard per il logging
            config_path: Percorso al file di configurazione JSON (opzionale)
        """
        # Crea la directory di log
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Carica la configurazione
        self.config = self._load_config(config_path)
        
        # Imposta il nome dell'esperimento
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"
        
        # Configura il logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Aggiungi un handler per il file
        log_file = os.path.join(self.log_dir, f"{self.experiment_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Inizializza il writer TensorBoard se richiesto
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.tb_log_dir = os.path.join(self.log_dir, "tensorboard", self.experiment_name)
            os.makedirs(self.tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tb_log_dir)
            self.logger.info(f"TensorBoard inizializzato in: {self.tb_log_dir}")
        
        # Inizializza il dizionario per le metriche
        self.metrics = {
            "steps": []
        }
        
        # Inizializza il file CSV per le metriche
        self.csv_file = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.csv")
        self.csv_header = ["Step"]
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)
        
        self.logger.info(f"BaseLogger inizializzato con output in: {self.log_dir}")
        self.start_time = time.time()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Carica la configurazione da un file JSON.
        
        Args:
            config_path: Percorso al file di configurazione
            
        Returns:
            Dizionario con la configurazione
        """
        default_config = {
            "logging": {
                "log_frequency": 100,
                "save_frequency": 1000,
                "metrics_to_track": ["loss"]
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Configurazione caricata da: {config_path}")
                return config
            except Exception as e:
                self.logger.error(f"Errore nel caricamento della configurazione: {e}")
                return default_config
        else:
            self.logger.warning("File di configurazione non trovato, utilizzo configurazione predefinita")
            return default_config
    
    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = ""
    ):
        """
        Registra le metriche per un passo di addestramento.
        
        Args:
            step: Passo di addestramento corrente
            metrics: Dizionario con le metriche (nome -> valore)
            prefix: Prefisso per i nomi delle metriche (opzionale)
        """
        # Verifica se è il momento di registrare le metriche
        log_frequency = self.config.get("logging", {}).get("log_frequency", 100)
        if step % log_frequency != 0:
            return
        
        # Registra il passo
        if step not in self.metrics["steps"]:
            self.metrics["steps"].append(step)
        
        # Registra le metriche
        for name, value in metrics.items():
            metric_name = f"{prefix}{name}" if prefix else name
            
            # Aggiungi la metrica al dizionario se non esiste
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
                
                # Aggiorna l'intestazione del CSV
                self.csv_header.append(metric_name)
                
                # Riscrive l'intestazione del CSV
                with open(self.csv_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                
                rows[0] = self.csv_header
                
                with open(self.csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
            
            # Aggiungi il valore alla lista
            self.metrics[metric_name].append(value)
            
            # Registra in TensorBoard
            if self.use_tensorboard:
                self.writer.add_scalar(f"Metrics/{metric_name}", value, step)
            
            self.logger.info(f"Step {step}: {metric_name} = {value:.6f}")
        
        # Aggiorna il file CSV
        self._update_csv(step, metrics, prefix)
        
        # Calcola e registra il tempo trascorso
        elapsed_time = time.time() - self.start_time
        if self.use_tensorboard:
            self.writer.add_scalar("Time/elapsed_hours", elapsed_time / 3600, step)
        
        # Forza il flush del writer TensorBoard
        if self.use_tensorboard:
            self.writer.flush()
    
    def _update_csv(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = ""
    ):
        """
        Aggiorna il file CSV con le metriche.
        
        Args:
            step: Passo di addestramento corrente
            metrics: Dizionario con le metriche (nome -> valore)
            prefix: Prefisso per i nomi delle metriche (opzionale)
        """
        # Prepara la riga da aggiungere
        row = [step]
        
        # Aggiungi i valori delle metriche nell'ordine dell'intestazione
        for name in self.csv_header[1:]:  # Salta "Step"
            # Rimuovi il prefisso dal nome della metrica se presente
            if prefix and name.startswith(prefix):
                metric_name = name[len(prefix):]
            else:
                metric_name = name
            
            # Aggiungi il valore se disponibile, altrimenti una stringa vuota
            if metric_name in metrics:
                row.append(metrics[metric_name])
            else:
                row.append("")
        
        # Aggiungi la riga al file CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_histogram(
        self,
        name: str,
        values: torch.Tensor,
        step: int
    ):
        """
        Registra un istogramma in TensorBoard.
        
        Args:
            name: Nome dell'istogramma
            values: Valori per l'istogramma
            step: Passo di addestramento corrente
        """
        if self.use_tensorboard:
            self.writer.add_histogram(name, values, step)
            self.logger.info(f"Step {step}: Istogramma {name} registrato")
    
    def log_model_gradients(
        self,
        model: torch.nn.Module,
        step: int
    ):
        """
        Registra gli istogrammi dei gradienti del modello.
        
        Args:
            model: Modello con gradienti
            step: Passo di addestramento corrente
        """
        if self.use_tensorboard:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, step)
            self.logger.info(f"Step {step}: Gradienti del modello registrati")
    
    def log_model_weights(
        self,
        model: torch.nn.Module,
        step: int
    ):
        """
        Registra gli istogrammi dei pesi del modello.
        
        Args:
            model: Modello con pesi
            step: Passo di addestramento corrente
        """
        if self.use_tensorboard:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f"Weights/{name}", param, step)
            self.logger.info(f"Step {step}: Pesi del modello registrati")
    
    def plot_metrics(self, save: bool = True) -> Dict[str, plt.Figure]:
        """
        Visualizza le metriche registrate.
        
        Args:
            save: Se True, salva le visualizzazioni
            
        Returns:
            Dizionario con le figure matplotlib
        """
        figures = {}
        
        # Verifica che ci siano dati sufficienti
        if len(self.metrics["steps"]) < 2:
            self.logger.warning("Dati insufficienti per visualizzare le metriche")
            return figures
        
        # Raggruppa le metriche per prefisso
        metric_groups = {}
        for name in self.metrics:
            if name == "steps":
                continue
            
            # Estrai il prefisso (tutto prima del primo '/')
            if '/' in name:
                prefix = name.split('/')[0]
            else:
                prefix = "Metrics"
            
            if prefix not in metric_groups:
                metric_groups[prefix] = []
            
            metric_groups[prefix].append(name)
        
        # Visualizza ogni gruppo di metriche
        for prefix, metric_names in metric_groups.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for name in metric_names:
                ax.plot(self.metrics["steps"], self.metrics[name], label=name)
            
            ax.set_xlabel("Passi di addestramento")
            ax.set_ylabel("Valore")
            ax.set_title(f"Evoluzione delle metriche - {prefix}")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            
            figures[prefix] = fig
            
            if save:
                filepath = os.path.join(self.log_dir, f"{self.experiment_name}_{prefix.lower()}.png")
                fig.savefig(filepath, dpi=300, bbox_inches="tight")
                self.logger.info(f"Grafico delle metriche salvato in: {filepath}")
        
        return figures
    
    def export_metrics(self, filepath: Optional[str] = None) -> str:
        """
        Esporta le metriche registrate in formato JSON.
        
        Args:
            filepath: Percorso del file di output (opzionale)
            
        Returns:
            Percorso del file salvato
        """
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")
        
        # Converti i dati in liste Python
        export_data = {}
        for name, values in self.metrics.items():
            export_data[name] = [float(x) if isinstance(x, (int, float, np.number)) else x for x in values]
        
        # Salva i dati
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metriche esportate in: {filepath}")
        return filepath
    
    def close(self):
        """
        Chiude il logger e il writer TensorBoard.
        """
        if self.use_tensorboard:
            self.writer.close()
        
        # Esporta le metriche
        self.export_metrics()
        
        # Visualizza e salva i grafici
        self.plot_metrics(save=True)
        
        self.logger.info("Logger chiuso")


class TrainingLogger(BaseLogger):
    """
    Logger specializzato per l'addestramento di modelli.
    
    Questa classe estende BaseLogger con metodi specifici per:
    - Registrare metriche di addestramento e validazione
    - Monitorare il tempo di addestramento
    - Tracciare le prestazioni del modello
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        config_path: Optional[str] = None
    ):
        """
        Inizializza il logger di addestramento.
        
        Args:
            log_dir: Directory per i file di log
            experiment_name: Nome dell'esperimento (opzionale)
            use_tensorboard: Se True, utilizza TensorBoard per il logging
            config_path: Percorso al file di configurazione JSON (opzionale)
        """
        super().__init__(log_dir, experiment_name, use_tensorboard, config_path)
        
        # Inizializza i contatori per le epoche e i batch
        self.epoch = 0
        self.batch = 0
        
        # Inizializza i timer
        self.epoch_start_time = None
        self.batch_start_time = None
        
        self.logger.info("TrainingLogger inizializzato")
    
    def start_epoch(self, epoch: int):
        """
        Inizia una nuova epoca di addestramento.
        
        Args:
            epoch: Numero dell'epoca
        """
        self.epoch = epoch
        self.epoch_start_time = time.time()
        self.logger.info(f"Inizio epoca {epoch}")
    
    def end_epoch(self, metrics: Dict[str, float]):
        """
        Termina un'epoca di addestramento.
        
        Args:
            metrics: Metriche dell'epoca
        """
        # Calcola il tempo trascorso
        epoch_time = time.time() - self.epoch_start_time
        
        # Registra le metriche
        self.log_metrics(self.epoch, metrics, prefix="Epoch/")
        
        # Registra il tempo dell'epoca
        if self.use_tensorboard:
            self.writer.add_scalar("Time/epoch", epoch_time, self.epoch)
        
        self.logger.info(f"Fine epoca {self.epoch}: tempo = {epoch_time:.2f}s")
    
    def start_batch(self, batch: int):
        """
        Inizia un nuovo batch di addestramento.
        
        Args:
            batch: Numero del batch
        """
        self.batch = batch
        self.batch_start_time = time.time()
    
    def end_batch(self, metrics: Dict[str, float], step: int):
        """
        Termina un batch di addestramento.
        
        Args:
            metrics: Metriche del batch
            step: Passo di addestramento globale
        """
        # Calcola il tempo trascorso
        batch_time = time.time() - self.batch_start_time
        
        # Registra le metriche
        self.log_metrics(step, metrics, prefix="Batch/")
        
        # Registra il tempo del batch
        if self.use_tensorboard and step % self.config.get("logging", {}).get("log_frequency", 100) == 0:
            self.writer.add_scalar("Time/batch", batch_time, step)
        
        # Log dettagliato solo ogni log_frequency passi
        if step % self.config.get("logging", {}).get("log_frequency", 100) == 0:
            self.logger.info(f"Batch {self.batch} (Step {step}): tempo = {batch_time:.4f}s")
    
    def log_validation(self, metrics: Dict[str, float], step: int):
        """
        Registra le metriche di validazione.
        
        Args:
            metrics: Metriche di validazione
            step: Passo di addestramento globale
        """
        self.log_metrics(step, metrics, prefix="Validation/")
        
        # Log dettagliato
        metrics_str = ", ".join([f"{name} = {value:.6f}" for name, value in metrics.items()])
        self.logger.info(f"Validazione (Step {step}): {metrics_str}")
    
    def log_learning_rate(self, lr: float, step: int):
        """
        Registra il learning rate.
        
        Args:
            lr: Learning rate
            step: Passo di addestramento globale
        """
        if self.use_tensorboard:
            self.writer.add_scalar("Training/learning_rate", lr, step)
        
        # Aggiungi al dizionario delle metriche
        if "learning_rate" not in self.metrics:
            self.metrics["learning_rate"] = []
        
        # Assicurati che la lista abbia la stessa lunghezza di steps
        while len(self.metrics["learning_rate"]) < len(self.metrics["steps"]):
            self.metrics["learning_rate"].append(None)
        
        # Trova l'indice corrispondente al passo corrente
        try:
            idx = self.metrics["steps"].index(step)
            self.metrics["learning_rate"][idx] = lr
        except ValueError:
            # Il passo non è stato registrato, aggiungi sia il passo che il valore
            self.metrics["steps"].append(step)
            self.metrics["learning_rate"].append(lr)
        
        # Log dettagliato solo ogni log_frequency passi
        if step % self.config.get("logging", {}).get("log_frequency", 100) == 0:
            self.logger.info(f"Step {step}: learning_rate = {lr:.6f}")
    
    def log_memory_usage(self, step: int):
        """
        Registra l'utilizzo della memoria GPU.
        
        Args:
            step: Passo di addestramento globale
        """
        if torch.cuda.is_available():
            # Registra l'utilizzo della memoria per ogni GPU
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
                
                if self.use_tensorboard:
                    self.writer.add_scalar(f"Memory/gpu{i}_allocated", memory_allocated, step)
                    self.writer.add_scalar(f"Memory/gpu{i}_reserved", memory_reserved, step)
                
                self.logger.info(
                    f"Step {step}: GPU {i} memoria allocata = {memory_allocated:.2f} GB, "
                    f"riservata = {memory_reserved:.2f} GB"
                )


# Funzioni di utilità per il logging

def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    experiment_name: Optional[str] = None
) -> Tuple[logging.Logger, str]:
    """
    Configura il logging.
    
    Args:
        log_dir: Directory per i file di log
        level: Livello di logging
        experiment_name: Nome dell'esperimento (opzionale)
        
    Returns:
        Tupla con il logger e il percorso del file di log
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Crea un logger con un nome univoco
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger_name = experiment_name or f"experiment_{timestamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Rimuovi gli handler esistenti
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Aggiungi un handler per la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Aggiungi un handler per il file
    log_file = os.path.join(log_dir, f"{logger_name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging configurato con output in: {log_file}")
    
    return logger, log_file


def log_system_info(logger: logging.Logger):
    """
    Registra informazioni sul sistema.
    
    Args:
        logger: Logger
    """
    import platform
    import psutil
    
    logger.info(f"Sistema operativo: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {psutil.cpu_count(logical=False)} core fisici, {psutil.cpu_count()} core logici")
    logger.info(f"Memoria: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB totale")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.device_count()} dispositivi")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("GPU: non disponibile")


def log_config(logger: logging.Logger, config: Dict[str, Any], save_path: Optional[str] = None):
    """
    Registra la configurazione.
    
    Args:
        logger: Logger
        config: Configurazione
        save_path: Percorso per salvare la configurazione (opzionale)
    """
    logger.info("Configurazione:")
    for section, params in config.items():
        logger.info(f"  {section}:")
        if isinstance(params, dict):
            for key, value in params.items():
                logger.info(f"    {key}: {value}")
        else:
            logger.info(f"    {params}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configurazione salvata in: {save_path}")
