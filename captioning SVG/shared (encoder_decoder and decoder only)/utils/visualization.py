"""
Modulo di visualizzazione condiviso per il progetto di captioning SVG.

Questo modulo fornisce funzionalità di visualizzazione comuni utilizzabili
sia dall'implementazione decoder-only che da quella encoder-decoder.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime

# Configurazione del logger
logger = logging.getLogger(__name__)

class SVGVisualizer:
    """
    Classe base per la visualizzazione di elementi SVG e risultati di addestramento.
    
    Questa classe fornisce metodi comuni per:
    - Visualizzare SVG e le loro rappresentazioni
    - Tracciare metriche di addestramento
    - Salvare visualizzazioni in vari formati
    """
    
    def __init__(
        self,
        output_dir: str = "visualizations",
        config_path: Optional[str] = None
    ):
        """
        Inizializza il visualizzatore.
        
        Args:
            output_dir: Directory di output per le visualizzazioni
            config_path: Percorso al file di configurazione JSON (opzionale)
        """
        # Carica la configurazione
        self.config = self._load_config(config_path)
        
        # Imposta la directory di output
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Inizializza il contatore per le visualizzazioni
        self.visualization_count = 0
        
        logger.info(f"SVGVisualizer inizializzato con output in: {self.output_dir}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Carica la configurazione da un file JSON.
        
        Args:
            config_path: Percorso al file di configurazione
            
        Returns:
            Dizionario con la configurazione
        """
        default_config = {
            "visualization": {
                "save_formats": ["png"],
                "colormap": "viridis",
                "dpi": 300
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configurazione caricata da: {config_path}")
                return config
            except Exception as e:
                logger.error(f"Errore nel caricamento della configurazione: {e}")
                return default_config
        else:
            logger.warning("File di configurazione non trovato, utilizzo configurazione predefinita")
            return default_config
    
    def plot_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        steps: List[int],
        title: str = "Training Metrics",
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Visualizza le metriche di addestramento.
        
        Args:
            metrics: Dizionario con le metriche (nome -> lista di valori)
            steps: Lista dei passi di addestramento
            title: Titolo del grafico
            save: Se True, salva la visualizzazione
            show: Se True, mostra la visualizzazione
            
        Returns:
            Figura matplotlib
        """
        # Crea la figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Visualizza ogni metrica
        for name, values in metrics.items():
            ax.plot(steps, values, label=name)
        
        # Imposta il titolo e le etichette
        ax.set_title(title)
        ax.set_xlabel("Passi di addestramento")
        ax.set_ylabel("Valore")
        ax.legend()
        ax.grid(True)
        
        # Regola il layout
        plt.tight_layout()
        
        # Salva la figura se richiesto
        if save:
            self._save_figure(fig, f"{title.lower().replace(' ', '_')}")
        
        # Mostra la figura se richiesto
        if show:
            plt.show()
        elif not save:
            plt.close(fig)
        
        return fig
    
    def plot_comparison(
        self,
        data: Dict[str, List[float]],
        labels: List[str],
        title: str = "Comparison",
        xlabel: str = "Categories",
        ylabel: str = "Values",
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Visualizza un confronto tra diverse categorie.
        
        Args:
            data: Dizionario con i dati (nome -> lista di valori)
            labels: Etichette per le categorie
            title: Titolo del grafico
            xlabel: Etichetta dell'asse x
            ylabel: Etichetta dell'asse y
            save: Se True, salva la visualizzazione
            show: Se True, mostra la visualizzazione
            
        Returns:
            Figura matplotlib
        """
        # Crea la figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calcola la larghezza delle barre
        n_groups = len(labels)
        n_bars = len(data)
        bar_width = 0.8 / n_bars
        
        # Visualizza ogni gruppo di barre
        for i, (name, values) in enumerate(data.items()):
            index = np.arange(n_groups) + i * bar_width
            ax.bar(index, values, bar_width, label=name)
        
        # Imposta il titolo e le etichette
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(n_groups) + (n_bars - 1) * bar_width / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, axis='y')
        
        # Regola il layout
        plt.tight_layout()
        
        # Salva la figura se richiesto
        if save:
            self._save_figure(fig, f"{title.lower().replace(' ', '_')}")
        
        # Mostra la figura se richiesto
        if show:
            plt.show()
        elif not save:
            plt.close(fig)
        
        return fig
    
    def plot_heatmap(
        self,
        data: np.ndarray,
        title: str = "Heatmap",
        xlabel: str = "X",
        ylabel: str = "Y",
        colorbar_label: str = "Value",
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Visualizza una mappa di calore.
        
        Args:
            data: Matrice di dati
            title: Titolo del grafico
            xlabel: Etichetta dell'asse x
            ylabel: Etichetta dell'asse y
            colorbar_label: Etichetta della barra dei colori
            x_labels: Etichette per l'asse x (opzionale)
            y_labels: Etichette per l'asse y (opzionale)
            save: Se True, salva la visualizzazione
            show: Se True, mostra la visualizzazione
            
        Returns:
            Figura matplotlib
        """
        # Crea la figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Visualizza la mappa di calore
        cmap = self.config.get("visualization", {}).get("colormap", "viridis")
        im = ax.imshow(data, cmap=cm.get_cmap(cmap))
        
        # Aggiungi una barra dei colori
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)
        
        # Imposta il titolo e le etichette
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Imposta le etichette degli assi se fornite
        if x_labels:
            # Mostra solo alcune etichette se ce ne sono troppe
            if len(x_labels) > 20:
                tick_indices = np.linspace(0, len(x_labels)-1, 10, dtype=int)
                ax.set_xticks(tick_indices)
                ax.set_xticklabels([x_labels[i] for i in tick_indices], rotation=45, ha="right")
            else:
                ax.set_xticks(range(len(x_labels)))
                ax.set_xticklabels(x_labels, rotation=45, ha="right")
        
        if y_labels:
            # Mostra solo alcune etichette se ce ne sono troppe
            if len(y_labels) > 20:
                tick_indices = np.linspace(0, len(y_labels)-1, 10, dtype=int)
                ax.set_yticks(tick_indices)
                ax.set_yticklabels([y_labels[i] for i in tick_indices])
            else:
                ax.set_yticks(range(len(y_labels)))
                ax.set_yticklabels(y_labels)
        
        # Regola il layout
        plt.tight_layout()
        
        # Salva la figura se richiesto
        if save:
            self._save_figure(fig, f"{title.lower().replace(' ', '_')}")
        
        # Mostra la figura se richiesto
        if show:
            plt.show()
        elif not save:
            plt.close(fig)
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, name: str):
        """
        Salva una figura in vari formati.
        
        Args:
            fig: Figura matplotlib
            name: Nome base del file
        """
        # Crea il nome del file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{name}_{timestamp}"
        
        # Salva in tutti i formati specificati
        save_formats = self.config.get("visualization", {}).get("save_formats", ["png"])
        dpi = self.config.get("visualization", {}).get("dpi", 300)
        
        for fmt in save_formats:
            filepath = os.path.join(self.output_dir, f"{filename_base}.{fmt}")
            fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches="tight")
            logger.info(f"Visualizzazione salvata in: {filepath}")
        
        # Incrementa il contatore
        self.visualization_count += 1


class SVGElementVisualizer(SVGVisualizer):
    """
    Classe per la visualizzazione di elementi SVG.
    
    Questa classe estende SVGVisualizer con metodi specifici per:
    - Visualizzare SVG e le loro rappresentazioni
    - Confrontare diversi SVG
    - Analizzare la struttura degli SVG
    """
    
    def __init__(
        self,
        output_dir: str = "svg_visualizations",
        config_path: Optional[str] = None,
        svg_element_labels: Optional[List[str]] = None
    ):
        """
        Inizializza il visualizzatore di elementi SVG.
        
        Args:
            output_dir: Directory di output per le visualizzazioni
            config_path: Percorso al file di configurazione JSON (opzionale)
            svg_element_labels: Etichette per gli elementi SVG (opzionale)
        """
        super().__init__(output_dir, config_path)
        
        # Imposta le etichette degli elementi SVG
        self.svg_element_labels = svg_element_labels or [
            "path", "rect", "circle", "ellipse", "line", "polyline", 
            "polygon", "text", "g", "defs", "use", "symbol", 
            "clipPath", "mask", "filter", "unknown"
        ]
    
    def visualize_svg_structure(
        self,
        element_types: np.ndarray,
        hierarchy: Optional[np.ndarray] = None,
        title: str = "SVG Structure",
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Visualizza la struttura di un SVG.
        
        Args:
            element_types: Array con i tipi degli elementi
            hierarchy: Matrice di adiacenza per la gerarchia (opzionale)
            title: Titolo del grafico
            save: Se True, salva la visualizzazione
            show: Se True, mostra la visualizzazione
            
        Returns:
            Figura matplotlib
        """
        # Crea la figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Visualizza i tipi degli elementi
        x = np.arange(len(element_types))
        ax.bar(x, np.ones_like(element_types), color=[plt.cm.tab20(t % 20) for t in element_types])
        
        # Imposta le etichette
        ax.set_xticks(x)
        ax.set_xticklabels([self.svg_element_labels[t] if t < len(self.svg_element_labels) else f"Type {t}" 
                           for t in element_types], rotation=45, ha="right")
        
        # Imposta il titolo
        ax.set_title(title)
        ax.set_xlabel("Elementi SVG")
        ax.set_yticks([])
        
        # Visualizza la gerarchia se fornita
        if hierarchy is not None:
            # Crea una seconda figura per la gerarchia
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            
            # Visualizza la matrice di adiacenza
            im = ax2.imshow(hierarchy, cmap="Blues")
            
            # Aggiungi una barra dei colori
            cbar = fig2.colorbar(im, ax=ax2)
            cbar.set_label("Relazione gerarchica")
            
            # Imposta le etichette
            labels = [self.svg_element_labels[t] if t < len(self.svg_element_labels) else f"Type {t}" 
                     for t in element_types]
            
            # Mostra solo alcune etichette se ce ne sono troppe
            if len(labels) > 20:
                tick_indices = np.linspace(0, len(labels)-1, 10, dtype=int)
                ax2.set_xticks(tick_indices)
                ax2.set_yticks(tick_indices)
                ax2.set_xticklabels([labels[i] for i in tick_indices], rotation=45, ha="right")
                ax2.set_yticklabels([labels[i] for i in tick_indices])
            else:
                ax2.set_xticks(range(len(labels)))
                ax2.set_yticks(range(len(labels)))
                ax2.set_xticklabels(labels, rotation=45, ha="right")
                ax2.set_yticklabels(labels)
            
            # Imposta il titolo
            ax2.set_title(f"{title} - Hierarchy")
            
            # Regola il layout
            plt.tight_layout()
            
            # Salva la figura se richiesto
            if save:
                self._save_figure(fig2, f"{title.lower().replace(' ', '_')}_hierarchy")
            
            # Mostra la figura se richiesto
            if show:
                plt.show()
            elif not save:
                plt.close(fig2)
        
        # Regola il layout
        plt.tight_layout()
        
        # Salva la figura se richiesto
        if save:
            self._save_figure(fig, f"{title.lower().replace(' ', '_')}_elements")
        
        # Mostra la figura se richiesto
        if show:
            plt.show()
        elif not save:
            plt.close(fig)
        
        return fig
    
    def visualize_svg_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "SVG Embeddings",
        method: str = "pca",
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Visualizza gli embedding degli SVG in 2D.
        
        Args:
            embeddings: Matrice di embedding [n_samples, n_features]
            labels: Etichette per i campioni (opzionale)
            title: Titolo del grafico
            method: Metodo di riduzione della dimensionalità ('pca' o 'tsne')
            save: Se True, salva la visualizzazione
            show: Se True, mostra la visualizzazione
            
        Returns:
            Figura matplotlib
        """
        # Riduci la dimensionalità a 2D
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        else:  # tsne
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2)
        
        # Applica la riduzione della dimensionalità
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Crea la figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Visualizza gli embedding
        if labels is not None:
            # Converti le etichette in numeri
            unique_labels = list(set(labels))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            label_ids = [label_to_id[label] for label in labels]
            
            # Visualizza i punti colorati per etichetta
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=label_ids,
                cmap="tab10",
                alpha=0.8
            )
            
            # Aggiungi una legenda
            legend1 = ax.legend(
                scatter.legend_elements()[0],
                unique_labels,
                title="Categorie"
            )
            ax.add_artist(legend1)
        else:
            # Visualizza i punti senza etichette
            ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                alpha=0.8
            )
        
        # Imposta il titolo e le etichette
        ax.set_title(f"{title} ({method.upper()})")
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.grid(True)
        
        # Regola il layout
        plt.tight_layout()
        
        # Salva la figura se richiesto
        if save:
            self._save_figure(fig, f"{title.lower().replace(' ', '_')}_{method}")
        
        # Mostra la figura se richiesto
        if show:
            plt.show()
        elif not save:
            plt.close(fig)
        
        return fig


# Funzioni di utilità per la visualizzazione

def tensor_to_numpy(tensor):
    """
    Converte un tensore PyTorch in un array NumPy.
    
    Args:
        tensor: Tensore PyTorch
        
    Returns:
        Array NumPy
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def setup_logging(log_dir="logs", level=logging.INFO):
    """
    Configura il logging.
    
    Args:
        log_dir: Directory per i file di log
        level: Livello di logging
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Configura il logger root
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Aggiungi un handler per il file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"visualization_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Aggiungi l'handler al logger root
    logging.getLogger('').addHandler(file_handler)
    
    return log_file
