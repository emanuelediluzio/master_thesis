"""
Modulo per l'aggiornamento dei file di utilità per utilizzare i componenti condivisi.

Questo script aggiorna i file esistenti per utilizzare i componenti condivisi
dalla directory shared/utils/.
"""

import os
import sys
import shutil
import importlib.util
import logging

# Configurazione del logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("update_utils")

def update_imports(file_path, shared_utils_path):
    """
    Aggiorna gli import in un file Python per utilizzare i moduli condivisi.
    
    Args:
        file_path: Percorso del file da aggiornare
        shared_utils_path: Percorso della directory shared/utils
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Aggiorna gli import per visualization.py
        if "attention_visualization.py" in file_path:
            # Aggiungi l'import per il modulo condiviso
            new_import = "from shared.utils.visualization import SVGVisualizer, tensor_to_numpy, setup_logging\n"
            if "import" in content and new_import not in content:
                # Trova la posizione dopo l'ultimo import
                import_lines = [i for i, line in enumerate(content.split('\n')) if 'import' in line]
                if import_lines:
                    last_import_line = import_lines[-1]
                    content_lines = content.split('\n')
                    content_lines.insert(last_import_line + 1, new_import)
                    content = '\n'.join(content_lines)
                else:
                    # Se non ci sono import, aggiungi all'inizio del file
                    content = new_import + content
            
            logger.info(f"Aggiornati gli import in {file_path}")
        
        # Aggiorna gli import per logger.py
        elif "logger.py" in file_path:
            # Aggiungi l'import per il modulo condiviso
            new_import = "from shared.utils.logging_utils import BaseLogger, TrainingLogger, setup_logging, log_system_info, log_config\n"
            if "import" in content and new_import not in content:
                # Trova la posizione dopo l'ultimo import
                import_lines = [i for i, line in enumerate(content.split('\n')) if 'import' in line]
                if import_lines:
                    last_import_line = import_lines[-1]
                    content_lines = content.split('\n')
                    content_lines.insert(last_import_line + 1, new_import)
                    content = '\n'.join(content_lines)
                else:
                    # Se non ci sono import, aggiungi all'inizio del file
                    content = new_import + content
            
            logger.info(f"Aggiornati gli import in {file_path}")
        
        # Scrivi il contenuto aggiornato
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"File aggiornato: {file_path}")
        
    except Exception as e:
        logger.error(f"Errore nell'aggiornamento di {file_path}: {e}")

def update_requirements(project_dir, shared_requirements_path):
    """
    Aggiorna i file requirements.txt per includere i requisiti condivisi.
    
    Args:
        project_dir: Directory del progetto
        shared_requirements_path: Percorso del file requirements.txt condiviso
    """
    try:
        # Leggi i requisiti condivisi
        with open(shared_requirements_path, 'r') as f:
            shared_requirements = f.read()
        
        # Cerca i file requirements.txt nel progetto
        for root, dirs, files in os.walk(project_dir):
            for file in files:
                if file == "requirements.txt" and "shared" not in root:
                    req_path = os.path.join(root, file)
                    
                    # Leggi i requisiti esistenti
                    with open(req_path, 'r') as f:
                        existing_requirements = f.read()
                    
                    # Aggiungi un riferimento ai requisiti condivisi
                    if "-r ../shared/requirements.txt" not in existing_requirements:
                        new_requirements = "-r ../shared/requirements.txt\n\n" + existing_requirements
                        
                        # Scrivi i requisiti aggiornati
                        with open(req_path, 'w') as f:
                            f.write(new_requirements)
                        
                        logger.info(f"Aggiornato il file requirements.txt: {req_path}")
    
    except Exception as e:
        logger.error(f"Errore nell'aggiornamento dei requisiti: {e}")

def create_init_files(shared_utils_path):
    """
    Crea i file __init__.py necessari nella directory shared.
    
    Args:
        shared_utils_path: Percorso della directory shared/utils
    """
    try:
        # Crea __init__.py nella directory shared
        shared_dir = os.path.dirname(shared_utils_path)
        init_path = os.path.join(shared_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write('"""Moduli condivisi per il progetto di captioning SVG."""\n')
            logger.info(f"Creato file {init_path}")
        
        # Crea __init__.py nella directory shared/utils
        utils_init_path = os.path.join(shared_utils_path, "__init__.py")
        if not os.path.exists(utils_init_path):
            with open(utils_init_path, 'w') as f:
                f.write('"""Utilità condivise per il progetto di captioning SVG."""\n\n')
                f.write('from .visualization import SVGVisualizer, tensor_to_numpy, setup_logging\n')
                f.write('from .logging_utils import BaseLogger, TrainingLogger, setup_logging as setup_logging_utils, log_system_info, log_config\n\n')
                f.write('__all__ = [\n')
                f.write('    "SVGVisualizer",\n')
                f.write('    "tensor_to_numpy",\n')
                f.write('    "setup_logging",\n')
                f.write('    "BaseLogger",\n')
                f.write('    "TrainingLogger",\n')
                f.write('    "setup_logging_utils",\n')
                f.write('    "log_system_info",\n')
                f.write('    "log_config",\n')
                f.write(']\n')
            logger.info(f"Creato file {utils_init_path}")
    
    except Exception as e:
        logger.error(f"Errore nella creazione dei file __init__.py: {e}")

def update_project_files(project_dir):
    """
    Aggiorna i file del progetto per utilizzare i componenti condivisi.
    
    Args:
        project_dir: Directory del progetto
    """
    # Percorsi
    shared_utils_path = os.path.join(project_dir, "shared", "utils")
    shared_requirements_path = os.path.join(project_dir, "shared", "requirements.txt")
    
    # Crea i file __init__.py
    create_init_files(shared_utils_path)
    
    # Aggiorna i file requirements.txt
    update_requirements(project_dir, shared_requirements_path)
    
    # Cerca i file da aggiornare
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                
                # Aggiorna gli import nei file specifici
                if file in ["attention_visualization.py", "logger.py"]:
                    update_imports(file_path, shared_utils_path)

if __name__ == "__main__":
    # Percorso del progetto
    project_dir = "/home/ubuntu/project"
    
    # Aggiorna i file del progetto
    update_project_files(project_dir)
    
    logger.info("Aggiornamento completato con successo!")
