#!/usr/bin/env python3
"""
Shutterstock AI Metadata Generator v1.0.0
Application unifiée avec:
- Analyseur d'images Shutterstock
- Gestionnaire Ollama intégré
- Pipeline de validation photos/métadonnées
- Upload FTPS
- Interface moderne avec CustomTkinter

Auteur: Emmanuel Grolleau
Date: Décembre 2025
"""

import subprocess
import os
import sys
import re
import time
import json
import csv
import logging
import threading

# Debug mode via environment variable
if os.environ.get("SHUTTERSTOCK_DEBUG") == "1":
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Shutterstock AI Metadata Generator DEBUG mode enabled")
import unicodedata
import shutil
import ftplib
import ssl
import base64
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Set, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime
from enum import Enum

# ============================================================================
# IMPORTS TIERS
# ============================================================================

try:
    from PIL import Image, ImageOps, ExifTags
    import piexif
    Image.MAX_IMAGE_PIXELS = 200_000_000
except ImportError:
    print("Installation requise: pip install Pillow piexif")
    sys.exit(1)

try:
    from ollama import Client
except ImportError:
    print("Installation requise: pip install ollama")
    sys.exit(1)

try:
    from pydantic import BaseModel, ValidationError, field_validator
except ImportError:
    print("Installation requise: pip install pydantic")
    sys.exit(1)

try:
    import customtkinter as ctk
    from CTkToolTip import CTkToolTip
    HAS_CTK = True
except ImportError:
    HAS_CTK = False
    print("Pour l'interface moderne: pip install customtkinter CTkToolTip")

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import tkinter as tk
from tkinter import messagebox, filedialog


# ============================================================================
# CONFIGURATION DU LOGGING
# ============================================================================

def setup_logging(log_file: str = "shutterstock_analyzer.log") -> logging.Logger:
    """Configure le système de logging"""
    logger = logging.getLogger("ShutterstockAnalyzer")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler fichier
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        pass

    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()


# ============================================================================
# CONSTANTES ET CONFIGURATION
# ============================================================================

class Config:
    """Configuration centralisée de l'application"""

    # Version de l'application
    APP_VERSION = "1.0.0"
    APP_NAME = "Shutterstock AI Metadata Generator"
    DEBUG_MODE = False  # Activer pour logs détaillés

    # Catégories Shutterstock
    CATEGORIES = frozenset({
        "Abstract", "Animals/Wildlife", "Arts", "Backgrounds/Textures",
        "Beauty/Fashion", "Buildings/Landmarks", "Business/Finance",
        "Celebrities", "Education", "Food and drink", "Healthcare/Medical",
        "Holidays", "Industrial", "Interiors", "Miscellaneous", "Nature",
        "Objects", "Parks/Outdoor", "People", "Religion", "Science",
        "Signs/Symbols", "Sports/Recreation", "Technology", "Transportation",
        "Vintage"
    })

    # Configuration FTPS
    FTPS = {
        'host': 'ftps.shutterstock.com',
        'port': 21,
        'timeout': 30,
        'upload_delay': 1.5
    }

    # Critères d'images (valeurs par défaut pour le préfiltrage)
    IMAGE_CRITERIA = {
        'jpeg': {
            'extensions': ['.jpg', '.jpeg'],
            'min_megapixels': 4,
            'max_file_size_mb': 50
        },
        'tiff': {
            'extensions': ['.tiff', '.tif'],
            'min_megapixels': 4,
            'max_file_size_mb': 4096
        },
        'eps': {
            'extensions': ['.eps'],
            'min_megapixels': 0,
            'max_megapixels': 25,
            'max_file_size_mb': 100
        }
    }

    # Options de préfiltrage par défaut
    PREFILTER_DEFAULTS = {
        'min_megapixels': 4.0,          # Résolution minimum en MP
        'max_file_size_mb': 50.0,        # Taille max en MB
        'check_orientation': True,       # Vérifier/corriger l'orientation EXIF
        'skip_duplicates': True,         # Ignorer les fichiers déjà traités
        'allowed_formats': ['jpeg', 'tiff'],  # Formats autorisés
    }

    # Limites métadonnées
    MAX_DESCRIPTION_LENGTH = 200
    MIN_KEYWORDS = 7
    MAX_KEYWORDS = 50
    MAX_CATEGORIES = 2
    BATCH_SIZE = 50

    # Modèles Ollama Vision avec descriptions
    AVAILABLE_MODELS = [
        "llama3.2-vision:11b",
        "llama3.2-vision:90b",
        "llava:7b",
        "llava:13b",
        "llava:34b",
        "bakllava:7b",
        "moondream:1.8b",
    ]

    # Descriptions des modèles pour les infobulles
    MODEL_DESCRIPTIONS = {
        "llama3.2-vision:11b": "Recommandé - Bon équilibre qualité/vitesse (7GB VRAM)",
        "llama3.2-vision:90b": "Haute qualité - Nécessite GPU puissant (48GB+ VRAM)",
        "llava:7b": "Rapide - Pour GPU modestes (4GB VRAM)",
        "llava:13b": "Équilibré - Qualité moyenne (8GB VRAM)",
        "llava:34b": "Haute qualité - GPU puissant requis (20GB VRAM)",
        "bakllava:7b": "Alternative LLaVA - Rapide (4GB VRAM)",
        "moondream:1.8b": "Ultra-léger - Pour CPU ou GPU faibles (2GB VRAM)",
    }

    # EXIF Orientations
    EXIF_ORIENTATIONS = {
        1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: 270, 7: 270, 8: 90,
    }

    # Textes d'aide et descriptions
    HELP_TEXTS = {
        'app_description': """
Shutterstock Analyzer est une application qui utilise l'IA pour analyser
vos photos et générer automatiquement les métadonnées requises par Shutterstock:
- Description (max 200 caractères)
- Mots-clés (7-50 mots)
- Catégories (1-2 parmi les catégories Shutterstock)

L'application vérifie également que vos images respectent les critères
techniques de Shutterstock (résolution, taille, format).
        """,

        'workflow': """
FLUX DE TRAVAIL TYPIQUE:
========================

1. PRÉPARATION
   - Placez vos photos dans un dossier source
   - Vérifiez qu'Ollama est installé et démarré

2. PRÉ-FILTRAGE (automatique)
   - Les photos valides → dossier "Valid"
   - Les photos invalides → dossier "Invalid"

3. ANALYSE IA
   - Chaque photo est analysée par le modèle Vision
   - Les métadonnées sont générées et sauvées en CSV

4. ORGANISATION
   - Photos analysées → dossier "Shutterstock"
   - Par lots de 50 images maximum
   - Un fichier CSV par lot

5. VALIDATION (Checklist)
   - Vérifie la correspondance photos/métadonnées
   - Identifie les fichiers manquants ou incomplets

6. UPLOAD FTPS
   - Envoi direct vers ftps.shutterstock.com
        """,

        'folder_structure': """
STRUCTURE DES DOSSIERS:
=======================

Avant traitement:
📁 Mon_Dossier_Photos/
   ├── photo1.jpg
   ├── photo2.jpg
   └── photo3.jpg

Après traitement:
📁 Mon_Dossier_Photos/
   ├── 📁 Valid/              ← Photos validées (pré-filtrage)
   ├── 📁 Invalid/            ← Photos rejetées (trop petites, etc.)
   ├── 📁 Shutterstock/       ← Lot 1 (max 50 photos)
   │   ├── photo1.jpg
   │   ├── photo2.jpg
   │   └── metadata.csv       ← Métadonnées du lot
   ├── 📁 Shutterstock_2/     ← Lot 2 (si > 50 photos)
   │   └── ...
   └── 📁 photos_unmatched/   ← Photos sans métadonnées
        """,

        'prefilter': """
Le pré-filtrage vérifie automatiquement:
- Résolution minimum (4 MP par défaut)
- Taille maximale du fichier
- Format d'image valide (JPEG, TIFF)
- Orientation EXIF correcte

Les photos qui ne passent pas ces critères sont
déplacées dans le dossier "Invalid" avec la raison.
        """,

        'checklist': """
La validation Checklist compare:
- Les photos présentes dans chaque lot
- Les entrées du fichier CSV de métadonnées

Elle identifie:
- Photos sans métadonnées (oubliées)
- Métadonnées sans photos (fichiers supprimés)
- Métadonnées incomplètes (description courte, peu de mots-clés)
        """,
    }


# ============================================================================
# ÉNUMÉRATIONS
# ============================================================================

class OllamaStatus(Enum):
    """États du système Ollama"""
    NOT_INSTALLED = "non_installé"
    NOT_RUNNING = "non_démarré"
    RUNNING = "en_cours"
    MODEL_MISSING = "modèle_manquant"
    READY = "prêt"
    ERROR = "erreur"


class ValidationStatus(Enum):
    """États de validation"""
    VALID = "valide"
    INVALID = "invalide"
    MISSING_METADATA = "metadata_manquante"
    MISSING_PHOTO = "photo_manquante"
    INCOMPLETE = "incomplet"


# ============================================================================
# MODÈLES DE DONNÉES
# ============================================================================

class ShutterstockMetadata(BaseModel):
    """Modèle Pydantic pour les métadonnées Shutterstock"""
    description: str
    keywords: List[str]
    categories: List[str]
    illustration: bool = False
    mature_content: bool = False
    editorial: bool = False

    @field_validator('description')
    @classmethod
    def validate_description(cls, v: str) -> str:
        v = v.strip()
        if len(v) > Config.MAX_DESCRIPTION_LENGTH:
            v = v[:Config.MAX_DESCRIPTION_LENGTH-3].rsplit(' ', 1)[0] + '...'
        return v

    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        cleaned = []
        seen = set()
        for kw in v:
            kw = kw.strip().lower()
            kw = Utils.remove_accents(kw)
            kw = re.sub(r'[^a-z0-9\s-]', '', kw)
            if kw and kw not in seen and len(kw) >= 2:
                cleaned.append(kw)
                seen.add(kw)
        return cleaned[:Config.MAX_KEYWORDS]

    @field_validator('categories')
    @classmethod
    def validate_categories(cls, v: List[str]) -> List[str]:
        valid = [c.strip() for c in v if c.strip() in Config.CATEGORIES]
        if not valid:
            valid = ["Miscellaneous"]
        return valid[:Config.MAX_CATEGORIES]


@dataclass
class ProcessingResult:
    """Résultat du traitement d'une image"""
    success: bool
    filename: str
    message: str
    metadata: Optional[ShutterstockMetadata] = None
    error_type: Optional[str] = None


@dataclass
class BatchInfo:
    """Information sur un lot de traitement"""
    batch_number: int
    folder_path: Path
    csv_path: Path
    image_count: int = 0


@dataclass
class FTPSUploadResult:
    """Résultat d'un upload FTPS"""
    success: bool
    filename: str
    message: str
    uploaded_size: int = 0


@dataclass
class ValidationResult:
    """Résultat de validation photo/métadonnées"""
    status: ValidationStatus
    photo_path: Optional[Path] = None
    metadata_row: Optional[dict] = None
    message: str = ""


@dataclass
class ChecklistReport:
    """Rapport de checklist photos/métadonnées"""
    total_photos: int = 0
    total_metadata: int = 0
    matched: int = 0
    photos_without_metadata: List[Path] = field(default_factory=list)
    metadata_without_photos: List[str] = field(default_factory=list)
    incomplete_metadata: List[Tuple[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# UTILITAIRES
# ============================================================================

class Utils:
    """Fonctions utilitaires"""

    @staticmethod
    def remove_accents(text: str) -> str:
        """Supprime les accents d'une chaîne"""
        nfkd = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in nfkd if not unicodedata.combining(c))

    @staticmethod
    def get_all_extensions() -> List[str]:
        """Retourne toutes les extensions d'images supportées"""
        extensions = []
        for criteria in Config.IMAGE_CRITERIA.values():
            extensions.extend(criteria['extensions'])
        return extensions

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Formate une taille en bytes"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    @staticmethod
    def extract_context_from_filename(filename: str) -> Dict[str, Any]:
        """Extrait des informations contextuelles du nom de fichier"""
        context = {
            'raw_name': filename,
            'cleaned_name': '',
            'hints': []
        }

        name_without_ext = Path(filename).stem
        parts = [p.strip() for p in name_without_ext.split(' - ') if p.strip()]

        for part in parts:
            if part.isdigit():
                continue

            date_match = re.match(r'(\d{4})[_/-](\d{2})[_/-](\d{2})', part)
            if date_match:
                year, month, day = date_match.groups()
                context['hints'].append(f"Photo taken on {year}/{month}/{day}")
                continue

            if re.match(r'^\d{4}$', part):
                context['hints'].append(f"Year: {part}")
                continue

            if len(part) > 2:
                clean_part = part.replace('_', ' ')
                context['hints'].append(clean_part)

        context['cleaned_name'] = ' '.join(context['hints']) if context['hints'] else name_without_ext
        return context

    @staticmethod
    def safe_file_exists(filepath: Path) -> bool:
        """Vérifie si un fichier existe de manière sécurisée"""
        try:
            return filepath.exists() and filepath.is_file()
        except (OSError, PermissionError):
            return False

    @staticmethod
    def get_timestamp() -> str:
        """Retourne un timestamp formaté"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')


# ============================================================================
# GESTIONNAIRE OLLAMA UNIFIÉ
# ============================================================================

class OllamaManager:
    """Gestionnaire unifié pour Ollama avec support GPU"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.ollama_process: Optional[subprocess.Popen] = None
        self.gpu_info = self._detect_gpu()
        self._session = self._create_session() if HAS_REQUESTS else None

    def _create_session(self) -> Optional[Any]:
        """Crée une session HTTP avec retry"""
        if not HAS_REQUESTS:
            return None

        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _detect_gpu(self) -> Dict[str, Any]:
        """Détecte les GPU disponibles"""
        gpu_info = {
            'available': False,
            'type': None,
            'name': None,
            'vram_mb': 0,
            'driver_version': None
        }

        # Essayer NVIDIA
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    gpu_info['available'] = True
                    gpu_info['type'] = 'NVIDIA'
                    gpu_info['name'] = parts[0].strip()
                    gpu_info['vram_mb'] = int(float(parts[1].strip()))
                    gpu_info['driver_version'] = parts[2].strip()
                    logger.info(f"GPU détecté: {gpu_info['name']} ({gpu_info['vram_mb']} MB)")
                    return gpu_info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Essayer AMD ROCm
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and 'vram' in result.stdout.lower():
                gpu_info['available'] = True
                gpu_info['type'] = 'AMD'
                gpu_info['name'] = 'AMD GPU (ROCm)'
                logger.info("GPU AMD détecté (ROCm)")
                return gpu_info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        logger.warning("Aucun GPU détecté - Mode CPU")
        return gpu_info

    def get_recommended_settings(self) -> Dict[str, Any]:
        """Retourne les paramètres recommandés selon le GPU"""
        vram = self.gpu_info.get('vram_mb', 0)

        if vram >= 24000:
            return {'model': 'llama3.2-vision:11b', 'num_gpu': 999, 'num_ctx': 4096}
        elif vram >= 12000:
            return {'model': 'llama3.2-vision:11b', 'num_gpu': 35, 'num_ctx': 2048}
        elif vram >= 8000:
            return {'model': 'llava:7b', 'num_gpu': 28, 'num_ctx': 2048}
        elif vram >= 6000:
            return {'model': 'moondream:1.8b', 'num_gpu': 20, 'num_ctx': 1024}
        elif vram >= 4000:
            return {'model': 'moondream:1.8b', 'num_gpu': 15, 'num_ctx': 1024}
        else:
            return {'model': 'moondream:1.8b', 'num_gpu': 0, 'num_ctx': 512}

    def find_ollama_executable(self) -> Optional[str]:
        """Trouve l'exécutable Ollama"""
        import getpass
        username = getpass.getuser()

        paths = [
            rf"C:\Users\{username}\AppData\Local\Programs\Ollama\ollama.exe",
            r"C:\Program Files\Ollama\ollama.exe",
            "ollama"
        ]

        for path in paths:
            if os.path.isfile(path):
                return path
            if path == "ollama":
                try:
                    cmd = ["where", "ollama"] if sys.platform == "win32" else ["which", "ollama"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return result.stdout.strip().split('\n')[0]
                except:
                    pass
        return None

    def is_serving(self) -> bool:
        """Vérifie si Ollama répond"""
        try:
            import urllib.request
            req = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
            return req.status == 200
        except:
            return False

    def start_server(self, force_gpu: bool = True) -> Tuple[bool, str]:
        """Démarre le serveur Ollama"""
        if self.is_serving():
            return True, "Ollama est déjà en cours d'exécution"

        ollama_path = self.find_ollama_executable()
        if not ollama_path:
            return False, "Ollama non trouvé. Installez-le depuis https://ollama.ai"

        try:
            env = os.environ.copy()

            if force_gpu and self.gpu_info['available']:
                if self.gpu_info['type'] == 'NVIDIA':
                    env['CUDA_VISIBLE_DEVICES'] = '0'
                    env['OLLAMA_KEEP_ALIVE'] = '10m'
                    env['OLLAMA_NUM_PARALLEL'] = '1'
                    env['OLLAMA_MAX_LOADED_MODELS'] = '1'
                elif self.gpu_info['type'] == 'AMD':
                    env['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
                    env['OLLAMA_KEEP_ALIVE'] = '10m'

            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 6

                self.ollama_process = subprocess.Popen(
                    [ollama_path, "serve"],
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env
                )
            else:
                self.ollama_process = subprocess.Popen(
                    [ollama_path, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=env
                )

            # Attendre jusqu'à 30 secondes pour le démarrage
            # Ollama v0.13+ peut prendre plus de temps au premier démarrage
            for i in range(30):
                time.sleep(1)
                if self.is_serving():
                    # Attendre 2 secondes supplémentaires pour que le serveur soit vraiment prêt
                    time.sleep(2)
                    gpu_status = f" (GPU: {self.gpu_info['name']})" if self.gpu_info['available'] else " (CPU)"
                    return True, f"Serveur Ollama démarré{gpu_status}"
                # Log progression toutes les 10 secondes
                if i > 0 and i % 10 == 0:
                    logger.info(f"Attente du serveur Ollama... ({i}s)")

            return False, "Timeout: le serveur n'a pas démarré après 30 secondes"

        except Exception as e:
            return False, f"Erreur au démarrage: {e}"

    def stop_server(self) -> Tuple[bool, str]:
        """Arrête le serveur Ollama"""
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             timeout=10)
                subprocess.run(["taskkill", "/F", "/IM", "ollama_llama_server.exe"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             timeout=10)
            else:
                subprocess.run(["pkill", "-f", "ollama"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             timeout=10)

            time.sleep(2)

            if not self.is_serving():
                return True, "Serveur Ollama arrêté"
            return False, "Impossible d'arrêter Ollama"

        except Exception as e:
            return False, f"Erreur: {e}"

    def kill_all_ollama_processes(self) -> bool:
        """Tue tous les processus Ollama de manière agressive"""
        try:
            if sys.platform == "win32":
                # Tuer tous les processus Ollama (sans capture pour éviter erreurs encodage)
                processes = ["ollama.exe", "ollama_llama_server.exe", "ollama app.exe"]
                for proc in processes:
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/IM", proc],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=10
                        )
                    except:
                        pass

                # Libérer le port 11434 si occupé
                try:
                    result = subprocess.run(
                        ["netstat", "-ano"],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='ignore',
                        timeout=10
                    )
                    for line in result.stdout.split('\n'):
                        if ':11434' in line:
                            parts = line.split()
                            if len(parts) >= 5:
                                pid = parts[-1]
                                if pid.isdigit():
                                    subprocess.run(
                                        ["taskkill", "/F", "/PID", pid],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        timeout=5
                                    )
                except:
                    pass
            else:
                subprocess.run(["pkill", "-9", "-f", "ollama"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             timeout=10)

            time.sleep(2)
            return True
        except Exception as e:
            logger.error(f"Erreur kill_all_ollama_processes: {e}")
            return False

    def clean_temp_files(self) -> bool:
        """Nettoie les fichiers temporaires Ollama"""
        try:
            ollama_local = Path(os.environ.get('LOCALAPPDATA', '')) / "Ollama"

            # Nettoyer le dossier tmp
            tmp_path = ollama_local / "tmp"
            if tmp_path.exists():
                shutil.rmtree(tmp_path, ignore_errors=True)
                logger.info("Dossier tmp Ollama nettoyé")

            # Nettoyer les logs à la racine (v0.13+)
            for log_pattern in ["server*.log", "app*.log"]:
                for log_file in ollama_local.glob(log_pattern):
                    try:
                        # Tronquer le fichier au lieu de le supprimer
                        with open(log_file, 'w') as f:
                            f.write("")
                        logger.info(f"Log nettoyé: {log_file.name}")
                    except:
                        pass

            # Nettoyer aussi l'ancien dossier logs si présent
            logs_path = ollama_local / "logs"
            if logs_path.exists():
                for log_file in logs_path.glob("*.log"):
                    try:
                        with open(log_file, 'w') as f:
                            f.write("")
                    except:
                        pass

            # Supprimer le fichier PID obsolète
            pid_file = ollama_local / "ollama.pid"
            if pid_file.exists():
                try:
                    pid_file.unlink()
                    logger.info("Fichier PID supprimé")
                except:
                    pass

            return True
        except Exception as e:
            logger.error(f"Erreur clean_temp_files: {e}")
            return False

    def check_port_available(self, port: int = 11434) -> bool:
        """Vérifie si le port est disponible"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('127.0.0.1', port))
                return result != 0  # True si le port est libre
        except:
            return True

    def repair_and_start(self, progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
        """Répare Ollama et le redémarre proprement"""
        messages = []

        def log_progress(msg):
            logger.info(msg)
            messages.append(msg)
            if progress_callback:
                progress_callback(msg)

        log_progress("Réparation Ollama en cours...")

        # Étape 1: Tuer tous les processus
        log_progress("[1/5] Arrêt des processus Ollama...")
        self.kill_all_ollama_processes()
        time.sleep(2)

        # Étape 2: Nettoyer les fichiers temporaires
        log_progress("[2/5] Nettoyage des fichiers temporaires...")
        self.clean_temp_files()

        # Étape 3: Vérifier que le port est libre
        log_progress("[3/5] Vérification du port 11434...")
        max_attempts = 5
        for attempt in range(max_attempts):
            if self.check_port_available(11434):
                log_progress("  Port 11434 libre")
                break
            else:
                log_progress(f"  Port occupé, tentative {attempt + 1}/{max_attempts}...")
                self.kill_all_ollama_processes()
                time.sleep(2)

        # Étape 4: Démarrer le serveur
        log_progress("[4/5] Démarrage du serveur Ollama...")
        success, start_msg = self.start_server(force_gpu=True)

        if not success:
            # Deuxième tentative sans GPU
            log_progress("  Échec avec GPU, tentative en mode CPU...")
            success, start_msg = self.start_server(force_gpu=False)

        # Étape 5: Vérification
        log_progress("[5/5] Vérification de la connexion...")
        if success and self.is_serving():
            # Vérifier les modèles
            model_success, models = self.list_models()
            if model_success:
                log_progress(f"  Modèles disponibles: {', '.join(models[:3])}")
            return True, "Ollama réparé et démarré avec succès"
        else:
            return False, f"Échec de la réparation: {start_msg}"

    def diagnose_problems(self) -> List[Dict[str, str]]:
        """Diagnostique les problèmes Ollama et retourne une liste de problèmes avec solutions"""
        problems = []

        # Vérifier si Ollama est installé
        ollama_path = self.find_ollama_executable()
        if not ollama_path:
            problems.append({
                'problem': "Ollama n'est pas installé",
                'solution': "Téléchargez Ollama depuis https://ollama.ai/download",
                'severity': 'critical'
            })
            return problems

        # Vérifier si le serveur répond
        if not self.is_serving():
            # Vérifier si le port est bloqué
            if not self.check_port_available(11434):
                problems.append({
                    'problem': "Le port 11434 est occupé par un autre processus",
                    'solution': "Utilisez 'Réparer Ollama' pour libérer le port",
                    'severity': 'high'
                })
            else:
                problems.append({
                    'problem': "Le serveur Ollama n'est pas démarré",
                    'solution': "Cliquez sur 'Démarrer' ou 'Réparer Ollama'",
                    'severity': 'high'
                })

        # Analyser les logs pour des problèmes connus
        logs = self.get_ollama_logs()
        log_content = ''.join(logs) if logs else ''

        if 'The handle is invalid' in log_content:
            problems.append({
                'problem': "Processus Ollama zombie détecté",
                'solution': "Un processus précédent ne s'est pas fermé correctement. Utilisez 'Réparer Ollama'",
                'severity': 'medium'
            })

        if 'connectex: No connection could be made' in log_content:
            problems.append({
                'problem': "Le serveur API Ollama ne démarre pas",
                'solution': "Le serveur interne refuse les connexions. Redémarrez Ollama ou votre PC",
                'severity': 'high'
            })

        if 'timeout scanning server log for inference compute' in log_content:
            problems.append({
                'problem': "Timeout détection GPU - Le serveur met trop de temps à démarrer",
                'solution': "Attendez quelques secondes puis réessayez, ou redémarrez Ollama",
                'severity': 'medium'
            })

        if 'context canceled' in log_content:
            problems.append({
                'problem': "Requêtes annulées - Le serveur n'était pas prêt",
                'solution': "Le serveur Ollama démarre lentement. Attendez 10-15 secondes après le démarrage",
                'severity': 'low'
            })

        if 'proxy error' in log_content.lower():
            problems.append({
                'problem': "Erreur de proxy interne Ollama",
                'solution': "Le serveur API n'est pas encore prêt. Patientez ou cliquez sur 'Réparer'",
                'severity': 'medium'
            })

        if 'out of memory' in log_content.lower():
            problems.append({
                'problem': "Mémoire GPU insuffisante",
                'solution': "Utilisez un modèle plus léger (moondream:1.8b ou llava:7b)",
                'severity': 'high'
            })

        if 'New update available' in log_content:
            problems.append({
                'problem': "Une mise à jour Ollama est disponible",
                'solution': "Mettez à jour vers la dernière version pour corriger les bugs",
                'severity': 'low'
            })

        # Vérifier les modèles
        if self.is_serving():
            success, models = self.list_models()
            if success and len(models) == 0:
                problems.append({
                    'problem': "Aucun modèle installé",
                    'solution': "Téléchargez un modèle avec le bouton 'Télécharger'",
                    'severity': 'high'
                })

        return problems

    def list_models(self) -> Tuple[bool, List[str]]:
        """Liste les modèles installés"""
        try:
            import urllib.request
            req = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10)
            data = json.loads(req.read().decode())
            models = [m['name'] for m in data.get('models', [])]
            return True, models
        except Exception as e:
            logger.error(f"Impossible de lister les modèles: {e}")
            return False, []

    def get_loaded_model(self) -> Optional[str]:
        """Retourne le modèle actuellement chargé en mémoire"""
        try:
            import urllib.request
            req = urllib.request.urlopen(f"{self.base_url}/api/ps", timeout=5)
            data = json.loads(req.read().decode())
            models = data.get('models', [])
            if models and len(models) > 0:
                return models[0].get('name', None)
            return None
        except Exception as e:
            logger.debug(f"Impossible de récupérer le modèle chargé: {e}")
            return None

    def get_ollama_logs(self) -> List[str]:
        """Récupère les logs Ollama depuis le dossier utilisateur"""
        import getpass
        all_logs = []
        username = getpass.getuser()
        ollama_local = Path(os.environ.get('LOCALAPPDATA', '')) / "Ollama"

        # Chemins corrects pour les logs Ollama sur Windows (v0.13+)
        log_files = [
            # Logs serveur (principaux)
            ollama_local / "server.log",
            ollama_local / "server-1.log",
            # Logs application
            ollama_local / "app.log",
            ollama_local / "app-1.log",
            # Anciens chemins (versions précédentes)
            ollama_local / "logs" / "server.log",
            Path(f"C:/Users/{username}/.ollama/logs/server.log"),
        ]

        # Lire tous les logs disponibles
        for log_path in log_files:
            try:
                if log_path.exists():
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        # Ajouter un header pour identifier le fichier
                        if lines:
                            all_logs.append(f"\n=== {log_path.name} ===\n")
                            # Prendre les 50 dernières lignes de chaque fichier
                            all_logs.extend(lines[-50:] if len(lines) > 50 else lines)
                            logger.info(f"Logs Ollama trouvés: {log_path}")
            except Exception as e:
                logger.debug(f"Erreur lecture logs {log_path}: {e}")

        return all_logs

    def pull_model(self, model_name: str, callback: Optional[Callable] = None) -> Tuple[bool, str]:
        """Télécharge un modèle"""
        ollama_path = self.find_ollama_executable()
        if not ollama_path:
            return False, "Ollama non trouvé"

        try:
            logger.info(f"Téléchargement du modèle {model_name}...")

            process = subprocess.Popen(
                [ollama_path, "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            for line in process.stdout:
                line = line.strip()
                if line and callback:
                    callback(line)

            process.wait()

            if process.returncode == 0:
                return True, f"Modèle {model_name} téléchargé"
            return False, "Erreur de téléchargement"

        except Exception as e:
            return False, f"Erreur: {e}"

    def load_model(self, model_name: str, num_gpu: Optional[int] = None) -> Tuple[bool, str]:
        """Charge un modèle en mémoire"""
        if not self.is_serving():
            return False, "Le serveur Ollama n'est pas démarré"

        try:
            import urllib.request

            if num_gpu is None:
                num_gpu = self.get_recommended_settings()['num_gpu']

            data = json.dumps({
                "model": model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 1, "num_gpu": num_gpu}
            }).encode()

            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )

            response = urllib.request.urlopen(req, timeout=180)

            if response.status == 200:
                gpu_info = f" (GPU layers: {num_gpu})" if num_gpu > 0 else " (CPU)"
                return True, f"Modèle {model_name} chargé{gpu_info}"
            return False, "Erreur de chargement"

        except Exception as e:
            return False, f"Erreur: {e}"

    def get_status(self) -> OllamaStatus:
        """Retourne l'état du système Ollama"""
        if not self.find_ollama_executable():
            return OllamaStatus.NOT_INSTALLED
        if not self.is_serving():
            return OllamaStatus.NOT_RUNNING
        success, models = self.list_models()
        if not success or not models:
            return OllamaStatus.MODEL_MISSING
        return OllamaStatus.READY

    def get_gpu_status_string(self) -> str:
        """Retourne une chaîne de statut GPU"""
        if not self.gpu_info['available']:
            return "Aucun GPU détecté (CPU)"
        vram_gb = self.gpu_info['vram_mb'] / 1024
        return f"{self.gpu_info['name']} ({vram_gb:.1f} GB VRAM)"


# ============================================================================
# VALIDATEUR DE CHECKLIST PHOTOS/MÉTADONNÉES
# ============================================================================

@dataclass
class BatchValidationResult:
    """Résultat de validation d'un lot"""
    batch_name: str
    batch_path: Path
    report: ChecklistReport
    csv_found: bool = True


@dataclass
class RecursiveChecklistReport:
    """Rapport de validation récursive de plusieurs lots"""
    source_dir: Path = None
    total_batches: int = 0
    batches_with_issues: int = 0
    batch_results: List[BatchValidationResult] = field(default_factory=list)
    global_stats: Dict[str, int] = field(default_factory=lambda: {
        'total_photos': 0,
        'total_metadata': 0,
        'total_matched': 0,
        'total_photos_without_metadata': 0,
        'total_metadata_without_photos': 0,
        'total_incomplete': 0
    })


class ChecklistValidator:
    """Validateur de correspondance photos/métadonnées avec support récursif"""

    def __init__(self, source_dir: Path):
        self.source_dir = Path(source_dir)
        self.report = ChecklistReport()
        self.recursive_report = RecursiveChecklistReport(source_dir=self.source_dir)

    def find_batch_folders(self, recursive: bool = False) -> List[Tuple[Path, Path]]:
        """
        Trouve tous les dossiers de lots avec leurs CSV associés.

        Args:
            recursive: Si True, cherche dans tous les sous-dossiers

        Returns:
            Liste de tuples (dossier_lot, chemin_csv)
        """
        batch_folders = []

        if recursive:
            # Recherche récursive de tous les CSV
            for csv_file in self.source_dir.rglob("*.csv"):
                # Vérifier que c'est un CSV de métadonnées Shutterstock
                if self._is_metadata_csv(csv_file):
                    batch_folder = csv_file.parent
                    batch_folders.append((batch_folder, csv_file))
        else:
            # Recherche uniquement dans le dossier source
            for csv_file in self.source_dir.glob("*.csv"):
                if self._is_metadata_csv(csv_file):
                    batch_folders.append((self.source_dir, csv_file))

            # Aussi chercher dans les sous-dossiers directs (Shutterstock, Shutterstock_2, etc.)
            for subdir in self.source_dir.iterdir():
                if subdir.is_dir():
                    for csv_file in subdir.glob("*.csv"):
                        if self._is_metadata_csv(csv_file):
                            batch_folders.append((subdir, csv_file))

        return batch_folders

    def _is_metadata_csv(self, csv_path: Path) -> bool:
        """Vérifie si un CSV est un fichier de métadonnées Shutterstock"""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                header = f.readline().lower()
                # Vérifier les colonnes typiques
                return 'filename' in header and ('description' in header or 'keywords' in header)
        except:
            return False

    def validate_all_batches(self, recursive: bool = False,
                            progress_callback: Optional[Callable] = None) -> RecursiveChecklistReport:
        """
        Valide tous les lots trouvés dans le dossier source.

        Args:
            recursive: Si True, cherche dans tous les sous-dossiers récursivement
            progress_callback: Callback(current, total, message)

        Returns:
            RecursiveChecklistReport avec tous les résultats
        """
        self.recursive_report = RecursiveChecklistReport(source_dir=self.source_dir)

        # Trouver tous les lots
        batch_folders = self.find_batch_folders(recursive=recursive)
        total_batches = len(batch_folders)
        self.recursive_report.total_batches = total_batches

        if total_batches == 0:
            logger.warning("Aucun lot trouvé avec des fichiers CSV de métadonnées")
            return self.recursive_report

        logger.info(f"Validation de {total_batches} lot(s)...")

        for i, (batch_folder, csv_path) in enumerate(batch_folders):
            if progress_callback:
                progress_callback(i, total_batches, f"Validation: {batch_folder.name}")

            # Valider ce lot
            report = self.validate_batch(batch_folder, csv_path)

            # Créer le résultat
            batch_result = BatchValidationResult(
                batch_name=batch_folder.name,
                batch_path=batch_folder,
                report=report,
                csv_found=True
            )
            self.recursive_report.batch_results.append(batch_result)

            # Mettre à jour les stats globales
            self.recursive_report.global_stats['total_photos'] += report.total_photos
            self.recursive_report.global_stats['total_metadata'] += report.total_metadata
            self.recursive_report.global_stats['total_matched'] += report.matched
            self.recursive_report.global_stats['total_photos_without_metadata'] += len(report.photos_without_metadata)
            self.recursive_report.global_stats['total_metadata_without_photos'] += len(report.metadata_without_photos)
            self.recursive_report.global_stats['total_incomplete'] += len(report.incomplete_metadata)

            # Compter les lots avec problèmes
            if report.photos_without_metadata or report.metadata_without_photos or report.incomplete_metadata:
                self.recursive_report.batches_with_issues += 1

        if progress_callback:
            progress_callback(total_batches, total_batches, "Validation terminée")

        return self.recursive_report

    def validate_batch(self, batch_folder: Path, csv_path: Path) -> ChecklistReport:
        """Valide un lot de photos contre son CSV de métadonnées"""
        self.report = ChecklistReport()

        # Collecter les photos
        photos = self._collect_photos(batch_folder)
        self.report.total_photos = len(photos)

        # Lire le CSV
        metadata_entries = self._read_csv(csv_path)
        self.report.total_metadata = len(metadata_entries)

        # Créer des ensembles pour la comparaison
        photo_names = {p.name for p in photos}
        metadata_names = set(metadata_entries.keys())

        # Trouver les correspondances
        matched = photo_names & metadata_names
        self.report.matched = len(matched)

        # Photos sans métadonnées
        photos_only = photo_names - metadata_names
        for name in photos_only:
            photo_path = batch_folder / name
            self.report.photos_without_metadata.append(photo_path)

        # Métadonnées sans photos
        metadata_only = metadata_names - photo_names
        for name in metadata_only:
            self.report.metadata_without_photos.append(name)

        # Vérifier les métadonnées incomplètes
        for name, entry in metadata_entries.items():
            issues = self._validate_metadata_entry(entry)
            if issues:
                self.report.incomplete_metadata.append((name, issues))

        return self.report

    def _collect_photos(self, folder: Path) -> List[Path]:
        """Collecte les photos d'un dossier"""
        photos = []
        for ext in Utils.get_all_extensions():
            photos.extend(folder.glob(f"*{ext}"))
            photos.extend(folder.glob(f"*{ext.upper()}"))
        return list(set(photos))

    def _read_csv(self, csv_path: Path) -> Dict[str, dict]:
        """Lit un CSV de métadonnées"""
        entries = {}
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('Filename', '')
                    if filename:
                        entries[filename] = row
        except Exception as e:
            self.report.errors.append(f"Erreur lecture CSV: {e}")
        return entries

    def _validate_metadata_entry(self, entry: dict) -> str:
        """Valide une entrée de métadonnées"""
        issues = []

        description = entry.get('Description', '')
        if not description or len(description) < 10:
            issues.append("description courte")

        keywords = entry.get('Keywords', '')
        if keywords:
            kw_list = [k.strip() for k in keywords.split(',')]
            if len(kw_list) < Config.MIN_KEYWORDS:
                issues.append(f"seulement {len(kw_list)} mots-clés")
        else:
            issues.append("pas de mots-clés")

        categories = entry.get('Categories', '')
        if not categories:
            issues.append("pas de catégorie")

        return ", ".join(issues) if issues else ""

    def move_unmatched_photos(self, unmatched_folder: Optional[Path] = None) -> int:
        """Déplace les photos sans métadonnées (pour un seul lot)"""
        if not unmatched_folder:
            unmatched_folder = self.source_dir / "photos_unmatched"

        unmatched_folder.mkdir(exist_ok=True)
        moved = 0

        for photo_path in self.report.photos_without_metadata:
            try:
                dest = unmatched_folder / photo_path.name
                if dest.exists():
                    dest = unmatched_folder / f"{photo_path.stem}_{Utils.get_timestamp()}{photo_path.suffix}"
                shutil.move(str(photo_path), str(dest))
                moved += 1
            except Exception as e:
                self.report.errors.append(f"Impossible de déplacer {photo_path.name}: {e}")

        return moved

    def move_all_unmatched_photos(self, create_subfolder_per_batch: bool = True) -> int:
        """
        Déplace toutes les photos sans métadonnées de tous les lots validés.

        Args:
            create_subfolder_per_batch: Si True, crée un sous-dossier par lot

        Returns:
            Nombre total de photos déplacées
        """
        total_moved = 0
        base_unmatched = self.source_dir / "photos_unmatched"
        base_unmatched.mkdir(exist_ok=True)

        for batch_result in self.recursive_report.batch_results:
            if not batch_result.report.photos_without_metadata:
                continue

            if create_subfolder_per_batch:
                unmatched_folder = base_unmatched / batch_result.batch_name
            else:
                unmatched_folder = base_unmatched

            unmatched_folder.mkdir(exist_ok=True)

            for photo_path in batch_result.report.photos_without_metadata:
                try:
                    dest = unmatched_folder / photo_path.name
                    if dest.exists():
                        dest = unmatched_folder / f"{photo_path.stem}_{Utils.get_timestamp()}{photo_path.suffix}"
                    shutil.move(str(photo_path), str(dest))
                    total_moved += 1
                except Exception as e:
                    logger.error(f"Impossible de déplacer {photo_path.name}: {e}")

        return total_moved

    def generate_log(self, output_path: Optional[Path] = None) -> Path:
        """Génère un fichier log de la validation (pour un seul lot)"""
        if not output_path:
            output_path = self.source_dir / f"checklist_report_{Utils.get_timestamp()}.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RAPPORT DE VALIDATION PHOTOS/MÉTADONNÉES\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total photos: {self.report.total_photos}\n")
            f.write(f"Total métadonnées: {self.report.total_metadata}\n")
            f.write(f"Correspondances: {self.report.matched}\n\n")

            if self.report.photos_without_metadata:
                f.write("-" * 40 + "\n")
                f.write("PHOTOS SANS MÉTADONNÉES:\n")
                f.write("-" * 40 + "\n")
                for photo in self.report.photos_without_metadata:
                    f.write(f"  - {photo.name}\n")
                f.write("\n")

            if self.report.metadata_without_photos:
                f.write("-" * 40 + "\n")
                f.write("MÉTADONNÉES SANS PHOTOS:\n")
                f.write("-" * 40 + "\n")
                for name in self.report.metadata_without_photos:
                    f.write(f"  - {name}\n")
                f.write("\n")

            if self.report.incomplete_metadata:
                f.write("-" * 40 + "\n")
                f.write("MÉTADONNÉES INCOMPLÈTES:\n")
                f.write("-" * 40 + "\n")
                for name, issues in self.report.incomplete_metadata:
                    f.write(f"  - {name}: {issues}\n")
                f.write("\n")

            if self.report.errors:
                f.write("-" * 40 + "\n")
                f.write("ERREURS:\n")
                f.write("-" * 40 + "\n")
                for error in self.report.errors:
                    f.write(f"  - {error}\n")

        return output_path

    def generate_recursive_log(self, output_path: Optional[Path] = None) -> Path:
        """Génère un rapport complet pour tous les lots validés"""
        if not output_path:
            output_path = self.source_dir / f"checklist_recursive_report_{Utils.get_timestamp()}.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RAPPORT DE VALIDATION RÉCURSIVE - TOUS LES LOTS\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dossier source: {self.recursive_report.source_dir}\n")
            f.write("=" * 70 + "\n\n")

            # Statistiques globales
            stats = self.recursive_report.global_stats
            f.write("STATISTIQUES GLOBALES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Lots analysés: {self.recursive_report.total_batches}\n")
            f.write(f"Lots avec problèmes: {self.recursive_report.batches_with_issues}\n")
            f.write(f"Total photos: {stats['total_photos']}\n")
            f.write(f"Total métadonnées: {stats['total_metadata']}\n")
            f.write(f"Total correspondances: {stats['total_matched']}\n")
            f.write(f"Photos sans métadonnées: {stats['total_photos_without_metadata']}\n")
            f.write(f"Métadonnées sans photos: {stats['total_metadata_without_photos']}\n")
            f.write(f"Métadonnées incomplètes: {stats['total_incomplete']}\n\n")

            # Détail par lot
            f.write("=" * 70 + "\n")
            f.write("DÉTAIL PAR LOT\n")
            f.write("=" * 70 + "\n\n")

            for batch_result in self.recursive_report.batch_results:
                report = batch_result.report
                has_issues = (report.photos_without_metadata or
                            report.metadata_without_photos or
                            report.incomplete_metadata)

                status_icon = "⚠️" if has_issues else "✅"
                f.write(f"\n{status_icon} LOT: {batch_result.batch_name}\n")
                f.write(f"   Chemin: {batch_result.batch_path}\n")
                f.write(f"   Photos: {report.total_photos} | Métadonnées: {report.total_metadata} | Correspondances: {report.matched}\n")

                if report.photos_without_metadata:
                    f.write(f"   ❌ Photos sans métadonnées ({len(report.photos_without_metadata)}):\n")
                    for photo in report.photos_without_metadata[:10]:
                        f.write(f"      - {photo.name}\n")
                    if len(report.photos_without_metadata) > 10:
                        f.write(f"      ... et {len(report.photos_without_metadata) - 10} autres\n")

                if report.metadata_without_photos:
                    f.write(f"   ❌ Métadonnées sans photos ({len(report.metadata_without_photos)}):\n")
                    for name in report.metadata_without_photos[:10]:
                        f.write(f"      - {name}\n")
                    if len(report.metadata_without_photos) > 10:
                        f.write(f"      ... et {len(report.metadata_without_photos) - 10} autres\n")

                if report.incomplete_metadata:
                    f.write(f"   ⚠️ Métadonnées incomplètes ({len(report.incomplete_metadata)}):\n")
                    for name, issues in report.incomplete_metadata[:10]:
                        f.write(f"      - {name}: {issues}\n")
                    if len(report.incomplete_metadata) > 10:
                        f.write(f"      ... et {len(report.incomplete_metadata) - 10} autres\n")

                f.write("\n")

            # Résumé final
            f.write("=" * 70 + "\n")
            f.write("RÉSUMÉ\n")
            f.write("=" * 70 + "\n")
            if self.recursive_report.batches_with_issues == 0:
                f.write("✅ Tous les lots sont valides!\n")
            else:
                f.write(f"⚠️ {self.recursive_report.batches_with_issues}/{self.recursive_report.total_batches} lots ont des problèmes\n")

        return output_path


# ============================================================================
# GESTIONNAIRE DE CONTINUITÉ D'EXÉCUTION
# ============================================================================

class ContinuityManager:
    """Gère la reprise des traitements interrompus"""

    def __init__(self, source_dir: Path):
        self.source_dir = Path(source_dir)
        self.state_file = self.source_dir / ".processing_state.json"

    def save_state(self, processed_files: Set[str], current_batch: int,
                   failed_files: Dict[str, str]) -> None:
        """Sauvegarde l'état du traitement"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'processed_files': list(processed_files),
            'current_batch': current_batch,
            'failed_files': failed_files
        }

        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> Optional[Dict]:
        """Charge l'état précédent"""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def clear_state(self) -> None:
        """Efface l'état sauvegardé"""
        if self.state_file.exists():
            self.state_file.unlink()

    def find_incomplete_metadata(self, batch_folder: Path, csv_path: Path) -> List[Path]:
        """Trouve les images avec métadonnées manquantes ou incomplètes"""
        incomplete = []

        # Lire le CSV existant
        existing_metadata = set()
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('Filename', '')
                    description = row.get('Description', '')
                    keywords = row.get('Keywords', '')

                    # Vérifier si complet
                    if filename and description and keywords:
                        kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
                        if len(kw_list) >= Config.MIN_KEYWORDS:
                            existing_metadata.add(filename)

        # Trouver les images sans métadonnées complètes
        for ext in Utils.get_all_extensions():
            for img in batch_folder.glob(f"*{ext}"):
                if img.name not in existing_metadata:
                    incomplete.append(img)
            for img in batch_folder.glob(f"*{ext.upper()}"):
                if img.name not in existing_metadata:
                    incomplete.append(img)

        return list(set(incomplete))

    def get_resume_info(self) -> Tuple[bool, str, int]:
        """Retourne les informations de reprise"""
        state = self.load_state()
        if not state:
            return False, "Aucun traitement interrompu", 0

        processed = len(state.get('processed_files', []))
        failed = len(state.get('failed_files', {}))
        timestamp = state.get('timestamp', 'inconnu')

        return True, f"Traitement interrompu le {timestamp}", processed


# ============================================================================
# UPLOADER FTPS
# ============================================================================

class FTPSUploader:
    """Gestionnaire d'upload FTPS vers Shutterstock"""

    def __init__(self, username: str, password: str,
                 host: str = Config.FTPS['host'],
                 port: int = Config.FTPS['port'],
                 upload_delay: float = Config.FTPS['upload_delay']):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.upload_delay = upload_delay

        self._connection: Optional[ftplib.FTP_TLS] = None
        self._stop_requested = False
        self.progress_callback: Optional[Callable] = None

    def connect(self) -> Tuple[bool, str]:
        """Établit la connexion FTPS"""
        try:
            self._connection = ftplib.FTP_TLS()
            self._connection.encoding = 'utf-8'

            logger.info(f"Connexion à {self.host}:{self.port}...")
            self._connection.connect(self.host, self.port, timeout=Config.FTPS['timeout'])

            self._connection.auth()
            self._connection.login(self.username, self.password)
            self._connection.prot_p()
            self._connection.set_pasv(True)

            logger.info("Connexion FTPS établie")
            return True, "Connexion réussie"

        except ftplib.error_perm as e:
            return False, f"Erreur d'authentification: {e}"
        except Exception as e:
            return False, f"Erreur de connexion: {e}"

    def disconnect(self) -> None:
        """Ferme la connexion"""
        if self._connection:
            try:
                self._connection.quit()
            except:
                pass
            self._connection = None

    def upload_file(self, local_path: Path) -> FTPSUploadResult:
        """Upload un fichier"""
        filename = local_path.name

        if not Utils.safe_file_exists(local_path):
            return FTPSUploadResult(False, filename, "Fichier non trouvé", 0)

        if not self._connection:
            return FTPSUploadResult(False, filename, "Non connecté", 0)

        try:
            file_size = local_path.stat().st_size
            uploaded = [0]

            def progress(data):
                uploaded[0] += len(data)
                if self.progress_callback and file_size > 0:
                    pct = (uploaded[0] / file_size) * 100
                    self.progress_callback(uploaded[0], file_size, f"Upload {filename}: {pct:.1f}%")

            with open(local_path, 'rb') as f:
                self._connection.storbinary(f'STOR {filename}', f, callback=progress)

            logger.info(f"Upload réussi: {filename} ({Utils.format_size(file_size)})")
            return FTPSUploadResult(True, filename, "OK", file_size)

        except Exception as e:
            return FTPSUploadResult(False, filename, str(e), 0)

    def upload_batch(self, files: List[Path],
                    progress_callback: Optional[Callable] = None) -> Dict[str, FTPSUploadResult]:
        """Upload un lot de fichiers"""
        results = {}
        total = len(files)

        for i, filepath in enumerate(files):
            if self._stop_requested:
                break

            if progress_callback:
                progress_callback(i, total, f"Upload {i+1}/{total}: {filepath.name}")

            results[filepath.name] = self.upload_file(filepath)

            if i < total - 1:
                time.sleep(self.upload_delay)

        return results

    def stop(self) -> None:
        """Arrête les uploads"""
        self._stop_requested = True

    def test_connection(self) -> Tuple[bool, str]:
        """Teste la connexion"""
        success, msg = self.connect()
        if success:
            self.disconnect()
        return success, msg


# ============================================================================
# ANALYSEUR D'IMAGES
# ============================================================================

class ImageAnalyzer:
    """Analyseur d'images avec IA Ollama"""

    def __init__(self, model: str = "llama3.2-vision:11b",
                 ollama_url: str = "http://localhost:11434",
                 max_workers: int = 2,
                 cooldown_seconds: float = 2.0):
        self.model = model
        self.ollama_url = ollama_url
        self.max_workers = max_workers
        self.cooldown_seconds = cooldown_seconds

        self.client = Client(host=ollama_url)
        self.ollama_manager = OllamaManager(ollama_url)

        self._stop_requested = False
        self._csv_lock = threading.Lock()
        self._processed_files: Set[str] = set()
        self._error_files: Dict[str, str] = {}

        self.progress_callback: Optional[Callable] = None
        self.continuity_manager: Optional[ContinuityManager] = None

        self.base_dir: Optional[Path] = None
        self.valid_dir: Optional[Path] = None
        self.invalid_dir: Optional[Path] = None

        self.current_batch: Optional[BatchInfo] = None
        self.batch_number = 0

        logger.info(f"Analyseur initialisé - Modèle: {model}")

    def update_settings(self, **kwargs) -> None:
        """Met à jour les paramètres"""
        if 'model' in kwargs:
            self.model = kwargs['model']
        if 'max_workers' in kwargs:
            self.max_workers = kwargs['max_workers']
        if 'cooldown_seconds' in kwargs:
            self.cooldown_seconds = kwargs['cooldown_seconds']
        if 'ollama_url' in kwargs:
            self.ollama_url = kwargs['ollama_url']
            self.client = Client(host=self.ollama_url)

    def verify_connection(self) -> Tuple[bool, List[str]]:
        """Vérifie la connexion Ollama"""
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models] if hasattr(models, 'models') else []
            return True, model_names
        except Exception as e:
            logger.error(f"Connexion Ollama échouée: {e}")
            return False, []

    def setup_directories(self, base_path: str) -> None:
        """Configure les dossiers de travail"""
        self.base_dir = Path(base_path)
        self.valid_dir = self.base_dir / "Valid"
        self.invalid_dir = self.base_dir / "Invalid"

        self.valid_dir.mkdir(exist_ok=True)
        self.invalid_dir.mkdir(exist_ok=True)

        self.continuity_manager = ContinuityManager(self.base_dir)

    def _create_new_batch(self) -> BatchInfo:
        """Crée un nouveau lot"""
        self.batch_number += 1

        folder_name = "Shutterstock" if self.batch_number == 1 else f"Shutterstock_{self.batch_number}"
        csv_name = "metadata.csv" if self.batch_number == 1 else f"metadata_{self.batch_number}.csv"

        folder_path = self.base_dir / folder_name
        folder_path.mkdir(exist_ok=True)

        csv_path = folder_path / csv_name
        self._write_csv_header(csv_path)

        return BatchInfo(self.batch_number, folder_path, csv_path)

    def _write_csv_header(self, csv_path: Path) -> None:
        """Écrit l'en-tête CSV"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Filename", "Description", "Keywords", "Categories",
                "Illustration", "Mature content", "Editorial"
            ])

    def _append_to_csv(self, csv_path: Path, metadata: ShutterstockMetadata, filename: str) -> None:
        """Ajoute une ligne au CSV"""
        with self._csv_lock:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    filename,
                    metadata.description,
                    ", ".join(metadata.keywords),
                    ", ".join(metadata.categories),
                    "yes" if metadata.illustration else "no",
                    "yes" if metadata.mature_content else "no",
                    "yes" if metadata.editorial else "no"
                ])

    def check_image_validity(self, image_path: Path) -> Tuple[bool, str, dict]:
        """Vérifie si une image est valide"""
        details = {'format': '', 'megapixels': 0, 'file_size_mb': 0, 'dimensions': (0, 0)}

        if not Utils.safe_file_exists(image_path):
            return False, f"Fichier introuvable", details

        try:
            ext = image_path.suffix.lower()

            format_type = None
            for fmt, criteria in Config.IMAGE_CRITERIA.items():
                if ext in criteria['extensions']:
                    format_type = fmt
                    break

            if not format_type:
                return False, f"Format non supporté: {ext}", details

            details['format'] = format_type.upper()
            criteria = Config.IMAGE_CRITERIA[format_type]

            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            details['file_size_mb'] = round(file_size_mb, 2)

            if file_size_mb > criteria['max_file_size_mb']:
                return False, f"Fichier trop volumineux: {file_size_mb:.1f}MB", details

            if format_type == 'eps':
                return True, "Format EPS accepté", details

            with Image.open(image_path) as img:
                width, height = img.size
                details['dimensions'] = (width, height)
                megapixels = (width * height) / 1_000_000
                details['megapixels'] = round(megapixels, 2)

                if megapixels > 100:
                    return False, f"Image trop grande: {megapixels:.1f}MP", details

                if megapixels < criteria.get('min_megapixels', 0):
                    return False, f"Résolution insuffisante: {megapixels:.1f}MP", details

            return True, f"Image valide: {megapixels:.1f}MP", details

        except Exception as e:
            return False, f"Erreur: {str(e)}", details

    def _build_prompt(self, context: Optional[Dict] = None) -> str:
        """Construit le prompt pour l'IA"""
        categories_list = ", ".join(sorted(Config.CATEGORIES))

        context_hints = ""
        if context and context.get('hints'):
            hints_text = "; ".join(context['hints'][:3])
            context_hints = f"Context from filename: {hints_text}\n"

        return f"""{context_hints}Analyze this image for stock photography. Return ONLY a JSON object, no other text.

JSON format required:
{{"description": "one sentence describing the image", "keywords": ["word1", "word2", "word3", "word4", "word5", "word6", "word7"], "categories": ["Category1"], "illustration": false, "mature_content": false, "editorial": false}}

Rules:
- description: max {Config.MAX_DESCRIPTION_LENGTH} chars, English, specific details
- keywords: {Config.MIN_KEYWORDS}-{Config.MAX_KEYWORDS} unique English words
- categories: 1-2 from [{categories_list}]
- illustration: true only if digital art
- mature_content: true only if adult content
- editorial: true if logos/brands/recognizable people

IMPORTANT: Output ONLY the JSON object. No explanation, no markdown, no ```json tags."""

    @lru_cache(maxsize=50)
    def _encode_image_base64(self, image_path: str) -> Optional[str]:
        """Encode une image en base64"""
        if not Utils.safe_file_exists(Path(image_path)):
            return None

        try:
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                max_dim = 1024
                max_pixels = 1_000_000
                current_pixels = img.size[0] * img.size[1]

                if current_pixels > max_pixels or max(img.size) > max_dim:
                    ratio = min(max_dim / max(img.size), (max_pixels / current_pixels) ** 0.5)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=75, optimize=True)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            logger.error(f"Erreur encodage: {e}")
            return None

    def _extract_json(self, content: str) -> Optional[dict]:
        """Extrait le JSON de la réponse"""
        content = content.strip()

        # Stratégie 1: Regex
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                json_str = json_match.group()
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                return json.loads(json_str)
            except:
                pass

        # Stratégie 2: Supprimer markdown
        content_clean = re.sub(r'```json\s*', '', content)
        content_clean = re.sub(r'```\s*', '', content_clean).strip()

        json_match = re.search(r'\{[\s\S]*\}', content_clean)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        # Stratégie 3: Extraction manuelle
        try:
            data = {}

            desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', content)
            if desc_match:
                data['description'] = desc_match.group(1)

            kw_match = re.search(r'"keywords"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            if kw_match:
                data['keywords'] = re.findall(r'"([^"]+)"', kw_match.group(1))

            cat_match = re.search(r'"categories"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            if cat_match:
                data['categories'] = re.findall(r'"([^"]+)"', cat_match.group(1))

            data['illustration'] = 'illustration": true' in content.lower()
            data['mature_content'] = 'mature_content": true' in content.lower()
            data['editorial'] = 'editorial": true' in content.lower()

            if data.get('description') and data.get('keywords'):
                return data
        except:
            pass

        return None

    def analyze_image(self, image_path: Path, max_retries: int = 3) -> Optional[ShutterstockMetadata]:
        """Analyse une image avec l'IA"""
        context = Utils.extract_context_from_filename(image_path.name)

        for attempt in range(1, max_retries + 1):
            if self._stop_requested:
                return None

            try:
                image_b64 = self._encode_image_base64(str(image_path))
                if not image_b64:
                    return None

                prompt = self._build_prompt(context)

                response = self.client.chat(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64]
                    }],
                    options={
                        "num_ctx": 2048,
                        "num_predict": 1000,
                        "temperature": 0.3
                    }
                )

                content = response.message.content
                data = self._extract_json(content)

                if data:
                    try:
                        metadata = ShutterstockMetadata(**data)
                        if len(metadata.keywords) >= Config.MIN_KEYWORDS:
                            logger.info(f"OK: {image_path.name} - {len(metadata.keywords)} mots-clés")
                            return metadata
                    except ValidationError:
                        pass

            except Exception as e:
                logger.error(f"Erreur analyse (tentative {attempt}): {e}")
                if "500" in str(e) or "stopped" in str(e).lower():
                    time.sleep(15 * attempt)

            if attempt < max_retries:
                time.sleep(self.cooldown_seconds * attempt)

        return None

    def _collect_images(self, source_dir: Path) -> List[Path]:
        """Collecte les images à traiter"""
        extensions = Utils.get_all_extensions()
        all_images = []

        exclude_folders = {'Valid', 'Invalid', 'Shutterstock', 'photos_unmatched'}
        for i in range(2, 100):
            exclude_folders.add(f'Shutterstock_{i}')

        for ext in extensions:
            for img in source_dir.glob(f"*{ext}"):
                if not any(p in img.parts for p in exclude_folders):
                    if Utils.safe_file_exists(img):
                        all_images.append(img)
            for img in source_dir.glob(f"*{ext.upper()}"):
                if not any(p in img.parts for p in exclude_folders):
                    if Utils.safe_file_exists(img):
                        all_images.append(img)

        return list(dict.fromkeys(all_images))

    def pre_filter_images(self, source_dir: Path) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """Pré-filtre les images"""
        logger.info("Pré-filtrage des images...")

        valid_images = []
        invalid_images = []

        all_images = self._collect_images(source_dir)
        total = len(all_images)

        if self.progress_callback:
            self.progress_callback(0, total, "Pré-filtrage...")

        for i, image_path in enumerate(all_images):
            if self._stop_requested:
                break

            is_valid, reason, _ = self.check_image_validity(image_path)

            if is_valid:
                dest = self.valid_dir / image_path.name
                if dest.exists():
                    dest = self.valid_dir / f"{image_path.stem}_{int(time.time())}{image_path.suffix}"

                try:
                    shutil.move(str(image_path), str(dest))
                    valid_images.append(dest)
                except Exception as e:
                    invalid_images.append((image_path, f"Erreur déplacement: {e}"))
            else:
                dest = self.invalid_dir / image_path.name
                if dest.exists():
                    dest = self.invalid_dir / f"{image_path.stem}_{int(time.time())}{image_path.suffix}"

                try:
                    shutil.move(str(image_path), str(dest))
                    invalid_images.append((dest, reason))
                except:
                    invalid_images.append((image_path, reason))

            if self.progress_callback:
                self.progress_callback(i + 1, total, f"Pré-filtrage: {i+1}/{total}")

        logger.info(f"Pré-filtrage: {len(valid_images)} valides, {len(invalid_images)} invalides")
        return valid_images, invalid_images

    def process_single_image(self, image_path: Path) -> ProcessingResult:
        """Traite une seule image"""
        filename = image_path.name

        if not Utils.safe_file_exists(image_path):
            return ProcessingResult(False, filename, "Fichier introuvable")

        try:
            if filename in self._processed_files:
                return ProcessingResult(True, filename, "Déjà traitée")

            metadata = self.analyze_image(image_path)

            if metadata is None:
                return ProcessingResult(False, filename, "Échec analyse IA")

            if self.current_batch is None or self.current_batch.image_count >= Config.BATCH_SIZE:
                self.current_batch = self._create_new_batch()

            dest_path = self.current_batch.folder_path / filename
            if dest_path.exists():
                dest_path = self.current_batch.folder_path / f"{image_path.stem}_{int(time.time())}{image_path.suffix}"

            shutil.move(str(image_path), str(dest_path))
            self._append_to_csv(self.current_batch.csv_path, metadata, dest_path.name)

            self.current_batch.image_count += 1
            self._processed_files.add(filename)

            # Sauvegarder l'état
            if self.continuity_manager:
                self.continuity_manager.save_state(
                    self._processed_files,
                    self.batch_number,
                    self._error_files
                )

            return ProcessingResult(True, filename, f"OK - {len(metadata.keywords)} mots-clés", metadata)

        except Exception as e:
            self._error_files[filename] = str(e)
            return ProcessingResult(False, filename, f"Erreur: {e}")

    def _scan_existing_metadata(self, source_dir: Path) -> Set[str]:
        """Scanne les fichiers metadata existants pour trouver les images déjà traitées"""
        already_processed = set()

        # Scanner les dossiers Shutterstock_X existants
        shutterstock_folders = [source_dir / "Shutterstock"]
        for i in range(2, 100):
            folder = source_dir / f"Shutterstock_{i}"
            if folder.exists():
                shutterstock_folders.append(folder)
            else:
                break

        for folder in shutterstock_folders:
            if not folder.exists():
                continue

            # Chercher les fichiers CSV
            csv_files = list(folder.glob("metadata*.csv"))
            for csv_path in csv_files:
                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            filename = row.get('Filename', '')
                            if filename:
                                already_processed.add(filename)
                    logger.info(f"Métadonnées scannées: {csv_path.name} ({len(already_processed)} entrées)")
                except Exception as e:
                    logger.warning(f"Erreur lecture CSV {csv_path}: {e}")

            # Aussi scanner les images présentes dans ce dossier
            for ext in Utils.get_all_extensions():
                for img in folder.glob(f"*{ext}"):
                    already_processed.add(img.name)
                for img in folder.glob(f"*{ext.upper()}"):
                    already_processed.add(img.name)

        return already_processed

    def _find_highest_batch_number(self, source_dir: Path) -> int:
        """Trouve le numéro de lot le plus élevé existant"""
        highest = 0
        if (source_dir / "Shutterstock").exists():
            highest = 1
        for i in range(2, 100):
            if (source_dir / f"Shutterstock_{i}").exists():
                highest = i
            else:
                break
        return highest

    def process_directory(self, directory: str, pre_filter: bool = True,
                         resume: bool = False) -> dict:
        """Traite un répertoire complet"""
        self._stop_requested = False

        stats = {
            'total': 0, 'processed': 0, 'success': 0,
            'failed': 0, 'invalid': 0, 'batches': 0
        }

        source_dir = Path(directory)
        if not source_dir.is_dir():
            return stats

        self.setup_directories(directory)

        # Vérifier reprise - Scanner automatiquement les metadata existantes
        if resume:
            # D'abord scanner les fichiers metadata existants
            existing_processed = self._scan_existing_metadata(source_dir)
            logger.info(f"Images déjà dans les lots Shutterstock: {len(existing_processed)}")

            # Ensuite charger l'état de continuité s'il existe
            if self.continuity_manager:
                state = self.continuity_manager.load_state()
                if state:
                    state_files = set(state.get('processed_files', []))
                    self._processed_files = existing_processed.union(state_files)
                    self.batch_number = max(state.get('current_batch', 0),
                                           self._find_highest_batch_number(source_dir))
                    self._error_files = state.get('failed_files', {})
                else:
                    self._processed_files = existing_processed
                    self.batch_number = self._find_highest_batch_number(source_dir)

            logger.info(f"Reprise: {len(self._processed_files)} images déjà traitées, lot actuel: {self.batch_number}")
        else:
            self.batch_number = 0
            self.current_batch = None
            self._processed_files.clear()
            self._error_files.clear()

        connected, _ = self.verify_connection()
        if not connected:
            return stats

        # Pré-filtrage - ne pas déplacer les images déjà dans Valid
        if pre_filter:
            # Collecter les images déjà dans Valid
            valid_in_folder = set()
            if self.valid_dir and self.valid_dir.exists():
                for ext in Utils.get_all_extensions():
                    for img in self.valid_dir.glob(f"*{ext}"):
                        valid_in_folder.add(img.name)
                    for img in self.valid_dir.glob(f"*{ext.upper()}"):
                        valid_in_folder.add(img.name)

            valid_images, invalid_images = self.pre_filter_images(source_dir)
            stats['invalid'] = len(invalid_images)

            # Ajouter les images déjà dans Valid qui n'ont pas été traitées
            if self.valid_dir and self.valid_dir.exists():
                for ext in Utils.get_all_extensions():
                    for img in self.valid_dir.glob(f"*{ext}"):
                        if img not in valid_images and img.name not in self._processed_files:
                            valid_images.append(img)
                    for img in self.valid_dir.glob(f"*{ext.upper()}"):
                        if img not in valid_images and img.name not in self._processed_files:
                            valid_images.append(img)
        else:
            valid_images = self._collect_images(self.valid_dir if self.valid_dir.exists() else source_dir)

        # Filtrer les déjà traitées
        valid_images = [img for img in valid_images if img.name not in self._processed_files]

        if not valid_images:
            logger.warning("Aucune image à traiter")
            return stats

        total = len(valid_images)
        stats['total'] = total

        if self.progress_callback:
            self.progress_callback(0, total, "Démarrage...")

        processed = 0
        success = 0
        failed = 0

        for image_path in valid_images:
            if self._stop_requested:
                break

            result = self.process_single_image(image_path)
            processed += 1

            if result.success:
                success += 1
            else:
                failed += 1

            if self.progress_callback:
                self.progress_callback(processed, total,
                    f"Analysé: {processed}/{total} (OK: {success}, Échec: {failed})")

            time.sleep(self.cooldown_seconds)

        stats['processed'] = processed
        stats['success'] = success
        stats['failed'] = failed
        stats['batches'] = self.batch_number

        # Nettoyer l'état si terminé
        if not self._stop_requested and self.continuity_manager:
            self.continuity_manager.clear_state()

        logger.info(f"Terminé: {success}/{processed} images traitées")
        return stats

    def stop(self) -> None:
        """Arrête le traitement"""
        self._stop_requested = True


# ============================================================================
# INTERFACE GRAPHIQUE MODERNE
# ============================================================================

def create_gui():
    """Crée l'interface graphique moderne"""

    if not HAS_CTK:
        print("CustomTkinter non disponible")
        return

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    class MainApplication(ctk.CTk):
        def __init__(self):
            super().__init__()

            self.title("Shutterstock AI Metadata Generator v1.0.0")
            self.geometry("1200x900")
            self.minsize(1000, 750)

            # Gestionnaires
            self.ollama_manager = OllamaManager()
            self.analyzer = ImageAnalyzer()
            self.analyzer.progress_callback = self.update_progress

            # État
            self.is_running = False
            self.is_uploading = False

            # Créer l'interface
            self.create_widgets()

            # Vérifications initiales
            self.after(1000, self.refresh_status)
            self.after(2000, self.auto_diagnose_ollama)

        def create_widgets(self):
            """Crée tous les widgets"""

            # Container principal avec scrollbar
            self.main_container = ctk.CTkScrollableFrame(self)
            self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

            # En-tête
            self.create_header()

            # Options rapides (Configuration IA, Gestion Ollama, CPU)
            self.create_quick_options()

            # Onglets principaux
            self.create_tabs()

        def create_header(self):
            """Crée l'en-tête"""
            header = ctk.CTkFrame(self.main_container)
            header.pack(fill="x", pady=(0, 15))

            # Titre
            ctk.CTkLabel(
                header,
                text="Shutterstock AI Metadata Generator v1.0.0",
                font=ctk.CTkFont(size=24, weight="bold")
            ).pack(side="left", padx=15, pady=15)

            # Statut Ollama
            status_frame = ctk.CTkFrame(header, fg_color="transparent")
            status_frame.pack(side="right", padx=15, pady=15)

            self.ollama_status_label = ctk.CTkLabel(
                status_frame,
                text="Vérification Ollama...",
                font=ctk.CTkFont(size=12)
            )
            self.ollama_status_label.pack(side="left", padx=(0, 10))

            ctk.CTkButton(
                status_frame,
                text="Démarrer Ollama",
                width=130,
                height=32,
                fg_color="#2ecc71",
                hover_color="#27ae60",
                command=self.start_ollama
            ).pack(side="left")

        def create_quick_options(self):
            """Crée le panneau des options rapides - Fusionné IA + Ollama"""

            # Frame principale
            quick_frame = ctk.CTkFrame(self.main_container)
            quick_frame.pack(fill="x", pady=(0, 15))

            # Container pour les 2 sections
            options_container = ctk.CTkFrame(quick_frame, fg_color="transparent")
            options_container.pack(fill="x", padx=15, pady=10)

            options_container.grid_columnconfigure(0, weight=2)
            options_container.grid_columnconfigure(1, weight=1)

            # Section 1: Modèle IA & Ollama (fusionnée)
            self.create_model_section(options_container, 0)

            # Section 2: GPU & Paramètres
            self.create_settings_section(options_container, 1)

        def create_model_section(self, parent, column):
            """Section Modèle IA & Serveur Ollama (fusionnée)"""
            frame = ctk.CTkFrame(parent)
            frame.grid(row=0, column=column, padx=5, pady=5, sticky="nsew")

            # Titre avec statut
            header_frame = ctk.CTkFrame(frame, fg_color="transparent")
            header_frame.pack(fill="x", padx=10, pady=(10, 5))

            ctk.CTkLabel(
                header_frame,
                text="Modèle IA",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(side="left")

            self.server_status = ctk.CTkLabel(
                header_frame,
                text="Vérification...",
                font=ctk.CTkFont(size=11)
            )
            self.server_status.pack(side="right")

            # Sélection du modèle (liste dynamique des modèles installés)
            model_select_frame = ctk.CTkFrame(frame, fg_color="transparent")
            model_select_frame.pack(fill="x", padx=10, pady=5)

            ctk.CTkLabel(model_select_frame, text="Modèle actif:").pack(side="left")

            # ComboBox qui sera mise à jour avec les modèles installés
            self.model_combo = ctk.CTkComboBox(
                model_select_frame,
                values=["Chargement..."],
                width=200,
                command=self.on_model_selected
            )
            self.model_combo.pack(side="left", padx=(5, 10))

            # Bouton rafraîchir la liste
            refresh_btn = ctk.CTkButton(
                model_select_frame,
                text="↻",
                width=30,
                height=28,
                command=self.refresh_models_list
            )
            refresh_btn.pack(side="left")
            CTkToolTip(refresh_btn, message="Actualiser la liste des modèles installés")

            # Indicateur modèle chargé
            self.loaded_model_label = ctk.CTkLabel(
                frame,
                text="Aucun modèle chargé",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            self.loaded_model_label.pack(anchor="w", padx=10, pady=(0, 5))

            # Boutons de contrôle Ollama
            control_frame = ctk.CTkFrame(frame, fg_color="transparent")
            control_frame.pack(fill="x", padx=10, pady=5)

            self.start_btn = ctk.CTkButton(
                control_frame,
                text="Démarrer",
                width=75,
                height=28,
                fg_color="#2ecc71",
                command=self.start_ollama
            )
            self.start_btn.pack(side="left", padx=(0, 5))

            self.stop_btn = ctk.CTkButton(
                control_frame,
                text="Arrêter",
                width=70,
                height=28,
                fg_color="#e74c3c",
                command=self.stop_ollama
            )
            self.stop_btn.pack(side="left", padx=(0, 5))

            self.repair_btn = ctk.CTkButton(
                control_frame,
                text="Réparer",
                width=70,
                height=28,
                fg_color="#9b59b6",
                hover_color="#8e44ad",
                command=self.repair_ollama
            )
            self.repair_btn.pack(side="left", padx=(0, 5))

            load_btn = ctk.CTkButton(
                control_frame,
                text="Charger",
                width=70,
                height=28,
                fg_color="#3498db",
                command=self.load_model
            )
            load_btn.pack(side="left")
            CTkToolTip(load_btn, message="Charge le modèle sélectionné en mémoire GPU")

            # Section téléchargement nouveau modèle
            download_frame = ctk.CTkFrame(frame, fg_color="transparent")
            download_frame.pack(fill="x", padx=10, pady=(10, 5))

            ctk.CTkLabel(
                download_frame,
                text="Télécharger un modèle:",
                font=ctk.CTkFont(size=11)
            ).pack(side="left")

            # ComboBox pour les modèles disponibles au téléchargement
            self.download_combo = ctk.CTkComboBox(
                download_frame,
                values=Config.AVAILABLE_MODELS,
                width=160,
                height=26
            )
            self.download_combo.set("llama3.2-vision:11b")
            self.download_combo.pack(side="left", padx=(5, 5))

            download_btn = ctk.CTkButton(
                download_frame,
                text="Télécharger",
                width=90,
                height=26,
                fg_color="#f39c12",
                hover_color="#e67e22",
                command=self.download_model
            )
            download_btn.pack(side="left")
            CTkToolTip(download_btn, message="Télécharge le modèle depuis ollama.ai\n(peut prendre plusieurs minutes)")

            # Label modèles installés
            self.models_label = ctk.CTkLabel(
                frame,
                text="Modèles installés: ...",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            self.models_label.pack(anchor="w", padx=10, pady=(5, 10))

        def create_settings_section(self, parent, column):
            """Section Paramètres & GPU"""
            frame = ctk.CTkFrame(parent)
            frame.grid(row=0, column=column, padx=5, pady=5, sticky="nsew")

            ctk.CTkLabel(
                frame,
                text="Paramètres",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=10, pady=(10, 5))

            # GPU Status
            self.gpu_status = ctk.CTkLabel(
                frame,
                text=self.ollama_manager.get_gpu_status_string(),
                font=ctk.CTkFont(size=11),
                text_color="#3498db"
            )
            self.gpu_status.pack(anchor="w", padx=10, pady=5)

            # Couches GPU
            gpu_frame = ctk.CTkFrame(frame, fg_color="transparent")
            gpu_frame.pack(fill="x", padx=10, pady=3)

            ctk.CTkLabel(gpu_frame, text="Couches GPU:").pack(side="left")
            self.gpu_layers = ctk.CTkEntry(gpu_frame, width=50)
            recommended = self.ollama_manager.get_recommended_settings()
            self.gpu_layers.insert(0, str(recommended['num_gpu']))
            self.gpu_layers.pack(side="left", padx=(5, 0))

            # Délai entre requêtes
            cooldown_frame = ctk.CTkFrame(frame, fg_color="transparent")
            cooldown_frame.pack(fill="x", padx=10, pady=3)

            ctk.CTkLabel(cooldown_frame, text="Délai (s):").pack(side="left")
            self.cooldown_entry = ctk.CTkEntry(cooldown_frame, width=50)
            self.cooldown_entry.insert(0, "2.0")
            self.cooldown_entry.pack(side="left", padx=(5, 0))
            CTkToolTip(self.cooldown_entry, message="Délai entre chaque analyse d'image")

            # Workers
            workers_frame = ctk.CTkFrame(frame, fg_color="transparent")
            workers_frame.pack(fill="x", padx=10, pady=3)

            ctk.CTkLabel(workers_frame, text="Workers:").pack(side="left")
            self.workers_entry = ctk.CTkEntry(workers_frame, width=50)
            self.workers_entry.insert(0, "2")
            self.workers_entry.pack(side="left", padx=(5, 0))
            CTkToolTip(self.workers_entry, message="Nombre de workers parallèles")

            # Boutons
            btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
            btn_frame.pack(fill="x", padx=10, pady=(10, 10))

            ctk.CTkButton(
                btn_frame,
                text="Actualiser GPU",
                width=100,
                height=26,
                command=self.refresh_gpu
            ).pack(side="left", padx=(0, 5))

            ctk.CTkButton(
                btn_frame,
                text="Tester",
                width=60,
                height=26,
                fg_color="#9b59b6",
                command=self.test_model
            ).pack(side="left")
            CTkToolTip(btn_frame.winfo_children()[-1], message="Teste le modèle avec une image")

        def create_tabs(self):
            """Crée les onglets principaux"""

            self.tabview = ctk.CTkTabview(self.main_container)
            self.tabview.pack(fill="both", expand=True)

            # Onglets
            self.tab_analyze = self.tabview.add("Analyse")
            self.tab_checklist = self.tabview.add("Checklist")
            self.tab_upload = self.tabview.add("Upload FTPS")
            self.tab_log = self.tabview.add("Journal")

            self.create_analyze_tab()
            self.create_checklist_tab()
            self.create_upload_tab()
            self.create_log_tab()

        def create_analyze_tab(self):
            """Onglet Analyse"""
            tab = self.tab_analyze

            # Sélection dossier
            dir_frame = ctk.CTkFrame(tab)
            dir_frame.pack(fill="x", pady=(10, 10), padx=10)

            dir_header = ctk.CTkFrame(dir_frame, fg_color="transparent")
            dir_header.pack(fill="x", padx=15, pady=(15, 5))

            ctk.CTkLabel(
                dir_header,
                text="Dossier source",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(side="left")

            # Bouton aide
            help_btn = ctk.CTkButton(
                dir_header,
                text="?",
                width=25,
                height=25,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=self.show_help
            )
            help_btn.pack(side="right")
            CTkToolTip(help_btn, message="Afficher l'aide et la structure des dossiers")

            dir_inner = ctk.CTkFrame(dir_frame, fg_color="transparent")
            dir_inner.pack(fill="x", padx=15, pady=(0, 15))

            self.dir_entry = ctk.CTkEntry(
                dir_inner,
                placeholder_text="Ex: C:/Mes_Photos (les images seront triées automatiquement)",
                width=500
            )
            self.dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
            CTkToolTip(self.dir_entry,
                      message="Dossier contenant vos photos à analyser.\n\n"
                              "Après traitement:\n"
                              "• Valid/ → Photos acceptées\n"
                              "• Invalid/ → Photos rejetées\n"
                              "• Shutterstock/ → Photos + métadonnées CSV")

            ctk.CTkButton(
                dir_inner,
                text="Parcourir",
                width=100,
                command=self.browse_folder
            ).pack(side="right")

            # Options de traitement
            opts_frame = ctk.CTkFrame(tab)
            opts_frame.pack(fill="x", pady=(0, 10), padx=10)

            ctk.CTkLabel(
                opts_frame,
                text="Options de traitement",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=15, pady=(10, 5))

            opts_inner = ctk.CTkFrame(opts_frame, fg_color="transparent")
            opts_inner.pack(fill="x", padx=15, pady=(0, 10))

            self.prefilter_var = ctk.BooleanVar(value=True)
            prefilter_cb = ctk.CTkCheckBox(
                opts_inner,
                text="Pré-filtrer",
                variable=self.prefilter_var
            )
            prefilter_cb.pack(side="left", padx=(0, 15))
            CTkToolTip(prefilter_cb,
                      message="Vérifie les critères Shutterstock:\n"
                              "• Résolution min: 4 MP\n"
                              "• Taille max: 50 MB\n"
                              "• Formats: JPEG, TIFF\n\n"
                              "Les photos invalides vont dans 'Invalid/'")

            self.resume_var = ctk.BooleanVar(value=False)
            resume_cb = ctk.CTkCheckBox(
                opts_inner,
                text="Reprendre",
                variable=self.resume_var
            )
            resume_cb.pack(side="left", padx=(0, 15))
            CTkToolTip(resume_cb,
                      message="Reprend un traitement interrompu.\n"
                              "Ignore les photos déjà analysées.")

            self.skip_analyzed_var = ctk.BooleanVar(value=True)
            skip_cb = ctk.CTkCheckBox(
                opts_inner,
                text="Ignorer existants",
                variable=self.skip_analyzed_var
            )
            skip_cb.pack(side="left", padx=(0, 15))
            CTkToolTip(skip_cb,
                      message="Ignore les photos qui ont déjà\n"
                              "des métadonnées dans le CSV.")

            # Options de préfiltrage avancées
            prefilter_frame = ctk.CTkFrame(tab)
            prefilter_frame.pack(fill="x", pady=(0, 10), padx=10)

            ctk.CTkLabel(
                prefilter_frame,
                text="Critères de préfiltrage",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=15, pady=(10, 5))

            prefilter_inner = ctk.CTkFrame(prefilter_frame, fg_color="transparent")
            prefilter_inner.pack(fill="x", padx=15, pady=(0, 10))

            # Résolution minimum
            ctk.CTkLabel(prefilter_inner, text="Résolution min (MP):").pack(side="left")
            self.min_mp_entry = ctk.CTkEntry(prefilter_inner, width=50, height=28)
            self.min_mp_entry.insert(0, str(Config.PREFILTER_DEFAULTS['min_megapixels']))
            self.min_mp_entry.pack(side="left", padx=(5, 15))
            CTkToolTip(self.min_mp_entry,
                      message="Résolution minimum en mégapixels.\n"
                              "Shutterstock exige 4 MP minimum.\n"
                              "Valeur par défaut: 4.0")

            # Taille max
            ctk.CTkLabel(prefilter_inner, text="Taille max (MB):").pack(side="left")
            self.max_size_entry = ctk.CTkEntry(prefilter_inner, width=50, height=28)
            self.max_size_entry.insert(0, str(Config.PREFILTER_DEFAULTS['max_file_size_mb']))
            self.max_size_entry.pack(side="left", padx=(5, 15))
            CTkToolTip(self.max_size_entry,
                      message="Taille maximale du fichier en MB.\n"
                              "Shutterstock limite à 50 MB pour JPEG.\n"
                              "Valeur par défaut: 50.0")

            # Correction orientation
            self.fix_orientation_var = ctk.BooleanVar(value=Config.PREFILTER_DEFAULTS['check_orientation'])
            orient_cb = ctk.CTkCheckBox(
                prefilter_inner,
                text="Corriger orientation",
                variable=self.fix_orientation_var
            )
            orient_cb.pack(side="left")
            CTkToolTip(orient_cb,
                      message="Corrige automatiquement l'orientation\n"
                              "des photos selon les données EXIF.")

            # Statistiques
            stats_frame = ctk.CTkFrame(tab)
            stats_frame.pack(fill="x", pady=(0, 15), padx=10)

            ctk.CTkLabel(
                stats_frame,
                text="Statistiques",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=15, pady=(15, 10))

            stats_inner = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stats_inner.pack(fill="x", padx=15, pady=(0, 15))

            self.stats_labels = {}
            for key, label, color in [
                ('total', 'Total', '#3498db'),
                ('success', 'Succès', '#2ecc71'),
                ('failed', 'Échecs', '#e74c3c'),
                ('invalid', 'Invalides', '#f39c12')
            ]:
                container = ctk.CTkFrame(stats_inner, width=100)
                container.pack(side="left", padx=15)

                value_lbl = ctk.CTkLabel(
                    container,
                    text="0",
                    font=ctk.CTkFont(size=28, weight="bold"),
                    text_color=color
                )
                value_lbl.pack()

                ctk.CTkLabel(container, text=label, text_color="gray").pack()
                self.stats_labels[key] = value_lbl

            # Progression
            prog_frame = ctk.CTkFrame(tab)
            prog_frame.pack(fill="x", pady=(0, 15), padx=10)

            self.progress_bar = ctk.CTkProgressBar(prog_frame, width=700)
            self.progress_bar.pack(padx=15, pady=15)
            self.progress_bar.set(0)

            self.status_label = ctk.CTkLabel(
                prog_frame,
                text="Prêt - Sélectionnez un dossier"
            )
            self.status_label.pack(padx=15, pady=(0, 15))

            # Boutons
            btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
            btn_frame.pack(fill="x", pady=(0, 15), padx=10)

            self.analyze_btn = ctk.CTkButton(
                btn_frame,
                text="Démarrer l'analyse",
                width=180,
                height=45,
                font=ctk.CTkFont(size=14, weight="bold"),
                command=self.start_analysis
            )
            self.analyze_btn.pack(side="left", padx=(0, 10))

            self.stop_analyze_btn = ctk.CTkButton(
                btn_frame,
                text="Arrêter",
                width=100,
                height=45,
                fg_color="#e74c3c",
                state="disabled",
                command=self.stop_analysis
            )
            self.stop_analyze_btn.pack(side="left", padx=(0, 10))

            ctk.CTkButton(
                btn_frame,
                text="Ouvrir résultats",
                width=140,
                height=45,
                command=self.open_results
            ).pack(side="left")

        def create_checklist_tab(self):
            """Onglet Checklist"""
            tab = self.tab_checklist

            ctk.CTkLabel(
                tab,
                text="Validation Photos / Métadonnées",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(anchor="w", padx=15, pady=(15, 10))

            # Sélection du dossier source
            batch_frame = ctk.CTkFrame(tab)
            batch_frame.pack(fill="x", pady=(0, 15), padx=10)

            ctk.CTkLabel(
                batch_frame,
                text="Dossier source à valider",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=15, pady=(15, 5))

            batch_inner = ctk.CTkFrame(batch_frame, fg_color="transparent")
            batch_inner.pack(fill="x", padx=15, pady=(0, 15))

            self.batch_entry = ctk.CTkEntry(
                batch_inner,
                placeholder_text="Ex: C:/Photos (contient les sous-dossiers Shutterstock, Shutterstock_2, ...)",
                width=500
            )
            self.batch_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

            ctk.CTkButton(
                batch_inner,
                text="Parcourir",
                width=100,
                command=self.browse_batch
            ).pack(side="right")

            # Options de validation
            options_frame = ctk.CTkFrame(tab)
            options_frame.pack(fill="x", pady=(0, 15), padx=10)

            ctk.CTkLabel(
                options_frame,
                text="Options de validation",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=15, pady=(15, 5))

            opts_inner = ctk.CTkFrame(options_frame, fg_color="transparent")
            opts_inner.pack(fill="x", padx=15, pady=(0, 15))

            # Option récursive
            self.recursive_var = ctk.BooleanVar(value=True)
            recursive_cb = ctk.CTkCheckBox(
                opts_inner,
                text="Recherche récursive (tous les sous-dossiers)",
                variable=self.recursive_var
            )
            recursive_cb.pack(side="left", padx=(0, 20))
            CTkToolTip(recursive_cb,
                      message="Recherche dans tous les sous-dossiers\n"
                              "pour trouver les lots avec fichiers CSV.\n\n"
                              "Idéal pour valider plusieurs lots à la fois.")

            # Ligne 2 d'options
            opts_inner2 = ctk.CTkFrame(options_frame, fg_color="transparent")
            opts_inner2.pack(fill="x", padx=15, pady=(0, 15))

            # Option déplacement
            self.move_unmatched_var = ctk.BooleanVar(value=True)
            move_cb = ctk.CTkCheckBox(
                opts_inner2,
                text="Déplacer photos orphelines",
                variable=self.move_unmatched_var,
                command=self._toggle_move_options
            )
            move_cb.pack(side="left", padx=(0, 15))
            CTkToolTip(move_cb,
                      message="Déplace les photos sans métadonnées\n"
                              "vers un dossier séparé.")

            # Option: copier dans dossier source (pas de sous-dossier unmatched)
            self.copy_to_source_var = ctk.BooleanVar(value=False)
            copy_source_cb = ctk.CTkCheckBox(
                opts_inner2,
                text="Copier vers source",
                variable=self.copy_to_source_var
            )
            copy_source_cb.pack(side="left", padx=(0, 15))
            CTkToolTip(copy_source_cb,
                      message="Copie les photos orphelines directement\n"
                              "dans le dossier source original.\n\n"
                              "Utile pour retraiter les images qui\n"
                              "n'ont pas eu de métadonnées générées.")

            # Option sous-dossiers par lot
            self.subfolder_per_batch_var = ctk.BooleanVar(value=True)
            subfolder_cb = ctk.CTkCheckBox(
                opts_inner2,
                text="Sous-dossier par lot",
                variable=self.subfolder_per_batch_var
            )
            subfolder_cb.pack(side="left")
            CTkToolTip(subfolder_cb,
                      message="Crée un sous-dossier par lot dans\n"
                              "'photos_unmatched' pour organiser les photos.\n\n"
                              "Désactivé si 'Copier vers source' est coché.")

            # Actions
            action_frame = ctk.CTkFrame(tab, fg_color="transparent")
            action_frame.pack(fill="x", pady=(0, 15), padx=10)

            self.validate_btn = ctk.CTkButton(
                action_frame,
                text="Valider tous les lots",
                width=180,
                height=40,
                font=ctk.CTkFont(weight="bold"),
                command=self.validate_all_batches
            )
            self.validate_btn.pack(side="left", padx=(0, 10))
            CTkToolTip(self.validate_btn,
                      message="Lance la validation de tous les lots\n"
                              "trouvés dans le dossier source")

            ctk.CTkButton(
                action_frame,
                text="Exporter le rapport",
                width=150,
                height=40,
                command=self.export_checklist_report
            ).pack(side="left", padx=(0, 10))

            # Barre de progression
            self.checklist_progress = ctk.CTkProgressBar(action_frame, width=200)
            self.checklist_progress.pack(side="left", padx=(20, 10))
            self.checklist_progress.set(0)

            self.checklist_status = ctk.CTkLabel(
                action_frame,
                text="",
                font=ctk.CTkFont(size=11)
            )
            self.checklist_status.pack(side="left")

            # Statistiques globales
            stats_frame = ctk.CTkFrame(tab)
            stats_frame.pack(fill="x", pady=(0, 10), padx=10)

            ctk.CTkLabel(
                stats_frame,
                text="Statistiques globales",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=15, pady=(10, 5))

            stats_inner = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stats_inner.pack(fill="x", padx=15, pady=(0, 10))

            self.checklist_stats = {}
            for key, label, color in [
                ('batches', 'Lots', '#3498db'),
                ('matched', 'OK', '#2ecc71'),
                ('no_meta', 'Sans meta', '#e74c3c'),
                ('no_photo', 'Sans photo', '#f39c12'),
                ('incomplete', 'Incomplets', '#9b59b6')
            ]:
                container = ctk.CTkFrame(stats_inner, width=80)
                container.pack(side="left", padx=10)

                value_lbl = ctk.CTkLabel(
                    container,
                    text="0",
                    font=ctk.CTkFont(size=20, weight="bold"),
                    text_color=color
                )
                value_lbl.pack()

                ctk.CTkLabel(container, text=label, text_color="gray",
                           font=ctk.CTkFont(size=10)).pack()
                self.checklist_stats[key] = value_lbl

            # Rapport détaillé
            report_frame = ctk.CTkFrame(tab)
            report_frame.pack(fill="both", expand=True, pady=(0, 10), padx=10)

            ctk.CTkLabel(
                report_frame,
                text="Rapport détaillé",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=15, pady=(10, 5))

            self.checklist_text = ctk.CTkTextbox(
                report_frame,
                height=250,
                font=ctk.CTkFont(family="Consolas", size=11)
            )
            self.checklist_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        def create_upload_tab(self):
            """Onglet Upload FTPS"""
            tab = self.tab_upload

            ctk.CTkLabel(
                tab,
                text="Upload FTPS vers Shutterstock",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(anchor="w", padx=15, pady=(15, 10))

            # Identifiants
            creds_frame = ctk.CTkFrame(tab)
            creds_frame.pack(fill="x", pady=(0, 15), padx=10)

            ctk.CTkLabel(
                creds_frame,
                text="Identifiants",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=15, pady=(15, 5))

            creds_inner = ctk.CTkFrame(creds_frame, fg_color="transparent")
            creds_inner.pack(fill="x", padx=15, pady=(0, 15))

            ctk.CTkLabel(creds_inner, text="Email:").grid(row=0, column=0, sticky="w", pady=5)
            self.ftps_user = ctk.CTkEntry(creds_inner, width=250)
            self.ftps_user.grid(row=0, column=1, padx=(10, 0), pady=5)

            ctk.CTkLabel(creds_inner, text="Mot de passe:").grid(row=1, column=0, sticky="w", pady=5)
            self.ftps_pass = ctk.CTkEntry(creds_inner, width=250, show="*")
            self.ftps_pass.grid(row=1, column=1, padx=(10, 0), pady=5)

            ctk.CTkButton(
                creds_inner,
                text="Tester",
                width=100,
                command=self.test_ftps
            ).grid(row=0, column=2, rowspan=2, padx=(20, 0))

            self.ftps_status = ctk.CTkLabel(creds_frame, text="")
            self.ftps_status.pack(anchor="w", padx=15, pady=(0, 10))

            # Dossier à uploader
            upload_frame = ctk.CTkFrame(tab)
            upload_frame.pack(fill="x", pady=(0, 15), padx=10)

            ctk.CTkLabel(
                upload_frame,
                text="Dossier à uploader",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=15, pady=(15, 5))

            upload_inner = ctk.CTkFrame(upload_frame, fg_color="transparent")
            upload_inner.pack(fill="x", padx=15, pady=(0, 15))

            self.upload_dir = ctk.CTkEntry(upload_inner, width=500)
            self.upload_dir.pack(side="left", fill="x", expand=True, padx=(0, 10))

            ctk.CTkButton(
                upload_inner,
                text="Parcourir",
                width=100,
                command=self.browse_upload
            ).pack(side="right")

            # Progression
            prog_frame = ctk.CTkFrame(tab)
            prog_frame.pack(fill="x", pady=(0, 15), padx=10)

            self.upload_progress = ctk.CTkProgressBar(prog_frame, width=700)
            self.upload_progress.pack(padx=15, pady=15)
            self.upload_progress.set(0)

            self.upload_status = ctk.CTkLabel(prog_frame, text="En attente...")
            self.upload_status.pack(padx=15, pady=(0, 15))

            # Boutons
            btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
            btn_frame.pack(fill="x", padx=10)

            self.upload_btn = ctk.CTkButton(
                btn_frame,
                text="Démarrer l'upload",
                width=180,
                height=45,
                fg_color="#2ecc71",
                command=self.start_upload
            )
            self.upload_btn.pack(side="left", padx=(0, 10))

            self.stop_upload_btn = ctk.CTkButton(
                btn_frame,
                text="Arrêter",
                width=100,
                height=45,
                fg_color="#e74c3c",
                state="disabled",
                command=self.stop_upload
            )
            self.stop_upload_btn.pack(side="left")

        def create_log_tab(self):
            """Onglet Journal"""
            tab = self.tab_log

            ctk.CTkLabel(
                tab,
                text="Journal d'activité",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(anchor="w", padx=15, pady=(15, 10))

            self.log_text = ctk.CTkTextbox(
                tab,
                font=ctk.CTkFont(family="Consolas", size=11)
            )
            self.log_text.pack(fill="both", expand=True, padx=15, pady=(0, 10))

            btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
            btn_frame.pack(fill="x", padx=15, pady=(0, 10))

            ctk.CTkButton(
                btn_frame,
                text="Effacer",
                width=100,
                command=lambda: self.log_text.delete("1.0", "end")
            ).pack(side="left", padx=(0, 10))

            ctk.CTkButton(
                btn_frame,
                text="Exporter",
                width=100,
                command=self.export_log
            ).pack(side="left")

            ctk.CTkButton(
                btn_frame,
                text="Logs Ollama",
                width=120,
                fg_color="#9b59b6",
                command=self.show_ollama_logs
            ).pack(side="left", padx=(10, 0))

            CTkToolTip(btn_frame.winfo_children()[-1],
                message="Affiche les logs du serveur Ollama\npour diagnostiquer les problèmes")

            # Setup log handler
            self.setup_log_handler()

        def setup_log_handler(self):
            """Configure le handler de log"""
            class GUILogHandler(logging.Handler):
                def __init__(self, text_widget, app):
                    super().__init__()
                    self.text_widget = text_widget
                    self.app = app

                def emit(self, record):
                    msg = self.format(record)
                    def append():
                        self.text_widget.insert("end", msg + '\n')
                        self.text_widget.see("end")
                    self.app.after(0, append)

            handler = GUILogHandler(self.log_text, self)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
            logger.addHandler(handler)

        # =====================================================================
        # MÉTHODES D'INTERFACE
        # =====================================================================

        def log_message(self, message: str):
            """Ajoute un message au journal"""
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.log_text.insert("end", f"[{timestamp}] {message}\n")
            self.log_text.see("end")

        def update_progress(self, current: int, total: int, message: str):
            """Met à jour la progression"""
            if total > 0:
                self.progress_bar.set(current / total)
            self.status_label.configure(text=message)

        def refresh_status(self):
            """Actualise tous les statuts"""
            def check():
                # Ollama
                if self.ollama_manager.is_serving():
                    self.after(0, lambda: self.ollama_status_label.configure(
                        text="Ollama: En cours", text_color="green"))
                    self.after(0, lambda: self.server_status.configure(
                        text="Serveur actif", text_color="green"))

                    success, models = self.ollama_manager.list_models()
                    loaded_model = self.ollama_manager.get_loaded_model()

                    if success:
                        # Mettre à jour la ComboBox avec les modèles installés
                        if models:
                            self.after(0, lambda m=models: self._update_model_combo(m))

                        # Afficher le modèle chargé
                        if loaded_model:
                            self.after(0, lambda lm=loaded_model: self.loaded_model_label.configure(
                                text=f"Chargé: {lm}", text_color="#2ecc71"))
                            model_display = f"Installés: {', '.join(models[:3])}" + ("..." if len(models) > 3 else "")
                            self.after(0, lambda md=model_display: self.models_label.configure(
                                text=md, text_color="gray"))
                        else:
                            self.after(0, lambda: self.loaded_model_label.configure(
                                text="Aucun modèle chargé", text_color="orange"))
                            self.after(0, lambda: self.models_label.configure(
                                text=f"Installés: {', '.join(models[:3])}" + ("..." if len(models) > 3 else ""),
                                text_color="gray"))
                    else:
                        self.after(0, lambda: self.models_label.configure(
                            text="Aucun modèle installé", text_color="orange"))
                else:
                    self.after(0, lambda: self.ollama_status_label.configure(
                        text="Ollama: Arrêté", text_color="red"))
                    self.after(0, lambda: self.server_status.configure(
                        text="Serveur arrêté", text_color="red"))
                    self.after(0, lambda: self.models_label.configure(
                        text="Démarrez Ollama pour voir les modèles", text_color="gray"))
                    self.after(0, lambda: self.loaded_model_label.configure(
                        text="Serveur non démarré", text_color="red"))

            threading.Thread(target=check, daemon=True).start()

        def _update_model_combo(self, models: list):
            """Met à jour la ComboBox avec les modèles installés"""
            if models:
                self.model_combo.configure(values=models)
                # Sélectionner le premier modèle si aucun n'est sélectionné
                current = self.model_combo.get()
                if current == "Chargement..." or current not in models:
                    self.model_combo.set(models[0])

        def refresh_models_list(self):
            """Rafraîchit la liste des modèles installés"""
            self.log_message("Actualisation de la liste des modèles...")

            def refresh():
                if not self.ollama_manager.is_serving():
                    self.after(0, lambda: messagebox.showwarning(
                        "Ollama non démarré",
                        "Démarrez d'abord le serveur Ollama pour voir les modèles installés."))
                    return

                success, models = self.ollama_manager.list_models()
                if success and models:
                    self.after(0, lambda m=models: self._update_model_combo(m))
                    self.after(0, lambda: self.log_message(f"Modèles trouvés: {', '.join(models)}"))
                else:
                    self.after(0, lambda: messagebox.showinfo(
                        "Aucun modèle",
                        "Aucun modèle installé. Utilisez 'Télécharger' pour en installer un."))

            threading.Thread(target=refresh, daemon=True).start()

        def on_model_selected(self, model_name: str):
            """Appelé quand un modèle est sélectionné dans la liste"""
            self.log_message(f"Modèle sélectionné: {model_name}")
            # Mettre à jour l'analyseur avec le nouveau modèle
            self.analyzer.update_settings(model=model_name)

        def refresh_gpu(self):
            """Actualise le statut GPU"""
            self.ollama_manager.gpu_info = self.ollama_manager._detect_gpu()
            self.gpu_status.configure(text=self.ollama_manager.get_gpu_status_string())
            recommended = self.ollama_manager.get_recommended_settings()
            self.gpu_layers.delete(0, "end")
            self.gpu_layers.insert(0, str(recommended['num_gpu']))

        def auto_diagnose_ollama(self):
            """Diagnostique automatiquement Ollama au démarrage et propose réparation si nécessaire"""
            def diagnose():
                # Vérifier si Ollama fonctionne
                if self.ollama_manager.is_serving():
                    return  # Tout va bien

                # Attendre un peu plus au cas où le serveur démarre
                time.sleep(3)
                if self.ollama_manager.is_serving():
                    return

                # Diagnostiquer les problèmes
                problems = self.ollama_manager.diagnose_problems()
                critical_problems = [p for p in problems if p['severity'] in ('critical', 'high')]

                if critical_problems:
                    def show_repair_dialog():
                        problem_list = "\n".join([f"• {p['problem']}" for p in critical_problems[:3]])
                        response = messagebox.askyesno(
                            "Problèmes Ollama détectés",
                            f"Des problèmes ont été détectés:\n\n{problem_list}\n\n"
                            "Voulez-vous lancer une réparation automatique?",
                            icon='warning'
                        )
                        if response:
                            self.repair_ollama()

                    self.after(0, show_repair_dialog)

            threading.Thread(target=diagnose, daemon=True).start()

        def start_ollama(self):
            """Démarre Ollama"""
            self.log_message("Démarrage d'Ollama...")

            def start():
                success, msg = self.ollama_manager.start_server()
                self.after(0, lambda: self.log_message(msg))
                self.after(0, self.refresh_status)

            threading.Thread(target=start, daemon=True).start()

        def stop_ollama(self):
            """Arrête Ollama"""
            def stop():
                success, msg = self.ollama_manager.stop_server()
                self.after(0, lambda: self.log_message(msg))
                self.after(0, self.refresh_status)

            threading.Thread(target=stop, daemon=True).start()

        def repair_ollama(self):
            """Répare Ollama en cas de problème"""
            self.log_message("Réparation d'Ollama en cours...")
            self.server_status.configure(text="Réparation en cours...", text_color="orange")
            self.repair_btn.configure(state="disabled")

            def repair():
                def progress_cb(msg):
                    self.after(0, lambda m=msg: self.log_message(m))
                    self.after(0, lambda m=msg: self.server_status.configure(text=m[:40]))

                success, msg = self.ollama_manager.repair_and_start(progress_callback=progress_cb)

                self.after(0, lambda: self.log_message(msg))
                self.after(0, self.refresh_status)
                self.after(0, lambda: self.repair_btn.configure(state="normal"))

                if success:
                    self.after(0, lambda: messagebox.showinfo("Succès",
                        "Ollama a été réparé et redémarré avec succès!"))
                else:
                    # Afficher le diagnostic
                    problems = self.ollama_manager.diagnose_problems()
                    if problems:
                        problem_text = "\n".join([f"• {p['problem']}\n  → {p['solution']}" for p in problems])
                        self.after(0, lambda: messagebox.showwarning("Problèmes détectés",
                            f"La réparation automatique a échoué.\n\nProblèmes identifiés:\n{problem_text}"))
                    else:
                        self.after(0, lambda: messagebox.showerror("Échec",
                            f"Impossible de réparer Ollama.\n\n{msg}\n\nEssayez de redémarrer votre ordinateur."))

            threading.Thread(target=repair, daemon=True).start()

        def download_model(self):
            """Télécharge un modèle Ollama"""
            model = self.download_combo.get()
            self.log_message(f"Téléchargement du modèle {model}...")
            self.server_status.configure(text=f"Téléchargement {model}...")

            def download():
                def progress_cb(line):
                    self.after(0, lambda: self.server_status.configure(text=line[:50]))

                success, msg = self.ollama_manager.pull_model(model, callback=progress_cb)
                self.after(0, lambda: self.log_message(msg))
                self.after(0, self.refresh_status)
                # Rafraîchir la liste des modèles installés
                self.after(0, self.refresh_models_list)

                if success:
                    self.after(0, lambda: messagebox.showinfo("Succès", f"Modèle {model} téléchargé!"))

            threading.Thread(target=download, daemon=True).start()

        def load_model(self):
            """Charge un modèle en mémoire GPU"""
            model = self.model_combo.get()
            if not model:
                messagebox.showwarning("Attention", "Sélectionnez d'abord un modèle installé")
                return
            self.log_message(f"Chargement du modèle {model}...")
            self.server_status.configure(text=f"Chargement {model}...")

            try:
                num_gpu = int(self.gpu_layers.get())
            except:
                num_gpu = None

            def load():
                success, msg = self.ollama_manager.load_model(model, num_gpu=num_gpu)
                self.after(0, lambda: self.log_message(msg))
                self.after(0, self.refresh_status)

                if success:
                    self.after(0, lambda: messagebox.showinfo("Succès", f"Modèle {model} chargé en mémoire!"))
                else:
                    self.after(0, lambda: messagebox.showerror("Erreur", msg))

            threading.Thread(target=load, daemon=True).start()

        def test_model(self):
            """Teste un modèle avec une image simple"""
            model = self.model_combo.get()
            if not model:
                messagebox.showwarning("Attention", "Sélectionnez d'abord un modèle installé")
                return
            self.log_message(f"Test du modèle {model}...")
            self.server_status.configure(text=f"Test {model}...")

            def test():
                try:
                    # Créer une image test simple (1x1 pixel blanc)
                    test_image = Image.new('RGB', (100, 100), color='white')
                    buffer = io.BytesIO()
                    test_image.save(buffer, format='JPEG')
                    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                    # Envoyer au modèle
                    response = self.analyzer.client.chat(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": "Describe this image in one sentence.",
                            "images": [img_b64]
                        }],
                        options={"num_predict": 50}
                    )

                    result = response.message.content
                    self.after(0, lambda: self.log_message(f"Test réussi: {result[:100]}"))
                    self.after(0, lambda: messagebox.showinfo("Test réussi",
                        f"Le modèle {model} fonctionne!\n\nRéponse: {result[:200]}"))

                except Exception as e:
                    self.after(0, lambda: self.log_message(f"Test échoué: {e}"))
                    self.after(0, lambda: messagebox.showerror("Test échoué",
                        f"Erreur avec le modèle {model}:\n{e}"))

                self.after(0, self.refresh_status)

            threading.Thread(target=test, daemon=True).start()

        def show_help(self):
            """Affiche la fenêtre d'aide"""
            help_window = ctk.CTkToplevel(self)
            help_window.title("Aide - Shutterstock Analyzer")
            help_window.geometry("700x600")
            help_window.transient(self)

            # Contenu
            notebook = ctk.CTkTabview(help_window)
            notebook.pack(fill="both", expand=True, padx=10, pady=10)

            # Onglet Description
            tab1 = notebook.add("Description")
            text1 = ctk.CTkTextbox(tab1, font=ctk.CTkFont(size=12))
            text1.pack(fill="both", expand=True, padx=5, pady=5)
            text1.insert("1.0", Config.HELP_TEXTS['app_description'])
            text1.configure(state="disabled")

            # Onglet Workflow
            tab2 = notebook.add("Flux de travail")
            text2 = ctk.CTkTextbox(tab2, font=ctk.CTkFont(family="Consolas", size=11))
            text2.pack(fill="both", expand=True, padx=5, pady=5)
            text2.insert("1.0", Config.HELP_TEXTS['workflow'])
            text2.configure(state="disabled")

            # Onglet Structure
            tab3 = notebook.add("Structure dossiers")
            text3 = ctk.CTkTextbox(tab3, font=ctk.CTkFont(family="Consolas", size=11))
            text3.pack(fill="both", expand=True, padx=5, pady=5)
            text3.insert("1.0", Config.HELP_TEXTS['folder_structure'])
            text3.configure(state="disabled")

            # Onglet Préfiltrage
            tab4 = notebook.add("Préfiltrage")
            text4 = ctk.CTkTextbox(tab4, font=ctk.CTkFont(size=12))
            text4.pack(fill="both", expand=True, padx=5, pady=5)
            text4.insert("1.0", Config.HELP_TEXTS['prefilter'])
            text4.configure(state="disabled")

            # Onglet Checklist
            tab5 = notebook.add("Checklist")
            text5 = ctk.CTkTextbox(tab5, font=ctk.CTkFont(size=12))
            text5.pack(fill="both", expand=True, padx=5, pady=5)
            text5.insert("1.0", Config.HELP_TEXTS['checklist'])
            text5.configure(state="disabled")

            # Bouton fermer
            ctk.CTkButton(
                help_window,
                text="Fermer",
                command=help_window.destroy
            ).pack(pady=10)

        def _toggle_move_options(self):
            """Gère l'état des options de déplacement"""
            # Cette méthode peut être étendue pour activer/désactiver d'autres options
            pass

        def browse_folder(self):
            """Parcourir pour sélectionner un dossier"""
            directory = filedialog.askdirectory()
            if directory:
                self.dir_entry.delete(0, "end")
                self.dir_entry.insert(0, directory)

        def browse_batch(self):
            """Parcourir pour sélectionner un lot"""
            directory = filedialog.askdirectory()
            if directory:
                self.batch_entry.delete(0, "end")
                self.batch_entry.insert(0, directory)

        def browse_upload(self):
            """Parcourir pour sélectionner un dossier upload"""
            directory = filedialog.askdirectory()
            if directory:
                self.upload_dir.delete(0, "end")
                self.upload_dir.insert(0, directory)

        def start_analysis(self):
            """Démarre l'analyse"""
            directory = self.dir_entry.get().strip()
            if not directory or not os.path.isdir(directory):
                messagebox.showerror("Erreur", "Sélectionnez un dossier valide")
                return

            # Mettre à jour les paramètres
            try:
                cooldown = float(self.cooldown_entry.get())
            except:
                cooldown = 2.0

            self.analyzer.update_settings(
                model=self.model_combo.get(),
                cooldown_seconds=cooldown,
                max_workers=int(self.workers_entry.get())
            )

            self.is_running = True
            self.analyze_btn.configure(state="disabled")
            self.stop_analyze_btn.configure(state="normal")

            for label in self.stats_labels.values():
                label.configure(text="0")

            def run():
                stats = self.analyzer.process_directory(
                    directory,
                    pre_filter=self.prefilter_var.get(),
                    resume=self.resume_var.get()
                )
                self.after(0, lambda: self.update_stats(stats))
                self.after(0, self.analysis_finished)

            threading.Thread(target=run, daemon=True).start()

        def update_stats(self, stats: dict):
            """Met à jour les statistiques"""
            for key in ['total', 'success', 'failed', 'invalid']:
                self.stats_labels[key].configure(text=str(stats.get(key, 0)))

        def analysis_finished(self):
            """Appelé quand l'analyse est terminée"""
            self.is_running = False
            self.analyze_btn.configure(state="normal")
            self.stop_analyze_btn.configure(state="disabled")

        def stop_analysis(self):
            """Arrête l'analyse"""
            self.analyzer.stop()
            self.status_label.configure(text="Arrêt en cours...")

        def open_results(self):
            """Ouvre le dossier résultats"""
            directory = self.dir_entry.get().strip()
            if directory:
                results = Path(directory) / "Shutterstock"
                if results.exists():
                    if sys.platform == 'win32':
                        os.startfile(results)
                    elif sys.platform == 'darwin':
                        subprocess.run(['open', str(results)])
                    else:
                        subprocess.run(['xdg-open', str(results)])

        def validate_all_batches(self):
            """Valide tous les lots de manière récursive"""
            source_dir = self.batch_entry.get().strip()
            if not source_dir or not os.path.isdir(source_dir):
                messagebox.showerror("Erreur", "Sélectionnez un dossier valide")
                return

            source_path = Path(source_dir)
            recursive = self.recursive_var.get()

            # Désactiver le bouton
            self.validate_btn.configure(state="disabled")
            self.checklist_progress.set(0)
            self.checklist_status.configure(text="Recherche des lots...")

            # Réinitialiser les stats
            for label in self.checklist_stats.values():
                label.configure(text="0")

            self.checklist_text.delete("1.0", "end")
            self.checklist_text.insert("end", "Validation en cours...\n\n")

            # Stocker le validateur pour export
            self.current_validator = None

            def run_validation():
                validator = ChecklistValidator(source_path)

                def progress_callback(current, total, message):
                    self.after(0, lambda: self.checklist_progress.set(current / total if total > 0 else 0))
                    self.after(0, lambda: self.checklist_status.configure(text=message))

                # Lancer la validation récursive
                recursive_report = validator.validate_all_batches(
                    recursive=recursive,
                    progress_callback=progress_callback
                )

                # Stocker pour export
                self.current_validator = validator

                # Mettre à jour l'interface
                self.after(0, lambda: self._display_recursive_report(validator, recursive_report))

            threading.Thread(target=run_validation, daemon=True).start()

        def _display_recursive_report(self, validator: ChecklistValidator,
                                     report: 'RecursiveChecklistReport'):
            """Affiche le rapport récursif dans l'interface"""

            # Mettre à jour les stats
            stats = report.global_stats
            self.checklist_stats['batches'].configure(text=str(report.total_batches))
            self.checklist_stats['matched'].configure(text=str(stats['total_matched']))
            self.checklist_stats['no_meta'].configure(text=str(stats['total_photos_without_metadata']))
            self.checklist_stats['no_photo'].configure(text=str(stats['total_metadata_without_photos']))
            self.checklist_stats['incomplete'].configure(text=str(stats['total_incomplete']))

            # Afficher le rapport détaillé
            self.checklist_text.delete("1.0", "end")

            if report.total_batches == 0:
                self.checklist_text.insert("end", "Aucun lot trouvé avec des fichiers CSV de métadonnées.\n\n")
                self.checklist_text.insert("end", "Assurez-vous que:\n")
                self.checklist_text.insert("end", "  - Le dossier contient des sous-dossiers avec des CSV\n")
                self.checklist_text.insert("end", "  - Les CSV ont les colonnes 'Filename' et 'Description'\n")
                self.validate_btn.configure(state="normal")
                return

            self.checklist_text.insert("end", "=" * 60 + "\n")
            self.checklist_text.insert("end", "RAPPORT DE VALIDATION RÉCURSIVE\n")
            self.checklist_text.insert("end", "=" * 60 + "\n\n")

            self.checklist_text.insert("end", f"Lots analysés: {report.total_batches}\n")
            self.checklist_text.insert("end", f"Lots avec problèmes: {report.batches_with_issues}\n")
            self.checklist_text.insert("end", f"Total photos: {stats['total_photos']}\n")
            self.checklist_text.insert("end", f"Total correspondances: {stats['total_matched']}\n\n")

            # Détail par lot
            for batch_result in report.batch_results:
                br = batch_result.report
                has_issues = (br.photos_without_metadata or
                            br.metadata_without_photos or
                            br.incomplete_metadata)

                icon = "!!" if has_issues else "OK"
                self.checklist_text.insert("end", f"\n[{icon}] {batch_result.batch_name}\n")
                self.checklist_text.insert("end", f"    Photos: {br.total_photos} | Meta: {br.total_metadata} | Match: {br.matched}\n")

                if br.photos_without_metadata:
                    self.checklist_text.insert("end", f"    Photos sans meta: {len(br.photos_without_metadata)}\n")
                    for p in br.photos_without_metadata[:5]:
                        self.checklist_text.insert("end", f"      - {p.name}\n")
                    if len(br.photos_without_metadata) > 5:
                        self.checklist_text.insert("end", f"      ... +{len(br.photos_without_metadata)-5} autres\n")

                if br.metadata_without_photos:
                    self.checklist_text.insert("end", f"    Meta sans photos: {len(br.metadata_without_photos)}\n")

                if br.incomplete_metadata:
                    self.checklist_text.insert("end", f"    Meta incompletes: {len(br.incomplete_metadata)}\n")

            # Déplacer les photos si demandé
            if self.move_unmatched_var.get() and stats['total_photos_without_metadata'] > 0:
                subfolder = self.subfolder_per_batch_var.get()
                moved = validator.move_all_unmatched_photos(create_subfolder_per_batch=subfolder)
                self.checklist_text.insert("end", f"\n{moved} photo(s) déplacée(s) vers photos_unmatched/\n")

            # Générer le rapport
            log_path = validator.generate_recursive_log()
            self.checklist_text.insert("end", f"\nRapport complet: {log_path}\n")

            self.checklist_status.configure(text="Validation terminée")
            self.validate_btn.configure(state="normal")

        def export_checklist_report(self):
            """Exporte le rapport de checklist"""
            if not hasattr(self, 'current_validator') or self.current_validator is None:
                messagebox.showinfo("Info", "Lancez d'abord une validation")
                return

            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Fichiers texte", "*.txt")],
                title="Exporter le rapport de validation"
            )

            if filepath:
                log_path = self.current_validator.generate_recursive_log(Path(filepath))
                self.log_message(f"Rapport exporté: {log_path}")
                messagebox.showinfo("Succès", f"Rapport exporté:\n{log_path}")

        def validate_batch(self):
            """Valide un seul lot (ancienne méthode, conservée pour compatibilité)"""
            # Redirige vers la nouvelle méthode
            self.validate_all_batches()

        def test_ftps(self):
            """Teste la connexion FTPS"""
            username = self.ftps_user.get().strip()
            password = self.ftps_pass.get().strip()

            if not username or not password:
                self.ftps_status.configure(text="Entrez vos identifiants", text_color="orange")
                return

            self.ftps_status.configure(text="Test en cours...", text_color="gray")

            def test():
                uploader = FTPSUploader(username, password)
                success, msg = uploader.test_connection()

                if success:
                    self.after(0, lambda: self.ftps_status.configure(
                        text="Connexion réussie!", text_color="green"))
                else:
                    self.after(0, lambda: self.ftps_status.configure(
                        text=msg, text_color="red"))

            threading.Thread(target=test, daemon=True).start()

        def start_upload(self):
            """Démarre l'upload"""
            username = self.ftps_user.get().strip()
            password = self.ftps_pass.get().strip()
            upload_dir = self.upload_dir.get().strip()

            if not username or not password:
                messagebox.showerror("Erreur", "Entrez vos identifiants")
                return

            if not upload_dir or not os.path.isdir(upload_dir):
                messagebox.showerror("Erreur", "Sélectionnez un dossier valide")
                return

            self.is_uploading = True
            self.upload_btn.configure(state="disabled")
            self.stop_upload_btn.configure(state="normal")

            self.ftps_uploader = FTPSUploader(username, password)

            def run():
                success, msg = self.ftps_uploader.connect()

                if not success:
                    self.after(0, lambda: self.upload_status.configure(text=msg))
                    self.after(0, self.upload_finished)
                    return

                # Collecter les fichiers (dossier principal + sous-dossiers)
                path = Path(upload_dir)
                files = []
                # Parcourir récursivement tous les sous-dossiers
                for ext in Utils.get_all_extensions():
                    files.extend(path.glob(f"*{ext}"))
                    files.extend(path.glob(f"*{ext.upper()}"))
                    files.extend(path.glob(f"**/*{ext}"))  # Sous-dossiers
                    files.extend(path.glob(f"**/*{ext.upper()}"))
                files.extend(path.glob("*.csv"))
                files.extend(path.glob("**/*.csv"))  # CSV dans sous-dossiers
                # Dédupliquer (glob peut retourner des doublons)
                files = list(set(files))

                if not files:
                    self.after(0, lambda: self.upload_status.configure(text="Aucun fichier"))
                    self.ftps_uploader.disconnect()
                    self.after(0, self.upload_finished)
                    return

                def progress(current, total, msg):
                    self.after(0, lambda: self.upload_progress.set(current/total if total else 0))
                    self.after(0, lambda: self.upload_status.configure(text=msg))

                results = self.ftps_uploader.upload_batch(files, progress)

                success_count = sum(1 for r in results.values() if r.success)
                self.after(0, lambda: self.upload_status.configure(
                    text=f"Terminé: {success_count}/{len(results)} fichiers"))

                self.ftps_uploader.disconnect()
                self.after(0, self.upload_finished)

            threading.Thread(target=run, daemon=True).start()

        def upload_finished(self):
            """Appelé quand l'upload est terminé"""
            self.is_uploading = False
            self.upload_btn.configure(state="normal")
            self.stop_upload_btn.configure(state="disabled")

        def stop_upload(self):
            """Arrête l'upload"""
            if hasattr(self, 'ftps_uploader') and self.ftps_uploader:
                self.ftps_uploader.stop()

        def export_log(self):
            """Exporte le journal"""
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Fichiers texte", "*.txt")]
            )
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get("1.0", "end"))
                self.log_message(f"Journal exporté: {filepath}")

        def show_ollama_logs(self):
            """Affiche les logs Ollama dans une fenêtre séparée"""
            logs = self.ollama_manager.get_ollama_logs()

            # Créer une fenêtre popup
            log_window = ctk.CTkToplevel(self)
            log_window.title("Logs Ollama - Diagnostic")
            log_window.geometry("900x600")
            log_window.transient(self)

            # Header
            header_frame = ctk.CTkFrame(log_window, fg_color="transparent")
            header_frame.pack(fill="x", padx=15, pady=(15, 10))

            ctk.CTkLabel(
                header_frame,
                text="Logs du serveur Ollama",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(side="left")

            # Bouton rafraîchir
            def refresh_logs():
                new_logs = self.ollama_manager.get_ollama_logs()
                log_textbox.delete("1.0", "end")
                if new_logs:
                    for line in new_logs:
                        # Colorer les erreurs en rouge
                        if 'ERROR' in line or 'error' in line:
                            log_textbox.insert("end", line, "error")
                        elif 'WARN' in line or 'warn' in line:
                            log_textbox.insert("end", line, "warning")
                        else:
                            log_textbox.insert("end", line)
                else:
                    log_textbox.insert("end", "Aucun log Ollama trouve.\n\n")
                    log_textbox.insert("end", "Chemins recherches:\n")
                    log_textbox.insert("end", "  - %LOCALAPPDATA%\\Ollama\\logs\\server.log\n")
                    log_textbox.insert("end", "  - %APPDATA%\\Ollama\\logs\\server.log\n")
                    log_textbox.insert("end", "  - ~/.ollama/logs/server.log\n")
                log_textbox.see("end")

            ctk.CTkButton(
                header_frame,
                text="Rafraichir",
                width=100,
                command=refresh_logs
            ).pack(side="right")

            # Zone de texte
            log_textbox = ctk.CTkTextbox(
                log_window,
                font=ctk.CTkFont(family="Consolas", size=10),
                wrap="none"
            )
            log_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 10))

            # Afficher les logs
            if logs:
                for line in logs:
                    if 'ERROR' in line or 'error' in line:
                        log_textbox.insert("end", line)
                    elif 'WARN' in line or 'warn' in line:
                        log_textbox.insert("end", line)
                    else:
                        log_textbox.insert("end", line)
                log_textbox.see("end")
            else:
                log_textbox.insert("end", "Aucun log Ollama trouve.\n\n")
                log_textbox.insert("end", "Chemins recherches:\n")
                log_textbox.insert("end", "  - %LOCALAPPDATA%\\Ollama\\logs\\server.log\n")
                log_textbox.insert("end", "  - %APPDATA%\\Ollama\\logs\\server.log\n")

            # Analyse automatique des problèmes
            analysis_frame = ctk.CTkFrame(log_window)
            analysis_frame.pack(fill="x", padx=15, pady=(0, 15))

            ctk.CTkLabel(
                analysis_frame,
                text="Analyse automatique:",
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(anchor="w", padx=10, pady=(10, 5))

            # Analyser les logs pour les problèmes courants
            problems = []
            log_content = ''.join(logs) if logs else ''

            if 'connectex: No connection could be made' in log_content:
                problems.append("Le serveur Ollama ne demarre pas correctement sur le port 11434")
            if 'The handle is invalid' in log_content:
                problems.append("Un processus Ollama precedent n'a pas ete ferme correctement")
            if 'out of memory' in log_content.lower():
                problems.append("Memoire GPU insuffisante - essayez un modele plus leger")
            if 'CUDA' in log_content and 'error' in log_content.lower():
                problems.append("Probleme avec les drivers CUDA/GPU")
            if not logs:
                problems.append("Logs non trouves - Ollama est peut-etre pas installe")

            if problems:
                for prob in problems:
                    ctk.CTkLabel(
                        analysis_frame,
                        text=f"  - {prob}",
                        text_color="#e74c3c",
                        font=ctk.CTkFont(size=11)
                    ).pack(anchor="w", padx=10)
            else:
                ctk.CTkLabel(
                    analysis_frame,
                    text="  Aucun probleme detecte dans les logs recents",
                    text_color="#2ecc71",
                    font=ctk.CTkFont(size=11)
                ).pack(anchor="w", padx=10)

            # Bouton fermer
            ctk.CTkButton(
                log_window,
                text="Fermer",
                width=100,
                command=log_window.destroy
            ).pack(pady=(0, 15))

    # Lancer l'application
    app = MainApplication()
    app.mainloop()


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    """Point d'entrée principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Shutterstock AI Metadata Generator v1.0.0")
    parser.add_argument('directory', nargs='?', help="Dossier source")
    parser.add_argument('--cli', action='store_true', help="Mode CLI")
    parser.add_argument('--no-filter', action='store_true', help="Désactiver le pré-filtrage")
    parser.add_argument('--resume', action='store_true', help="Reprendre traitement")
    parser.add_argument('--model', default='llama3.2-vision:11b', help="Modèle Ollama")
    parser.add_argument('--cooldown', type=float, default=2.0, help="Délai entre requêtes")

    args = parser.parse_args()

    if args.cli:
        if not args.directory:
            print("Erreur: Spécifiez un dossier en mode CLI")
            sys.exit(1)

        analyzer = ImageAnalyzer(model=args.model, cooldown_seconds=args.cooldown)

        def progress(current, total, msg):
            if total > 0:
                pct = (current / total) * 100
                bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
                print(f"\r[{bar}] {pct:.1f}% - {msg}", end='', flush=True)

        analyzer.progress_callback = progress

        try:
            analyzer.process_directory(
                args.directory,
                pre_filter=not args.no_filter,
                resume=args.resume
            )
            print()
        except KeyboardInterrupt:
            print("\nInterrompu")
            analyzer.stop()
    else:
        create_gui()


if __name__ == "__main__":
    main()
