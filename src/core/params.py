"""
Modèle de paramètres Shutterstock Analyzer.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParamMeta:
    """Métadonnées d'un paramètre."""

    label: str
    help_text: str = ""
    category: str = "essential"
    default: Any = None
    choices: list = field(default_factory=list)
    depends_on: Optional[str] = None
    placeholder: str = ""
    unit: str = ""


@dataclass
class ShutterstockParams:
    """Paramètres complets de l'analyseur Shutterstock."""

    # === Source ===
    source_folder: str = ""
    prefilter_enabled: bool = True
    resume_mode: bool = False
    skip_analyzed: bool = True

    # === Modèle IA ===
    model_name: str = "llama3.2-vision:11b"

    # === Avancé: Préfiltrage ===
    min_megapixels: float = 4.0
    max_file_size_mb: float = 50.0
    fix_orientation: bool = True

    # === Avancé: Performance ===
    gpu_layers: int = 35
    cooldown: float = 2.0
    workers: int = 2

    # === Avancé: FTPS ===
    ftps_username: str = ""
    ftps_password: str = ""

    # === Avancé: Debug ===
    debug_mode: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ShutterstockParams":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def get_modified_fields(self, defaults: "ShutterstockParams" = None) -> List[str]:
        if defaults is None:
            defaults = ShutterstockParams()
        return [f for f in self.__dataclass_fields__ if getattr(self, f) != getattr(defaults, f)]


PARAMS_META: Dict[str, ParamMeta] = {
    "source_folder": ParamMeta(
        label="Dossier source",
        help_text="Dossier contenant vos photos à analyser. Les sous-dossiers Valid/Invalid/Shutterstock seront créés.",
        category="essential",
        placeholder="Ex: C:/Mes_Photos",
    ),
    "model_name": ParamMeta(
        label="Modèle IA",
        help_text="Modèle Ollama Vision pour l'analyse d'images",
        category="essential",
        default="llama3.2-vision:11b",
        choices=[
            "llama3.2-vision:11b",
            "llama3.2-vision:90b",
            "llava:7b",
            "llava:13b",
            "llava:34b",
            "bakllava:7b",
            "moondream:1.8b",
        ],
    ),
    "prefilter_enabled": ParamMeta(
        label="Pré-filtrer",
        help_text="Vérifie résolution, taille et format avant analyse",
        category="essential",
        default=True,
    ),
    "resume_mode": ParamMeta(
        label="Reprendre",
        help_text="Reprend un traitement interrompu (ignore les photos déjà analysées)",
        category="secondary",
        default=False,
    ),
    "skip_analyzed": ParamMeta(
        label="Ignorer existants",
        help_text="Ignore les photos qui ont déjà des métadonnées",
        category="secondary",
        default=True,
    ),
    "min_megapixels": ParamMeta(
        label="Résolution min",
        help_text="Résolution minimum en mégapixels (Shutterstock exige 4 MP)",
        category="advanced",
        default=4.0,
        unit="MP",
    ),
    "max_file_size_mb": ParamMeta(
        label="Taille max fichier",
        help_text="Taille maximale du fichier en MB",
        category="advanced",
        default=50.0,
        unit="MB",
    ),
    "gpu_layers": ParamMeta(
        label="Couches GPU",
        help_text="Nombre de couches du modèle chargées sur le GPU",
        category="advanced",
        default=35,
    ),
    "cooldown": ParamMeta(
        label="Délai entre requêtes",
        help_text="Pause entre chaque analyse d'image (en secondes)",
        category="advanced",
        default=2.0,
        unit="s",
    ),
    "workers": ParamMeta(
        label="Workers parallèles", help_text="Nombre de traitements en parallèle", category="advanced", default=2
    ),
}
