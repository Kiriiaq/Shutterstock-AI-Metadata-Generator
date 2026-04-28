"""
Gestionnaire de profils Shutterstock Analyzer.
"""

import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from .params import ShutterstockParams


class ConfigManager:
    """Gestionnaire de profils."""

    SCHEMA_VERSION = 1

    def __init__(self):
        self.profiles_dir = Path.home() / ".shutterstock_analyzer" / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.current_profile: Optional[str] = None
        self._current_data = {}
        if not (self.profiles_dir / "default.json").exists():
            self.save_profile("default", ShutterstockParams())

    def list_profiles(self) -> List[str]:
        return sorted([p.stem for p in self.profiles_dir.glob("*.json")])

    def load_profile(self, name: str) -> ShutterstockParams:
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            return ShutterstockParams()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.current_profile = name
        self._current_data = data
        return ShutterstockParams.from_dict(data.get("params", {}))

    def save_profile(self, name: str, params: ShutterstockParams):
        data = {
            "schema_version": self.SCHEMA_VERSION,
            "app_name": "ShutterstockAnalyzer",
            "profile_name": name,
            "created_at": self._current_data.get("created_at", datetime.now().isoformat()),
            "modified_at": datetime.now().isoformat(),
            "params": params.to_dict()
        }
        path = self.profiles_dir / f"{name}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.current_profile = name

    def delete_profile(self, name: str) -> bool:
        if name == "default":
            return False
        path = self.profiles_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def export_json(self, name: str, export_path: Path):
        params = self.load_profile(name)
        data = {"schema_version": self.SCHEMA_VERSION, "params": params.to_dict()}
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_json(self, import_path: Path) -> str:
        with open(import_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        name = data.get("profile_name", import_path.stem)
        params = ShutterstockParams.from_dict(data.get("params", {}))
        self.save_profile(name, params)
        return name

    def reset_to_defaults(self) -> ShutterstockParams:
        return ShutterstockParams()
