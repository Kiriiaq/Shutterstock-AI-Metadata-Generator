# Rapport de qualification — ShutterstockAnalyzer

> Document généré à partir de la méthodologie « Qualification d'outil à IHM »
> (template Edvance — EPR/I&C). Toutes les sections sont prêtes à être
> complétées au fur et à mesure des passes manuelles.

---

## 1. Identification

| Champ | Valeur |
|---|---|
| **Outil** | ShutterstockAnalyzer |
| **Version** | v2.1.0 (pivot Pro = évaluation qualité) |
| **Date génération** | 2026-05-27 |
| **Testeur principal** | _(à compléter)_ |
| **Environnement** | Windows 10/11 — Python 3.11 — sans Ollama |
| **Stack** | Python 3.11 + CustomTkinter + ExifTool + Ollama (optionnel) |
| **Build EXE** | `dist/ShutterstockAnalyzer.exe` — 25,2 Mo (PROD secret embedded) |
| **Édition testée** | Community **et** Pro (les deux modes doivent être couverts — voir § 2.4) |

**Dépendances critiques :** Pillow, openpyxl, customtkinter, requests, ExifTool (externe).

---

## 2. Périmètre testé

### 2.1 Fonctionnalités IHM couvertes

- Topbar : toggle theme, bouton Aide, chips santé (Backend / ExifTool / Ollama)
- Workspace : 7 panneaux (Sources, Éditeur IPTC, Analyse IA, Modèle IA,
  Validation, Historique, Paramètres)
- Modaux : 5 vues détails (settings, audit, ai_control, validate, **expert_report**)
- Raccourcis : Ctrl+1/2/3 focus, Esc fermeture modal / annulation batch,
  Ctrl+? aide

### 2.2 Pipelines testés

- Heuristique pur (sans IA) — chemin par défaut
- IA optionnelle (Ollama, mode `hybrid`)
- Export double CSV (Adobe + Shutterstock)
- Lecture/écriture IPTC via ExifTool

### 2.3 Hors-périmètre

- Upload FTPS vers Shutterstock (legacy, non couvert)
- Worker pool multi-processus (config par défaut = threads)

### 2.4 Gating Pro / Community (pivot v2.1.0)

Tester chaque chemin dans **les deux modes** (Community sans licence active +
Pro avec une clé `pro_solo` ou `lifetime` injectée via Settings → Licence) :

| Surface | Community attendu | Pro attendu |
|---|---|---|
| Settings → Licence | « 🆓 Édition Community… 2 aperçus gratuits » | « ✅ Édition Pro Solo · email · expire JJ/MM/AAAA » |
| Rapport expert (1ᵉʳ / 2ᵉ image) | Bandeau « 🎁 il reste N/2 » | Pas de bandeau |
| Rapport expert (3ᵉ image) | Écran upsell (benefits + Acheter Pro + J'ai déjà une clé) | Rapport rendu normalement |
| Rapport expert — bouton « Exporter CSV double » | Disabled, label « 🔒 Exporter CSV double — Pro » | Enabled, export OK |
| Rapport expert — case « Enrichir avec IA » | Disabled, label « 🔒 Enrichir avec IA — Pro » | Enabled |
| Export Batch — radio Plateforme | « 🔒 Les deux — Pro », défaut Adobe | « Les deux », défaut both |
| Export Batch — case « Enrichir avec IA » | Disabled, label « 🔒 Enrichir avec IA — Pro » | Enabled |
| Export Batch — Start avec « Les deux » coché | Toast « Pro requis pour export double » | Démarre |
| Export Batch — Start avec IA coché | Toast « Pro requis pour enrichissement IA » | Démarre |
| Export Batch — Start avec > 50 fichiers | Toast « Pro requis pour > 50 images » | Démarre |

---

## 3. Synthèse chiffrée

### 3.1 Tests automatisés (pytest)

| Item | Valeur |
|---|---|
| **Total tests automatisés** | 120 |
| **Verts** | 120 |
| **Rouges** | 0 |
| **Durée** | ~7 s |
| **Commande** | `pytest tests/ -q` |

Détail par module :
- `tests/test_core/test_expert_report.py` — 16 tests (builder heuristique, IA optionnelle, sérialisation, filtres anti-marques/anti-stuffing)
- `tests/test_core/test_csv_exporter.py` — 4 tests (CSV Adobe, CSV Shutterstock, double export, fix P0)
- `tests/test_core/test_platform_compliance.py` — 10 tests (Adobe, Shutterstock, posture lâche)
- `tests/test_core/test_export_batch.py` — 10 tests (pipeline batch, IPTC write-back, FTP)
- `tests/test_core/test_ftp_uploader.py` — 8 tests (FTPS, échec partiel)
- `tests/test_core/test_ollama_facade.py` — 9 tests (Ollama probe, preload, fallback)
- `tests/test_core/test_licensing.py` — **30 tests** : community default (3), key generation (6), tamper resistance (5), feature gating (3), expiration (2), facade integration (2), **TestPivotFeatures (4) + TestCommunityExpertReportQuota (5)** ← nouveautés v2.1.0
- `tests/test_core/test_config.py` — 3 tests (ShutterstockParams)
- `tests/test_utils/test_validators.py` — 28 tests existants
- `tests/smoke/test_smoke.py` — 1 test
- `tests/ui/test_app_v3_shell.py` — 1 test (ouvre les 5 modaux dont expert_report)

### 3.2 Tests manuels — matrice IHM

Suivi dans [`matrice_tests.xlsx`](matrice_tests.xlsx) (feuille **Synthèse**) et dans
[`validation_ihm.html`](validation_ihm.html) (export Markdown / JSON).

| Catégorie | Total | OK | NOK | NA | Taux OK |
|---|---|---|---|---|---|
| IHM | 15 | _(à compléter)_ | | 15 | — |
| Paramètres | 3 | | | 3 | — |
| Entrées | 9 | | | 9 | — |
| Sorties | 5 | | | 5 | — |
| Cas limites | 6 | | | 6 | — |
| Performance | 3 | | | 3 | — |
| Robustesse | 5 | | | 5 | — |
| Régression | 3 | | | 3 | — |
| **TOTAL** | **49** | | | **49** | **—** |

### 3.3 Run de référence (run_tests.py)

| Item | Valeur |
|---|---|
| **Inputs traités** | 15 / 15 ✓ |
| **Durée** | 5,4 s (≈ 360 ms/image, ExifTool inclus) |
| **CSV Adobe produit** | `outputs_reels/reels_adobe.csv` — 15 lignes |
| **CSV Shutterstock produit** | `outputs_reels/reels_shutterstock.csv` — 15 lignes |
| **Comparaison vs référence** | ✓ cell-for-cell match |

---

## 4. Anomalies détectées

> À compléter au fil des passes manuelles. Format :

| ID | Sévérité | Description | Reproductibilité | Contournement |
|---|---|---|---|---|
| _(vide pour l'instant)_ | | | | |

### 4.1 Anomalies connues (déjà corrigées dans v2.0.0 / v2.1.0)

| Issue | Statut | Référence |
|---|---|---|
| Bug P0 — CSV Shutterstock séparait les keywords par espace au lieu de virgule | **Corrigé** | `metadata_models.py:349` (cf. audit `audit/AUDIT_EXPERT_MICROSTOCK.md` §9) |
| `Platform.ADOBE_STOCK` défini mais jamais utilisé | **Corrigé** | catégories + mapping + export double CSV |
| Pas de détection visuelle (bruit, focus…) en heuristique pur | **Documenté** | Posture lâche : filtres techniques en warnings, détection visuelle uniquement si IA activée |

---

## 5. Couverture fonctionnelle (exigences × tests)

| Exigence | Tests | Couverture |
|---|---|---|
| **REQ-IHM-01..15** (interface utilisateur) | T-001 → T-015 | 100 % (manuel) |
| **REQ-PARAM-01..03** (mode IA optionnel) | T-020 → T-022 | 100 % |
| **REQ-IN-01..09** (formats d'entrée) | T-030 → T-038 | 100 % (15 inputs Pillow réels) |
| **REQ-OUT-01..05** (sorties CSV) | T-040 → T-044 | 100 % + référence cell-for-cell |
| **REQ-LIM-01..06** (cas limites) | T-050 → T-055 | 100 % + 30 tests pytest dédiés |
| **REQ-PERF-01..03** (performance) | T-060 → T-062 | partiel (baseline = 360 ms/img) |
| **REQ-ROB-01..05** (robustesse) | T-070 → T-074 | 100 % (5 scénarios dégradés) |
| **REQ-REG-01..03** (régression) | T-080 → T-082 | 100 % (pytest + build + legacy API) |

**Bilan couverture** : 49 tests pour 48 exigences. **Taux théorique 100 %**.

---

## 6. Conclusion

### 6.1 Décision (à compléter par le testeur)

- [ ] **GO inconditionnel** — tous les tests OK, anomalies = 0 ou seulement mineures
- [ ] **GO conditionnel** — anomalies majeures contournables, lister les conditions
- [ ] **NO-GO** — au moins une anomalie critique non corrigée

### 6.2 Conditions de GO (si applicable)

_(à compléter)_

### 6.3 Signature

| Rôle | Nom | Date | Signature |
|---|---|---|---|
| Testeur | | | |
| Validateur QA | | | |
| Chef de projet | | | |

---

## 7. Annexes

### 7.1 Livrables QA

| Fichier | Description |
|---|---|
| [`matrice_tests.xlsx`](matrice_tests.xlsx) | Matrice 49 tests, 2 feuilles (Tests + Synthèse), validations & couleurs conditionnelles |
| [`validation_ihm.html`](validation_ihm.html) | Checklist interactive autonome (localStorage, export JSON/MD) |
| [`inputs/`](inputs/) | 15 images Pillow réelles (nominal, vide, low MP, volumineux, PNG, CMYK, UTF-8, corrompu, brands, stuffing, chemin avec accents…) |
| [`outputs_reference/`](outputs_reference/) | CSV de référence pour non-régression (cell-for-cell) |
| [`outputs_reels/`](outputs_reels/) | Sorties produites par `run_tests.py` (peuplé à l'exécution) |
| [`scripts/run_tests.py`](scripts/run_tests.py) | Orchestrateur : `inputs/` → `outputs_reels/` |
| [`scripts/compare_outputs.py`](scripts/compare_outputs.py) | Diff `outputs_reels/` vs `outputs_reference/` |
| [`scripts/_make_matrix.py`](scripts/_make_matrix.py) | Génération de la matrice XLSX |
| [`scripts/_make_inputs.py`](scripts/_make_inputs.py) | Génération des images Pillow |
| [`scripts/_make_reference.py`](scripts/_make_reference.py) | Rafraîchissement de la référence (idempotent) |

### 7.2 Audits associés

- [`../audit/AUDIT_EXPERT_MICROSTOCK.md`](../audit/AUDIT_EXPERT_MICROSTOCK.md) — audit initial qui a déclenché la refonte multi-plateforme

### 7.3 Commandes utiles

```bash
# Suite automatisée
pytest tests/ -q

# Régénérer la matrice XLSX
python test/scripts/_make_matrix.py

# Régénérer les images de test (Pillow)
python test/scripts/_make_inputs.py

# Lancer le pipeline sur tous les inputs
python test/scripts/run_tests.py

# Comparer reels vs référence
python test/scripts/compare_outputs.py

# Rafraîchir la référence après changement intentionnel
python test/scripts/_make_reference.py

# Construire les EXEs
python build.py all
```

### 7.4 Logs / build

| Item | Lien |
|---|---|
| Logs Ollama session courante | `~/.shutterstock_ai/logs/` |
| Base SQLite | `~/.shutterstock_ai/shutterstock_ai.db` |
| Préférences UI | `~/.shutterstock_ai/ui_prefs.json` |
| Build PyInstaller | `dist/ShutterstockAnalyzer{,-debug}.exe` |

---

*Fin du rapport. Document à figer (PDF + signature) une fois la matrice
manuelle complétée.*
