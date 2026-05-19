# Audit — Mise en conformité « Expert microstock top 1 % »

> Cible : aligner le pipeline IA actuel (Ollama + `VisionAnalyzer`) sur le
> cahier des charges « expert senior microstock, référencement Adobe Stock
> + Shutterstock ». Date : 2026-05-18 — Branche : `main`.

---

## 0. TL;DR — Le verdict

**État actuel** : le projet est un **générateur de métadonnées Shutterstock
mono-plateforme** (titre + description + 7–50 keywords + 1-2 catégories).
Le prompt actuel est générique, sans contrôle technique, sans logique
Adobe Stock, sans scoring commercial, sans analyse des risques de rejet,
sans tendances marketing.

**Distance à la spec** : ~30 % d'alignement. Les fondations existent
(`Platform.ADOBE_STOCK` est défini mais jamais utilisé, le parsing est
extensible, l'UI a déjà un panneau d'analyse), mais **7 sections sur 8
du livrable demandé ne sont pas produites** par le pipeline.

**Effort de mise à niveau** : refonte ciblée du prompt + extension du
parser + ajout d'un modèle de sortie enrichi + nouveau panneau UI
« Rapport expert ». Pas de réécriture du backend ExifTool / Ollama.

---

## 1. Cartographie de l'existant

| Brique | Fichier | Rôle | Conformité spec |
|---|---|---|---|
| Prompt FULL | `src/modules/ai/prompt_templates.py:84` | Génère TITLE / DESCRIPTION / KEYWORDS / CATEGORIES | Partielle (Shutterstock only) |
| Limites plateforme | `src/modules/ai/prompt_templates.py:43` | `PlatformLimits` Shutterstock, Adobe, Getty… | Adobe défini mais **jamais sélectionné** |
| Parser réponse | `src/modules/ai/prompt_templates.py:233` | 5 champs (TITLE/DESC/KW/CAT/EDITORIAL) | Ne lit pas score / risques / usages |
| Modèle sortie | `src/modules/models/metadata_models.py:255` `ShutterstockMetadata` | dataclass 4 champs + flags | Pas de champ Adobe, scores, risques, usages |
| Pipeline | `src/modules/integration.py:1036` | hard-code `Platform.SHUTTERSTOCK` | Bloque tout multi-plateforme |
| Catégories | `src/modules/models/metadata_models.py:44` `SHUTTERSTOCK_CATEGORIES` | 26 catégories Shutterstock | Pas de mapping Adobe Stock |
| Validation | `src/modules/integration.py:473` | Score complétude / qualité / SEO sur la **métadonnée** | Aucun contrôle **technique d'image** (bruit, focus, watermark…) |
| Pré-filtre | `src/modules/integration.py:400` | Taille fichier 50 Mo + min 4 MP | Pas de max 100 MP / 45 Mo Adobe, pas de sRGB |
| UI Analyse | `app/views/workspace.py:767` | Démarrer / barre / log texte | Affiche uniquement le titre et un ✓/✗, **pas le rapport** |
| CSV export | `src/modules/models/metadata_models.py:349` | `" ".join(keywords)` | **Bug** : Shutterstock attend des keywords séparés par `,` |

---

## 2. Lecture critique du prompt actuel

Le prompt `FULL_ANALYSIS_TEMPLATE` (`src/modules/ai/prompt_templates.py:84-98`)
demande seulement :

```
TITLE / DESCRIPTION / KEYWORDS / CATEGORIES
```

**Ce qui manque vs la spec** :

1. Aucun rôle/persona « reviewer Adobe + Shutterstock + acheteur marketing + SEO microstock ».
2. Aucune contrainte technique (sRGB / 4–100 MP / 45 Mo / 50 Mo / JPEG / pas de marque).
3. Aucune détection de défauts (bruit, focus mou, JPEG artefacts, halos, sur-traitement, défauts IA, doigts, texte illisible, watermark, logos, marques, releases, propriété privée).
4. Aucune hiérarchisation des 10 premiers keywords (poids SEO).
5. Aucun score global (potentiel commercial / qualité / SEO / risque rejet).
6. Aucune description double (Adobe vs Shutterstock).
7. Aucune catégorie séparée pour Adobe.
8. Aucune analyse de risques de rejet ni d'améliorations recommandées.
9. Aucun panel d'usages marketing (publicité, bannière, fintech, santé, lifestyle…).
10. Pas de garde-fou « éviter keyword stuffing / variantes / mots non visibles ».

Conclusion : **le prompt est ~2 fois trop court** pour produire le livrable demandé.

---

## 3. Plan de modifications

### Section A — Prompt « expert microstock »

**Fichier** : `src/modules/ai/prompt_templates.py`

- [A1] Ajouter `PromptType.EXPERT_REPORT` (nouvelle sortie structurée).
- [A2] Ajouter une constante `EXPERT_SYSTEM_PROMPT` reprenant la spec
  (rôles, règles Adobe + Shutterstock, anti-stuffing, structure 8 sections).
- [A3] Ajouter `EXPERT_TEMPLATE` qui demande explicitement au modèle de
  retourner un **JSON strict** :

```json
{
  "scores": {"commercial": 0, "technical": 0, "seo": 0, "rejection_risk": 0},
  "title_adobe": "...",
  "title_shutterstock": "...",
  "description": "...",
  "keywords": ["...", "..."],
  "category_adobe_primary": "...",
  "category_adobe_secondary": "...",
  "category_shutterstock": "...",
  "rejection_risks": [{"issue": "...", "cause": "...", "fix": "..."}],
  "improvements": ["..."],
  "marketing_uses": ["publicité", "fintech", ...],
  "buyer_profiles": ["agence marketing", ...],
  "trends": ["..."],
  "technical_flags": {
    "noise": false, "soft_focus": false, "jpeg_artifacts": false,
    "oversharpen": false, "hdr_overprocessed": false, "halos": false,
    "oversaturated": false, "ai_artifacts": false, "bad_hands": false,
    "unreadable_text": false, "watermark": false, "logo_or_brand": false,
    "protected_building": false, "needs_model_release": false,
    "needs_property_release": false
  }
}
```

- [A4] Préciser dans le prompt : « **TU ÉCRIS DU JSON UNIQUEMENT**.
  Pas de markdown, pas de prologue, pas de balise de code ». Indispensable
  avec LLaMA 3.2 Vision / LLaVA qui ont tendance à fuir en prose.
- [A5] Imposer 50 keywords max, anglais, **10 premiers = commerciaux**,
  séparés par virgules dans la sortie.
- [A6] Lister les marques / logos / bâtiments protégés à signaler (clauses
  releases) — copier la liste de la spec.

### Section B — Parser & modèle de données

**Fichier** : `src/modules/ai/prompt_templates.py` + `src/modules/models/metadata_models.py`

- [B1] Implémenter `_parse_expert_response()` :
  - utiliser `json.loads()` avec un *try / repair* (`re.search(r"\{.*\}", text, re.DOTALL)`)
    pour récupérer le JSON même si le modèle a ajouté du texte autour.
  - fallback : si le JSON est invalide, retomber sur l'ancien parser
    `TITLE: … KEYWORDS: …` pour ne pas casser le batch.
- [B2] Nouvelle dataclass `ExpertMetadataReport` à côté de `ShutterstockMetadata` :
  - Champs : scores (4), titres (Adobe + Shutterstock), description,
    keywords, catégories (Adobe primary/secondary + Shutterstock),
    `rejection_risks: list[RejectionRisk]`,
    `improvements: list[str]`, `marketing_uses: list[str]`,
    `buyer_profiles: list[str]`, `trends: list[str]`,
    `technical_flags: TechnicalFlags`.
  - Méthode `to_adobe_csv_row()` et `to_shutterstock_csv_row()`.
  - Méthode `summary_markdown()` pour l'export rapide.
- [B3] Constante `ADOBE_STOCK_CATEGORIES` (les 21 catégories officielles
  Adobe : Animals, Buildings, Business, Drinks, … cf. site Adobe Stock
  Contributor) à ajouter dans `metadata_models.py`.
- [B4] Mapping Shutterstock ↔ Adobe (table de correspondance simple).

### Section C — Pipeline d'analyse

**Fichier** : `src/modules/ai/vision_analyzer.py` + `src/modules/integration.py`

- [C1] `VisionAnalyzer.analyze_image()` : nouveau paramètre `mode="expert"`
  qui sélectionne `PromptType.EXPERT_REPORT` et retourne un
  `ExpertAnalysisResult` au lieu de l'`AnalysisResult` actuel.
- [C2] `ShutterstockAIv2.init_ai()` : ne plus forcer `Platform.SHUTTERSTOCK`.
  Lire `platform` depuis les settings, défaut `"expert"` (= multi-stock).
- [C3] `ShutterstockAIv2.analyze_image_ai()` : retourner également les
  nouveaux champs (`scores`, `rejection_risks`, `marketing_uses`…) dans
  le dict de résultat pour que l'UI puisse les afficher.
- [C4] Ajouter `analyze_image_expert()` dédié (chemin neuf, sans casser
  l'existant) qui ne sérialise QUE le rapport expert.

### Section D — Pré-filtres techniques (avant Ollama)

**Fichier** : `src/utils/validators.py` (existant) — à étendre.

- [D1] Vérifier `format == JPEG` (Adobe + Shutterstock).
- [D2] Vérifier `color_space in {"sRGB", "Uncalibrated"}` via `metadata_reader`.
- [D3] Vérifier `4 MP ≤ megapixels ≤ 100 MP` (Adobe). Shutterstock : pas
  de plafond MP officiel mais 50 Mo en taille fichier.
- [D4] Vérifier `file_size ≤ 45 MB` pour Adobe, `≤ 50 MB` pour Shutterstock.
- [D5] Ajouter le résultat (`adobe_ready: bool`, `shutterstock_ready: bool`)
  dans la `ValidationResult` (ajouter ces deux booléens dans le dataclass
  `ValidationResult` de `metadata_models.py:418`).
- [D6] Détection bruit / flou côté Python (heuristique légère, optionnelle)
  via `Pillow` + variance Laplacien — nice-to-have, **secondaire**.

### Section E — Vérifications post-analyse (anti-spam)

**Fichier** : `src/modules/ai/prompt_templates.py` (méthode `_parse_keywords` ligne 290).

- [E1] Filtrer les keywords non visibles : ajouter un rejet des termes
  trop génériques (`stock`, `image`, `photo`, `picture`, `wallpaper`)
  sauf si présents dans le titre.
- [E2] Détecter le **keyword stuffing** : si > 60 % des keywords partagent
  la même racine (`run / running / runner / runs`), tronquer aux 2 plus
  pertinents et ajouter un warning à `ExpertMetadataReport.warnings`.
- [E3] Détecter et **rejeter les marques connues** (liste statique : Apple,
  Nike, Coca-Cola, Microsoft, Google, BMW…) — déjà couvert partiellement
  par le prompt mais à doubler côté code.
- [E4] Garantir **anglais** : si présence de caractères accentués hors
  noms propres, marquer le keyword comme `suspect` et exclure.

### Section F — UI : panneau « Rapport expert »

**Fichier** : nouveau `app/views/expert_report.py` + intégration `app/views/workspace.py`.

- [F1] Nouveau panneau / vue affichant les 8 sections de la spec :
  1. SCORE GLOBAL (4 jauges 0-10)
  2. TITRE Adobe / Shutterstock (2 entrées éditables)
  3. DESCRIPTION (textarea)
  4. KEYWORDS (chips réordonnables, 10 premiers mis en évidence)
  5. CATÉGORIES (Adobe primary/secondary + Shutterstock)
  6. RISQUES DE REJET (tableau issue / cause / fix)
  7. AMÉLIORATIONS (liste à puces)
  8. USAGES MARKETING (chips)
- [F2] Le panneau `ANALYSE IA` actuel (`workspace.py:767`) doit passer en
  « mode démarrage / progression », et **rediriger** vers le nouveau
  panneau Rapport expert dès qu'un résultat tombe — sinon l'écran
  d'accueil est saturé.
- [F3] Bouton « Exporter rapport » (Markdown + JSON + CSV double-plateforme).
- [F4] Bouton « Régénérer le titre Adobe seul » / « Régénérer keywords »
  (utiliser `regenerate_field` existant — `vision_analyzer.py:402`).

### Section G — Exports double plateforme

**Fichier** : `src/modules/engines/metadata_writer.py` + nouveau `src/modules/export/csv_exporter.py`.

- [G1] **Corriger le bug actuel** : `metadata_models.py:349`
  ```python
  "Keywords": " ".join(self.keywords),   # ❌ Shutterstock attend des virgules
  ```
  → remplacer par `", ".join(self.keywords)`. À vérifier sur leur CSV
  template (col `Keywords` en `,`).
- [G2] Générer `metadata_shutterstock.csv` (existant, à corriger) ET
  `metadata_adobe_stock.csv` (nouveau) :
  - Colonnes Adobe : `Filename, Title, Keywords, Category, Releases`.
  - Colonnes Shutterstock : `Filename, Description, Keywords, Categories, Editorial, Mature, Illustration`.
- [G3] Écrire l'IPTC **`headline = title_shutterstock`** par défaut
  (modifiable dans paramètres) plutôt que `title` générique.

### Section H — Paramètres / configuration

**Fichier** : `src/core/config_manager.py` + `app/views/settings.py`.

- [H1] Ajouter clés :
  - `analysis.mode` ∈ {`shutterstock`, `adobe`, `expert`} — défaut `expert`.
  - `analysis.platforms` (multi-select) — défaut `["adobe", "shutterstock"]`.
  - `analysis.language_keywords` — défaut `"en"`.
  - `analysis.enforce_english` — défaut `true`.
  - `analysis.max_keywords_adobe` — défaut `49` (10 prioritaires + 39).
  - `analysis.max_keywords_shutterstock` — défaut `50`.
- [H2] Settings UI : ajouter une section « Plateformes & SEO » dans
  `app/views/settings.py`.

### Section I — Tests

**Fichier** : `tests/test_core/` + `tests/smoke/`.

- [I1] `test_expert_prompt_parsing.py` : vérifier que `_parse_expert_response()`
  accepte un JSON propre, un JSON entouré de texte, un JSON mal formé
  (fallback), et un texte purement non-JSON (échec contrôlé).
- [I2] `test_expert_model.py` : vérifier la sérialisation
  `ExpertMetadataReport.to_adobe_csv_row()` / `to_shutterstock_csv_row()`.
- [I3] `test_keyword_quality.py` : vérifier les filtres anti-stuffing,
  anti-marques, anti-non-anglais.
- [I4] `tests/smoke/test_expert_end_to_end.py` : un appel mock-ollama
  (réponse JSON pré-enregistrée) → `ExpertMetadataReport` complet.
- [I5] Mettre à jour `BASELINE_TESTS.md` avec les nouvelles entrées.

### Section J — Documentation

- [J1] `README.md` : ajouter la section « Multi-plateforme (Adobe Stock + Shutterstock) ».
- [J2] `docs/expert_mode.md` (nouveau) : décrire la sortie complète,
  l'usage en CLI, l'export double.
- [J3] `JOURNAL_CORRECTIONS.md` : entrée datée pour cette refonte.

---

## 4. Tableau de couverture spec → action

| Exigence spec | État | Action |
|---|---|---|
| Adobe Stock — JPEG / sRGB | ❌ | [D1] [D2] |
| Adobe — 4 MP ≤ img ≤ 100 MP | partiel (min OK, max KO) | [D3] |
| Adobe — ≤ 45 Mo | ❌ | [D4] |
| Adobe — 10 premiers keywords prioritaires | ❌ | [A3] [A5] [B2] |
| Adobe — pas de variantes trop similaires | ❌ | [E2] |
| Shutterstock — 7–50 keywords | ✅ | — |
| Shutterstock — ≤ 50 Mo | partiel (50 Mo en dur, OK) | — |
| Détection bruit / focus / artefacts JPEG | ❌ | [A3] [F1] |
| Détection sur-sharpening / HDR / halos / sursat. | ❌ | [A3] [F1] |
| Défauts IA / doigts / texte illisible | ❌ | [A3] [F1] |
| Watermark / logo / marque | ❌ | [A3] [E3] [F1] |
| Bâtiment protégé / model release / property release | ❌ | [A3] [F1] |
| Score commercial /10 | ❌ | [A3] [B2] [F1] |
| Score technique /10 | ❌ | [A3] [B2] [F1] |
| Score SEO /10 | ❌ | [A3] [B2] [F1] |
| Risque de rejet /10 | ❌ | [A3] [B2] [F1] |
| Titre Adobe + titre Shutterstock distincts | ❌ | [A3] [B2] [F1] |
| Description commerciale orientée marketing | partiel | [A3] |
| 50 keywords classés du plus important au moins | ❌ | [A3] [A5] |
| Adobe : 2 catégories / Shutterstock : 1 catégorie | partiel | [A3] [B2] [B3] |
| Liste des risques de rejet (issue / cause / fix) | ❌ | [A3] [B2] [F1] |
| Améliorations recommandées (retouche, recadrage…) | ❌ | [A3] [B2] [F1] |
| Usages marketing (pub, bannière, fintech, santé…) | ❌ | [A3] [B2] [F1] |
| Pas de keyword stuffing | partiel | [A3] [E1] [E2] |
| Export double-plateforme | ❌ | [G2] |
| **Bug CSV Shutterstock (espaces au lieu de virgules)** | ❌ | [G1] |

---

## 5. Priorisation

### P0 — Bloquant (à faire avant toute promesse expert)

1. **[G1]** Fix `to_csv_row` (espaces → virgules) — 5 lignes, à corriger
   immédiatement même hors refonte. Régression silencieuse sur tous les
   exports actuels.
2. **[A1][A2][A3][A4]** Nouveau prompt expert + JSON strict.
3. **[B1][B2]** Parser JSON + dataclass `ExpertMetadataReport`.
4. **[C1][C2]** Brancher `analyze_image()` en mode expert.

### P1 — Indispensable pour le livrable spec

5. **[F1][F2]** Vue UI Rapport expert (les 8 sections).
6. **[D1][D2][D3][D4][D5]** Pré-filtres techniques Adobe / Shutterstock.
7. **[G2]** Export double CSV.
8. **[B3]** `ADOBE_STOCK_CATEGORIES` + mapping.

### P2 — Qualité / robustesse

9. **[E1][E2][E3][E4]** Filtres anti-spam / anti-marques / anglais.
10. **[H1][H2]** Settings UI multi-plateforme.
11. **[I1]→[I5]** Tests.
12. **[J1][J2][J3]** Documentation.

### P3 — Nice to have

13. **[D6]** Heuristique bruit / flou Python (variance Laplacien).
14. **[F3][F4]** Régénération champ par champ + export rapport.

---

## 6. Estimation d'effort (ordre de grandeur)

| Lot | Étapes | Effort |
|---|---|---|
| P0 | A1-A4, B1-B2, C1-C2, G1 | 1 journée |
| P1 | F1-F2, D1-D5, G2, B3 | 2 journées |
| P2 | E1-E4, H1-H2, I1-I5 | 1,5 journée |
| P3 | D6, F3-F4, J1-J3 | 1 journée |
| **Total** | | **≈ 5,5 jours dev** |

---

## 7. Risques de mise en œuvre

| Risque | Mitigation |
|---|---|
| LLaMA 3.2 Vision peut sortir un JSON cassé (ajout de markdown / prose). | Parser tolérant + fallback texte (B1) + 1 retry avec temperature ↓ 0.2 |
| 4 scores /10 = hallucination probable du LLM. | Cadrer dans le prompt : « base-toi UNIQUEMENT sur les défauts visibles, pas d'optimisme » + post-traitement (si `noise=true` → cap technical_score à 5) |
| Variantes IA-fingers / texte illisible : faux positifs sur photos macro. | Demander dans le prompt une **confiance** (`low/medium/high`) par flag |
| Spec attend `keywords` en anglais, mais l'app est en français. | Forcer `language="en"` côté pipeline (paramètre Settings H1) |
| L'UI n'a aujourd'hui qu'un panneau d'analyse — risque de surcharge. | Nouvelle vue dédiée `expert_report.py` accessible via Router (F1-F2) |
| Régression sur le mode mono-Shutterstock actuel. | Mode `"shutterstock"` conservé tel quel ; le mode `"expert"` est un chemin neuf. Anciens tests inchangés |

---

## 8. Critères d'acceptation

L'audit sera considéré comme adressé quand, pour une image test
(`tests/fixtures/sample.jpg`) :

- [ ] `api.analyze_image_expert(path)` retourne un `ExpertMetadataReport`
  contenant les 8 sections de la spec.
- [ ] Les 4 scores sont des entiers 0-10.
- [ ] Le rapport contient ≥ 30 keywords, en anglais, dont les 10
  premiers couvrent : sujet, action, contexte, émotion, usage commercial.
- [ ] L'export produit `*_adobe.csv` (5 colonnes) + `*_shutterstock.csv`
  (7 colonnes) — keywords séparés par virgules dans les deux.
- [ ] La vue `ExpertReportView` affiche les scores, les chips de
  keywords, les risques de rejet et les usages marketing.
- [ ] `pytest tests/` reste vert (smoke + unit) avec ≥ 5 nouveaux tests
  pour le mode expert.
- [ ] `ruff check` reste propre.

---

## 9. Annexe — Diff minimal pour le bug P0 (CSV)

```diff
--- a/src/modules/models/metadata_models.py
+++ b/src/modules/models/metadata_models.py
@@ def to_csv_row(self) -> Dict[str, str]:
     return {
         "Filename": self.filename,
         "Description": self.description,
-        "Keywords": " ".join(self.keywords),
+        "Keywords": ", ".join(self.keywords),
         "Categories": ",".join(self.categories),
         ...
```

> Vérifier sur le template officiel Shutterstock Contributor Portal
> (colonne **Keywords**) avant merge — la séparation comma est la valeur
> par défaut documentée, mais certaines versions de l'import acceptent
> les deux. Test à faire avec un mini-CSV 1 ligne dans la sandbox du
> portail.

---

*Fin de l'audit. Lot P0 prêt à être lancé sur demande.*
