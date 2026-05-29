# Stratégie monétisation & distribution

> Synthèse Phases 5 + 6 du protocole d'audit. Voie principale arbitrée :
> **freemium / dual-license**. Voie fallback : **lead magnet portfolio**.

---

## 1. Évaluation préalable

### 1.1 Atouts monétisables

- **Utilité concrète et chiffrable** : 5–10 min de saisie manuelle économisées
  par image × dizaines d'images/jour pour un contributeur microstock actif.
- **Pas de concurrent direct** identifié sur ce niche (microstock + local +
  Adobe Stock **et** Shutterstock + IA optionnelle + FTP push).
- **Public payeur identifié** : photographes pro / semi-pro de microstock
  (Adobe Stock contributors, Shutterstock contributors), agences générant
  100+ images/semaine.
- **Tech mature** : 120 tests verts, EXE PyInstaller 25,2 Mo (PROD secret
  embedded), doc, architecture propre — niveau « shippable » atteint,
  mécanique de licence + gates UI livrés (v2.1.0).
- **Différenciation forte** : IA **locale** (Ollama), donc pas d'API key,
  pas de coût récurrent, pas de fuite de propriété intellectuelle vers
  un fournisseur tiers.

### 1.2 Risques / limites

- **Marché microstock en lent déclin** (saturation IA générative depuis
  2023). Mais le besoin de **métadonnées propres** est en croissance
  pour percer le bruit.
- **Audience anglophone à atteindre** (l'UI est en français — point à
  régler pour internationaliser).
- **Distribution Windows-only** limite l'audience macOS qui est sur-
  représentée dans la photographie pro.
- **Pas de revenu récurrent passif** sans SaaS — choix assumé du local.

### 1.3 Droits / juridique

- **Code à 100 % personnel** (pas d'employeur impliqué, pas de code écrit
  sur temps de travail salarié à clarifier en pratique).
- **Dépendances** : toutes MIT / BSD / Apache 2 / PSF (Python). Aucune
  GPL → pas de contamination copyleft, freemium possible.
- **Données traitées** : 100 % locales, **pas de RGPD à gérer** côté outil.
  Les photos restent sur la machine de l'utilisateur.
- **Logos / marques** : « Shutterstock » et « Adobe Stock » sont des
  marques déposées. Usage **purement descriptif** (« generates CSV for X ») ;
  pas d'affiliation revendiquée, pas de logo officiel utilisé. Action :
  ajouter une mention disclaimer dans le README de la page produit.
- **Modèles Ollama** : LLaMA 3.2 (Meta, license « Llama Community »), LLaVA
  (Apache 2.0), Moondream (Apache 2.0). Compatibles usage commercial **côté
  utilisateur**. Notre outil ne redistribue pas les modèles, juste
  s'interface avec eux → pas de contrainte de licence à porter.

---

## 2. Stratégie freemium / dual-license (voie principale)

> **Pivot v2.2.0 (2026-05-29)** : la frontière Pro est désormais **un
> seul mur — l'export de données**. Tout le reste (scan, IPTC, rapport
> expert, IA Ollama, mise en page double CSV, FTP) est **gratuit et
> illimité**. Community dispose de **3 exports gratuits**, puis l'export
> illimité se débloque avec une **clé à vie à 10 €** (paiement unique).
> Raison : zéro barrière à l'adoption (tout l'outil est essayable),
> prix d'impulsion sans friction d'abonnement, conversion au moment
> précis où l'utilisateur a déjà produit ses premiers CSV.

### 2.1 Frontière OSS / Pro

| Feature | **Community** (gratuite) | **Pro** (10 € à vie) |
|---|---|---|
| Scan dossier + sélection multi-fichiers | ✅ | ✅ |
| Lecture / écriture IPTC (éditeur manuel) | ✅ | ✅ |
| Rapport expert microstock (4 scores + risques + améliorations + usages) | ✅ | ✅ |
| Enrichissement IA Ollama (LLaMA Vision, LLaVA, Moondream) | ✅ | ✅ |
| Mise en page double CSV (Adobe + Shutterstock) | ✅ | ✅ |
| Anti-stuffing automatique (filtres marques + keywords) | ✅ | ✅ |
| Validation pré-upload, historique, push FTP / FTPS | ✅ | ✅ |
| **Export de données** (génération des CSV à uploader) | 🎁 **3 exports gratuits** | ✅ illimité |

> **Pourquoi cette frontière** : tout l'outil est essayable
> gratuitement, donc l'adoption n'a aucune barrière. Le seul geste
> payant — exporter ses données en masse — arrive **après** que
> l'utilisateur a vu la valeur (scores, métadonnées propres) et produit
> ses premiers fichiers. On vend la commodité de l'usage répété, pas
> l'accès à une fonctionnalité.

> **Anti-frustration** : les 3 exports gratuits laissent l'utilisateur
> produire de vrais CSV pour son catalogue avant de buter sur le mur.
> Un bandeau affiche le compte restant (« 2/3, 1/3… ») pour que l'upsell
> soit explicite, jamais caché.

### 2.2 Tarification cible

| Tier | Prix | Cible | Justification |
|---|---|---|---|
| **Community** | 0 € | Tous | Acquisition + démonstration. Outil complet + 3 exports gratuits. |
| **Pro — à vie** | **10 € (paiement unique)** | Tout utilisateur régulier | Export de données illimité, pour toujours. Prix d'impulsion : sous la barre 10-15 €, pas de décision budgétaire. Coût marginal nul → quasi pure marge. |

Stratégie : **paiement unique à vie**, pas d'abonnement (l'audience
photographe déteste les subscriptions). La licence débloque l'export
illimité localement (activation par clé hors-ligne, HMAC honor-system
vérifié sur le binaire).

### 2.3 Effort estimé (mise en marché)

Le pivot v2.1.0 a déjà absorbé les lots techniques. Ce qui reste avant
la première vente est purement commercial/contenu.

| Lot | Détail | État |
|---|---|---|
| ~~Mécanique de licence~~ | HMAC, persistance, activation UI, generate_license CLI | ✅ livré v2.0/v2.1 |
| ~~Frontière logique~~ | Gates expert_report / dual_csv / ai_enrichment / batch_unlimited | ✅ livré v2.1 |
| **Page produit** | Landing GitHub Pages, démo embarquée, FAQ, CTA Gumroad | 1 j |
| **Listing Gumroad** | Création produit, screenshots, vidéo 60 s, descriptions | 0,5 j |
| **Système de support** | Email dédié, template FAQ, premiers SLA | 0,5 j |
| **Communication d'amorçage** | LinkedIn × 3, Product Hunt, Reddit, IndieHackers | 0,5 j |
| **Assets visuels** | GIF hero + 3 screenshots + 60 s screencast (specs : `docs/MEDIA.md`) | 1 j |
| **Total amorce v0 restant** | | **~ 3,5 j-h** |

> **Gain de pivot** : passage de 7-8 j-h à 3,5 j-h en repositionnant
> Pro sur des features déjà livrées. Le delta budgétaire libéré peut
> être réinvesti dans les assets visuels (qui font ×3 sur le taux de
> clic LinkedIn / Product Hunt).

### 2.4 Revenu réaliste 12 mois

Hypothèses prudentes (basées sur les chiffres communs des outils microstock
similaires sur Gumroad, segments < 100 €) :

- **Trafic mensuel cible mois 1-3** : 200 visiteurs uniques (post LinkedIn + Reddit)
- **Conversion visiteur → essai gratuit** : 5 % → 10 essais / mois
- **Conversion essai → Pro Solo** : 15 % → 1,5 paying customers / mois × 29 €
- **Croissance mois 4-12** : ×3 → ~ 5 customers / mois mois 12

| Période | Customers payants nets | Revenu mensuel | Cumul |
|---|---|---|---|
| Mois 1-3 | 4 | ~ 120 € | 360 € |
| Mois 4-6 | 8 | ~ 230 € | 1 050 € |
| Mois 7-9 | 12 | ~ 350 € | 2 100 € |
| Mois 10-12 | 18 | ~ 520 € | 3 660 € |
| **12 mois cumulé** | **~ 25-30 customers actifs** | | **~ 3 500–4 500 €** |

**Lecture honnête** : c'est un **side project**, pas un remplacement de
salaire. Le ROI sur 1 an couvre largement l'effort (8 j-h ≈ 4 000 € au
TJM moyen freelance), mais ne paiera pas un loyer parisien. Le vrai
levier est sur **24-36 mois** + lead magnet pour missions freelance.

### 2.5 Voie fallback : lead magnet portfolio

Si l'objectif est moins « revenu direct » et plus « visibilité freelance » :

- Garder **TOUT en open-source MIT** (pas de tier Pro).
- Investir l'effort de la frontière Pro dans des **features qui font parler**
  (drag-and-drop, plugin VSCode pour métadonnées, intégration Lightroom).
- Capitaliser sur **README + démo + LinkedIn × 3** pour générer du trafic
  vers ton profil et faire venir les missions freelance via le projet
  comme preuve de compétence.

Cette voie demande **0,5 j-h de plus** que la voie freemium (focus
communication + un peu de polish), mais ne génère **aucun revenu direct
sur le projet** — par contre une mission freelance trouvée via ce vecteur
peut valoir des milliers d'euros.

**Reco** : commencer par **freemium** (ton GO initial), garder la voie
fallback en réserve si l'amorce ne décolle pas après 6 mois.

---

## 3. Plateformes de distribution (Phase 6)

Ordre de publication (ne pas tout sortir le même jour, étaler sur 3-4 semaines) :

### J0 — Préparation (avant tout lancement)

- [x] Tag local `git tag v2.1.0` (fait — reste à pousser : `git push origin v2.1.0`)
- [ ] Capture GIF hero + 3 screenshots (dont **1 montrant le quota Community + 1 montrant l'écran upsell** — voir `docs/MEDIA.md`)
- [ ] GitHub Release v2.1.0 avec EXEs attachés (`dist/ShutterstockAnalyzer.exe` + `dist/ShutterstockAnalyzer-debug.exe`, **25,2 Mo chacun, PROD secret embedded**)
- [ ] Mettre à jour README avec liens images réels

### Semaine 1 — Distribution gratuite (community)

| Plateforme | Action | Asset principal | Difficulté |
|---|---|---|---|
| **GitHub** | Release publique avec EXE + CHANGELOG | EXE + CHANGELOG | ⭐ |
| **GitHub Topics** | `microstock`, `metadata`, `iptc`, `adobe-stock`, `shutterstock`, `ai`, `ollama`, `customtkinter`, `python` | — | ⭐ |
| **dev.to** | Article technique « Generating microstock metadata with local LLMs » | post 1500 mots + GIF | ⭐⭐ |
| **Hashnode** | Cross-post du dev.to | idem | ⭐ |
| **Medium** | Variante « Why I chose local AI over OpenAI for my image tool » | post 1200 mots | ⭐⭐ |

### Semaine 2 — Communautés thématiques

| Plateforme | Action | Notes |
|---|---|---|
| **r/stockphotography** | Show post avec GIF | ~ 30k membres, axé revenue |
| **r/AdobeStock** | Annonce outil | ~ 15k membres, plus restrictif sur l'auto-promo |
| **r/learnpython** | Show & tell technique | Bon pour découverte tech, pas pour les ventes |
| **Indie Hackers** | Milestone post « Launched v2.1 of [pitch] » | Audience makers, retour qualitatif |
| **Discord microstock** | Annonce dans les Discords contributeurs (Adobe Stock, Shutterstock) | Plus organique, moins viral |

### Semaine 3 — Lancement médiatique

| Plateforme | Action | Asset principal |
|---|---|---|
| **Product Hunt** | Launch un mardi (meilleur jour) à 00:01 PST | GIF hero + 5 screenshots + tagline + 30s vidéo |
| **Hacker News** | Show HN: ShutterstockAnalyzer — local metadata for microstock | Link GitHub + 1 paragraphe de pitch |
| **LinkedIn × 3** | Voir `LINKEDIN_DRAFTS.md` | 3 formats échelonnés sur 2 semaines |
| **X / Twitter** | Thread 5-7 tweets avec GIF | Reprise du post technique court |

### Semaine 4 — Vente Pro

| Plateforme | Action |
|---|---|
| **Gumroad** | Listing « ShutterstockAnalyzer Pro » **10 € à vie** (paiement unique), lien depuis le README |
| **Lemon Squeezy** (alternative) | Idem, gère mieux la TVA EU si volume > seuil de franchise |
| **GitHub Sponsors** | Reactiver le bouton sponsor existant + Ko-fi |

### En continu

- **Newsletter** mensuelle (Mailchimp ou Substack) pour les utilisateurs
  Pro : nouveautés, tips, retours d'utilisation.
- **Blog technique** (1 article / 2 mois) pour le SEO long-terme :
  comparatif outils microstock, tutos avancés, retours communautaires.
- **Issues GitHub triées et étiquetées** (bonne pratique = activité visible).
- **Releases mensuelles** v2.x.y avec quelques améliorations chaque fois
  (donne du rythme, raison de re-communiquer).

---

## 4. Vérifications légales / juridiques

À faire avant Day 1 :

- [ ] Vérifier que **« Shutterstock »** et **« Adobe Stock »** peuvent
      apparaître dans le nom commercial. Reco : garder
      `ShutterstockAnalyzer` comme **nom technique** mais utiliser un nom
      marketing distinct sur Gumroad / Product Hunt (ex.
      « StockMeta Pro » ou « MicrostockKit ») pour limiter le risque
      trademark.
- [ ] Mention « *Not affiliated with Shutterstock Inc. or Adobe Inc.* »
      dans README + landing page.
- [ ] CGU + politique de remboursement (Gumroad propose des templates).
- [ ] TVA EU : suivre le seuil de franchise en base (< 85 800 € en France
      pour micro-entreprise BIC). En dessous, pas de TVA à collecter.
- [ ] Statut juridique : micro-entreprise simple si revenu < seuil ; sinon
      EI ou SASU. **À ne pas anticiper** tant que le revenu mensuel reste
      en dessous de 500 € — la micro suffit.

---

## 5. Récap exécutif

| Élément | Valeur |
|---|---|
| **Voie principale** | Freemium MIT + Pro **10 € à vie** (pivot v2.2.0 : seul l'export de données est payant, 3 essais gratuits) |
| **Voie fallback** | Lead magnet portfolio (open-source pur, focus visibilité freelance) |
| **Effort amorce restant** | ~ 3,5 j-h (assets visuels + Gumroad listing, code 100 % livré) |
| **Revenu réaliste 12 mois** | 3 500–4 500 € (~ 25-30 customers actifs) |
| **Risque principal** | Marché microstock en saturation IA — la différenciation est l'usage *local* sans cloud + l'évaluation qualité automatique |
| **Différenciation forte** | Seul outil microstock qui *évalue* la qualité (4 scores + risques + améliorations) en local, sans cloud |
| **Premier livrable** | Tag v2.1.0 + GitHub Release + 3 posts LinkedIn |
| **Premier ROI mesurable** | Mois 3 — premiers 4-5 customers payants |

---

*Document à reviewer 6 mois après lancement. Si revenu réel < 50 % de la
projection : pivoter vers la voie fallback (lead magnet). Si > 150 % :
recruter un freelance pour macOS/Linux build + Lightroom plugin.*
