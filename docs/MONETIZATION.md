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
- **Tech mature** : 90 tests, EXE PyInstaller, doc, architecture propre
  — niveau « shippable » atteint.
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

### 2.1 Frontière OSS / Pro

| Feature | Édition **Community** (MIT, gratuite) | Édition **Pro** (commerciale, payante) |
|---|---|---|
| Scan dossier + sélection multi-fichiers | ✅ | ✅ |
| Lire / écrire IPTC | ✅ | ✅ |
| Rapport expert heuristique (sans IA) | ✅ | ✅ |
| Rapport expert IA Ollama | ✅ | ✅ |
| Export CSV Adobe **OU** Shutterstock | ✅ | ✅ |
| Export double CSV (Adobe + Shutterstock) | ✅ | ✅ |
| Validation pré-upload | ✅ | ✅ |
| Historique opérations | ✅ | ✅ |
| **Batch ≤ 50 images / run** | ✅ | ✅ |
| **Batch illimité (> 50, parallélisation worker pool)** | 🔴 | ✅ |
| **Push FTP unique post-export** | ✅ (déjà gratuit) | ✅ |
| **FTP scheduling** (push horaire, run-and-forget) | 🔴 | ✅ |
| **Templates IPTC customs** (export/import) | 🔴 | ✅ |
| **Multi-comptes FTP** (Adobe + Shutterstock simultané) | 🔴 | ✅ |
| **Profils prompt IA par catégorie** (Business, Nature…) | 🔴 | ✅ |
| **Stats avancées** (revenus simulés, top keywords du compte) | 🔴 | ✅ |
| **Support prioritaire** (réponse < 48 h ouvrées) | 🔴 | ✅ |
| **Licence d'usage entreprise** (≤ 5 postes) | 🔴 | ✅ |

> **Note** : le push FTP basique reste dans l'édition Community pour ne
> pas casser le workflow d'usage actuel. C'est le **scheduling et le
> multi-comptes** qui justifient le tier Pro.

### 2.2 Tarification cible

| Tier | Prix | Cible | Justification |
|---|---|---|---|
| **Community** | 0 € | Photographes occasionnels, étudiants, curieux | Acquisition, démonstration, retours utilisateurs |
| **Pro Solo** | **29 € / an** | Photographe pro indépendant | ~ 2-3 heures économisées / mois sur 1 an |
| **Pro Studio** | **89 € / an** | Studio / agence, ≤ 5 postes | Multi-poste + support |
| **One-shot Lifetime** | **79 € unique** | Achat sans renouvellement | Alternative aux abonnements |

Stratégie : **abonnement annuel par défaut**, lifetime en alternative pour
les réfractaires (audience photographe = souvent allergique aux
subscriptions). Une licence Pro débloque les features Pro localement
(activation par clé hors-ligne, vérification cryptographique sur le
binaire).

### 2.3 Effort estimé (mise en marché)

| Lot | Détail | Effort |
|---|---|---|
| **Mécanique de licence** | Génération clés, vérification offline, persistance dans `~/.shutterstock_ai/license` | 2 j |
| **Build dual** | Branche `pro/` privée + features pro derrière `if license.is_pro()` checks | 1 j |
| **Frontière logique** | Batch > 50, scheduling FTP (background loop), multi-comptes UI | 2 j |
| **Page produit** | Landing GitHub Pages, démo embarquée, FAQ, CTA Gumroad | 1 j |
| **Listing Gumroad** | Création produit, screenshots, vidéo 60 s, descriptions Adobe + Shutterstock keywords | 0,5 j |
| **Système de support** | Email dédié, template FAQ, premiers SLA | 0,5 j |
| **Communication d'amorçage** | LinkedIn × 3, Product Hunt, Reddit, IndieHackers | 0,5 j |
| **Total amorce v0** | | **~ 7-8 j-h** |

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

- [ ] Tag `git tag v2.0.0 && git push --tags`
- [ ] Capture GIF hero + 3 screenshots
- [ ] GitHub Release v2.0.0 avec EXEs attachés
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
| **Indie Hackers** | Milestone post « Launched v2.0 of [pitch] » | Audience makers, retour qualitatif |
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
| **Gumroad** | Listing « ShutterstockAnalyzer Pro » 29 €/an et 79 € lifetime, lien depuis le README v2 |
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
| **Voie principale** | Freemium dual-license MIT + Pro 29-89 €/an |
| **Voie fallback** | Lead magnet portfolio (open-source pur, focus visibilité freelance) |
| **Effort amorce** | ~ 7-8 j-h |
| **Revenu réaliste 12 mois** | 3 500–4 500 € (~ 25-30 customers actifs) |
| **Risque principal** | Marché microstock en saturation IA — la différenciation est l'usage *local* sans cloud |
| **Différenciation forte** | Seul outil microstock multi-plateforme avec IA locale + FTP intégré |
| **Premier livrable** | Tag v2.0.0 + GitHub Release + 3 posts LinkedIn |
| **Premier ROI mesurable** | Mois 3 — premiers 4-5 customers payants |

---

*Document à reviewer 6 mois après lancement. Si revenu réel < 50 % de la
projection : pivoter vers la voie fallback (lead magnet). Si > 150 % :
recruter un freelance pour macOS/Linux build + Lightroom plugin.*
