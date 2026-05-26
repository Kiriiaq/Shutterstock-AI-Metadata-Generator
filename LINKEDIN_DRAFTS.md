# LinkedIn / Réseaux sociaux — drafts v2.1.0

> 3 formats prêts à copier-coller. Adapter le lien GitHub avant publication.
> Audience : developers + photographes microstock + recruteurs tech.
>
> **Pivot v2.1.0 (2026-05-27)** — les drafts pivotent du pitch « 80 % de
> la valeur sans IA » (pure-tech) au pitch « évaluation qualité de tes
> images en local » (acheteur). Le freemium est explicite : Community
> couvre le workflow IPTC complet ; Pro 29 €/an débloque l'évaluation
> (rapport expert, dual CSV, IA Ollama, batch illimité).

---

## Format 1 — Post technique court (≈ 1 100 caractères)

> **Pour quoi ?** Annonce neutre, format dev/tech, peu d'effort, premier
> jet d'amorçage. À publier le jour J du tag v2.1.0.

```text
🏷 ShutterstockAnalyzer v2.1 — un outil local pour évaluer la qualité de tes images microstock avant de les envoyer.

Le problème : un contributeur Adobe Stock / Shutterstock passe 5-10 minutes par image à écrire titre, description, mots-clés, catégories — sans savoir laquelle vaut le coup d'uploader. Pour 50 photos par semaine, c'est 5h de saisie + 0 visibilité commerciale.

Ma solution, 100 % locale, sans cloud :
✅ Édition Community (MIT, gratuite) — scan, édition IPTC, export CSV mono-plateforme, push FTP. Le workflow de base, end-to-end. + 2 aperçus gratuits du rapport expert pour tester.
✅ Édition Pro (29 €/an ou 79 € lifetime) — rapport expert illimité (4 scores : commercial / technique / SEO / risque rejet), dual CSV Adobe + Shutterstock simultané, enrichissement IA via Ollama (LLaMA Vision, LLaVA), anti-stuffing automatique, batch > 50 images.

Stack : Python 3.11 + CustomTkinter, ~20 000 LOC, 120 tests verts, EXE PyInstaller 25 Mo. HMAC pour les licences, zéro télémétrie, zéro photo qui quitte ta machine.

1 enseignement : monétiser ce qui apporte un *jugement* (les scores) plutôt qu'un cap technique (batch ∞). Le user paie pour l'expertise, pas pour la levée d'un quota.

👉 Code + EXE : [LIEN_GITHUB]
👉 Pro : [LIEN_GUMROAD]

#Python #Microstock #IndieDev
```

**Caractères : ~ 1 200** (limite LinkedIn ~ 3 000).
**Hashtags** : 3 ciblés, pas de pavé.
**Vérifications avant publication** :
- [ ] Lien GitHub canonique (release v2.1.0 publiée)
- [ ] Lien Gumroad live (sinon, retirer la 2ᵉ ligne CTA)
- [ ] GIF hero attaché en pièce jointe (boost engagement × 3)
- [ ] Publier mardi/jeudi 10h-12h heure FR (peak engagement)

---

## Format 2 — Carrousel 8 slides

> **Pour quoi ?** Format LinkedIn carousel (PDF 8 pages converti par
> LinkedIn). Très visuel, fort engagement, à publier ~ 1 semaine après
> le post court pour relancer.
>
> **Format technique** : 1080 × 1350 px par slide (ratio 4:5), PDF
> agrégé. Outils : Canva, Figma, PowerPoint export PDF.

---

### Slide 1 — HOOK

```
[Titre énorme]
Économisez 4 heures par semaine
sur vos métadonnées microstock.

[Sous-titre]
Adobe Stock + Shutterstock + local AI.
Open-source.

[Visuel central]
GIF still (1ʳᵉ frame) ou screenshot
Workspace de l'app

[Pied de page]
ShutterstockAnalyzer v2.0
```

---

### Slide 2 — PROBLÈME

```
[Titre]
Le contributeur microstock perd 5 à 10 minutes
par image en saisie manuelle.

[Bloc chiffré]
50 images / semaine
× 7 minutes / image
= 5 h 50 min de saisie répétitive

[Visuel]
Capture d'écran "avant" :
formulaire Adobe Stock avec champs vides
```

---

### Slide 3 — INSIGHT TECH

```
[Titre]
80 % de la valeur arrive
SANS intelligence artificielle.

[3 colonnes]
🔹 Métadonnées IPTC existantes
   → titre, mots-clés, byline

🔹 Propriétés image
   → résolution, format, espace
   colorimétrique → scores techniques

🔹 Règles métier
   → 50 keywords max, top 10 priorité,
   filtres anti-marques

[Bilan]
Résultat : un rapport complet en 200 ms.
```

---

### Slide 4 — FEATURE 1 : DOUBLE EXPORT

```
[Titre]
Adobe Stock + Shutterstock
en un seul export

[Capture] Modal Export Batch avec radio
"Les deux" sélectionné

[Texte]
2 formats CSV générés côte à côte :
- Adobe : Filename, Title, Keywords, Category
- Shutterstock : Filename, Description, Keywords...

UTF-8 BOM + virgules conformes
aux templates contributeurs.
```

---

### Slide 5 — FEATURE 2 : IA OPTIONNELLE

```
[Titre]
L'IA n'est PAS un prérequis.

[Texte]
☐ Décoché par défaut.

Si activée :
→ Ollama local (llama3.2-vision, LLaVA)
→ Pas de clé API, pas de coût récurrent
→ Vos photos ne quittent jamais votre disque

[Capture] Bandeau Ollama avec dropdown
modèle + bouton "⬇ Charger"
```

---

### Slide 6 — FEATURE 3 : FTP INTÉGRÉ

```
[Titre]
Push FTP vers le portail contributeur
en un clic.

[Texte]
Tester la connexion · Choisir Adobe ou
Shutterstock · Lancer · Done.

FTPS (TLS) par défaut.
Credentials jamais persistés.
Stdlib Python — aucune dépendance externe.

[Capture] Bandeau FTP révélé avec
champs hôte / user / mdp masqué
```

---

### Slide 7 — STACK & MÉTRIQUES

```
[Titre]
Sous le capot

[Tableau]
20 167  LOC Python
82      fichiers
120     tests verts (7 s)
49      tests qualification IHM
25 Mo   EXE PyInstaller
4       dépendances runtime

[Stack]
Python 3.11 · CustomTkinter ·
Pillow · SQLite · stdlib ftplib ·
ExifTool subprocess · Ollama HTTP
HMAC pour les licences (stdlib)

[Note]
Architecture : 1 facade backend,
UI 100 % indépendante.
Heuristique d'abord, IA en option.
```

---

### Slide 8 — CTA

```
[Titre]
Disponible maintenant.

[Bullet list]
🆓 Édition Community — gratuite, MIT, MicroStock complet
   + 2 aperçus gratuits du rapport expert
💼 Édition Pro — 29 €/an ou 79 € à vie
   Rapport expert illimité, dual CSV, IA Ollama,
   anti-stuffing, batch > 50

[Bloc CTA]
⭐ github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator
🛒 gumroad.com/l/shutterstockanalyzer-pro
☕ ko-fi.com/kiriiaq

[Pied]
Commentez si vous êtes contributeur microstock.
J'écris un retour d'expérience tech la semaine prochaine.

#Python #IndieDev #Microstock
```

---

## Format 3 — Post storytelling (≈ 1 900 caractères)

> **Pour quoi ?** Format long, raconte la genèse + l'apprentissage. Très
> bon pour les recruteurs et les contacts freelance. À publier ~ 2-3
> semaines après les deux premiers, quand le projet a accumulé un peu
> de visibilité.

```text
J'ai écrit un outil pour me débarrasser des 4 heures de saisie par semaine que me coûtaient mes uploads Adobe Stock et Shutterstock. Trois leçons valent le partage.

Le contexte : pour chaque image envoyée sur Adobe Stock ou Shutterstock, le contributeur doit fournir un titre descriptif, une description orientée acheteur, 7 à 50 keywords classés, une ou deux catégories, et flagger éditorial / mature / illustration. À 5-10 minutes par image, 50 images par semaine = 5h50 de saisie répétitive qui ne génère aucun revenu direct.

Première leçon : avant de coller de l'IA partout, regarder ce que les heuristiques peuvent faire. J'ai écrit un builder qui combine les métadonnées IPTC existantes du fichier, les propriétés techniques (résolution, espace colorimétrique, poids), et des règles métier (filtres anti-marques, anti-stuffing, top 10 keywords commerciaux prioritaires). Résultat : un rapport en 8 sections, 4 scores 0-10, généré en 200 ms par image. Zéro appel HTTP, zéro modèle chargé en RAM. Sur un laptop modeste, 15 images = 3 secondes. L'IA en option enrichit le résultat, elle n'est pas un prérequis.

Deuxième leçon : la posture lax bat la posture stricte sur ce type de produit. Adobe et Shutterstock ont leurs propres reviewers humains. Mon outil émet des warnings (résolution sous 4 MP, espace CMYK, plus de 50 keywords) mais ne bloque jamais l'export. Les utilisateurs préfèrent un outil qui les laisse décider à un outil qui refuse de produire le CSV.

Troisième leçon : l'IA locale change le rapport coût-bénéfice. Ollama + LLaMA 3.2 Vision tournent sur n'importe quel PC moderne, sans clé API, sans télémétrie, sans qu'aucune photo ne quitte le disque. Pour un photographe pro qui traite des images sous NDA, c'est non-négociable.

Stack : Python 3.11, CustomTkinter, SQLite local, ftplib stdlib pour le push FTP direct, PyInstaller pour packager en EXE 25 Mo.

Code + EXE : github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator

Si tu fais du microstock ou si tu shipes des outils desktop en Python : je suis preneur de retours ou d'idées d'extension. Et si tu cherches un freelance pour un projet similaire (UI desktop, pipeline data, intégration AI locale), DM ouvert.

#Python #IndieDev #FreelanceTech
```

**Caractères : ~ 1 950** (limite LinkedIn 3 000).

**Vérifications avant publication** :
- [ ] Ne PAS commencer par "I'm thrilled to..." / "Je suis fier de..."
- [ ] Lien en clair (LinkedIn dégrade les liens raccourcis)
- [ ] CTA double : (1) GitHub repo (2) DM ouvert pour freelance
- [ ] Pas plus de 3 hashtags
- [ ] Publier sans emoji si audience corporate

---

## Templates de réponses aux commentaires

À garder sous la main pour ne pas improviser :

### « Comment ça se compare à [outil X] ? »

> Bonne question. La différence clé : [outil X] tourne en SaaS / via clé
> API, donc tes photos transitent par leurs serveurs. ShutterstockAnalyzer
> tourne 100 % localement, avec ou sans IA. Pour des images sous NDA ou
> propriété client, c'est non-négociable. Côté features, le double export
> Adobe + Shutterstock simultané n'existe pas chez les concurrents que
> j'ai testés — chacun cible une plateforme.

### « Pourquoi pas macOS / Linux ? »

> Le code est portable. Le seul Windows-specific dans le bundle est
> l'ICO et un AppUserModelID. Un build macOS / Linux est sur la roadmap
> (issue #X à ouvrir). Si tu veux contribuer ou tester un build .app /
> AppImage, DM ouvert.

### « Le code est open-source mais il y a une version Pro ? »

> Oui — dual-license. Le core est MIT (gratuit, hackable, redistribuable)
> et couvre tout le workflow de base : scan, édition IPTC, export CSV
> mono-plateforme, push FTP. Tu as aussi 2 aperçus gratuits du rapport
> expert pour tester la valeur. La version Pro (29 €/an ou 79 € à vie)
> débloque le rapport expert illimité — c'est le module qui te dit
> *pourquoi* une image vaut le coup d'uploader, avec 4 scores et des
> améliorations concrètes — ainsi que l'export dual Adobe+Shutterstock
> simultané, l'enrichissement IA via Ollama local, et le batch > 50
> images. Tu paies pour le jugement automatisé, pas pour la levée d'un
> quota technique.

### « Tu acceptes les PRs ? »

> Oui, mais ouvre une issue d'abord pour les changements > 50 LOC.
> `CONTRIBUTING.md` détaille la quality bar (tests + ruff + format). Les
> petites corrections vont vite, les gros refactos je préfère en parler
> avant pour éviter qu'on parte en directions opposées.

### « Tu disponible pour du freelance ? »

> Oui — UI desktop Python, pipelines data, intégrations AI locales, audits
> d'architecture. DM ouvert, on en discute. Tarif jour disponible sur
> demande.

---

## Calendrier de publication suggéré

| Semaine | Lundi | Mardi | Mercredi | Jeudi | Vendredi |
|---|---|---|---|---|---|
| **S+0** (release) | — | **Post technique court** + GIF | — | — | — |
| **S+1** | — | — | — | **Carousel 8 slides** | — |
| **S+2** | — | — | Hacker News Show HN | — | — |
| **S+3** | — | Product Hunt launch | — | **Post storytelling** | — |
| **S+4** | Reddit r/stockphotography | — | Reddit r/learnpython | — | — |
| **S+5+** | Newsletter mensuelle / blog technique / engagement aux commentaires reçus | | | | |

**Cadence** : 1 post LinkedIn par semaine pendant 4 semaines, puis 1 / 2
semaines en rythme de croisière. Ne pas surcharger l'audience.

---

*Drafts à réviser une fois le repo public et les assets visuels produits
(GIF hero notamment). Bloc CTA et liens à valider canonique.*
