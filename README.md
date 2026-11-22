# Transformation de Perspective et Orthorectification de Documents sous OpenCV (C++)

**Auteur :** Mohammed SALHI  
**Travail :** Devoir individuel — Traitement d'images et Vision par Ordinateur (C++ / OpenCV)

---

## 1. Contexte et Objectif du Projet

L'application d'une **transformation de perspective (homographie planaire 2D)** permet de corriger les distorsions projectives subies par un document plat photographié sous un angle quelconque. À partir de la photographie d'une page posée sur un fond, l'objectif de ce travail individuel est d'isoler le quadrilatère délimitant la feuille, puis de calculer et d'appliquer la matrice de transformation projective afin que la page apparaisse comme une **vue de dessus redressée (orthorectifiée)**.

Conformément au cahier des charges de l'énoncé, le programme C++ développé avec OpenCV intègre les quatre étapes fondamentales :
1. **Détection automatique personnalisée des 4 coins du document** (adaptée aux fonds simples comme aux fonds texturés) ;
2. **Affichage visuel du résultat de cette détection** (superposition du quadrilatère et identification des 4 sommets ordonnés) ;
3. **Phase d'ajustement interactif des points détectés** (manipulation à la souris en temps réel par *glisser-déposer / Drag & Drop*) ;
4. **Application de la transformation de perspective** et génération de la vue orthorectifiée.

---

## 2. Justification du Choix des Méthodes selon le Type d'Image

L'énoncé exige la prise en charge de deux catégories d'images : les **images à fond simple** et les **images à fond texturé**. Une méthode unique ne pouvant être optimale simultanément sur ces deux topologies, nous avons conçu deux pipelines spécialisés ainsi qu'un estimateur automatique de complexité de fond.

### 2.1. Cas n°1 : Images à fond simple (fond uniforme et contrasté par rapport à la page)

* **Caractéristiques du signal :** Lorsqu'une page claire est posée sur un support uniforme et contrasté (exemple : `pa.JPG`, `pa_.jpeg`), l'histogramme des intensités lumineuses est nettement **bimodal** : un mode correspond aux pixels de l'arrière-plan et l'autre aux pixels de la feuille.
* **Méthode retenue (`detectOnSimpleBackground`) :**
  1. **Conversion en niveaux de gris et lissage Gaussien ($5 \times 5$)** : Réduit le bruit électronique du capteur photographique sans déplacer les frontières du document.
  2. **Seuillage global automatique d'Otsu (`cv::THRESH_OTSU`)** : Détermine analytiquement le seuil optimal $T^*$ qui maximise la variance inter-classes $\sigma_B^2(T) = \omega_0(T)\omega_1(T)[\mu_0(T) - \mu_1(T)]^2$ entre le fond et la page.
  3. **Fermeture morphologique (`cv::MORPH_CLOSE`, noyau rectangulaire $9 \times 9$)** : Opération de dilatation suivie d'une érosion permettant de combler les caractères imprimés sombres situés à l'intérieur de la page blanche afin d'obtenir une composante connexe pleine et unique.
  4. **Extraction de l'enveloppe convexe (`cv::convexHull`) et approximation polygonale de Ramer-Douglas-Peucker (`cv::approxPolyDP`)** : Compense les éventuelles occlusions mineures ou coins légèrement courbés et extrait exactement les $4$ sommets du quadrilatère convexe dominant.
* **Justification du choix :** Sur un fond uniforme, une approche par région (binarisation statistique d'Otsu) est mathématiquement optimale, insensible aux gradients internes du texte imprimé après fermeture morphologique, et présente un coût calculatoire minimal.

---

### 2.2. Cas n°2 : Images à fond texturé (fond complexe, boisé ou bruité)

* **Caractéristiques du signal :** Lorsque la page repose sur un support fortement texturé ou bruité (veines de bois sombre dans `cell_pic.jpg`, surface granuleuse dans `receipt.jpg`, motifs ou variations d'éclairage), l'hypothèse de bimodalité de l'histogramme tombe en défaut. Un seuillage global fusionne certaines zones du fond avec la page et génère de faux contours.
* **Méthode retenue (`detectOnTexturedBackground`) :**
  1. **Filtrage bilatéral (`cv::bilateralFilter`, $d=9, \sigma_{\text{color}}=75, \sigma_{\text{space}}=75$)** : Contrairement au flou gaussien qui dégrade les bords, le filtre bilatéral pondère chaque voisin à la fois par sa proximité spatiale et par sa proximité radiométrique :
     $$I^{\text{filtré}}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \, G_{\sigma_s}(\|p - q\|) \, G_{\sigma_r}(|I(p) - I(q)|)$$
     Il lisse ainsi les textures répétitives de l'arrière-plan (veines du bois, grain de la table) tout en **préservant intégralement la discontinuité franche que constitue le bord physique de la feuille**.
  2. **Gradient morphologique (`cv::MORPH_GRADIENT`, noyau $5 \times 5$) + Fermeture morphologique large ($15 \times 15$)** : Met en évidence les transitions géométriques majeures et reconnecte les segments de contour interrompus par la texture du support.
  3. **Repli multi-échelle par détecteur de Canny (`cv::Canny`)** : Si le gradient morphologique ne converge pas vers un quadrilatère convexe, une extraction d'arêtes par hystérésis de Canny suivie d'une dilatation permet de capturer la silhouette externe de la page.
  4. **Approximation de Ramer-Douglas-Peucker sur enveloppe convexe** : Isole le quadrilatère final à 4 sommets.
* **Justification du choix :** Sur fond complexe, la séparation ne doit plus reposer sur l'intensité brute mais sur la **persistance structurelle des contours**. Le couplage du filtre bilatéral (suppression de la texture haute fréquence) et du gradient morphologique permet de discriminer les frontières rectilignes macroscopiques de la page face aux micro-textures stochastiques du fond.

---

### 2.3. Critère automatique de sélection et mécanisme de repli

Au lancement en mode automatique (`--auto`), le programme évalue la **variance du Laplacien** $\text{Var}(\nabla^2 I)$ exclusivement sur une bande périphérique de $10\%$ le long des bords de l'image (zone appartenant à l'arrière-plan) :
- Une faible variance traduit un fond homogène $\rightarrow$ sélection de la méthode **Fond Simple**.
- Une variance élevée traduit un fond bruité ou texturé $\rightarrow$ sélection de la méthode **Fond Texturé**.
- En outre, un mécanisme de repli croisé (*fallback*) bascule automatiquement sur la seconde méthode si la première ne trouve pas un quadrilatère convexe valide, et l'utilisateur peut à tout moment basculer manuellement entre les deux méthodes via les touches **`1`** et **`2`**.

---

## 3. Modèle Mathématique de la Transformation de Perspective

Une fois les 4 coins détectés (et éventuellement ajustés à la souris), ils sont ordonnés canoniquement dans le sens horaire : **Haut-Gauche ($TL$)**, **Haut-Droit ($TR$)**, **Bas-Droit ($BR$)** et **Bas-Gauche ($BL$)** grâce aux sommes $(x+y)$ et différences $(y-x)$ des coordonnées :
$$TL = \arg\min_{P_i}(x_i + y_i), \quad BR = \arg\max_{P_i}(x_i + y_i), \quad TR = \arg\min_{P_i}(y_i - x_i), \quad BL = \arg\max_{P_i}(y_i - x_i)$$

Afin de respecter les proportions géométriques de la page originale sans écrasement, la largeur $W$ et la hauteur $H$ de l'image redressée sont déterminées par les longueurs euclidiennes maximales des côtés opposés :
$$W = \max\left(\|BR - BL\|_2, \; \|TR - TL\|_2\right), \qquad H = \max\left(\|TR - BR\|_2, \; \|TL - BL\|_2\right)$$

La matrice d'homographie $H \in \mathbb{R}^{3 \times 3}$ reliant les coordonnées sources $(x, y)$ aux coordonnées orthorectifiées $(x', y') \in [0, W-1] \times [0, H-1]$ vérifie en coordonnées homogènes :
$$\begin{bmatrix} s \, x' \\ s \, y' \\ s \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Elle est résolue exactement à partir des 4 correspondances via `cv::getPerspectiveTransform`, puis appliquée à l'image entière par interpolation bilinéaire avec `cv::warpPerspective`.

---

## 4. Structure du Projet

```text
Change-image-perspective-with-OpenCV/
├── main.cpp          # Implémentation C++17 complète (Mohammed SALHI)
├── Makefile          # Script de compilation GNU Make
├── CMakeLists.txt    # Configuration de compilation CMake
├── README.md         # Rapport technique et justifications des méthodes
└── data/             # Images de test fournies avec l'énoncé
    ├── cell_pic.jpg  # Page photographiée sur table en bois sombre nervuré
    ├── pa.JPG        # Document incliné sur surface claire
    ├── pa_.jpeg      # Dépliant photographié en perspective forte
    └── receipt.jpg   # Ticket de caisse sur fond sombre granuleux
```

---

## 5. Compilation et Exécution

### Prérequis
- Compilateur C++ compatible **C++17** (`g++`)
- Bibliothèque **OpenCV 4.x** (`libopencv-dev`)

### Compilation
Avec `make` :
```bash
make clean && make
```
Ou avec `CMake` :
```bash
mkdir -p build && cd build && cmake .. && make
```

### Exécution sur les images de test
```bash
# Exécution avec détection automatique du type de fond
./perspective_transform data/pa.JPG
./perspective_transform data/pa_.jpeg
./perspective_transform data/cell_pic.jpg
./perspective_transform data/receipt.jpg

# Forcer explicitement la méthode Fond Simple ou Fond Texturé
./perspective_transform data/pa.JPG --simple
./perspective_transform data/cell_pic.jpg --textured
```

---

## 6. Fonctionnement de la Phase d'Ajustement Interactif

Lors de l'exécution, deux fenêtres s'ouvrent simultanément :
1. **`Mohammed SALHI - Detection & Ajustement des 4 Coins`** : Affiche l'image originale, le quadrilatère vert semi-transparent détecté et les 4 poignées circulaires numérotées `C1 (HG)`, `C2 (HD)`, `C3 (BD)`, `C4 (BG)`.
2. **`Mohammed SALHI - Vue de Dessus (Orthorectifiee)`** : Affiche en temps réel le document redressé.

| Commande | Action réalisée |
| :--- | :--- |
| **Clic gauche + Glisser (Souris)** | Déplace interactivement l'un des 4 coins (*Drag & Drop*) et recalcule la perspective en temps réel |
| **Touche `1`** | Applique la méthode **Fond Simple** (Otsu + Fermeture morphologique) |
| **Touche `2`** | Applique la méthode **Fond Texturé** (Filtre bilatéral + Gradient morphologique / Canny) |
| **Touche `A`** | Relance la sélection **Auto-adaptative** |
| **Touche `R`** | Réinitialise les 4 points pour permettre une sélection manuelle intégrale à la souris |
| **Touche `S`** | Enregistre l'image redressée sur le disque (`<nom>_redresse.<ext>` et `document_redresse.jpg`) |
| **Touche `Q` / `Échap`** | Ferme l'application |
