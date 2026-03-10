# 🚗 VehicleEye — Compteur de véhicules YOLO

Détection, classification et comptage de véhicules sur flux vidéo,  
accessible depuis n'importe quel smartphone via un lien web.

---

## 📦 Contenu du projet

```
vehicle-counter/
├── app.py              ← Application principale Streamlit
├── requirements.txt    ← Dépendances Python
└── README.md           ← Ce fichier
```

---

## 🎯 Fonctionnalités

| Catégorie  | Emoji | Classe COCO |
|------------|-------|-------------|
| Piéton     | 🚶   | 0           |
| Vélo       | 🚲   | 1           |
| Voiture    | 🚗   | 2           |
| Moto       | 🏍️   | 3           |
| Bus        | 🚌   | 5           |
| Camion     | 🚛   | 7           |

**3 modes d'utilisation :**
- 📷 **Photo** — importer une image
- 🎬 **Vidéo** — analyser un fichier vidéo frame par frame
- 📱 **Caméra live** — prendre une photo directement depuis un smartphone

---

## 🚀 Déploiement sur Streamlit Cloud (GRATUIT)

### Étape 1 — Mettre le code sur GitHub

1. Créer un compte sur [github.com](https://github.com) (si pas déjà fait)
2. Créer un **nouveau repository** public : `vehicle-counter`
3. Uploader les fichiers `app.py` et `requirements.txt` directement via l'interface web GitHub

### Étape 2 — Déployer sur Streamlit Cloud

1. Aller sur [share.streamlit.io](https://share.streamlit.io)
2. Se connecter avec son compte GitHub
3. Cliquer sur **"New app"**
4. Sélectionner :
   - **Repository** : `votre-username/vehicle-counter`
   - **Branch** : `main`
   - **Main file** : `app.py`
5. Cliquer sur **"Deploy !"**

⏱️ Le déploiement prend ~3 minutes.

### Étape 3 — Utiliser depuis smartphone

Une URL de la forme `https://votre-username-vehicle-counter.streamlit.app`  
sera générée — ouvrez-la dans n'importe quel navigateur mobile !

---

## ⚡ Test rapide sur Google Colab

Si vous voulez tester avant de déployer :

```python
# Installer les dépendances
!pip install streamlit ultralytics opencv-python-headless pyngrok

# Lancer l'app avec un tunnel public
from pyngrok import ngrok
import subprocess

proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
public_url = ngrok.connect(8501)
print("🌐 URL publique :", public_url)
```

---

## 🎛️ Paramètres disponibles

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| Modèle YOLO | `n` (rapide) / `s` (équilibré) / `m` (précis) | `yolov8n.pt` |
| Confiance min. | Seuil de détection (0.2 à 0.9) | 0.4 |
| Labels | Afficher le nom sur les boîtes | Oui |
| Confiance affichée | Afficher le % de confiance | Non |

> 💡 **Pour smartphone** : utiliser `yolov8n.pt` (le plus léger, ~6 MB)

---

## 📊 Performance estimée

| Modèle    | Taille | Précision | Vitesse sur CPU |
|-----------|--------|-----------|-----------------|
| yolov8n   | 6 MB   | ★★★☆☆    | ~0.5 s/frame    |
| yolov8s   | 22 MB  | ★★★★☆    | ~1.2 s/frame    |
| yolov8m   | 52 MB  | ★★★★★    | ~3 s/frame      |

*Les modèles sont téléchargés automatiquement au premier lancement.*

---

## 🛠️ Évolutions possibles

- [ ] Comptage cumulatif avec tracking (ByteTrack)
- [ ] Export CSV des comptages
- [ ] Ligne virtuelle pour comptage de passage
- [ ] Alertes si seuil dépassé
- [ ] Support flux RTSP (caméra IP)
