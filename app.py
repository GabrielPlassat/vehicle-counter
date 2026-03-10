import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
import io

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VehicleEye – Compteur YOLO",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Catégories ciblées (classes COCO) ────────────────────────────────────────
CATEGORIES = {
    0:  {"label": "Piéton",   "emoji": "🚶", "color": (255, 200,  50)},
    1:  {"label": "Vélo",     "emoji": "🚲", "color": ( 50, 220, 100)},
    2:  {"label": "Voiture",  "emoji": "🚗", "color": ( 50, 150, 255)},
    3:  {"label": "Moto",     "emoji": "🏍️", "color": (255, 100,  50)},
    5:  {"label": "Bus",      "emoji": "🚌", "color": (200,  50, 255)},
    7:  {"label": "Camion",   "emoji": "🚛", "color": (255,  50, 100)},
    9:  {"label": "Feu",      "emoji": "🚦", "color": (255, 255,  50)},
   24:  {"label": "Sac à dos","emoji": "🎒", "color": (100, 200, 200)},
}
TARGET_IDS = set(CATEGORIES.keys())

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0d0f14; }

.hero {
    background: linear-gradient(135deg, #0d0f14 0%, #131926 60%, #0a1628 100%);
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(50,150,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800;
    color: #ffffff; margin: 0 0 0.3rem;
    letter-spacing: -1px;
}
.hero h1 span { color: #3296ff; }
.hero p { color: #8899bb; font-size: 1rem; margin: 0; }

.counter-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 10px;
    margin: 1rem 0;
}
.counter-card {
    background: #131926;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
    transition: border-color 0.2s;
}
.counter-card.active { border-color: #3296ff; }
.counter-card .emoji { font-size: 1.6rem; margin-bottom: 4px; }
.counter-card .count {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem; font-weight: 800;
    color: #3296ff; line-height: 1;
}
.counter-card .name { font-size: 0.72rem; color: #8899bb; margin-top: 2px; }

.total-badge {
    background: linear-gradient(90deg, #1a2840, #1e3050);
    border: 1px solid #3296ff44;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 1rem;
}
.total-badge .total-num {
    font-family: 'Syne', sans-serif;
    font-size: 3rem; font-weight: 800; color: #3296ff; line-height: 1;
}
.total-badge .total-label { color: #8899bb; font-size: 0.9rem; }

.stButton > button {
    background: #3296ff; color: white; border: none;
    border-radius: 10px; font-family: 'DM Sans'; font-weight: 500;
    padding: 0.6rem 1.2rem; font-size: 1rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #1a7fe0; }

.mode-tab {
    background: #131926; border: 1px solid #1e2d45;
    border-radius: 10px; padding: 0.8rem 1rem;
    color: #8899bb; cursor: pointer; text-align: center;
    font-size: 0.85rem;
}

div[data-testid="stVerticalBlock"] { gap: 0.6rem; }

.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }

.tag {
    display: inline-block; background: #1e2d45; color: #8899bb;
    border-radius: 6px; padding: 2px 8px; font-size: 0.75rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Vehicle<span>Eye</span> 🎯</h1>
  <p>Détection & comptage en temps réel — propulsé par YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# ── Chargement du modèle ──────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

# ── Sidebar : paramètres ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    model_choice = st.selectbox(
        "Modèle YOLO",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="n = rapide (mobile), s = équilibré, m = précis"
    )
    confidence = st.slider("Confiance min.", 0.2, 0.9, 0.4, 0.05)
    show_labels = st.toggle("Afficher les labels", True)
    show_conf = st.toggle("Afficher la confiance", False)
    st.markdown("---")
    st.markdown("**Classes détectées :**")
    for cid, info in CATEGORIES.items():
        st.markdown(f"{info['emoji']} `{info['label']}`")

with st.spinner("Chargement du modèle YOLO..."):
    model = load_model(model_choice)

# ── Fonction de détection ─────────────────────────────────────────────────────
def detect_frame(frame_bgr, conf_thresh):
    results = model(frame_bgr, conf=conf_thresh, verbose=False)[0]
    counts = {cid: 0 for cid in CATEGORIES}
    annotated = frame_bgr.copy()

    for box in results.boxes:
        cid = int(box.cls[0])
        if cid not in TARGET_IDS:
            continue
        conf = float(box.conf[0])
        counts[cid] += 1
        info = CATEGORIES[cid]
        color = info["color"]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label = info["label"]
            if show_conf:
                label += f" {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return annotated, counts

def render_counters(counts):
    total = sum(counts.values())
    cards = ""
    for cid, info in CATEGORIES.items():
        n = counts.get(cid, 0)
        active = "active" if n > 0 else ""
        cards += f"""
        <div class="counter-card {active}">
          <div class="emoji">{info['emoji']}</div>
          <div class="count">{n}</div>
          <div class="name">{info['label']}</div>
        </div>"""

    st.markdown(f"""
    <div class="total-badge">
      <div class="total-num">{total}</div>
      <div>
        <div style="color:#fff;font-weight:600;font-size:1.1rem">objets détectés</div>
        <div class="total-label">sur cette frame</div>
      </div>
    </div>
    <div class="counter-grid">{cards}</div>
    """, unsafe_allow_html=True)

# ── Onglets de mode ───────────────────────────────────────────────────────────
tab_photo, tab_video, tab_cam = st.tabs(["📷 Photo", "🎬 Vidéo", "📱 Caméra live"])

# ─── MODE PHOTO ───────────────────────────────────────────────────────────────
with tab_photo:
    st.markdown("##### Importer une image")
    uploaded_img = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"],
                                    label_visibility="collapsed")
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        annotated, counts = detect_frame(frame_bgr, confidence)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(annotated_rgb, use_container_width=True, caption="Résultat de détection")
        with col2:
            render_counters(counts)

# ─── MODE VIDÉO ───────────────────────────────────────────────────────────────
with tab_video:
    st.markdown("##### Importer une vidéo")
    uploaded_vid = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"],
                                     label_visibility="collapsed", key="vid")

    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            frame_skip = st.slider("Analyser 1 frame sur :", 1, 10, 3,
                                   help="Augmenter pour aller plus vite")
        with col_ctrl2:
            max_frames = st.slider("Frames max à analyser :", 10, min(300, total_frames),
                                   min(100, total_frames))

        if st.button("▶️  Lancer l'analyse"):
            stframe = st.empty()
            counter_placeholder = st.empty()
            progress = st.progress(0)
            status = st.empty()

            frame_idx = 0
            analyzed = 0
            cumulative = {cid: 0 for cid in CATEGORIES}

            while cap.isOpened() and analyzed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                annotated, counts = detect_frame(frame, confidence)
                for cid in CATEGORIES:
                    cumulative[cid] = max(cumulative[cid], counts[cid])

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_rgb, use_container_width=True)

                with counter_placeholder.container():
                    render_counters(counts)

                analyzed += 1
                progress.progress(min(analyzed / max_frames, 1.0))
                status.caption(f"Frame {frame_idx} / {total_frames}  •  {analyzed} analysées")

            cap.release()
            st.success("✅ Analyse terminée !")
            st.markdown("**Maximums observés sur la vidéo :**")
            render_counters(cumulative)

# ─── MODE CAMÉRA LIVE ─────────────────────────────────────────────────────────
with tab_cam:
    st.markdown("##### Prise de photo depuis votre caméra")
    st.info("📱 Sur smartphone, cette option utilise directement votre appareil photo.")

    cam_img = st.camera_input("Prendre une photo")
    if cam_img:
        img = Image.open(cam_img).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        annotated, counts = detect_frame(frame_bgr, confidence)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(annotated_rgb, use_container_width=True, caption="Détection")
        with col2:
            render_counters(counts)

        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="JPEG", quality=90)
        st.download_button("⬇️ Télécharger le résultat", buf.getvalue(),
                           file_name="vehicleeye_result.jpg", mime="image/jpeg")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:0.8rem'>"
    "VehicleEye · YOLOv8 (Ultralytics) · Streamlit · "
    "<a href='https://github.com/ultralytics/ultralytics' style='color:#3296ff'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
