"""
VehicleEye — Détection, tracking ByteTrack, comptage ligne virtuelle, export CSV
Aucune dépendance cv2 : tout le dessin se fait via Pillow.
"""

import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import tempfile, io, csv, os
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VehicleEye", page_icon="🚗",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Catégories COCO ciblées ───────────────────────────────────────────────────
CATEGORIES = {
    0: {"label": "Piéton",  "emoji": "🚶", "color": (255, 200,  50)},
    1: {"label": "Vélo",    "emoji": "🚲", "color": ( 50, 220, 100)},
    2: {"label": "Voiture", "emoji": "🚗", "color": ( 50, 150, 255)},
    3: {"label": "Moto",    "emoji": "🏍️", "color": (255, 100,  50)},
    5: {"label": "Bus",     "emoji": "🚌", "color": (200,  50, 255)},
    7: {"label": "Camion",  "emoji": "🚛", "color": (255,  50, 100)},
}
TARGET_IDS = set(CATEGORIES.keys())

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.hero{background:linear-gradient(135deg,#0d0f14,#131926,#0a1628);border:1px solid #1e2d45;
      border-radius:16px;padding:2rem 2.5rem 1.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-40px;right:-40px;width:200px;height:200px;
      background:radial-gradient(circle,rgba(50,150,255,.12),transparent 70%);border-radius:50%;}
.hero h1{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;color:#fff;margin:0 0 .3rem;letter-spacing:-1px;}
.hero h1 span{color:#3296ff;}
.hero p{color:#8899bb;font-size:1rem;margin:0;}
.counter-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;margin:.8rem 0;}
.counter-card{background:#131926;border:1px solid #1e2d45;border-radius:12px;padding:12px 8px;text-align:center;}
.counter-card.active{border-color:#3296ff;}
.counter-card .emoji{font-size:1.5rem;margin-bottom:3px;}
.counter-card .count{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;color:#3296ff;line-height:1;}
.counter-card .sub{font-size:.65rem;color:#8899bb;margin-top:1px;}
.total-badge{background:linear-gradient(90deg,#1a2840,#1e3050);border:1px solid #3296ff44;
      border-radius:12px;padding:1rem 1.5rem;display:flex;align-items:center;gap:1rem;margin-bottom:1rem;}
.total-badge .num{font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;color:#3296ff;line-height:1;}
.total-badge .lbl{color:#8899bb;font-size:.85rem;}
.line-info{background:#131926;border:1px solid #f59e0b44;border-radius:10px;padding:.8rem 1rem;
      color:#f59e0b;font-size:.85rem;margin:.5rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Vehicle<span>Eye</span> 🎯</h1>
  <p>Tracking ByteTrack · Ligne virtuelle · Export CSV — propulsé par YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# ── Modèle ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    return YOLO(name)

with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    model_choice = st.selectbox("Modèle YOLO",
        ["yolov8n.pt","yolov8s.pt","yolov8m.pt"],
        help="n=rapide (mobile), s=équilibré, m=précis")
    confidence = st.slider("Confiance min.", 0.2, 0.9, 0.4, 0.05)
    show_labels = st.toggle("Labels sur les boîtes", True)
    show_track  = st.toggle("Afficher l'ID de tracking", True)
    show_trail  = st.toggle("Trajectoires (traînées)", True)
    st.markdown("---")
    st.markdown("**Classes :**")
    for info in CATEGORIES.values():
        st.markdown(f"{info['emoji']} `{info['label']}`")

with st.spinner("Chargement du modèle…"):
    model = load_model(model_choice)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def cx_cy(x1,y1,x2,y2):
    return (x1+x2)//2, (y1+y2)//2

def crosses_line(prev_cy, curr_cy, line_y):
    """Détecte si un objet a franchi line_y entre deux frames."""
    if prev_cy is None:
        return False
    return (prev_cy < line_y <= curr_cy) or (prev_cy > line_y >= curr_cy)

def draw_annotations(pil_img, boxes_info, line_y=None, trails=None, show_lbl=True, show_tid=True):
    """Dessine boîtes, labels, ligne virtuelle et traînées sur une image PIL."""
    draw = ImageDraw.Draw(pil_img, "RGBA")
    W, H = pil_img.size

    # ── Ligne virtuelle ──
    if line_y is not None:
        ly = int(line_y * H)
        draw.line([(0, ly), (W, ly)], fill=(255, 80, 80, 220), width=3)
        draw.rectangle([0, ly-14, 120, ly+2], fill=(255,80,80,180))
        draw.text((4, ly-13), "LIGNE COMPTAGE", fill=(255,255,255))

    # ── Traînées ──
    if trails and show_trail:
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                alpha = int(200 * i / len(pts))
                draw.line([pts[i-1], pts[i]], fill=(100,200,255,alpha), width=2)

    # ── Boîtes & labels ──
    for (x1,y1,x2,y2,cid,tid,conf) in boxes_info:
        info  = CATEGORIES[cid]
        color = info["color"]
        draw.rectangle([x1,y1,x2,y2], outline=color+(200,), width=3)
        if show_lbl:
            parts = [info["label"]]
            if show_tid and tid is not None:
                parts.append(f"#{tid}")
            txt = " ".join(parts)
            tw  = draw.textlength(txt)
            draw.rectangle([x1, y1-18, x1+int(tw)+8, y1], fill=color+(220,))
            draw.text((x1+4, y1-17), txt, fill=(0,0,0))

    return pil_img

def render_counters(counts_frame, counts_cumul, counts_line):
    """Affiche les compteurs frame + cumulatif + franchissements."""
    total_f = sum(counts_frame.values())
    total_c = sum(counts_cumul.values())
    total_l = sum(counts_line.values())

    st.markdown(f"""
    <div class="total-badge">
      <div class="num">{total_c}</div>
      <div>
        <div style="color:#fff;font-weight:600;font-size:1rem">IDs uniques trackés</div>
        <div class="lbl">Frame actuelle : {total_f} · Ligne : {total_l} franchissements</div>
      </div>
    </div>""", unsafe_allow_html=True)

    cards = ""
    for cid, info in CATEGORIES.items():
        f = counts_frame.get(cid, 0)
        c = counts_cumul.get(cid, 0)
        l = counts_line.get(cid, 0)
        active = "active" if c > 0 else ""
        cards += f"""
        <div class="counter-card {active}">
          <div class="emoji">{info['emoji']}</div>
          <div class="count">{c}</div>
          <div class="sub">frame:{f} · ligne:{l}</div>
          <div class="sub" style="color:#aaa">{info['label']}</div>
        </div>"""
    st.markdown(f'<div class="counter-grid">{cards}</div>', unsafe_allow_html=True)

def make_csv(log_rows):
    """Génère le contenu CSV à partir des lignes de log."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["frame","timestamp","track_id","categorie","x1","y1","x2","y2",
                     "centre_x","centre_y","confiance","franchissement_ligne"])
    writer.writerows(log_rows)
    return buf.getvalue().encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE pour tracking multi-frames
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "cumul_ids":    defaultdict(set),   # cid -> set de track_ids vus
        "line_counts":  defaultdict(int),   # cid -> nb de franchissements
        "prev_cy":      {},                 # tid -> cy de la frame précédente
        "trails":       defaultdict(list),  # tid -> liste de points (cx,cy)
        "log_rows":     [],                 # lignes CSV
        "frame_idx":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_state():
    for k in ["cumul_ids","line_counts","prev_cy","trails","log_rows","frame_idx"]:
        if k in st.session_state:
            del st.session_state[k]
    init_state()

init_state()

# ─────────────────────────────────────────────────────────────────────────────
# MOTEUR DE TRAITEMENT D'UNE FRAME
# ─────────────────────────────────────────────────────────────────────────────
def process_frame(pil_img, conf_thresh, line_y_ratio=None):
    """
    Fait tourner YOLO + ByteTrack sur une image PIL.
    Met à jour le session_state (cumul, ligne, trails, log).
    Retourne (image annotée PIL, counts_frame, counts_cumul, counts_line).
    """
    st.session_state.frame_idx += 1
    fidx = st.session_state.frame_idx
    ts   = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    img_rgb = pil_img.convert("RGB")
    W, H    = img_rgb.size

    # YOLOv8 + ByteTrack (persist=True active le tracker entre frames)
    results = model.track(
        np.array(img_rgb),
        conf=conf_thresh,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )[0]

    counts_frame = defaultdict(int)
    boxes_info   = []

    for box in results.boxes:
        cid = int(box.cls[0])
        if cid not in TARGET_IDS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        tid  = int(box.id[0]) if box.id is not None else None
        cx, cy = cx_cy(x1,y1,x2,y2)

        counts_frame[cid] += 1

        # ── Tracking cumulatif ──
        if tid is not None:
            st.session_state.cumul_ids[cid].add(tid)

            # ── Traînée ──
            if show_trail:
                trail = st.session_state.trails[tid]
                trail.append((cx, cy))
                if len(trail) > 30:
                    trail.pop(0)

            # ── Franchissement ligne ──
            crossed = False
            if line_y_ratio is not None:
                line_y_px = int(line_y_ratio * H)
                prev = st.session_state.prev_cy.get(tid)
                if crosses_line(prev, cy, line_y_px):
                    st.session_state.line_counts[cid] += 1
                    crossed = True
            st.session_state.prev_cy[tid] = cy

        boxes_info.append((x1,y1,x2,y2,cid,tid,conf))

        # ── Log CSV ──
        st.session_state.log_rows.append([
            fidx, ts, tid,
            CATEGORIES[cid]["label"],
            x1, y1, x2, y2, cx, cy,
            f"{conf:.2f}",
            "OUI" if crossed else "non"
        ])

    counts_cumul = {cid: len(ids) for cid, ids in st.session_state.cumul_ids.items()}
    counts_line  = dict(st.session_state.line_counts)

    annotated = draw_annotations(
        img_rgb.copy(),
        boxes_info,
        line_y=line_y_ratio,
        trails=st.session_state.trails,
        show_lbl=show_labels,
        show_tid=show_track,
    )
    return annotated, dict(counts_frame), counts_cumul, counts_line

# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────────────────────────────────────
tab_photo, tab_video, tab_cam = st.tabs(["📷 Photo", "🎬 Vidéo", "📱 Caméra live"])

# ══ PHOTO ════════════════════════════════════════════════════════════════════
with tab_photo:
    st.markdown("##### Importer une image")

    col_up, col_line = st.columns([3,2])
    with col_up:
        upl = st.file_uploader("", type=["jpg","jpeg","png","webp"],
                               label_visibility="collapsed")
    with col_line:
        use_line_p = st.toggle("Activer la ligne virtuelle", False, key="lp")
        line_pos_p = st.slider("Position de la ligne (% hauteur)",
                               10, 90, 50, key="lpp",
                               disabled=not use_line_p) / 100

    if upl:
        reset_state()
        img = Image.open(upl)
        line_y = line_pos_p if use_line_p else None
        annotated, cf, cc, cl = process_frame(img, confidence, line_y)

        c1, c2 = st.columns([3,2])
        with c1:
            st.image(annotated, use_container_width=True)
        with c2:
            render_counters(cf, cc, cl)
            if st.session_state.log_rows:
                st.download_button("⬇️ Export CSV",
                    make_csv(st.session_state.log_rows),
                    file_name=f"vehicleeye_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv")
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=90)
        st.download_button("⬇️ Télécharger l'image annotée",
                           buf.getvalue(), "vehicleeye.jpg", "image/jpeg")

# ══ VIDÉO ════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown("##### Importer une vidéo")
    st.info("La vidéo est analysée frame par frame. Clip court recommandé (< 60 s).")

    upl_v = st.file_uploader("", type=["mp4","mov","avi","mkv"],
                              label_visibility="collapsed", key="vid")
    if upl_v:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            frame_skip = st.slider("1 frame sur :", 1, 10, 3)
        with col_b:
            max_frames = st.slider("Frames max :", 20, 300, 80)
        with col_c:
            use_line_v = st.toggle("Ligne virtuelle", True, key="lv")
            line_pos_v = st.slider("Position ligne (%)", 10, 90, 50, key="lpv",
                                   disabled=not use_line_v) / 100

        if use_line_v:
            st.markdown(
                f'<div class="line-info">🔴 Ligne à <b>{int(line_pos_v*100)}%</b> de la hauteur de l\'image. '
                'Tout véhicule la franchissant sera comptabilisé séparément.</div>',
                unsafe_allow_html=True)

        if st.button("▶️ Lancer l'analyse"):
            reset_state()
            try:
                import imageio.v3 as iio
            except ImportError:
                st.error("Dépendance manquante : `imageio[ffmpeg]` — vérifiez requirements.txt")
                st.stop()

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(upl_v.read()); tfile.flush()

            stframe  = st.empty()
            cnt_ph   = st.empty()
            prog     = st.progress(0)
            status   = st.empty()
            idx = analyzed = 0
            line_y = line_pos_v if use_line_v else None

            for frame_np in iio.imiter(tfile.name, plugin="pyav"):
                if analyzed >= max_frames:
                    break
                idx += 1
                if idx % frame_skip != 0:
                    continue

                pil_f = Image.fromarray(frame_np)
                annotated, cf, cc, cl = process_frame(pil_f, confidence, line_y)

                stframe.image(annotated, use_container_width=True)
                with cnt_ph.container():
                    render_counters(cf, cc, cl)

                analyzed += 1
                prog.progress(min(analyzed / max_frames, 1.0))
                status.caption(f"Frame {idx} · {analyzed}/{max_frames} analysées")

            os.unlink(tfile.name)
            st.success("✅ Analyse terminée !")

            st.markdown("### 📊 Résultats finaux")
            render_counters({}, dict(st.session_state.cumul_ids), dict(st.session_state.line_counts))

            if st.session_state.log_rows:
                st.download_button(
                    "⬇️ Télécharger le CSV complet",
                    make_csv(st.session_state.log_rows),
                    file_name=f"vehicleeye_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                )
                with st.expander("👁️ Aperçu du CSV (20 dernières lignes)"):
                    import pandas as pd
                    cols = ["frame","timestamp","track_id","categorie","x1","y1","x2","y2",
                            "centre_x","centre_y","confiance","franchissement_ligne"]
                    df = pd.DataFrame(st.session_state.log_rows[-20:], columns=cols)
                    st.dataframe(df, use_container_width=True)

# ══ CAMÉRA ════════════════════════════════════════════════════════════════════
with tab_cam:
    st.markdown("##### Prise de vue — caméra smartphone")
    st.info("📱 Sur smartphone, cette option active directement l'appareil photo.")

    use_line_c = st.toggle("Ligne virtuelle", True, key="lc")
    line_pos_c = st.slider("Position ligne (%)", 10, 90, 50, key="lpc",
                           disabled=not use_line_c) / 100

    cam_img = st.camera_input("Prendre une photo")
    if cam_img:
        reset_state()
        img = Image.open(cam_img)
        line_y = line_pos_c if use_line_c else None
        annotated, cf, cc, cl = process_frame(img, confidence, line_y)

        c1, c2 = st.columns([3,2])
        with c1:
            st.image(annotated, use_container_width=True)
        with c2:
            render_counters(cf, cc, cl)
            if st.session_state.log_rows:
                st.download_button("⬇️ Export CSV",
                    make_csv(st.session_state.log_rows),
                    file_name=f"vehicleeye_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv")

        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=90)
        st.download_button("⬇️ Télécharger l'image annotée",
                           buf.getvalue(), "vehicleeye_cam.jpg", "image/jpeg")

# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:.8rem;margin-top:2rem'>"
    "VehicleEye · YOLOv8 + ByteTrack (Ultralytics) · Pillow · Streamlit</div>",
    unsafe_allow_html=True)
