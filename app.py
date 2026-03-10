"""
VehicleEye — Détection DETR (HuggingFace), tracker IoU maison, ligne virtuelle, CSV
Zéro dépendance cv2 / ultralytics. Compatible Python 3.14+
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tempfile, io, csv, os
from datetime import datetime
from collections import defaultdict
import torch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VehicleEye", page_icon="🚗",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Classes COCO ciblées (identiques à YOLO) ──────────────────────────────────
CATEGORIES = {
    "person":     {"label": "Piéton",  "emoji": "🚶", "color": (255, 200,  50)},
    "bicycle":    {"label": "Vélo",    "emoji": "🚲", "color": ( 50, 220, 100)},
    "car":        {"label": "Voiture", "emoji": "🚗", "color": ( 50, 150, 255)},
    "motorcycle": {"label": "Moto",    "emoji": "🏍️", "color": (255, 100,  50)},
    "bus":        {"label": "Bus",     "emoji": "🚌", "color": (200,  50, 255)},
    "truck":      {"label": "Camion",  "emoji": "🚛", "color": (255,  50, 100)},
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.hero{background:linear-gradient(135deg,#0d0f14,#131926,#0a1628);border:1px solid #1e2d45;
      border-radius:16px;padding:2rem 2.5rem 1.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.hero h1{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:#fff;margin:0 0 .3rem;letter-spacing:-1px;}
.hero h1 span{color:#3296ff;}
.hero p{color:#8899bb;font-size:.95rem;margin:0;}
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
.line-info{background:#131926;border:1px solid #f59e0b44;border-radius:10px;
      padding:.8rem 1rem;color:#f59e0b;font-size:.85rem;margin:.5rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Vehicle<span>Eye</span> 🎯</h1>
  <p>Détection DETR · Tracking IoU · Ligne virtuelle · Export CSV — Python 3.14 ✅</p>
</div>
""", unsafe_allow_html=True)

# ── Chargement modèle DETR (HuggingFace, sans cv2) ───────────────────────────
@st.cache_resource
def load_model():
    from transformers import DetrImageProcessor, DetrForObjectDetection
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model     = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.eval()
    return processor, model

with st.spinner("⏳ Chargement du modèle DETR (HuggingFace)… ~1 min au premier démarrage"):
    processor, det_model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    confidence  = st.slider("Confiance min.", 0.3, 0.95, 0.6, 0.05)
    show_labels = st.toggle("Labels sur les boîtes", True)
    show_trail  = st.toggle("Trajectoires (traînées)", True)
    st.markdown("---")
    for info in CATEGORIES.values():
        st.markdown(f"{info['emoji']} `{info['label']}`")
    st.markdown("---")
    st.caption("Modèle : DETR ResNet-50 (COCO)\nTracking : IoU Centroid Tracker\nAucune dépendance cv2")

# ══════════════════════════════════════════════════════════════════════════════
# DÉTECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect(pil_img, conf_thresh):
    """Retourne liste de dicts {label, box:[x1,y1,x2,y2], score}."""
    inputs  = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = det_model(**inputs)
    W, H = pil_img.size
    results = processor.post_process_object_detection(
        outputs, threshold=conf_thresh,
        target_sizes=torch.tensor([[H, W]])
    )[0]
    detections = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = det_model.config.id2label[label_id.item()]
        if label not in CATEGORIES:
            continue
        x1,y1,x2,y2 = map(int, box.tolist())
        detections.append({"label": label, "box": [x1,y1,x2,y2], "score": float(score)})
    return detections

# ══════════════════════════════════════════════════════════════════════════════
# TRACKER IoU CENTROID (pure Python, sans cv2 ni ultralytics)
# ══════════════════════════════════════════════════════════════════════════════
class IoUTracker:
    """Tracker léger basé sur IoU entre boîtes consécutives."""
    def __init__(self, iou_thresh=0.3, max_lost=10):
        self.iou_thresh = iou_thresh
        self.max_lost   = max_lost
        self.tracks     = {}   # tid -> {box, label, lost, cy_prev}
        self.next_id    = 1

    @staticmethod
    def iou(a, b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        if inter == 0: return 0.0
        ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/ua if ua else 0.0

    def update(self, detections):
        """detections = liste de dicts {label, box, score}"""
        matched_tids = set()
        assigned     = [False]*len(detections)

        # Associer détections aux tracks existants par IoU max
        for tid, track in list(self.tracks.items()):
            best_iou, best_di = 0, -1
            for di, det in enumerate(detections):
                if assigned[di]: continue
                if det["label"] != track["label"]: continue
                s = self.iou(track["box"], det["box"])
                if s > best_iou:
                    best_iou, best_di = s, di
            if best_iou >= self.iou_thresh and best_di >= 0:
                self.tracks[tid]["box"]  = detections[best_di]["box"]
                self.tracks[tid]["lost"] = 0
                assigned[best_di] = True
                matched_tids.add(tid)
            else:
                self.tracks[tid]["lost"] += 1

        # Créer de nouveaux tracks pour les détections non assignées
        for di, det in enumerate(detections):
            if not assigned[di]:
                self.tracks[self.next_id] = {
                    "box": det["box"], "label": det["label"],
                    "lost": 0, "cy_prev": None
                }
                self.next_id += 1

        # Supprimer les tracks perdus
        for tid in [t for t,v in self.tracks.items() if v["lost"] > self.max_lost]:
            del self.tracks[tid]

        # Retourner les tracks actifs avec leur ID
        return [(tid, v["label"], v["box"]) for tid, v in self.tracks.items() if v["lost"] == 0]

# ══════════════════════════════════════════════════════════════════════════════
# DESSIN (Pillow uniquement)
# ══════════════════════════════════════════════════════════════════════════════
def draw_frame(pil_img, tracked, line_y_ratio, trails):
    img   = pil_img.convert("RGBA")
    over  = Image.new("RGBA", img.size, (0,0,0,0))
    draw  = ImageDraw.Draw(over)
    W, H  = img.size

    # Ligne virtuelle
    if line_y_ratio is not None:
        ly = int(line_y_ratio * H)
        draw.line([(0,ly),(W,ly)], fill=(255,80,80,220), width=3)
        draw.rectangle([0,ly-16,140,ly+2], fill=(255,80,80,180))
        draw.text((4, ly-15), "LIGNE COMPTAGE", fill=(255,255,255,255))

    # Traînées
    if show_trail:
        for tid, pts in trails.items():
            if len(pts) < 2: continue
            for i in range(1,len(pts)):
                alpha = int(200*i/len(pts))
                draw.line([pts[i-1],pts[i]], fill=(100,200,255,alpha), width=2)

    # Boîtes
    for (tid, label, box) in tracked:
        info  = CATEGORIES[label]
        color = info["color"]+(200,)
        x1,y1,x2,y2 = box
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        if show_labels:
            txt = f"{info['label']} #{tid}"
            tw  = draw.textlength(txt)
            draw.rectangle([x1,y1-18,x1+int(tw)+8,y1], fill=color)
            draw.text((x1+4,y1-17), txt, fill=(0,0,0,255))

    return Image.alpha_composite(img, over).convert("RGB")

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defs = {
        "tracker":      None,
        "cumul_ids":    defaultdict(set),
        "line_counts":  defaultdict(int),
        "trails":       defaultdict(list),
        "log_rows":     [],
        "frame_idx":    0,
    }
    for k,v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_state():
    for k in ["tracker","cumul_ids","line_counts","trails","log_rows","frame_idx"]:
        if k in st.session_state: del st.session_state[k]
    init_state()
    st.session_state.tracker = IoUTracker()

init_state()
if st.session_state.tracker is None:
    st.session_state.tracker = IoUTracker()

# ══════════════════════════════════════════════════════════════════════════════
# TRAITEMENT D'UNE FRAME
# ══════════════════════════════════════════════════════════════════════════════
def process_frame(pil_img, conf_thresh, line_y_ratio=None):
    st.session_state.frame_idx += 1
    fidx = st.session_state.frame_idx
    ts   = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    W, H = pil_img.size

    detections = detect(pil_img, conf_thresh)
    tracked    = st.session_state.tracker.update(detections)

    counts_frame = defaultdict(int)

    for (tid, label, box) in tracked:
        x1,y1,x2,y2 = box
        cx = (x1+x2)//2
        cy = (y1+y2)//2

        counts_frame[label] += 1
        st.session_state.cumul_ids[label].add(tid)

        # Traînée
        trail = st.session_state.trails[tid]
        trail.append((cx,cy))
        if len(trail) > 30: trail.pop(0)

        # Franchissement
        crossed = False
        if line_y_ratio is not None:
            line_px  = int(line_y_ratio * H)
            prev_cy  = st.session_state.tracker.tracks.get(tid,{}).get("cy_prev")
            if prev_cy is not None:
                if (prev_cy < line_px <= cy) or (prev_cy > line_px >= cy):
                    st.session_state.line_counts[label] += 1
                    crossed = True
            if tid in st.session_state.tracker.tracks:
                st.session_state.tracker.tracks[tid]["cy_prev"] = cy

        score = next((d["score"] for d in detections
                      if d["label"]==label and
                         abs((d["box"][0]+d["box"][2])//2-cx)<5), 0.0)

        st.session_state.log_rows.append([
            fidx, ts, tid,
            CATEGORIES[label]["label"],
            x1,y1,x2,y2, cx, cy,
            f"{score:.2f}",
            "OUI" if crossed else "non"
        ])

    annotated = draw_frame(pil_img, tracked, line_y_ratio, st.session_state.trails)
    counts_cumul = {lb: len(ids) for lb,ids in st.session_state.cumul_ids.items()}
    return annotated, dict(counts_frame), counts_cumul, dict(st.session_state.line_counts)

# ══════════════════════════════════════════════════════════════════════════════
# UI COMPTEURS
# ══════════════════════════════════════════════════════════════════════════════
def render_counters(counts_frame, counts_cumul, counts_line):
    total_c = sum(counts_cumul.values())
    total_l = sum(counts_line.values())
    total_f = sum(counts_frame.values())
    st.markdown(f"""
    <div class="total-badge">
      <div class="num">{total_c}</div>
      <div>
        <div style="color:#fff;font-weight:600;font-size:1rem">IDs uniques trackés</div>
        <div class="lbl">Frame : {total_f} · Franchissements ligne : {total_l}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    cards = "".join(
        f'<div class="counter-card {"active" if counts_cumul.get(lb,0)>0 else ""}">'
        f'<div class="emoji">{info["emoji"]}</div>'
        f'<div class="count">{counts_cumul.get(lb,0)}</div>'
        f'<div class="sub">frame:{counts_frame.get(lb,0)} ligne:{counts_line.get(lb,0)}</div>'
        f'<div class="sub" style="color:#aaa">{info["label"]}</div></div>'
        for lb, info in CATEGORIES.items()
    )
    st.markdown(f'<div class="counter-grid">{cards}</div>', unsafe_allow_html=True)

def make_csv(rows):
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["frame","timestamp","track_id","categorie",
                "x1","y1","x2","y2","centre_x","centre_y",
                "confiance","franchissement_ligne"])
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PRÉVISUALISATION LIGNE (sans détection)
# ══════════════════════════════════════════════════════════════════════════════
def draw_line_preview(pil_img, line_y_ratio):
    """Dessine uniquement la ligne virtuelle sur une image — sans lancer DETR."""
    img  = pil_img.convert("RGBA")
    over = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(over)
    W, H = img.size
    ly   = int(line_y_ratio * H)

    # Ligne principale + halo
    draw.line([(0, ly), (W, ly)], fill=(255, 80, 80, 60), width=10)
    draw.line([(0, ly), (W, ly)], fill=(255, 80, 80, 220), width=3)

    # Étiquette
    label = f"LIGNE — {int(line_y_ratio*100)}% de la hauteur"
    tw    = draw.textlength(label)
    lx    = int(W/2 - tw/2)
    draw.rectangle([lx-6, ly-20, lx+int(tw)+6, ly-2], fill=(255,80,80,200))
    draw.text((lx, ly-19), label, fill=(255,255,255,255))

    # Flèches latérales pour indiquer la ligne
    for x in [12, W-12]:
        draw.polygon([(x,ly-8),(x-7,ly-18),(x+7,ly-18)], fill=(255,80,80,220))
        draw.polygon([(x,ly+8),(x-7,ly+18),(x+7,ly+18)], fill=(255,80,80,220))

    return Image.alpha_composite(img, over).convert("RGB")

# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════════
tab_photo, tab_video, tab_cam = st.tabs(["📷 Photo", "🎬 Vidéo", "📱 Caméra live"])

# ── PHOTO ─────────────────────────────────────────────────────────────────────
with tab_photo:
    upl = st.file_uploader("Importer une image",
          type=["jpg","jpeg","png","webp"], label_visibility="visible")

    if upl:
        img = Image.open(upl).convert("RGB")

        # Contrôles ligne
        use_line_p = st.toggle("Ligne virtuelle", False, key="lp")
        line_pos_p = st.slider(
            "📍 Position de la ligne (glissez pour voir)",
            10, 90, 50, key="lpp", disabled=not use_line_p
        ) / 100

        # Prévisualisation live (ne relance PAS la détection)
        c1, c2 = st.columns([3, 2])
        with c1:
            if use_line_p:
                st.image(draw_line_preview(img, line_pos_p),
                         use_container_width=True,
                         caption=f"Prévisualisation — ligne à {int(line_pos_p*100)}%")
            else:
                st.image(img, use_container_width=True, caption="Image importée")

        with c2:
            st.markdown("##### Lancer l'analyse")
            st.caption("La ligne est positionnée ? Cliquez pour détecter.")
            if st.button("🔍 Analyser", key="run_photo"):
                reset_state()
                annotated, cf, cc, cl = process_frame(img, confidence,
                                                      line_pos_p if use_line_p else None)
                st.image(annotated, use_container_width=True, caption="Résultat")
                render_counters(cf, cc, cl)
                if st.session_state.log_rows:
                    st.download_button("⬇️ Export CSV",
                        make_csv(st.session_state.log_rows),
                        f"vehicleeye_{datetime.now():%Y%m%d_%H%M%S}.csv","text/csv")
                buf = io.BytesIO()
                annotated.save(buf, format="JPEG", quality=90)
                st.download_button("⬇️ Image annotée", buf.getvalue(),
                                   "vehicleeye.jpg","image/jpeg")

# ── VIDÉO ─────────────────────────────────────────────────────────────────────
with tab_video:
    st.info("⚠️ Vidéo analysée frame par frame. Recommandé : clips < 30 s, résolution 720p max.")
    upl_v = st.file_uploader("Importer une vidéo",
            type=["mp4","mov","avi","mkv"], label_visibility="visible", key="vid")

    if upl_v:
        # Extraire la 1ère frame pour la prévisualisation
        import av as _av, tempfile as _tf
        _tmp = _tf.NamedTemporaryFile(delete=False, suffix=".mp4")
        _tmp.write(upl_v.getvalue()); _tmp.flush()
        _container = _av.open(_tmp.name)
        _first_frame = None
        for _pkt in _container.demux(video=0):
            for _frm in _pkt.decode():
                _first_frame = _frm.to_image()
                break
            if _first_frame: break
        _container.close()

        ca,cb,cc_ = st.columns(3)
        with ca: frame_skip = st.slider("1 frame sur :", 1, 10, 4)
        with cb: max_frames = st.slider("Frames max :", 10, 150, 50)
        with cc_:
            use_line_v = st.toggle("Ligne virtuelle", True, key="lv")
            line_pos_v = st.slider(
                "📍 Position de la ligne (glissez pour voir)",
                10, 90, 50, key="lpv", disabled=not use_line_v
            ) / 100

        # Prévisualisation live sur la 1ère frame
        if _first_frame:
            if use_line_v:
                st.image(draw_line_preview(_first_frame, line_pos_v),
                         use_container_width=True,
                         caption=f"Prévisualisation 1ère frame — ligne à {int(line_pos_v*100)}%")
            else:
                st.image(_first_frame, use_container_width=True,
                         caption="1ère frame de la vidéo")

        if st.button("▶️ Lancer l'analyse"):
            reset_state()
            try:
                import av
            except ImportError:
                st.error("Dépendance manquante : av — vérifiez requirements.txt"); st.stop()

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(upl_v.read()); tfile.flush()

            stframe = st.empty(); cnt_ph = st.empty()
            prog = st.progress(0); status = st.empty()
            idx = analyzed = 0
            line_y = line_pos_v if use_line_v else None

            container = av.open(tfile.name)
            for packet in container.demux(video=0):
                for frame in packet.decode():
                    if analyzed >= max_frames: break
                    idx += 1
                    if idx % frame_skip != 0: continue
                    pil_f = frame.to_image()          # renvoie directement un PIL.Image
                    annotated, cf, cc2, cl = process_frame(pil_f, confidence, line_y)
                    stframe.image(annotated, use_container_width=True)
                    with cnt_ph.container(): render_counters(cf, cc2, cl)
                    analyzed += 1
                    prog.progress(min(analyzed/max_frames, 1.0))
                    status.caption(f"Frame {idx} · {analyzed}/{max_frames} analysées")
                if analyzed >= max_frames: break
            container.close()

            os.unlink(tfile.name)
            st.success("✅ Analyse terminée !")
            st.markdown("### 📊 Résultats finaux")
            render_counters(
                {},
                {lb: len(ids) for lb, ids in st.session_state.cumul_ids.items()},
                dict(st.session_state.line_counts)
            )

            if st.session_state.log_rows:
                st.download_button("⬇️ Télécharger le CSV",
                    make_csv(st.session_state.log_rows),
                    f"vehicleeye_{datetime.now():%Y%m%d_%H%M%S}.csv","text/csv")
                with st.expander("👁️ Aperçu CSV (20 dernières lignes)"):
                    import pandas as pd
                    cols = ["frame","timestamp","track_id","categorie",
                            "x1","y1","x2","y2","cx","cy","conf","ligne"]
                    st.dataframe(pd.DataFrame(
                        st.session_state.log_rows[-20:], columns=cols),
                        use_container_width=True)

# ── CAMÉRA ────────────────────────────────────────────────────────────────────
with tab_cam:
    st.info("📱 Sur smartphone, cette option active directement l'appareil photo.")

    use_line_c = st.toggle("Ligne virtuelle", True, key="lc")
    line_pos_c = st.slider(
        "📍 Position de la ligne (glissez pour voir)",
        10, 90, 50, key="lpc", disabled=not use_line_c
    ) / 100

    cam_img = st.camera_input("Prendre une photo")

    if cam_img:
        img = Image.open(cam_img).convert("RGB")

        # Prévisualisation live
        if use_line_c:
            st.image(draw_line_preview(img, line_pos_c),
                     use_container_width=True,
                     caption=f"Prévisualisation — ligne à {int(line_pos_c*100)}%")

        if st.button("🔍 Analyser", key="run_cam"):
            reset_state()
            annotated, cf, cc, cl = process_frame(img, confidence,
                                                  line_pos_c if use_line_c else None)
            c1,c2 = st.columns([3,2])
            with c1: st.image(annotated, use_container_width=True)
            with c2:
                render_counters(cf, cc, cl)
                if st.session_state.log_rows:
                    st.download_button("⬇️ Export CSV",
                        make_csv(st.session_state.log_rows),
                        f"vehicleeye_{datetime.now():%Y%m%d_%H%M%S}.csv","text/csv")
            buf = io.BytesIO()
            annotated.save(buf,format="JPEG",quality=90)
            st.download_button("⬇️ Image annotée", buf.getvalue(),
                               "vehicleeye_cam.jpg","image/jpeg")

st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:.8rem;margin-top:2rem'>"
    "VehicleEye · DETR ResNet-50 (HuggingFace) · Tracker IoU · Pillow · Streamlit"
    "</div>", unsafe_allow_html=True)
