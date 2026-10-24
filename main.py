"""
WasteWise AI - From Waste to Resource
SOCIOTHON 2026 MVP

Single-file Streamlit app:
  AI scanner -> disposal guidance (India 4-stream) -> pickup request ->
  collector workflow -> evidence-backed recovery -> impact analytics.

Run:  streamlit run main.py
"""

import hashlib
import io
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "wastewise.db"
MODEL_PATH = APP_DIR / "models" / "efficientnet_lite0.onnx"
LABELS_PATH = APP_DIR / "models" / "imagenet_labels.txt"
EVIDENCE_DIR = APP_DIR / "evidence_uploads"

STATUSES = ["Reported", "Accepted", "Collected", "Recovery Confirmed"]
PENDING_STATUSES = ("Reported", "Accepted")

# Thresholds on the aggregated material score (sum of ImageNet class probabilities
# that map to one material). Below LOW we do not offer an AI suggestion at all.
CONF_HIGH = 0.45
CONF_LOW = 0.20

STREAM_WET = "Wet waste"
STREAM_DRY = "Dry waste"
STREAM_SANITARY = "Sanitary waste"
STREAM_SPECIAL = "Special-care waste"
STREAM_REVIEW = "Needs human review"

STREAM_STYLE = {
    STREAM_WET: {"color": "#16a34a", "bin": "Green bin", "icon": "🟢"},
    STREAM_DRY: {"color": "#2563eb", "bin": "Blue bin", "icon": "🔵"},
    STREAM_SANITARY: {"color": "#dc2626", "bin": "Red / sealed bag", "icon": "🔴"},
    STREAM_SPECIAL: {"color": "#f59e0b", "bin": "Black / special-care box", "icon": "🟠"},
    STREAM_REVIEW: {"color": "#6b7280", "bin": "Hold for review", "icon": "⚪"},
}

STATUS_STYLE = {
    "Reported": "#64748b",
    "Accepted": "#2563eb",
    "Collected": "#f59e0b",
    "Recovery Confirmed": "#16a34a",
}

# Material -> default stream + handling guidance (context can override the stream).
MATERIALS = {
    "Plastic": {
        "stream": STREAM_DRY,
        "steps": [
            "Empty and rinse bottles/containers; squeeze to save space.",
            "Keep caps on; remove food residue so it stays recyclable.",
            "Multi-layer packets (chips, sachets) are low-value: keep separate from PET/HDPE.",
        ],
        "recovery": "Recycler / Material Recovery Facility (MRF)",
    },
    "Paper": {
        "stream": STREAM_DRY,
        "steps": [
            "Keep dry and clean; flatten or bundle sheets.",
            "Tissue paper and food-soiled paper are NOT recyclable (see context checks).",
        ],
        "recovery": "Paper recycler / kabadiwala",
    },
    "Cardboard": {
        "stream": STREAM_DRY,
        "steps": [
            "Flatten boxes; remove tape and plastic film where possible.",
            "Greasy pizza-box parts go to wet waste; clean parts stay dry.",
        ],
        "recovery": "Paper/board recycler",
    },
    "Metal": {
        "stream": STREAM_DRY,
        "steps": [
            "Rinse cans and tins; crush cans if possible.",
            "Pressurised aerosol cans are special-care waste, not regular metal.",
        ],
        "recovery": "Scrap metal dealer / MRF",
    },
    "Glass": {
        "stream": STREAM_DRY,
        "steps": [
            "Rinse bottles and jars; remove lids (metal/plastic go separately).",
            "Broken glass: wrap in newspaper, label 'BROKEN GLASS', hand over separately.",
        ],
        "recovery": "Glass recycler / bottle collection",
    },
    "Textile": {
        "stream": STREAM_DRY,
        "steps": [
            "Clean, wearable items: donate or reuse first.",
            "Worn-out textiles: keep dry and bag separately for textile recyclers.",
        ],
        "recovery": "Donation drive / textile recycler",
    },
    "Organic / food": {
        "stream": STREAM_WET,
        "steps": [
            "Put food scraps, peels and garden waste in the green (wet) bin.",
            "No plastic bags inside; drain excess liquid.",
            "Ideal for campus composting or biogas.",
        ],
        "recovery": "Campus composter / biogas plant",
    },
    "Sanitary": {
        "stream": STREAM_SANITARY,
        "steps": [
            "Wrap diapers, sanitary pads, used tissues and bandages in paper.",
            "Mark the packet with a red dot / 'SANITARY' label.",
            "Never mix with wet or dry waste; hand over separately.",
        ],
        "recovery": "Safe disposal (incinerator / authorised facility) - not recycled",
    },
    "E-waste": {
        "stream": STREAM_SPECIAL,
        "steps": [
            "Do not throw in any regular bin; do not break open.",
            "Remove batteries if easily removable and keep them together.",
            "Hand over to an authorised e-waste collector / take-back point.",
        ],
        "recovery": "Authorised e-waste recycler (E-Waste Rules)",
    },
    "Hazardous (medicines, chemicals, aerosols, sharps)": {
        "stream": STREAM_SPECIAL,
        "steps": [
            "Keep in original container; never pour chemicals down the drain.",
            "Sharps (needles/blades): put in a rigid closed container.",
            "Expired medicines: return to a pharmacy / special-care collection.",
        ],
        "recovery": "Authorised hazardous / biomedical handler",
    },
    "Mixed / unsure": {
        "stream": STREAM_REVIEW,
        "steps": [
            "Do not guess: keep the item aside.",
            "Ask a collector or sanitation staff to inspect it.",
        ],
        "recovery": "Decided after inspection",
    },
}
MATERIAL_NAMES = list(MATERIALS.keys())
MATERIAL_COLORS = {
    "Plastic": "#2563eb", "Paper": "#38bdf8", "Cardboard": "#a16207", "Metal": "#64748b",
    "Glass": "#06b6d4", "Textile": "#a855f7", "Organic / food": "#16a34a", "Sanitary": "#dc2626",
    "E-waste": "#f59e0b", "Hazardous (medicines, chemicals, aerosols, sharps)": "#ea580c",
    "Mixed / unsure": "#9ca3af",
}

# ImageNet label -> material. Only clear, defensible mappings are listed; everything
# else counts as "unmapped" and lowers the material score (we never force a guess).
IMAGENET_TO_MATERIAL = {
    # Plastic
    "water bottle": "Plastic", "pop bottle": "Plastic", "plastic bag": "Plastic",
    "water jug": "Plastic", "soap dispenser": "Plastic", "lotion": "Plastic",
    "sunscreen": "Plastic", "bottlecap": "Plastic", "packet": "Plastic",
    "nipple": "Plastic", "pencil box": "Plastic", "bucket": "Plastic",
    "crate": "Plastic", "tray": "Plastic", "shopping basket": "Plastic",
    "plate rack": "Plastic",
    # Glass
    "beer bottle": "Glass", "wine bottle": "Glass", "beer glass": "Glass",
    "red wine": "Glass", "whiskey jug": "Glass", "goblet": "Glass", "vase": "Glass",
    "perfume": "Glass",
    # Metal
    "milk can": "Metal", "frying pan": "Metal", "wok": "Metal", "caldron": "Metal",
    "pot": "Metal", "nail": "Metal", "screw": "Metal", "chain": "Metal",
    "padlock": "Metal", "coffeepot": "Metal", "teapot": "Metal", "can opener": "Metal",
    "mailbox": "Metal", "hammer": "Metal", "screwdriver": "Metal", "safety pin": "Metal",
    # Paper
    "envelope": "Paper", "menu": "Paper", "comic book": "Paper", "book jacket": "Paper",
    "crossword puzzle": "Paper", "binder": "Paper",
    # Cardboard
    "carton": "Cardboard",
    # Organic / food
    "banana": "Organic / food", "orange": "Organic / food", "lemon": "Organic / food",
    "fig": "Organic / food", "pineapple": "Organic / food", "strawberry": "Organic / food",
    "pomegranate": "Organic / food", "custard apple": "Organic / food",
    "jackfruit": "Organic / food", "Granny Smith": "Organic / food",
    "head cabbage": "Organic / food", "broccoli": "Organic / food",
    "cauliflower": "Organic / food", "cucumber": "Organic / food",
    "zucchini": "Organic / food", "acorn squash": "Organic / food",
    "butternut squash": "Organic / food", "spaghetti squash": "Organic / food",
    "bell pepper": "Organic / food", "artichoke": "Organic / food",
    "cardoon": "Organic / food", "mushroom": "Organic / food", "corn": "Organic / food",
    "ear": "Organic / food", "acorn": "Organic / food", "hip": "Organic / food",
    "French loaf": "Organic / food", "bagel": "Organic / food",
    "pretzel": "Organic / food", "cheeseburger": "Organic / food",
    "hotdog": "Organic / food", "mashed potato": "Organic / food",
    "meat loaf": "Organic / food", "pizza": "Organic / food", "potpie": "Organic / food",
    "burrito": "Organic / food", "carbonara": "Organic / food",
    "trifle": "Organic / food", "ice cream": "Organic / food",
    "ice lolly": "Organic / food", "guacamole": "Organic / food",
    "consomme": "Organic / food", "hot pot": "Organic / food", "dough": "Organic / food",
    "plate": "Organic / food",
    # Textile
    "jean": "Textile", "sock": "Textile", "sweatshirt": "Textile", "jersey": "Textile",
    "cardigan": "Textile", "running shoe": "Textile", "sandal": "Textile",
    "Loafer": "Textile", "clog": "Textile", "pillow": "Textile", "wool": "Textile", "miniskirt": "Textile",
    "trench coat": "Textile", "bath towel": "Textile",
    # Sanitary
    "diaper": "Sanitary", "Band Aid": "Sanitary", "toilet tissue": "Sanitary",
    "paper towel": "Sanitary",
    # E-waste
    "cellular telephone": "E-waste", "laptop": "E-waste", "notebook": "E-waste",
    "desktop computer": "E-waste", "computer keyboard": "E-waste", "mouse": "E-waste",
    "remote control": "E-waste", "iPod": "E-waste", "hard disc": "E-waste",
    "modem": "E-waste", "monitor": "E-waste", "screen": "E-waste",
    "television": "E-waste", "hand-held computer": "E-waste", "printer": "E-waste",
    "loudspeaker": "E-waste", "cassette player": "E-waste", "cassette": "E-waste",
    "electric fan": "E-waste", "toaster": "E-waste", "typewriter keyboard": "E-waste",
    "dial telephone": "E-waste", "microphone": "E-waste", "digital watch": "E-waste",
    "joystick": "E-waste", "CD player": "E-waste", "tape player": "E-waste",
    "radio": "E-waste", "power drill": "E-waste", "hand blower": "E-waste",
    "iron": "E-waste", "switch": "E-waste", "digital clock": "E-waste",
    # Hazardous / special care
    "syringe": "Hazardous (medicines, chemicals, aerosols, sharps)",
    "pill bottle": "Hazardous (medicines, chemicals, aerosols, sharps)",
    "hair spray": "Hazardous (medicines, chemicals, aerosols, sharps)",
    "lighter": "Hazardous (medicines, chemicals, aerosols, sharps)",
    "matchstick": "Hazardous (medicines, chemicals, aerosols, sharps)",
}

# Item-specific notes shown when the model's top ImageNet class is one of these.
ITEM_NOTES = {
    "paper towel": "Tissue/paper towels are not recyclable. Hygiene use → Sanitary; food-soiled → Wet.",
    "toilet tissue": "Used tissue is sanitary waste. Clean, unused rolls are not waste.",
    "plate": "The model sees a plate (often with food). Food scraps → Wet; the plate itself is not waste unless disposable.",
    "packet": "Chips/snack packets are multi-layer plastic (MLP): dry waste, but low recycling value.",
    "perfume": "Perfume bottles may contain flammable residue: empty fully before treating as glass.",
    "bottlecap": "Bottle caps can be plastic or metal: keep them with their bottle for recycling.",
}

CONTEXT_FLAGS = {
    "food": "Contaminated with food or liquid",
    "sanitary": "Used for hygiene / medical purposes (diaper, pad, bandage, tissue)",
    "broken": "Broken, sharp or has jagged edges",
    "electronic": "Contains a battery, circuit board or electrical parts",
    "chemical": "Held medicines, chemicals, paint, pesticides or is pressurised (aerosol)",
}

CAMPUS_LOCATIONS = [
    "Boys Hostel A", "Boys Hostel B", "Girls Hostel 1", "Girls Hostel 2",
    "Central Canteen", "Academic Block 1", "Academic Block 2", "Central Library",
    "Admin Building", "Sports Complex", "Labs Block", "Staff Quarters",
]

RECOVERY_FACILITIES = [
    "Campus Material Recovery Facility (MRF)",
    "Authorised scrap dealer / kabadiwala",
    "Campus composting unit",
    "Campus biogas plant",
    "Authorised e-waste recycler",
    "Authorised hazardous / biomedical handler",
    "Municipal sanitary waste facility",
]


# --------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pickups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                location TEXT NOT NULL,
                material TEXT NOT NULL,
                stream TEXT NOT NULL,
                est_weight_kg REAL NOT NULL CHECK (est_weight_kg > 0),
                notes TEXT,
                requester TEXT,
                source TEXT NOT NULL DEFAULT 'manual',
                ai_confidence REAL,
                needs_review INTEGER NOT NULL DEFAULT 0,
                review_reason TEXT,
                status TEXT NOT NULL DEFAULT 'Reported',
                accepted_at TEXT,
                collected_at TEXT,
                collected_weight_kg REAL,
                recovered_at TEXT,
                recovered_weight_kg REAL,
                recovery_facility TEXT,
                evidence_type TEXT,
                evidence_ref TEXT,
                evidence_file TEXT,
                is_demo INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS status_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pickup_id INTEGER NOT NULL REFERENCES pickups(id) ON DELETE CASCADE,
                at TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                note TEXT
            );
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                method TEXT NOT NULL,
                ai_material TEXT,
                ai_confidence REAL,
                top_imagenet_label TEXT,
                final_material TEXT NOT NULL,
                stream TEXT NOT NULL,
                needs_review INTEGER NOT NULL DEFAULT 0,
                is_demo INTEGER NOT NULL DEFAULT 0
            );
            """
        )


def now_iso():
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def log_status(conn, pickup_id, from_status, to_status, note="", at=None):
    conn.execute(
        "INSERT INTO status_log (pickup_id, at, from_status, to_status, note) VALUES (?,?,?,?,?)",
        (pickup_id, at or now_iso(), from_status, to_status, note),
    )


def create_pickup(location, material, stream, weight, notes="", requester="",
                  source="manual", ai_confidence=None, needs_review=False,
                  review_reason="", is_demo=False, created_at=None):
    created_at = created_at or now_iso()
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO pickups (created_at, location, material, stream, est_weight_kg, notes,
                   requester, source, ai_confidence, needs_review, review_reason, status, is_demo)
               VALUES (?,?,?,?,?,?,?,?,?,?,?, 'Reported', ?)""",
            (created_at, location, material, stream, float(weight), notes, requester, source,
             ai_confidence, int(needs_review), review_reason, int(is_demo)),
        )
        pid = cur.lastrowid
        log_status(conn, pid, None, "Reported", "Request submitted", at=created_at)
    return pid


def get_pickup(pid):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM pickups WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None


def accept_pickup(pid, at=None):
    p = get_pickup(pid)
    if not p or p["status"] != "Reported":
        return False, "Only 'Reported' requests can be accepted."
    at = at or now_iso()
    with get_conn() as conn:
        conn.execute("UPDATE pickups SET status='Accepted', accepted_at=? WHERE id=?", (at, pid))
        log_status(conn, pid, "Reported", "Accepted", "Collector accepted", at=at)
    return True, f"Request #{pid} accepted."


def collect_pickup(pid, weight, at=None):
    p = get_pickup(pid)
    if not p or p["status"] != "Accepted":
        return False, "Only 'Accepted' requests can be marked collected."
    if weight is None or weight <= 0:
        return False, "Collected weight must be greater than 0 kg."
    at = at or now_iso()
    with get_conn() as conn:
        conn.execute(
            "UPDATE pickups SET status='Collected', collected_at=?, collected_weight_kg=? WHERE id=?",
            (at, float(weight), pid),
        )
        log_status(conn, pid, "Accepted", "Collected", f"Collected {weight:.2f} kg", at=at)
    return True, f"Request #{pid} marked collected ({weight:.2f} kg)."


def confirm_recovery(pid, recovered_weight, facility, evidence_type, evidence_ref="",
                     evidence_file="", at=None):
    """evidence_type: 'document' (file uploaded), 'reference' (receipt/manifest no.),
    'simulated' (explicit demo confirmation)."""
    p = get_pickup(pid)
    if not p or p["status"] != "Collected":
        return False, "Only 'Collected' requests can be confirmed."
    if evidence_type not in ("document", "reference", "simulated"):
        return False, "Recovery needs evidence (file or reference) or an explicit simulated confirmation."
    if evidence_type == "reference" and not evidence_ref.strip():
        return False, "Enter a receipt / manifest reference number."
    if evidence_type == "document" and not evidence_file:
        return False, "Upload an evidence file."
    if not facility.strip():
        return False, "Select or enter the recovery facility."
    collected = p["collected_weight_kg"] or 0.0
    if p["stream"] == STREAM_SANITARY:
        recovered_weight = 0.0  # safely disposed, never counted as recovered material
    if recovered_weight is None or recovered_weight < 0:
        return False, "Recovered weight cannot be negative."
    if recovered_weight > collected + 1e-9:
        return False, f"Recovered weight cannot exceed collected weight ({collected:.2f} kg)."
    at = at or now_iso()
    with get_conn() as conn:
        conn.execute(
            """UPDATE pickups SET status='Recovery Confirmed', recovered_at=?, recovered_weight_kg=?,
                   recovery_facility=?, evidence_type=?, evidence_ref=?, evidence_file=? WHERE id=?""",
            (at, float(recovered_weight), facility, evidence_type, evidence_ref, evidence_file, pid),
        )
        note = f"{recovered_weight:.2f} kg via {facility} [{evidence_type}]"
        log_status(conn, pid, "Collected", "Recovery Confirmed", note, at=at)
    return True, f"Recovery confirmed for request #{pid}."


def log_scan(method, ai_material, ai_conf, top_label, final_material, stream, needs_review,
             is_demo=False, created_at=None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO scans (created_at, method, ai_material, ai_confidence, top_imagenet_label,
                   final_material, stream, needs_review, is_demo) VALUES (?,?,?,?,?,?,?,?,?)""",
            (created_at or now_iso(), method, ai_material, ai_conf, top_label, final_material,
             stream, int(needs_review), int(is_demo)),
        )


def load_pickups(include_demo=True):
    q = "SELECT * FROM pickups" + ("" if include_demo else " WHERE is_demo=0") + " ORDER BY id DESC"
    with get_conn() as conn:
        return pd.read_sql_query(q, conn)


def load_scans(include_demo=True):
    q = "SELECT * FROM scans" + ("" if include_demo else " WHERE is_demo=0") + " ORDER BY id DESC"
    with get_conn() as conn:
        return pd.read_sql_query(q, conn)


def load_log(limit=200):
    with get_conn() as conn:
        return pd.read_sql_query(
            """SELECT l.at, l.pickup_id, l.from_status, l.to_status, l.note, p.location, p.is_demo
               FROM status_log l JOIN pickups p ON p.id = l.pickup_id
               ORDER BY l.id DESC LIMIT ?""",
            conn, params=(limit,),
        )


def demo_counts():
    with get_conn() as conn:
        p = conn.execute("SELECT COUNT(*) FROM pickups WHERE is_demo=1").fetchone()[0]
        s = conn.execute("SELECT COUNT(*) FROM scans WHERE is_demo=1").fetchone()[0]
    return p, s


def clear_demo_data():
    with get_conn() as conn:
        conn.execute("DELETE FROM status_log WHERE pickup_id IN (SELECT id FROM pickups WHERE is_demo=1)")
        conn.execute("DELETE FROM pickups WHERE is_demo=1")
        conn.execute("DELETE FROM scans WHERE is_demo=1")


def seed_demo_data(n=28, seed=2026):
    """Insert clearly-flagged DEMO records spread over the last 14 days.
    Recoveries in demo data use the 'simulated' evidence type."""
    rng = random.Random(seed)
    weights_by_material = {
        "Plastic": (0.5, 6), "Paper": (1, 8), "Cardboard": (1, 10), "Metal": (0.3, 4),
        "Glass": (0.5, 5), "Organic / food": (3, 25), "Textile": (1, 5),
        "Sanitary": (0.5, 3), "E-waste": (0.5, 6),
        "Hazardous (medicines, chemicals, aerosols, sharps)": (0.2, 1.5),
    }
    facility_for_stream = {
        STREAM_DRY: "Campus Material Recovery Facility (MRF)",
        STREAM_WET: "Campus composting unit",
        STREAM_SPECIAL: "Authorised e-waste recycler",
        STREAM_SANITARY: "Municipal sanitary waste facility",
    }
    materials = list(weights_by_material.keys())
    mat_weights = [16, 10, 10, 5, 5, 20, 4, 4, 5, 2]
    base = datetime.now().replace(microsecond=0) - timedelta(days=14)
    for _ in range(n):
        mat = rng.choices(materials, weights=mat_weights)[0]
        stream = MATERIALS[mat]["stream"]
        lo, hi = weights_by_material[mat]
        est = round(rng.uniform(lo, hi), 1)
        created = (base + timedelta(days=rng.uniform(0, 13.5), hours=rng.uniform(0, 8))).replace(microsecond=0)
        review = stream == STREAM_SPECIAL
        pid = create_pickup(
            rng.choice(CAMPUS_LOCATIONS), mat, stream, est, notes="DEMO DATA",
            requester="Demo user", source=rng.choice(["ai", "manual"]),
            needs_review=review, review_reason="Special-care item" if review else "",
            is_demo=True, created_at=created.isoformat(sep=" "),
        )
        # Progress through the workflow with realistic probabilities.
        r = rng.random()
        t = created
        if r < 0.85:
            t = (t + timedelta(hours=rng.uniform(1, 12))).replace(microsecond=0)
            accept_pickup(pid, at=t.isoformat(sep=" "))
        if r < 0.70:
            t = (t + timedelta(hours=rng.uniform(2, 24))).replace(microsecond=0)
            cw = round(est * rng.uniform(0.8, 1.15), 2)
            collect_pickup(pid, cw, at=t.isoformat(sep=" "))
            if r < 0.50:
                t = (t + timedelta(hours=rng.uniform(12, 48))).replace(microsecond=0)
                rw = round(cw * rng.uniform(0.75, 0.97), 2)
                confirm_recovery(
                    pid, rw, facility_for_stream[stream], "simulated",
                    evidence_ref="DEMO-SIMULATED", at=min(t, datetime.now()).isoformat(sep=" "),
                )
    for _ in range(12):
        mat = rng.choices(materials, weights=mat_weights)[0]
        log_scan(rng.choice(["AI (confirmed)", "AI (corrected)", "Manual"]), mat,
                 round(rng.uniform(0.45, 0.95), 2), None, mat, MATERIALS[mat]["stream"],
                 MATERIALS[mat]["stream"] == STREAM_SPECIAL, is_demo=True,
                 created_at=(base + timedelta(days=rng.uniform(0, 14))).replace(microsecond=0).isoformat(sep=" "))


# --------------------------------------------------------------------------------------
# AI model (ONNX Runtime, EfficientNet-Lite0 pretrained on ImageNet)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Returns (session, labels, error_message). Never raises."""
    try:
        import onnxruntime as ort
    except Exception as e:  # pragma: no cover - depends on local install
        return None, None, f"onnxruntime is not installed ({e})."
    if not MODEL_PATH.exists():
        return None, None, f"Model file not found at {MODEL_PATH.name}."
    if not LABELS_PATH.exists():
        return None, None, f"Label file not found at {LABELS_PATH.name}."
    try:
        sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        labels = [l.strip() for l in LABELS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        out_dim = sess.get_outputs()[0].shape[-1]
        if out_dim != len(labels):
            return None, None, f"Model outputs {out_dim} classes but {len(labels)} labels were found."
        return sess, labels, None
    except Exception as e:
        return None, None, f"Could not load model: {e}"


def preprocess(img: Image.Image, size: int, layout: str) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB")
    # Centre crop to square, then resize (matches standard ImageNet eval preprocessing).
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((size, size), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5  # EfficientNet-Lite: [-1, 1]
    if layout == "NCHW":
        arr = arr.transpose(2, 0, 1)
    return arr[None, ...]


def classify_image(image_bytes: bytes):
    """Returns dict with real model output, or {'error': ...}."""
    sess, labels, err = load_model()
    if err:
        return {"error": err}
    try:
        img = Image.open(io.BytesIO(image_bytes))
        inp = sess.get_inputs()[0]
        shape = inp.shape  # e.g. [1, 224, 224, 3] or [1, 3, 224, 224]
        layout = "NCHW" if shape[1] == 3 else "NHWC"
        size = shape[2] if isinstance(shape[2], int) else 224
        x = preprocess(img, size, layout)
        out = sess.run(None, {inp.name: x})[0][0].astype(np.float64)
        if out.min() < 0 or abs(out.sum() - 1.0) > 1e-2:  # logits -> softmax
            e = np.exp(out - out.max())
            out = e / e.sum()
    except UnidentifiedImageError:
        return {"error": "The file is not a readable image."}
    except Exception as e:
        return {"error": f"Inference failed: {e}"}

    top_idx = np.argsort(out)[::-1][:5]
    top5 = [(labels[i], float(out[i]), IMAGENET_TO_MATERIAL.get(labels[i])) for i in top_idx]
    scores = {}
    for i, p in enumerate(out):
        m = IMAGENET_TO_MATERIAL.get(labels[i])
        if m:
            scores[m] = scores.get(m, 0.0) + float(p)
    unmapped = max(0.0, 1.0 - sum(scores.values()))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_mat, best_score = (ranked[0] if ranked else (None, 0.0))
    if best_score >= CONF_HIGH:
        level = "high"
    elif best_score >= CONF_LOW:
        level = "low"
    else:
        level = "none"
    return {
        "top5": top5, "scores": ranked, "unmapped": unmapped,
        "material": best_mat, "confidence": best_score, "level": level,
        "top_label": top5[0][0],
    }


@st.cache_data(show_spinner=False, max_entries=64)
def classify_cached(image_hash: str, image_bytes: bytes):
    return classify_image(image_bytes)


# --------------------------------------------------------------------------------------
# Guidance engine (material + context -> stream)
# --------------------------------------------------------------------------------------
def compute_guidance(material: str, ctx: dict, ai_level: str = None):
    info = MATERIALS[material]
    stream = info["stream"]
    steps = list(info["steps"])
    review_reasons = []
    overrides = []

    if ctx.get("chemical"):
        stream = STREAM_SPECIAL
        overrides.append("Held chemicals/medicines or is pressurised → special-care waste.")
        review_reasons.append("Potentially hazardous contents")
    if ctx.get("electronic") and material not in ("E-waste",):
        stream = STREAM_SPECIAL
        overrides.append("Contains battery/electronics → special-care (e-waste) stream.")
        review_reasons.append("Battery / electronics present")
    if ctx.get("sanitary") and stream != STREAM_SPECIAL:
        stream = STREAM_SANITARY
        overrides.append("Used for hygiene/medical purposes → sanitary waste, regardless of material.")
        steps = list(MATERIALS["Sanitary"]["steps"])
    if ctx.get("broken"):
        steps.insert(0, "Wrap sharp/broken pieces in thick paper and label 'SHARP' before handover.")
        review_reasons.append("Sharp / broken item - handle with gloves")
    if ctx.get("food") and stream == STREAM_DRY:
        if material in ("Paper", "Cardboard"):
            stream = STREAM_WET
            overrides.append("Food-soiled paper/cardboard cannot be recycled → wet waste (compostable).")
        else:
            steps.insert(0, "Rinse off food/liquid first. If it cannot be cleaned, it is a dry reject (low value).")

    if stream == STREAM_SPECIAL:
        review_reasons.append("Special-care stream requires trained handling")
    if material == "Mixed / unsure":
        review_reasons.append("Material not identified")
    if ai_level == "low":
        review_reasons.append("Low AI confidence - confirm the material")

    collection = {
        STREAM_DRY: (True, "Yes - schedule a dry-waste pickup (bundle several items to save trips)."),
        STREAM_WET: (True, "Yes for bulk (canteen/hostel mess). Single items: use the nearest green bin."),
        STREAM_SANITARY: (True, "Yes - separate sealed pickup only, never mixed with recyclables."),
        STREAM_SPECIAL: (True, "Yes - special-care pickup; collector must review before handling."),
        STREAM_REVIEW: (False, "Not yet - get the item inspected first, then request the right stream."),
    }[stream]
    recovery = MATERIALS["Sanitary"]["recovery"] if stream == STREAM_SANITARY else info["recovery"]
    return {
        "material": material, "stream": stream, "steps": steps, "overrides": overrides,
        "review_reasons": list(dict.fromkeys(review_reasons)),
        "collect": collection[0], "collect_text": collection[1], "recovery": recovery,
    }


# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------
CSS = """
<style>
:root { --ww-green:#16a34a; --ww-green-d:#15803d; --ww-dark:#0f172a; --ww-muted:#64748b; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#052e16 0%,#0f172a 100%); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton button { background:#16a34a22; border:1px solid #16a34a88; }
.ww-hero { background: linear-gradient(120deg,#052e16 0%,#14532d 55%,#16a34a 100%);
  color:#fff; padding: 28px 32px; border-radius: 18px; margin-bottom: 18px; }
.ww-hero h1 { color:#fff; margin:0 0 6px 0; font-size: 2.1rem; }
.ww-hero p { color:#dcfce7; margin:0; font-size:1.05rem; }
.ww-kpi { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:16px 18px;
  box-shadow: 0 1px 2px rgba(15,23,42,.05); height:100%; }
.ww-kpi .lbl { color:#64748b; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; }
.ww-kpi .val { color:#0f172a; font-size:1.85rem; font-weight:700; line-height:1.2; margin-top:4px;}
.ww-kpi .sub { color:#64748b; font-size:.82rem; margin-top:2px; }
.ww-kpi.accent { border-left: 5px solid #16a34a; }
.ww-badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem;
  font-weight:600; color:#fff; margin-right:6px; }
.ww-demo { background:#fef3c7; color:#92400e !important; border:1px solid #f59e0b; }
.ww-card { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:18px 20px; }
.ww-stream { border-radius:14px; padding:16px 20px; color:#fff; }
.ww-stream h3 { color:#fff; margin:0; }
.ww-step { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:14px; height:100%; }
.ww-step b { color:#15803d; }
.ww-muted { color:#64748b; font-size:.9rem; }
</style>
"""


def badge(text, color):
    return f'<span class="ww-badge" style="background:{color}">{text}</span>'


def demo_badge():
    return '<span class="ww-badge ww-demo">DEMO DATA</span>'


def kpi(col, label, value, sub="", accent=False):
    col.markdown(
        f'<div class="ww-kpi {"accent" if accent else ""}"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def fmt_kg(x):
    return f"{x:,.1f} kg"


def go_to(page):
    st.session_state["page"] = page


def compute_metrics(df: pd.DataFrame):
    """Single source of truth for all KPI numbers."""
    if df.empty:
        return dict(reported=0.0, n=0, collected=0.0, n_collected=0, recovered=0.0,
                    recovered_evidence=0.0, recovered_sim=0.0, pending=0,
                    in_transit=0, recovery_rate=None, disposed=0.0, review=0)
    collected_mask = df["status"].isin(["Collected", "Recovery Confirmed"])
    confirmed = df[df["status"] == "Recovery Confirmed"]
    rec = confirmed["recovered_weight_kg"].fillna(0)
    sim_mask = confirmed["evidence_type"] == "simulated"
    collected_confirmed = confirmed["collected_weight_kg"].fillna(0)
    non_san = confirmed["stream"] != STREAM_SANITARY
    denom = collected_confirmed[non_san].sum()
    return dict(
        reported=float(df["est_weight_kg"].sum()),
        n=int(len(df)),
        collected=float(df.loc[collected_mask, "collected_weight_kg"].fillna(0).sum()),
        n_collected=int(collected_mask.sum()),
        recovered=float(rec.sum()),
        recovered_evidence=float(rec[~sim_mask].sum()),
        recovered_sim=float(rec[sim_mask].sum()),
        pending=int(df["status"].isin(PENDING_STATUSES).sum()),
        in_transit=int((df["status"] == "Collected").sum()),
        recovery_rate=(float(rec[non_san].sum()) / denom) if denom > 0 else None,
        disposed=float(collected_confirmed[~non_san].sum()),
        review=int(((df["needs_review"] == 1) & (df["status"] != "Recovery Confirmed")).sum()),
    )


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def page_home():
    st.markdown(
        '<div class="ww-hero"><h1>♻️ WasteWise AI</h1>'
        "<p><b>From waste to resource.</b> Identify an item, get India's 4-stream disposal guidance, "
        "request a campus pickup and track it all the way to <b>verified recovery</b>.</p></div>",
        unsafe_allow_html=True,
    )
    m = compute_metrics(load_pickups(True))
    c = st.columns(4)
    kpi(c[0], "Reported", fmt_kg(m["reported"]), f'{m["n"]} requests', accent=True)
    kpi(c[1], "Collected", fmt_kg(m["collected"]), f'{m["n_collected"]} pickups')
    kpi(c[2], "Recovery confirmed", fmt_kg(m["recovered"]),
        f'evidence-backed {fmt_kg(m["recovered_evidence"])} · simulated {fmt_kg(m["recovered_sim"])}')
    kpi(c[3], "Pending pickups", m["pending"], f'{m["review"]} flagged for review')
    p_demo, _ = demo_counts()
    if p_demo:
        st.caption(f"Includes {p_demo} DEMO DATA requests. Toggle them off in Impact Analytics.")

    st.markdown("### How it works")
    steps = [
        ("1. Scan", "Upload or snap a photo. A pretrained vision model suggests the material, with an honest confidence score."),
        ("2. Segregate", "Context checks map it to Wet / Dry / Sanitary / Special-care, with handling steps and hazard flags."),
        ("3. Collect", "One tap creates a pickup request. Collectors accept, collect and weigh it."),
        ("4. Verify", "Recovery counts only with evidence (receipt / photo / manifest) - not when merely reported."),
    ]
    cols = st.columns(4)
    for col, (t, d) in zip(cols, steps):
        col.markdown(f'<div class="ww-step"><b>{t}</b><br><span class="ww-muted">{d}</span></div>',
                     unsafe_allow_html=True)

    st.markdown("### Get started")
    b = st.columns(4)
    b[0].button("📷 Scan an item", on_click=go_to, args=("AI Scanner",), width="stretch", type="primary")
    b[1].button("🚚 Request pickup", on_click=go_to, args=("Request Pickup",), width="stretch")
    b[2].button("🧑‍🔧 Collector dashboard", on_click=go_to, args=("Collector Dashboard",), width="stretch")
    b[3].button("📊 Impact analytics", on_click=go_to, args=("Impact Analytics",), width="stretch")

    sess, labels, err = load_model()
    if err:
        st.warning(f"AI model unavailable: {err} The scanner still works with transparent manual selection.")
    else:
        st.success("AI model loaded: EfficientNet-Lite0 (ImageNet-pretrained) running locally via ONNX Runtime - no internet or API key needed.")


def render_guidance(g):
    style = STREAM_STYLE[g["stream"]]
    st.markdown(
        f'<div class="ww-stream" style="background:{style["color"]}">'
        f'<div style="opacity:.85;font-size:.85rem">RECOMMENDED STREAM</div>'
        f'<h3>{style["icon"]} {g["stream"]}</h3>'
        f'<div style="opacity:.92">Bin: {style["bin"]} · Material: {g["material"]} · Recovery route: {g["recovery"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    for o in g["overrides"]:
        st.info(f"Context rule applied: {o}")
    if g["review_reasons"]:
        st.warning("⚠️ **Flagged for human review:** " + "; ".join(g["review_reasons"]))
    st.markdown("**How to handle it**")
    for s in g["steps"]:
        st.markdown(f"- {s}")
    if g["collect"]:
        st.success(f"**Collection request appropriate?** {g['collect_text']}")
    else:
        st.error(f"**Collection request appropriate?** {g['collect_text']}")


def page_scanner():
    st.markdown("## 📷 AI Waste Scanner")
    st.caption("Upload or capture a photo of a single waste item on a plain background for best results.")
    sess, labels, model_err = load_model()
    if model_err:
        st.warning(f"**AI model unavailable** - {model_err} Use manual selection below. "
                   "No prediction or confidence score will be shown.")

    src = st.radio("Image source", ["Upload image", "Use camera"], horizontal=True, label_visibility="collapsed")
    file = (st.file_uploader("Upload a waste item photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
            if src == "Upload image" else st.camera_input("Take a photo"))

    result = None
    image_bytes = None
    left, right = st.columns([1, 1.25], gap="large")
    if file is not None:
        image_bytes = file.getvalue()
        if not image_bytes:
            st.error("The uploaded file is empty.")
            image_bytes = None
    if image_bytes:
        try:
            preview = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
            left.image(preview, caption="Uploaded item", width="stretch")
        except Exception:
            left.error("Could not read this file as an image. Try a JPG or PNG.")
            image_bytes = None
    if image_bytes and not model_err:
        with right:
            with st.spinner("Analysing image with the on-device model..."):
                result = classify_cached(hashlib.sha256(image_bytes).hexdigest(), image_bytes)
            if "error" in result:
                st.error(f"AI analysis failed: {result['error']} Please select the material manually.")
                result = None
            else:
                lvl = result["level"]
                if lvl == "high":
                    st.markdown(badge("AI SUGGESTION", "#16a34a"), unsafe_allow_html=True)
                    st.markdown(f"### {result['material']}")
                elif lvl == "low":
                    st.markdown(badge("LOW CONFIDENCE - PLEASE CONFIRM", "#f59e0b"), unsafe_allow_html=True)
                    st.markdown(f"### Possibly {result['material']}")
                else:
                    st.markdown(badge("NOT RECOGNISED", "#6b7280"), unsafe_allow_html=True)
                    st.markdown("### No reliable material match")
                    st.caption("The model did not match this image to a waste material it knows. "
                               "Select the material manually below - no guess is made.")
                if lvl != "none":
                    st.progress(min(result["confidence"], 1.0),
                                text=f"Material score: {result['confidence']:.0%}")
                    st.caption("Material score = sum of the model's probabilities for ImageNet classes "
                               "that map to this material.")
                if result["top_label"] in ITEM_NOTES:
                    st.info(ITEM_NOTES[result["top_label"]])
                with st.expander("Model evidence (raw top-5 predictions)", expanded=lvl != "high"):
                    ev = pd.DataFrame(
                        [{"ImageNet class": l, "Probability": f"{p:.1%}", "Maps to": m or "- (not a mapped material)"}
                         for l, p, m in result["top5"]]
                    )
                    st.dataframe(ev, hide_index=True, width="stretch")
                    if result["scores"]:
                        sc = pd.DataFrame(result["scores"][:5], columns=["Material", "Score"])
                        fig = px.bar(sc, x="Score", y="Material", orientation="h", range_x=[0, 1],
                                     color_discrete_sequence=["#16a34a"])
                        fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
                        st.plotly_chart(fig, width="stretch")
                    st.caption(f"Probability mass on classes not mapped to any material: {result['unmapped']:.0%}")
    elif not image_bytes:
        right.markdown(
            '<div class="ww-card"><b>No image yet.</b><br><span class="ww-muted">Upload a photo to get an AI '
            "suggestion, or skip straight to manual selection below.</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Confirm material & context")
    st.caption("Material alone does not decide the stream. A quick context check prevents wrong segregation.")
    ai_mat = result["material"] if result and result["level"] in ("high", "low") else None
    options = ["- Select material -"] + MATERIAL_NAMES
    default_idx = options.index(ai_mat) if ai_mat in options else 0
    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        chosen = st.selectbox(
            "Material" + (" (pre-filled from AI - change if wrong)" if ai_mat else " (manual selection)"),
            options, index=default_idx, key=f"mat_{hashlib.md5(image_bytes or b'').hexdigest()}",
        )
        ctx = {k: st.checkbox(v, key=f"ctx_{k}") for k, v in CONTEXT_FLAGS.items()}
    if chosen == options[0]:
        c2.info("Select a material to see disposal guidance.")
        return
    if ai_mat is None:
        method = "Manual"
    elif chosen == ai_mat:
        method = "AI (confirmed)"
    else:
        method = "AI (corrected)"
    g = compute_guidance(chosen, ctx, result["level"] if result and method != "Manual" else None)
    with c2:
        render_guidance(g)
        st.caption(f"Decision method: **{method}**")
        b1, b2 = st.columns(2)
        scan_args = (method, ai_mat, result["confidence"] if result else None,
                     result["top_label"] if result else None, chosen, g["stream"], bool(g["review_reasons"]))
        prefill = {
            "material": chosen, "stream": g["stream"], "source": "ai" if ai_mat else "manual",
            "ai_confidence": result["confidence"] if (result and ai_mat) else None,
            "needs_review": bool(g["review_reasons"]),
            "review_reason": "; ".join(g["review_reasons"]),
        }
        b1.button("🚚 Request pickup for this item", type="primary", width="stretch",
                  disabled=not g["collect"], on_click=start_pickup_from_scan, args=(scan_args, prefill))
        if b2.button("📝 Log scan only", width="stretch"):
            log_scan(*scan_args)
            st.toast("Scan logged to analytics.", icon="✅")


def start_pickup_from_scan(scan_args, prefill):
    log_scan(*scan_args)
    st.session_state["prefill"] = prefill
    st.session_state["page"] = "Request Pickup"


def page_request():
    st.markdown("## 🚚 Request a Pickup")
    pre = st.session_state.get("prefill")
    if pre:
        st.success(f"Pre-filled from your scan: **{pre['material']}** → **{pre['stream']}**")
        if pre.get("needs_review"):
            st.warning(f"This request will be flagged for collector review: {pre['review_reason']}")
    with st.form("pickup_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        loc_choice = c1.selectbox("Campus location / hostel", CAMPUS_LOCATIONS + ["Other (type below)"])
        loc_other = c1.text_input("Other location / room details", placeholder="e.g. Block C, Room 214")
        mat_idx = MATERIAL_NAMES.index(pre["material"]) if pre else 0
        material = c2.selectbox("Waste category", MATERIAL_NAMES, index=mat_idx)
        weight = c2.number_input("Approximate weight (kg)", min_value=0.1, max_value=500.0, value=1.0, step=0.5)
        requester = c1.text_input("Your name (optional)")
        notes = c2.text_input("Notes for collector (optional)", placeholder="e.g. 2 bags near the mess entrance")
        submitted = st.form_submit_button("Submit pickup request", type="primary", width="stretch")
    if submitted:
        location = loc_other.strip() if loc_choice.startswith("Other") else loc_choice
        if loc_choice.startswith("Other") and not location:
            st.error("Please type the location.")
            return
        if loc_other.strip() and not loc_choice.startswith("Other"):
            location = f"{loc_choice} - {loc_other.strip()}"
        if weight <= 0:
            st.error("Weight must be greater than 0 kg.")
            return
        use_pre = pre and pre["material"] == material
        stream = pre["stream"] if use_pre else MATERIALS[material]["stream"]
        if stream == STREAM_REVIEW:
            st.error("'Mixed / unsure' items must be inspected first. Use the scanner context checks or pick a specific category.")
            return
        needs_review = bool(use_pre and pre["needs_review"]) or stream == STREAM_SPECIAL
        reason = (pre["review_reason"] if use_pre and pre["review_reason"] else
                  ("Special-care stream" if stream == STREAM_SPECIAL else ""))
        pid = create_pickup(
            location, material, stream, weight, notes.strip(), requester.strip(),
            source=pre["source"] if use_pre else "manual",
            ai_confidence=pre["ai_confidence"] if use_pre else None,
            needs_review=needs_review, review_reason=reason,
        )
        st.session_state.pop("prefill", None)
        st.success(f"✅ Pickup request **#{pid}** submitted: {material} ({stream}), {weight:.1f} kg at {location}. "
                   "Status: **Reported**.")
        st.button("Open collector dashboard →", on_click=go_to, args=("Collector Dashboard",))

    st.markdown("### Your recent requests")
    df = load_pickups(True).head(8)
    if df.empty:
        st.info("No pickup requests yet. Submit the first one above.")
    else:
        show = df[["id", "created_at", "location", "material", "stream", "est_weight_kg", "status", "is_demo"]].copy()
        show["is_demo"] = show["is_demo"].map({1: "DEMO", 0: ""})
        show.columns = ["#", "Created", "Location", "Category", "Stream", "Est. kg", "Status", "Data"]
        st.dataframe(show, hide_index=True, width="stretch")


def pickup_card(p):
    style = STREAM_STYLE.get(p["stream"], STREAM_STYLE[STREAM_REVIEW])
    header = (f"**#{p['id']} · {p['location']}** &nbsp; "
              + badge(p["status"], STATUS_STYLE[p["status"]])
              + badge(p["stream"], style["color"])
              + (demo_badge() if p["is_demo"] else "")
              + (badge("⚠ REVIEW", "#b45309") if p["needs_review"] else ""))
    st.markdown(header, unsafe_allow_html=True)
    meta = f"{p['material']} · est. {p['est_weight_kg']:.1f} kg · reported {p['created_at']}"
    if p["collected_weight_kg"]:
        meta += f" · collected {p['collected_weight_kg']:.2f} kg"
    if p["source"] == "ai" and p["ai_confidence"] is not None:
        meta += f" · AI-assisted ({p['ai_confidence']:.0%})"
    st.caption(meta)
    if p["needs_review"] and p["review_reason"]:
        st.caption(f"⚠ Review: {p['review_reason']}")
    if p["notes"]:
        st.caption(f"📝 {p['notes']}")


def page_collector():
    st.markdown("## 🧑‍🔧 Collector Dashboard")
    st.caption("Workflow: Reported → Accepted → Collected → Recovery Confirmed (evidence required).")
    if msg := st.session_state.pop("collector_msg", None):
        (st.success if msg[0] else st.error)(msg[1])

    df = load_pickups(True)
    counts = df["status"].value_counts() if not df.empty else pd.Series(dtype=int)
    c = st.columns(4)
    for col, s in zip(c, STATUSES):
        kpi(col, s, int(counts.get(s, 0)), "requests", accent=(s == "Reported"))
    st.write("")

    f1, f2 = st.columns([2, 1])
    stage = f1.radio("Queue", ["Reported", "Accepted", "Collected", "Recovery Confirmed"], horizontal=True)
    show_demo = f2.toggle("Show DEMO DATA", value=True)
    q = df[df["status"] == stage] if not df.empty else df
    if not show_demo and not q.empty:
        q = q[q["is_demo"] == 0]
    if q.empty:
        st.info(f"No requests in **{stage}**. " +
                ("Create one from Request Pickup." if stage == "Reported" else "Move requests forward from the previous stage."))
    for _, row in q.iterrows():
        p = row.to_dict()
        pid = int(p["id"])
        with st.container(border=True):
            info, act = st.columns([1.6, 1], gap="large")
            with info:
                pickup_card(p)
            with act:
                if p["status"] == "Reported":
                    if st.button("✅ Accept request", key=f"acc_{pid}", type="primary", width="stretch"):
                        st.session_state["collector_msg"] = accept_pickup(pid)
                        st.rerun()
                elif p["status"] == "Accepted":
                    with st.form(f"col_{pid}"):
                        w = st.number_input("Collected weight (kg)", min_value=0.0, max_value=1000.0,
                                            value=float(p["est_weight_kg"]), step=0.1, key=f"w_{pid}")
                        if st.form_submit_button("📦 Mark collected", type="primary", width="stretch"):
                            st.session_state["collector_msg"] = collect_pickup(pid, w)
                            st.rerun()
                elif p["status"] == "Collected":
                    recovery_form(p)
                else:
                    ev = {"document": "📄 Evidence file", "reference": "🧾 Receipt / manifest ref",
                          "simulated": "🧪 SIMULATED demo confirmation"}.get(p["evidence_type"], "-")
                    label = "Safely disposed" if p["stream"] == STREAM_SANITARY else "Recovered"
                    st.markdown(f"**{label}:** {p['recovered_weight_kg'] or 0:.2f} kg  \n"
                                f"**Facility:** {p['recovery_facility']}  \n**Evidence:** {ev}"
                                + (f" ({p['evidence_ref']})" if p["evidence_ref"] else ""))
                    if p["evidence_file"]:
                        fp = EVIDENCE_DIR / p["evidence_file"]
                        if fp.exists() and fp.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                            st.image(str(fp), width=160)

    with st.expander("📜 Audit trail (every status change is logged)"):
        log = load_log()
        if log.empty:
            st.caption("No activity yet.")
        else:
            log["is_demo"] = log["is_demo"].map({1: "DEMO", 0: ""})
            log.columns = ["Time", "#", "From", "To", "Note", "Location", "Data"]
            st.dataframe(log, hide_index=True, width="stretch")


def recovery_form(p):
    pid = int(p["id"])
    sanitary = p["stream"] == STREAM_SANITARY
    collected = float(p["collected_weight_kg"] or 0)
    with st.form(f"rec_{pid}"):
        st.markdown("**Confirm safe disposal**" if sanitary else "**Confirm recovery**")
        fac = st.selectbox("Facility", RECOVERY_FACILITIES, key=f"fac_{pid}",
                           index=6 if sanitary else (2 if p["stream"] == STREAM_WET else
                                                     4 if p["stream"] == STREAM_SPECIAL else 0))
        if sanitary:
            st.caption("Sanitary waste is safely disposed, not recycled: it is never counted as recovered.")
            rw = 0.0
        else:
            rw = st.number_input(f"Recovered weight (kg, ≤ {collected:.2f})", min_value=0.0,
                                 max_value=max(collected, 0.0), value=collected, step=0.1, key=f"rw_{pid}")
        ev_file = st.file_uploader("Evidence: weighbridge slip / receipt / photo", type=["png", "jpg", "jpeg", "pdf"],
                                   key=f"ef_{pid}")
        ev_ref = st.text_input("…or receipt / manifest number", key=f"er_{pid}")
        sim = st.checkbox("No real evidence - mark as SIMULATED demo confirmation", key=f"sim_{pid}")
        if st.form_submit_button("🌱 Confirm", type="primary", width="stretch"):
            fname = ""
            if ev_file is not None:
                EVIDENCE_DIR.mkdir(exist_ok=True)
                suffix = Path(ev_file.name).suffix.lower() or ".bin"
                fname = f"pickup{pid}_{datetime.now():%Y%m%d%H%M%S}{suffix}"
                (EVIDENCE_DIR / fname).write_bytes(ev_file.getvalue())
                etype = "document"
            elif ev_ref.strip():
                etype = "reference"
            elif sim:
                etype = "simulated"
            else:
                st.error("Add an evidence file, a receipt number, or tick the simulated demo box.")
                return
            st.session_state["collector_msg"] = confirm_recovery(pid, rw, fac, etype, ev_ref.strip(), fname)
            st.rerun()


def page_analytics():
    st.markdown("## 📊 Impact Analytics")
    p_demo, s_demo = demo_counts()
    c_top = st.columns([2, 1])
    include_demo = c_top[1].toggle("Include DEMO DATA", value=True,
                                   help="Turn off to see only real observations recorded in this app.")
    df = load_pickups(include_demo)
    scans = load_scans(include_demo)
    if include_demo and p_demo:
        c_top[0].markdown(demo_badge() + f" Showing {p_demo} demo requests + real records. "
                          "Demo recoveries use SIMULATED evidence.", unsafe_allow_html=True)
    else:
        c_top[0].markdown(badge("REAL DATA ONLY", "#16a34a"), unsafe_allow_html=True)

    if df.empty:
        st.info("No pickup data yet. Submit a pickup request, or load DEMO DATA from the sidebar to preview the dashboard.")
        return
    m = compute_metrics(df)
    c = st.columns(4)
    kpi(c[0], "Total reported", fmt_kg(m["reported"]), f'{m["n"]} requests (estimated weight)', accent=True)
    kpi(c[1], "Total collected", fmt_kg(m["collected"]), f'{m["n_collected"]} pickups (weighed)')
    kpi(c[2], "Recovery confirmed", fmt_kg(m["recovered"]),
        f'evidence-backed {fmt_kg(m["recovered_evidence"])} · simulated {fmt_kg(m["recovered_sim"])}')
    kpi(c[3], "Pending pickups", m["pending"], f'{m["in_transit"]} collected, awaiting verification')
    st.write("")
    c = st.columns(4)
    kpi(c[0], "Recovery rate", f'{m["recovery_rate"]:.0%}' if m["recovery_rate"] is not None else "-",
        "recovered ÷ collected (confirmed, excl. sanitary)")
    kpi(c[1], "Safely disposed", fmt_kg(m["disposed"]), "sanitary - not counted as recovered")
    kpi(c[2], "Flagged for review", m["review"], "open hazardous / uncertain items")
    kpi(c[3], "Scans logged", len(scans),
        f'{int((scans["method"] != "Manual").sum()) if not scans.empty else 0} AI-assisted')
    st.caption("Reported = requester estimate. Collected = collector-weighed. Recovered = confirmed with evidence "
               "(or SIMULATED for demos). Reported or collected material is never counted as recycled.")

    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Waste by category (reported kg)")
        by_mat = df.groupby("material", as_index=False)["est_weight_kg"].sum().sort_values("est_weight_kg", ascending=False)
        fig = px.pie(by_mat, names="material", values="est_weight_kg", hole=0.55,
                     color="material", color_discrete_map=MATERIAL_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent", insidetextorientation="horizontal")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Reported → Collected → Recovered, by stream")
        rows = []
        for stream, g in df.groupby("stream"):
            mm = compute_metrics(g)
            rows += [(stream, "Reported", mm["reported"]), (stream, "Collected", mm["collected"]),
                     (stream, "Recovered", mm["recovered"])]
        sdf = pd.DataFrame(rows, columns=["Stream", "Stage", "kg"])
        fig = px.bar(sdf, x="Stream", y="kg", color="Stage", barmode="group",
                     color_discrete_map={"Reported": "#94a3b8", "Collected": "#f59e0b", "Recovered": "#16a34a"})
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, legend_title=None)
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown("#### Daily trend (kg)")
        d = df.copy()
        rep = d.assign(day=pd.to_datetime(d["created_at"], format="mixed").dt.date).groupby("day")["est_weight_kg"].sum().rename("Reported")
        cd = d.dropna(subset=["collected_at"])
        col = cd.assign(day=pd.to_datetime(cd["collected_at"], format="mixed").dt.date).groupby("day")["collected_weight_kg"].sum().rename("Collected")
        rd = d[d["status"] == "Recovery Confirmed"]
        rec = rd.assign(day=pd.to_datetime(rd["recovered_at"], format="mixed").dt.date).groupby("day")["recovered_weight_kg"].sum().rename("Recovered")
        trend = pd.concat([rep, col, rec], axis=1).fillna(0).sort_index()
        fig = go.Figure()
        for name, color in [("Reported", "#94a3b8"), ("Collected", "#f59e0b"), ("Recovered", "#16a34a")]:
            if name in trend:
                fig.add_trace(go.Scatter(x=trend.index, y=trend[name], name=name, mode="lines+markers",
                                         line=dict(color=color, width=3)))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Pipeline status")
        sc = df["status"].value_counts().reindex(STATUSES, fill_value=0)
        fig = go.Figure(go.Funnel(y=sc.index, x=sc.values,
                                  marker=dict(color=[STATUS_STYLE[s] for s in sc.index])))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Hotspots by location (reported kg)")
    loc = df.groupby("location", as_index=False)["est_weight_kg"].sum().sort_values("est_weight_kg").tail(10)
    fig = px.bar(loc, x="est_weight_kg", y="location", orientation="h", color_discrete_sequence=["#15803d"])
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="kg", yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    with st.expander("All records"):
        out = df.copy()
        out["is_demo"] = out["is_demo"].map({1: "DEMO", 0: "REAL"})
        st.dataframe(out, hide_index=True, width="stretch")
        st.download_button("⬇️ Download CSV", out.to_csv(index=False).encode("utf-8"),
                           "wastewise_pickups.csv", "text/csv")


def page_about():
    st.markdown("## ℹ️ About WasteWise AI")
    st.markdown(
        """
**Problem.** Campuses generate tonnes of mixed waste. Apps that only *identify* waste stop short:
nobody knows whether the item was segregated correctly, collected, or actually recovered.

**Solution.** WasteWise AI closes the loop: **identify → segregate (Wet / Dry / Sanitary / Special-care) →
request pickup → collect & weigh → confirm recovery with evidence → measure impact.**

**AI, honestly.** A pretrained EfficientNet-Lite0 (ImageNet, 1,000 classes) runs locally through ONNX Runtime.
Its class probabilities are mapped to waste materials and summed into a material score. Low scores trigger
manual confirmation; the model never forces a guess, and every prediction can be corrected by the user.

**Integrity rules.**
- Reported weight is an estimate; collected weight is measured by the collector.
- Recovery is counted only after confirmation with evidence (file / receipt number), or an explicit
  SIMULATED confirmation that is reported separately.
- Sanitary waste is safely disposed and never counted as recovered.
- Demo records are flagged `DEMO DATA` and can be excluded from every chart.
"""
    )


# --------------------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------------------
PAGES = {
    "Home": ("🏠", page_home),
    "AI Scanner": ("📷", page_scanner),
    "Request Pickup": ("🚚", page_request),
    "Collector Dashboard": ("🧑‍🔧", page_collector),
    "Impact Analytics": ("📊", page_analytics),
    "About": ("ℹ️", page_about),
}


def sidebar():
    with st.sidebar:
        st.markdown("## ♻️ WasteWise AI")
        st.caption("From waste to resource")
        st.radio("Navigate", list(PAGES.keys()), key="page",
                 format_func=lambda p: f"{PAGES[p][0]}  {p}", label_visibility="collapsed")
        st.divider()
        m = compute_metrics(load_pickups(True))
        st.markdown(f"**Pending pickups:** {m['pending']}  \n**Recovered:** {fmt_kg(m['recovered'])}")
        st.divider()
        st.markdown("**Demo controls**")
        p_demo, s_demo = demo_counts()
        if p_demo:
            st.caption(f"{p_demo} DEMO requests + {s_demo} DEMO scans loaded.")
            if st.button("🗑️ Clear DEMO DATA", width="stretch"):
                clear_demo_data()
                st.toast("Demo data removed. Real records kept.", icon="🗑️")
                st.rerun()
        else:
            st.caption("Load labelled sample records to preview analytics.")
            if st.button("✨ Load DEMO DATA", width="stretch"):
                seed_demo_data()
                st.toast("Demo data loaded (flagged as DEMO).", icon="✨")
                st.rerun()
        st.divider()
        _, _, err = load_model()
        st.caption("🧠 AI model: " + ("offline fallback (manual mode)" if err else "EfficientNet-Lite0 · on-device"))


def main():
    st.set_page_config(page_title="WasteWise AI", page_icon="♻️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_db()
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    sidebar()
    PAGES[st.session_state["page"]][1]()


if __name__ == "__main__":
    main()
