"""
Citrus Disease Detection V2 - Flask App (FINAL FIXED)
Model: ResNet-50, 4-channel RGB+NDVI, 9 classes
All PDF bugs fixed. Low confidence warning added.

Copyright (c) 2026 Sanket Dhumal. All Rights Reserved.
This source code is proprietary. See LICENSE for terms.
"""

import os, io, base64, sqlite3, datetime, json, requests
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if present


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps
import cv2
from flask import (Flask, request, render_template, jsonify,
                   send_file, session, redirect, url_for, g, send_from_directory)
from fpdf import FPDF

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-fallback-key")

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "results" / "fusion_resnet50_v2_seed42.pth"
NDVI_PATH  = BASE_DIR / "ndvi_20230212.npy"
DB_PATH    = BASE_DIR / "instance" / "predictions.db"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Devanagari-capable font (Hindi / Marathi PDF reports). Helvetica (FPDF's
# built-in font) only supports Latin-1, so हिंदी/मराठी text would render as
# garbage or empty boxes unless we embed a real Unicode font for those PDFs.
FONTS_DIR    = BASE_DIR / "fonts"
NOTO_REGULAR = FONTS_DIR / "NotoSansDevanagari-Regular.ttf"
NOTO_BOLD    = FONTS_DIR / "NotoSansDevanagari-Bold.ttf"

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
DEFAULT_CITY = os.environ.get("DEFAULT_CITY", "Chhatrapati Sambhajinagar")

# ── Class names (alphabetical - must match training) ──────────────────────────
CLASS_NAMES = [
    "Anthracnose",
    "Citrus_Blackspot",
    "Citrus_Canker",
    "Citrus_Greening_HLB",
    "Citrus_Leafminer",
    "Citrus_Nutrient_Deficiency",
    "Healthy_Leaf",
    "Multiple_Diseases",
    "Young_Healthy_Leaf",
]

DISPLAY_NAMES = {
    "en": {
        "Anthracnose":               "Anthracnose",
        "Citrus_Blackspot":          "Citrus Blackspot",
        "Citrus_Canker":             "Citrus Canker",
        "Citrus_Greening_HLB":       "Citrus Greening (HLB)",
        "Citrus_Leafminer":          "Citrus Leafminer",
        "Citrus_Nutrient_Deficiency":"Nutrient Deficiency",
        "Healthy_Leaf":              "Healthy Leaf",
        "Multiple_Diseases":         "Multiple Diseases",
        "Young_Healthy_Leaf":        "Young Healthy Leaf",
    },
    "hi": {
        "Anthracnose":               "एन्थ्रेकनोज",
        "Citrus_Blackspot":          "साइट्रस काला धब्बा",
        "Citrus_Canker":             "साइट्रस कैंकर",
        "Citrus_Greening_HLB":       "साइट्रस ग्रीनिंग (HLB)",
        "Citrus_Leafminer":          "साइट्रस पत्ती खनक",
        "Citrus_Nutrient_Deficiency":"पोषक तत्व की कमी",
        "Healthy_Leaf":              "स्वस्थ पत्ती",
        "Multiple_Diseases":         "एकाधिक रोग",
        "Young_Healthy_Leaf":        "युवा स्वस्थ पत्ती",
    },
    "mr": {
        "Anthracnose":               "अँथ्रॅकनोज",
        "Citrus_Blackspot":          "लिंबू काळे डाग",
        "Citrus_Canker":             "लिंबू कँकर",
        "Citrus_Greening_HLB":       "लिंबू ग्रीनिंग (HLB)",
        "Citrus_Leafminer":          "लिंबू पानखाऊ कीड",
        "Citrus_Nutrient_Deficiency":"पोषक तत्वांची कमतरता",
        "Healthy_Leaf":              "निरोगी पान",
        "Multiple_Diseases":         "एकाधिक रोग",
        "Young_Healthy_Leaf":        "तरुण निरोगी पान",
    },
}

# ── PDF report labels (structural text, not disease content) ─────────────────
# Note: the treatment paragraph text itself (cause/pesticide/prevention/
# recovery sentences in TREATMENTS below) is only authored in English. These
# labels translate the report's headings/scaffolding; if you also want the
# treatment paragraphs professionally translated into hi/mr, that's a
# separate, larger task best reviewed by a fluent speaker given it's
# agronomic/pesticide guidance.
PDF_LABELS = {
    "en": {
        "title":             "Citrus Disease Detection Report",
        "date_time":         "Date / Time",
        "plot_location":     "Plot / Location",
        "not_specified":     "Not specified",
        "prediction":        "PREDICTION",
        "confidence":        "Confidence",
        "severity":          "Severity",
        "gradcam":           "Grad-CAM heatmap",
        "treatment_heading": "TREATMENT & MANAGEMENT",
        "cause":             "Cause",
        "pesticide":         "Recommended Treatment",
        "prevention":        "Prevention",
        "recovery":          "Expected Recovery",
        "sev_values":        {"Severe": "Severe", "Moderate": "Moderate", "Mild": "Mild", "None": "None"},
    },
    "hi": {
        "title":             "साइट्रस रोग डिटेक्शन रिपोर्ट",
        "date_time":         "दिनांक / समय",
        "plot_location":     "प्लॉट / स्थान",
        "not_specified":     "निर्दिष्ट नहीं",
        "prediction":        "पूर्वानुमान",
        "confidence":        "विश्वास",
        "severity":          "गंभीरता",
        "gradcam":           "ग्रैड-कैम हीटमैप",
        "treatment_heading": "उपचार एवं प्रबंधन",
        "cause":             "कारण",
        "pesticide":         "अनुशंसित उपचार",
        "prevention":        "रोकथाम",
        "recovery":          "अपेक्षित सुधार",
        "sev_values":        {"Severe": "गंभीर", "Moderate": "मध्यम", "Mild": "हल्का", "None": "कोई नहीं"},
    },
    "mr": {
        "title":             "लिंबू रोग डिटेक्शन अहवाल",
        "date_time":         "दिनांक / वेळ",
        "plot_location":     "प्लॉट / स्थान",
        "not_specified":     "नमूद केलेले नाही",
        "prediction":        "अंदाज",
        "confidence":        "आत्मविश्वास",
        "severity":          "तीव्रता",
        "gradcam":           "ग्रॅड-कॅम हीटमॅप",
        "treatment_heading": "उपचार आणि व्यवस्थापन",
        "cause":             "कारण",
        "pesticide":         "शिफारस केलेला उपचार",
        "prevention":        "प्रतिबंध",
        "recovery":          "अपेक्षित सुधारणा",
        "sev_values":        {"Severe": "तीव्र", "Moderate": "मध्यम", "Mild": "सौम्य", "None": "काहीही नाही"},
    },
}

TREATMENTS = {
    "Anthracnose": {
        "cause":      "Fungal pathogen Colletotrichum gloeosporioides; thrives in humid, warm conditions.",
        "pesticide":  "Copper oxychloride (0.3%) or Mancozeb (0.25%) sprays every 10-14 days.",
        "prevention": "Prune dead wood, improve air circulation, avoid overhead irrigation.",
        "recovery":   "3-6 weeks with consistent treatment.",
        "is_disease": True,
    },
    "Citrus_Blackspot": {
        "cause":      "Fungal pathogen Phyllosticta citricarpa; spreads via wind-borne spores.",
        "pesticide":  "Copper-based fungicides (Copper hydroxide 77 WP) at 2-week intervals.",
        "prevention": "Remove fallen leaves/fruit, apply dormant copper sprays.",
        "recovery":   "4-8 weeks; infected fruit must be discarded.",
        "is_disease": True,
    },
    "Citrus_Canker": {
        "cause":      "Bacterial pathogen Xanthomonas citri; spreads via rain splash and wind.",
        "pesticide":  "Copper bactericides (Copper hydroxide + Mancozeb). Notify local agriculture dept.",
        "prevention": "Windbreaks, avoid working in wet orchards, disinfect tools.",
        "recovery":   "Canker is not curable; manage spread. Severe cases may require tree removal.",
        "is_disease": True,
    },
    "Citrus_Greening_HLB": {
        "cause":      "Bacterial pathogen Candidatus Liberibacter; spread by Asian citrus psyllid insect.",
        "pesticide":  "Control psyllid with Imidacloprid or Thiamethoxam. No cure for infected trees.",
        "prevention": "Certified disease-free nursery stock. Remove and destroy infected trees immediately.",
        "recovery":   "No recovery - infected trees decline over 3-5 years. Early removal is essential.",
        "is_disease": True,
    },
    "Citrus_Leafminer": {
        "cause":      "Larvae of moth Phyllocnistis citrella tunnel through young leaves.",
        "pesticide":  "Spinosad or Abamectin sprays targeting new flush growth.",
        "prevention": "Synchronize flushes, remove heavily infested shoots.",
        "recovery":   "2-4 weeks after controlling the pest on new growth.",
        "is_disease": True,
    },
    "Citrus_Nutrient_Deficiency": {
        "cause":      "Iron, zinc, or magnesium deficiency - often from poor soil pH or drainage.",
        "pesticide":  "Foliar spray: ZnSO4 (0.5%) for zinc, FeSO4 (0.5%) for iron, MgSO4 (1%) for Mg.",
        "prevention": "Soil test annually. Maintain pH 6.0-7.0. Apply balanced NPK fertiliser.",
        "recovery":   "2-4 weeks after corrective foliar/soil application.",
        "is_disease": False,
    },
    "Healthy_Leaf": {
        "cause":      "No disease detected.",
        "pesticide":  "No treatment required.",
        "prevention": "Continue regular monitoring, balanced nutrition, and irrigation schedule.",
        "recovery":   "Plant is healthy.",
        "is_disease": False,
    },
    "Multiple_Diseases": {
        "cause":      "Multiple pathogens detected simultaneously - inspect closely.",
        "pesticide":  "Treat the most severe disease first. Consult an agriculture extension officer.",
        "prevention": "Reduce plant stress, improve orchard hygiene, avoid excess nitrogen.",
        "recovery":   "Varies - address each disease individually.",
        "is_disease": True,
    },
    "Young_Healthy_Leaf": {
        "cause":      "Young healthy growth detected.",
        "pesticide":  "No treatment required.",
        "prevention": "Protect young flush from psyllid and leafminer with preventive sprays.",
        "recovery":   "Plant is healthy.",
        "is_disease": False,
    },
}

UI = {
    "en": {
        "title":          "Citrus Disease Detector",
        "upload_prompt":  "Upload or take a photo of a citrus leaf",
        "detect_btn":     "Detect Disease",
        "result_heading": "Detection Result",
        "severity":       "Severity",
        "confidence":     "Confidence",
        "cause":          "Cause",
        "pesticide":      "Recommended Treatment",
        "prevention":     "Prevention",
        "recovery":       "Expected Recovery",
        "download_pdf":   "Download PDF Report",
        "history":        "Prediction History",
        "weather_risk":   "Weather Disease Risk",
        "lang_label":     "Language",
        "healthy_msg":    "Your citrus plant looks healthy!",
    },
    "hi": {
        "title":          "साइट्रस रोग डिटेक्टर",
        "upload_prompt":  "साइट्रस पत्ती की फोटो अपलोड करें या लें",
        "detect_btn":     "रोग पहचानें",
        "result_heading": "पहचान परिणाम",
        "severity":       "गंभीरता",
        "confidence":     "विश्वास",
        "cause":          "कारण",
        "pesticide":      "अनुशंसित उपचार",
        "prevention":     "रोकथाम",
        "recovery":       "अपेक्षित सुधार",
        "download_pdf":   "PDF रिपोर्ट डाउनलोड करें",
        "history":        "पूर्वानुमान इतिहास",
        "weather_risk":   "मौसम रोग जोखिम",
        "lang_label":     "भाषा",
        "healthy_msg":    "आपका साइट्रस पौधा स्वस्थ दिखता है!",
    },
    "mr": {
        "title":          "लिंबू रोग डिटेक्टर",
        "upload_prompt":  "लिंबू पानाचा फोटो अपलोड करा किंवा घ्या",
        "detect_btn":     "रोग ओळखा",
        "result_heading": "ओळख निकाल",
        "severity":       "तीव्रता",
        "confidence":     "आत्मविश्वास",
        "cause":          "कारण",
        "pesticide":      "शिफारस केलेला उपचार",
        "prevention":     "प्रतिबंध",
        "recovery":       "अपेक्षित सुधारणा",
        "download_pdf":   "PDF अहवाल डाउनलोड करा",
        "history":        "अंदाज इतिहास",
        "weather_risk":   "हवामान रोग धोका",
        "lang_label":     "भाषा",
        "healthy_msg":    "तुमचे लिंबू झाड निरोगी दिसत आहे!",
    },
}

# ── PDF helper ────────────────────────────────────────────────────────────────
# English PDFs use FPDF's built-in Helvetica font, which only supports
# Latin-1 — so English text still gets stripped to Latin-1 to avoid crashes.
# Hindi/Marathi PDFs use the embedded Noto Sans Devanagari font instead
# (see make_pdf), which is full Unicode, so their text passes through as-is.
def _pdf_safe(text, unicode_font=False):
    if not isinstance(text, str):
        text = str(text)
    if unicode_font:
        return text
    return text.encode('latin-1', errors='ignore').decode('latin-1')


@lru_cache(maxsize=1)
def devanagari_fonts_available():
    return NOTO_REGULAR.exists() and NOTO_BOLD.exists()


# ── Model loading ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_model():
    model = models.resnet50(weights=None)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    nn.init.zeros_(new_conv.weight)
    model.conv1 = new_conv
    model.fc = nn.Linear(2048, len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ── NDVI loading ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_ndvi():
    ndvi_raw = np.load(str(NDVI_PATH)).astype(np.float32)
    ndvi_raw = np.clip(ndvi_raw, 0, 1)
    ndvi_norm = (ndvi_raw - ndvi_raw.min()) / (ndvi_raw.max() - ndvi_raw.min() + 1e-5)
    return ndvi_norm


# ── Leaf auto-crop ────────────────────────────────────────────────────────────
def crop_to_leaf(img_pil, pad_frac=0.08):
    """Crop the photo down to the leaf's bounding box so it fills the frame
    the way training images do, instead of squashing the whole scene
    (leaf + background) into 224x224. Falls back to the full image if no
    leaf-colored region is found."""
    arr = np.array(img_pil.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 30), (85, 255, 255))
    brown_mask = cv2.inRange(hsv, (8, 40, 20), (25, 200, 180))
    mask = cv2.bitwise_or(green_mask, brown_mask)

    # Clean up small noise so stray background pixels don't skew the box
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:  # not enough leaf-colored pixels to trust a crop
        return img_pil

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = arr.shape[:2]
    pad_x = int((x1 - x0) * pad_frac)
    pad_y = int((y1 - y0) * pad_frac)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)

    # Guard against a degenerate (near-zero-area) crop
    if (x1 - x0) < 20 or (y1 - y0) < 20:
        return img_pil

    return Image.fromarray(arr[y0:y1, x0:x1])


# ── Image loading (EXIF-safe) ───────────────────────────────────────────────
def open_image_upright(raw_bytes):
    """Open image bytes as PIL RGB, correcting for EXIF orientation.
    Phone cameras store photos with an orientation tag rather than
    physically rotating the pixel data; PIL.Image.open() ignores that tag
    by default, so a portrait photo taken on a phone can load sideways.
    ImageOps.exif_transpose() applies the tag so every downstream check
    (color mask, scene classifier, disease model) sees the photo the same
    way a person would see it on screen."""
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)  # no-op if no orientation tag present
    return img.convert("RGB")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(raw_bytes, use_crop=True):
    img_pil = open_image_upright(raw_bytes)
    if use_crop:
        img_pil = crop_to_leaf(img_pil)
    img_pil = img_pil.resize((224, 224))

    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    rgb_tensor = rgb_transform(img_pil)

    # NDVI channel: the model expects a 4th channel, but we have no real
    # per-photo NDVI for phone/internet images. Using the fixed, unrelated
    # Nagpur satellite tile here injects the same irrelevant signal into
    # every prediction regardless of what's actually in the photo. Fall
    # back to a neutral mid-value channel instead, so the 4th channel
    # contributes ~no information rather than actively misleading the
    # model with mismatched context.
    ndvi_tensor = torch.full((1, 224, 224), 0.5)

    fused = torch.cat([rgb_tensor, ndvi_tensor], dim=0)
    return fused.unsqueeze(0)


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def generate_gradcam(model, img_tensor_4ch, class_idx):
    features, grads = {}, {}

    def fwd_hook(m, inp, out):
        features["map"] = out.detach()

    def bwd_hook(m, gin, gout):
        grads["map"] = gout[0].detach()

    target_layer = model.layer4[-1]
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    with torch.enable_grad():
        out = model(img_tensor_4ch)
        out[0, class_idx].backward()

    fh.remove()
    bh.remove()

    pooled = grads["map"].mean(dim=[2, 3], keepdim=True)
    cam = (features["map"] * pooled).sum(dim=1).squeeze()
    cam = torch.relu(cam).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_u8 = (cam * 255).astype(np.uint8)
    cam_r = cv2.resize(cam_u8, (224, 224))
    heatmap = cv2.applyColorMap(cam_r, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    orig = img_tensor_4ch.squeeze()[:3].permute(1, 2, 0).numpy()
    orig = (orig * std + mean).clip(0, 1)
    orig_u8 = (orig * 255).astype(np.uint8)

    blended = cv2.addWeighted(orig_u8, 0.55, heatmap, 0.45, 0)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Severity ──────────────────────────────────────────────────────────────────
def get_severity(confidence, is_disease):
    if not is_disease:
        return "None", "success"
    if confidence >= 0.88:
        return "Severe", "danger"
    elif confidence >= 0.70:
        return "Moderate", "warning"
    else:
        return "Mild", "info"


# ── NDVI satellite context panel ────────────────────────────────────────────
# This renders the ONE real Sentinel-2 NDVI tile we have (the same array the
# model's 4th input channel uses) as an honest, clearly-labeled REGIONAL
# vegetation-context visual for the user/agronomist. It is NOT a per-leaf
# measurement and is NOT recomputed per prediction - it's the same static
# scene the model was trained against, shown for transparency/context only.
NDVI_SOURCE_LABEL = "Sentinel-2 L2A NDVI — Nagpur Agricultural Region, Maharashtra (2023-02-12)"

def _ndvi_to_rgb(ndvi_norm):
    """Map a 0-1 normalised NDVI array to an RdYlGn-style colour image
    (red = low vegetation vigour, yellow = moderate, green = high) without
    pulling in matplotlib as a dependency."""
    stops = np.array([
        [0.00, 165,  0,  38],   # red
        [0.30, 244, 109, 67],   # orange
        [0.50, 254, 224, 139],  # yellow
        [0.70, 166, 217, 106],  # light green
        [1.00,  26, 152,  80],  # dark green
    ])
    xs = stops[:, 0]
    r = np.interp(ndvi_norm, xs, stops[:, 1])
    g = np.interp(ndvi_norm, xs, stops[:, 2])
    b = np.interp(ndvi_norm, xs, stops[:, 3])
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return rgb


def _ndvi_health_label(mean_val):
    if mean_val < 0.20:
        return "Bare / Stressed", "danger"
    elif mean_val < 0.40:
        return "Sparse Vegetation", "warning"
    elif mean_val < 0.60:
        return "Moderate Vegetation Health", "info"
    elif mean_val < 0.80:
        return "Healthy Vegetation", "success"
    else:
        return "Dense / Very Healthy Vegetation", "success"


@lru_cache(maxsize=1)
def get_ndvi_context():
    """Build the colourised NDVI image + summary stats once (cached),
    since it's the same static source scene every time."""
    ndvi = load_ndvi()  # 0-1 normalised, 224x224
    rgb = _ndvi_to_rgb(ndvi)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    mean_val = float(ndvi.mean())
    label, css_class = _ndvi_health_label(mean_val)

    return {
        "image": img_b64,
        "source": NDVI_SOURCE_LABEL,
        "mean_ndvi": round(mean_val, 3),
        "health_label": label,
        "css_class": css_class,
    }


# ── Weather ───────────────────────────────────────────────────────────────────
def get_weather_risk(lat=19.8993, lon=75.3195, city="Chhatrapati Sambhajinagar"):
    if not OPENWEATHER_API_KEY:
        return None
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric")
        r    = requests.get(url, timeout=4)
        data = r.json()
        temp  = data["main"]["temp"]
        humid = data["main"]["humidity"]
        desc  = data["weather"][0]["description"].title()
        if humid > 80 and 20 < temp < 38:
            risk, risk_class = "High", "danger"
            tip = "High humidity + warm temperature: ideal for Anthracnose & Blackspot. Apply preventive fungicide."
        elif humid > 65:
            risk, risk_class = "Moderate", "warning"
            tip = "Moderate humidity: monitor closely. Ensure good air circulation in the orchard."
        else:
            risk, risk_class = "Low", "success"
            tip = "Current weather conditions are not favourable for most citrus diseases."
        return dict(temp=temp, humidity=humid, description=desc,
                    risk=risk, risk_class=risk_class, tip=tip, city=city)
    except Exception:
        return None


# ── PDF report ────────────────────────────────────────────────────────────────
def make_pdf(disease, display_name, confidence, severity, treatment,
             cam_b64, timestamp, plot_id, lang="en"):
    labels = PDF_LABELS.get(lang, PDF_LABELS["en"])

    # Use the embedded Unicode Devanagari font for hi/mr so हिंदी/मराठी text
    # renders correctly; otherwise stick to FPDF's built-in Helvetica.
    use_unicode = lang in ("hi", "mr") and devanagari_fonts_available()
    FONT = "NotoDev" if use_unicode else "Helvetica"

    def safe(text):
        return _pdf_safe(text, unicode_font=use_unicode)

    pdf = FPDF()
    pdf.add_page()

    if use_unicode:
        pdf.add_font("NotoDev", "", str(NOTO_REGULAR))
        pdf.add_font("NotoDev", "B", str(NOTO_BOLD))
        # No italic weight embedded; fall back to regular for "I" style.
        pdf.add_font("NotoDev", "I", str(NOTO_REGULAR))

    # ── Header banner ─────────────────────────────────────────────────────────
    pdf.set_fill_color(45, 140, 45)          # --green-dark
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_y(10)
    pdf.set_font(FONT, "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, safe(labels["title"]), ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # ── Meta row ──────────────────────────────────────────────────────────────
    pdf.set_font(FONT, "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(95, 6, safe(f"{labels['date_time']} : {timestamp}"), ln=0)
    pdf.cell(0,  6, safe(f"{labels['plot_location']} : {plot_id or labels['not_specified']}"), ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y() + 2, 210 - pdf.r_margin, pdf.get_y() + 2)
    pdf.ln(6)

    # ── Grad-CAM image (left) + prediction summary (right) ───────────────────
    cam_w = 75
    has_cam = False
    cam_path = "/tmp/cam_tmp.png"
    if cam_b64:
        try:
            with open(cam_path, "wb") as f:
                f.write(base64.b64decode(cam_b64))
            has_cam = True
        except Exception:
            pass

    y_section = pdf.get_y()

    if has_cam:
        pdf.image(cam_path, x=pdf.l_margin, y=y_section, w=cam_w)
        pdf.set_xy(pdf.l_margin + cam_w + 5, y_section)
    else:
        pdf.set_xy(pdf.l_margin, y_section)

    info_x = pdf.get_x()
    info_w = 210 - pdf.r_margin - info_x

    # ── Prediction heading ────────────────────────────────────────────────────
    pdf.set_font(FONT, "B", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(info_w, 7, safe(labels["prediction"]), ln=True)
    pdf.set_xy(info_x, pdf.get_y())

    # Disease name
    pdf.set_font(FONT, "B", 15)
    pdf.set_text_color(45, 140, 45)
    pdf.cell(info_w, 9, safe(display_name), ln=True)
    pdf.set_xy(info_x, pdf.get_y())

    # Confidence
    pdf.set_font(FONT, "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(info_w, 7, safe(f"{labels['confidence']} : {confidence*100:.1f}%"), ln=True)
    pdf.set_xy(info_x, pdf.get_y() + 3)

    # Severity badge — pick text color for contrast
    sev_bg = {
        "Severe":   (220, 53,  69),
        "Moderate": (230, 130,  0),   # darker orange instead of yellow
        "Mild":     (10,  150, 200),
        "None":     (25,  135,  84),
    }.get(severity, (108, 117, 125))
    pdf.set_fill_color(*sev_bg)
    pdf.set_text_color(255, 255, 255)    # white always readable on these darks
    pdf.set_font(FONT, "B", 11)
    pdf.set_xy(info_x, pdf.get_y())
    sev_display = labels.get("sev_values", {}).get(severity, severity)
    pdf.cell(info_w, 8, safe(f"  {labels['severity']} : {sev_display}  "), ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)

    # Cam label
    if has_cam:
        pdf.set_xy(pdf.l_margin, pdf.get_y())
        pdf.set_font(FONT, "I", 8)
        pdf.set_text_color(130, 130, 130)
        pdf.cell(cam_w, 5, safe(labels["gradcam"]), align="C", ln=True)

    pdf.ln(6)
    pdf.set_text_color(0, 0, 0)

    # ── Section heading helper ────────────────────────────────────────────────
    def section_heading(title):
        pdf.set_fill_color(45, 140, 45)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font(FONT, "B", 10)
        pdf.cell(0, 7, safe(f"  {title}"), ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

    # ── Treatment table ───────────────────────────────────────────────────────
    section_heading(labels["treatment_heading"])
    page_w  = pdf.w - pdf.l_margin - pdf.r_margin
    label_w = 58
    value_w = page_w - label_w

    rows = [
        (labels["cause"],     treatment.get("cause", "")),
        (labels["pesticide"], treatment.get("pesticide", "")),
        (labels["prevention"],treatment.get("prevention", "")),
        (labels["recovery"],  treatment.get("recovery", "")),
    ]

    for label, value in rows:
        y_start = pdf.get_y()
        pdf.set_font(FONT, "B", 10)
        pdf.set_fill_color(232, 245, 232)
        pdf.set_xy(pdf.l_margin, y_start)
        pdf.cell(label_w, 8, safe(label), border=1, fill=True, ln=0)
        pdf.set_font(FONT, "", 10)
        pdf.set_fill_color(255, 255, 255)
        pdf.set_xy(pdf.l_margin + label_w, y_start)
        pdf.multi_cell(value_w, 8, safe(value), border=1)

    pdf.ln(4)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        DB_PATH.parent.mkdir(exist_ok=True)
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            filename   TEXT,
            disease    TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            severity   TEXT    NOT NULL,
            plot_id    TEXT    DEFAULT '',
            cam_image  TEXT
        )
    """)
    con.commit()
    con.close()


@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()


# ── Leaf image validator ───────────────────────────────────────────────────────
def is_valid_leaf_image(raw_bytes):
    """
    Returns (is_valid, reason).
    Uses HSV color space to detect green/yellow-green leaf pixels.
    Rejects cameras, random objects, blank images, and non-leaf photos.
    """
    try:
        img = open_image_upright(raw_bytes).resize((224, 224))
        arr = np.array(img, dtype=np.uint8)

        # Check 1: brightness (widened to tolerate real-world phone lighting)
        brightness = arr.mean()
        if brightness < 12:
            return False, "Image is too dark. Please use better lighting."
        if brightness > 252:
            return False, "Image is too bright. Please avoid direct flash."

        # Check 2: not blank
        if arr.std() < 6:
            return False, "Image appears blank. Please upload a real leaf photo."

        # Check 3: HSV color-pixel ratio.
        # Two masks combined: (a) green/yellow-green for healthy/young tissue,
        # (b) a wider hue band that also picks up brown/tan/rust diseased
        # patches, dried edges, and leafminer trails, which real-world phone
        # photos of damaged leaves often show a lot more of than studio
        # dataset images do.
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        # Narrowed from the old (20,100) hue band, which bled into golds/
        # yellows and let things like logos/badges pass as "leaf-colored".
        green_mask = cv2.inRange(hsv, (35, 40, 30), (85, 255, 255))
        brown_mask = cv2.inRange(hsv, (8, 40, 20), (25, 200, 180))
        leaf_mask  = cv2.bitwise_or(green_mask, brown_mask)
        leaf_pct   = (leaf_mask > 0).sum() / (224 * 224) * 100

        if leaf_pct < 15:
            return False, "No leaf detected. Please upload a clear photo of a citrus leaf."

        return True, "OK"

    except Exception as e:
        return False, f"Could not read image: {str(e)}"


# ── Scene gate: is this even a plant/leaf photo at all? ────────────────────────
# Color heuristics (above) only check "does this photo contain green/brown
# pixels" — which almost anything can satisfy (laptops, code editors, wood
# furniture, walls, clothing, skin, etc). To actually catch "this is a
# laptop" or "this is a logo", we need a model that understands *objects*,
# not just colors. We use a small pretrained ImageNet classifier (ships
# with torchvision, no custom training data needed) as a coarse gate:
# if its top prediction is confidently a known non-plant object (laptop,
# keyboard, monitor, person, furniture, etc.), we reject before the photo
# ever reaches the 9-class disease model.
NON_LEAF_KEYWORDS = [
    # Electronics / screens
    "laptop", "notebook", "keyboard", "computer", "screen", "monitor", "television",
    "cellular", "phone", "desktop", "mouse", "printer", "remote control",
    "microphone", "loudspeaker", "hard disc",
    # Furniture / household
    "desk", "chair", "table", "furniture", "couch", "sofa", "lamp",
    # People / clothing
    "person", "face", "clothing", "shirt", "jean", "shoe", "sandal",
    # Stationery / paper objects (catches notebooks, registers, books)
    "book", "paper", "envelope", "web site", "menu", "binder", "pencil",
    # Vehicles
    "car", "vehicle", "truck", "bus", "bicycle",
    # Food / fruit (non-plant objects that are green or citrus-colored)
    "granny smith", "lemon", "orange", "apple", "banana", "cucumber",
    "tennis ball", "golf ball", "balloon",
]


@lru_cache(maxsize=1)
def load_scene_classifier():
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    model = mobilenet_v2(weights=weights)
    model.eval()
    return model, weights


def scene_gate_check(raw_bytes):
    """Returns (ok, reason).
    Rejects if any of the top-5 ImageNet predictions is a known non-plant
    object with cumulative confidence above a low threshold. Checking top-5
    (not just top-1) catches cases like the laptop where the correct class
    appears at rank 2-3 even though rank-1 has low absolute confidence."""
    try:
        model, weights = load_scene_classifier()
        preprocess = weights.transforms()
        img = open_image_upright(raw_bytes)
        x = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top5_prob, top5_idx = probs.topk(5)
        categories = weights.meta["categories"]
        top_labels = [categories[i].lower() for i in top5_idx.tolist()]
        top_probs  = top5_prob.tolist()

        print("SCENE GATE DEBUG:",
              list(zip(top_labels, [round(p, 3) for p in top_probs])))

        # Accumulate probability mass from non-leaf classes across all top-5.
        # If enough of the model's probability is going to known non-plant
        # objects, reject — even if no single prediction crosses 0.25.
        non_leaf_mass = 0.0
        top_non_leaf  = None
        for label, prob in zip(top_labels, top_probs):
            if any(kw in label for kw in NON_LEAF_KEYWORDS):
                non_leaf_mass += prob
                if top_non_leaf is None:
                    top_non_leaf = label

        # Reject thresholds (tuned from real test cases):
        # - single top-1 prediction clearly non-leaf  (>= 0.15)
        # - or cumulative non-leaf mass across top-5  (>= 0.20)
        top1_label, top1_prob = top_labels[0], top_probs[0]
        top1_is_non_leaf = any(kw in top1_label for kw in NON_LEAF_KEYWORDS)

        if top1_is_non_leaf and top1_prob >= 0.15:
            return False, top1_label
        if non_leaf_mass >= 0.20:
            return False, top_non_leaf or top1_label

        return True, top1_label
    except Exception as e:
        print("SCENE GATE ERROR (gate skipped, falling back to color-only check):", repr(e))
        return True, "scene-gate-unavailable"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/set_lang/<lang>")
def set_lang(lang):
    if lang in ("en", "hi", "mr"):
        session["lang"] = lang
    return redirect(request.referrer or url_for("index"))


@app.route("/manifest.json")
def pwa_manifest():
    """Served from root (not /static/) so it can declare scope: '/' cleanly."""
    return send_from_directory(BASE_DIR, "manifest.json", mimetype="application/manifest+json")


@app.route("/sw.js")
def service_worker():
    """Service worker MUST be served from root scope (not /static/sw.js) —
    otherwise its default scope would only cover /static/, and 'Add to
    Home Screen' installability would not apply to the whole app."""
    response = send_from_directory(BASE_DIR, "sw.js", mimetype="application/javascript")
    response.headers["Service-Worker-Allowed"] = "/"
    return response


@app.route("/")
def index():
    lang    = session.get("lang", "en")
    weather = get_weather_risk()
    ndvi_context = get_ndvi_context()
    return render_template("index.html", ui=UI[lang], lang=lang, weather=weather, ndvi=ndvi_context)


@app.route("/weather/data")
def weather_data():
    """JSON endpoint for live weather polling from the browser. Always hits
    the OpenWeatherMap API fresh (no caching) so it reflects real-time
    conditions, the way a live weather app would."""
    weather = get_weather_risk()
    if weather is None:
        return jsonify(available=False)
    weather["available"] = True
    weather["fetched_at"] = datetime.datetime.now().strftime("%H:%M:%S")
    return jsonify(weather)


@app.route("/predict", methods=["POST"])
def predict():
    lang = session.get("lang", "en")
    if "image" not in request.files:
        return jsonify(error="No image uploaded"), 400

    file    = request.files["image"]
    plot_id = request.form.get("plot_id", "")
    if not file.filename:
        return jsonify(error="Empty filename"), 400

    ts_str    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname     = f"{ts_str}_{file.filename}"
    save_path = UPLOAD_DIR / fname
    file.seek(0)
    raw_bytes = file.read()
    with open(save_path, "wb") as f:
        f.write(raw_bytes)

    def reject_response(reason):
        return jsonify(
            error=None,
            low_confidence=True,
            warning=reason,
            disease="Unknown",
            display="Not a Leaf Image",
            confidence="--",
            confidence_raw=0,
            severity="Unknown",
            sev_class="secondary",
            treatment={
                "cause": "Invalid image uploaded.",
                "pesticide": "N/A",
                "prevention": "Please upload a clear, close-up photo of a single citrus leaf.",
                "recovery": "N/A",
                "is_disease": False,
            },
            cam_image=None,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            plot_id=plot_id,
            top3=[],
            ui=UI[lang],
            lang=lang,
            filename=fname,
        )

    # Validate image before prediction — two gates:
    # 1) color heuristic (cheap, catches blank/dark/no-green-or-brown images)
    # 2) pretrained scene classifier (catches actual non-plant OBJECTS like
    #    laptops, logos, faces, furniture — things color alone can't rule out)
    valid, reason = is_valid_leaf_image(raw_bytes)
    if not valid:
        return reject_response(reason)

    scene_ok, scene_label = scene_gate_check(raw_bytes)
    if not scene_ok:
        return reject_response(
            f"This looks like a photo of \"{scene_label}\", not a citrus leaf. "
            "Please upload a clear, close-up photo of a single leaf."
        )

    img_tensor = preprocess_image(raw_bytes)

    model = load_model()
    with torch.no_grad():
        # Test-time augmentation: average predictions over the original
        # crop and a horizontal flip. This smooths out single-crop noise
        # that tends to hit harder on real-world/internet photos than on
        # clean dataset images, without requiring any retraining.
        logits_orig = model(img_tensor)
        flipped     = torch.flip(img_tensor, dims=[3])
        logits_flip = model(flipped)
        probs = (torch.softmax(logits_orig, dim=1)[0] +
                 torch.softmax(logits_flip, dim=1)[0]) / 2
    conf, idx  = probs.max(0)
    confidence = conf.item()
    disease    = CLASS_NAMES[idx.item()]

    try:
        img_tensor_grad = img_tensor.clone().requires_grad_(True)
        cam_b64 = generate_gradcam(model, img_tensor_grad, idx.item())
    except Exception:
        cam_b64 = None

    treatment           = TREATMENTS[disease]
    severity, sev_class = get_severity(confidence, treatment["is_disease"])
    display             = DISPLAY_NAMES[lang].get(disease, disease)
    timestamp           = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db = get_db()
    db.execute(
        "INSERT INTO predictions (timestamp, filename, disease, confidence, severity, plot_id, cam_image)"
        " VALUES (?,?,?,?,?,?,?)",
        (timestamp, fname, disease, confidence, severity, plot_id, cam_b64)
    )
    db.commit()

    top3_vals, top3_idx = probs.topk(3)
    top3 = [
        {"name": DISPLAY_NAMES[lang].get(CLASS_NAMES[i], CLASS_NAMES[i]),
         "prob": f"{v*100:.1f}%"}
        for v, i in zip(top3_vals.tolist(), top3_idx.tolist())
    ]

    # Low confidence warning
    low_confidence = confidence < 0.60
    warning = (
        "Low confidence - please upload a clear, close-up photo of a single "
        "citrus leaf with good lighting and plain background."
    ) if low_confidence else None

    return jsonify(
        disease=disease,
        display=display,
        confidence=f"{confidence*100:.1f}%",
        confidence_raw=confidence,
        low_confidence=low_confidence,
        warning=warning,
        severity=severity,
        sev_class=sev_class,
        treatment=treatment,
        cam_image=cam_b64,
        timestamp=timestamp,
        plot_id=plot_id,
        top3=top3,
        ui=UI[lang],
        lang=lang,
        filename=fname,
    )


@app.route("/report")
def download_report():
    lang       = session.get("lang", "en")
    disease    = request.args.get("disease", "Healthy_Leaf")
    confidence = float(request.args.get("confidence", 1.0))
    severity   = request.args.get("severity", "None")
    plot_id    = request.args.get("plot_id", "")
    filename   = request.args.get("filename", "")
    timestamp  = request.args.get("timestamp",
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    cam_b64 = None
    try:
        db  = get_db()
        row = db.execute(
            "SELECT cam_image FROM predictions WHERE filename=? ORDER BY id DESC LIMIT 1",
            (filename,)
        ).fetchone()
        if row:
            cam_b64 = row["cam_image"]
    except Exception:
        pass

    treatment = TREATMENTS.get(disease, TREATMENTS["Healthy_Leaf"])
    display   = DISPLAY_NAMES[lang].get(disease, disease)

    buf = make_pdf(disease, display, confidence, severity,
                   treatment, cam_b64, timestamp, plot_id, lang)
    safe_name = f"citrus_report_{disease}_{timestamp[:10]}.pdf"
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name=safe_name)


@app.route("/history")
def history():
    lang = session.get("lang", "en")
    db   = get_db()
    rows = db.execute(
        "SELECT id, timestamp, disease, confidence, severity, plot_id, filename"
        " FROM predictions ORDER BY id DESC LIMIT 100"
    ).fetchall()
    records = [dict(r) for r in rows]
    for r in records:
        r["display"]        = DISPLAY_NAMES[lang].get(r["disease"], r["disease"])
        r["confidence_pct"] = f"{r['confidence']*100:.1f}%"
    return render_template("history.html", ui=UI[lang], lang=lang, records=records)


@app.route("/history/data")
def history_data():
    db   = get_db()
    rows = db.execute(
        "SELECT disease, COUNT(*) as cnt FROM predictions GROUP BY disease ORDER BY cnt DESC"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Wipes all prediction records (and their saved photos) so the app
    can be reset to a clean slate before a demo/evaluation run."""
    db   = get_db()
    rows = db.execute("SELECT filename FROM predictions").fetchall()

    db.execute("DELETE FROM predictions")
    db.commit()

    # Best-effort cleanup of the saved leaf images on disk. Failures here
    # (e.g. a file already missing) shouldn't block clearing the DB.
    for r in rows:
        fname = r["filename"]
        if not fname:
            continue
        try:
            (UPLOAD_DIR / fname).unlink(missing_ok=True)
        except Exception:
            pass

    return redirect(url_for("history"))


# ── Startup ───────────────────────────────────────────────────────────────────
with app.app_context():
    init_db()

if __name__ == "__main__":
    init_db()
    # host="0.0.0.0" makes the server reachable from other devices on the
    # same network (e.g. your phone) at http://<your-laptop-LAN-IP>:5000
    #
    # PORT env var lets this same command work unchanged on platforms like
    # Render that assign their own port. FLASK_DEBUG defaults OFF so a
    # forgotten `python app.py` in production doesn't expose the debugger;
    # set FLASK_DEBUG=1 in your local .env for the interactive reloader.
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", debug=debug, port=port)
