import io
from typing import Dict
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# OCR (CPU)
from paddleocr import PaddleOCR

# Tiny LLM (CPU)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------------------- lazy loaders (cached) --------------------
@st.cache_resource(show_spinner=False)
def load_ocr():
    # English OCR; CPU only
    return PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)

@st.cache_resource(show_spinner=False)
def load_llm():
    model_name = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, mdl


# -------------------- utils --------------------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def detect_chart_type(bgr: np.ndarray) -> str:
    """Heuristic chart classifier: pie / bar / histogram / line / other (CPU-fast)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(gray, 50, 150)

    # pie (circle)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(bgr.shape[:2])//4, param1=100, param2=30,
        minRadius=30, maxRadius=0
    )
    if circles is not None and len(circles[0]) >= 1:
        return "pie"

    # bars
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - thr
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_vert, iterations=1)
    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 80]
    tall_rects = [r for r in rects if r[3] > r[2] * 1.2]

    if len(tall_rects) >= 3:
        xs = sorted([r[0] for r in tall_rects])
        span = (max(xs) - min(xs)) if xs else 0
        packed = span / max(1, len(xs)) < 25
        return "histogram" if packed else "bar"

    # line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=30, maxLineGap=10)
    if lines is not None and len(lines) > 25:
        return "line"

    return "other"

def bar_like_peaks(bgr: np.ndarray) -> Dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - thr
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_vert, iterations=1)
    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = sorted([cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 80],
                   key=lambda r: r[0])

    n = len(rects)
    if n == 0:
        return {"n_bars": 0}

    heights = np.array([h for (_, _, _, h) in rects], dtype=np.float32)
    xs = np.array([x for (x, _, _, _) in rects], dtype=np.float32)
    left_half = xs < (xs.min() + xs.max()) / 2
    left_share = float(left_half.sum()) / n
    tallest_idx = int(np.argmax(heights))

    return {"n_bars": int(n),
            "left_share": round(left_share, 3),
            "max_height_index": tallest_idx}

def extract_text(img: Image.Image, ocr) -> Dict[str, str]:
    result = ocr.ocr(np.array(img), cls=True)
    lines = []
    if result and result[0]:
        for line in result[0]:
            text = line[1][0]
            if text and text.strip():
                lines.append(text.strip())
    title = lines[0] if lines else ""
    subtitle = lines[1] if len(lines) > 1 else ""
    return {
        "title": title,
        "subtitle": subtitle,
        "raw_text": " ".join(lines)
    }

def llm_phrase(findings: Dict, tokenizer, model, style="concise") -> str:
    chart_type = findings.get("chart_type", "chart")
    text = findings.get("ocr", {}).get("raw_text", "")
    insights = findings.get("insights", "")
    prompt = f"""
You are a data analyst. Write a {style} 2–3 sentence interpretation of a {chart_type}.
Base your statements on the OCR text and the hints. Avoid inventing numbers that are not present.

OCR text:
{text}

Hints:
{insights}

Output:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=140, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# -------------------- UI --------------------
st.set_page_config(page_title="Chart Interpreter (CPU)", layout="wide")
st.title("📊 Chart Interpreter — CPU-only")
st.caption("Upload a chart image. The app uses PaddleOCR + light CV + FLAN-T5-small to produce a plain-English interpretation (no paid APIs).")

uploaded = st.file_uploader("Upload a chart image", type=["png", "jpg", "jpeg", "webp", "bmp"])

if uploaded is None:
    st.info("Choose a chart image to begin.")
    st.stop()

img = Image.open(uploaded)
st.image(img, caption="Input", use_column_width=True)

with st.spinner("Analyzing…"):
    ocr = load_ocr()
    tok, mdl = load_llm()
    ocr_res = extract_text(img, ocr)
    bgr = pil_to_cv(img)
    ctype = detect_chart_type(bgr)

    addl = {}
    if ctype in ("bar", "histogram"):
        addl = bar_like_peaks(bgr)

    # readable hints (also shown to user for transparency)
    rule_notes = []
    if ocr_res.get("title"):
        rule_notes.append(f"Title suggests: {ocr_res['title']}.")
    if ctype in ("bar", "histogram") and addl.get("n_bars", 0) >= 3:
        if addl["left_share"] >= 0.6:
            rule_notes.append("Bars cluster to the left (more low values).")
        else:
            rule_notes.append("Bars are fairly spread across the range.")
    rules_text = " ".join(rule_notes)

    findings = {
        "chart_type": ctype,
        "ocr": ocr_res,
        "insights": f"Detected={ctype}; bar_stats={addl}; rules='{rules_text}'"
    }

    try:
        interpretation = llm_phrase(findings, tok, mdl)
    except Exception:
        interpretation = rules_text or "The chart shows a clear high-level pattern based on the detected layout and text."

st.subheader("📝 Interpretation")
st.write(interpretation)

with st.expander("Show OCR & heuristic details"):
    st.json({"chart_type": ctype, "ocr": ocr_res, "bar_like_stats": addl})
