# FastAPI OCR: RapidOCR (ONNX, CPU) + spaCy NER + layout logic
# POST /ocr  -> { ok, fullName, company, text }

import os, re
from io import BytesIO
from typing import List, Tuple
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from rapidocr_onnxruntime import RapidOCR         # fast, CPU-only

import spacy                                      # NER to classify PERSON vs ORG
nlp = spacy.load("en_core_web_sm")                # free English model

# Optional handwriting fallback (turn on with env var ENABLE_HANDWRITING=1)
ENABLE_HANDWRITING = os.getenv("ENABLE_HANDWRITING", "0") == "1"
try:
    import easyocr
    eocr = easyocr.Reader(['en'], gpu=False) if ENABLE_HANDWRITING else None
except Exception:
    eocr = None

ocr = RapidOCR()                                  # loads ONNX models lazily

EVENT_TERMS = {
  "EXHIBITOR","ATTENDEE","SPEAKER","STAFF","VIP","MEDIA","PASS","CONFERENCE",
  "EVENT","EXPO","REGISTRATION","GUEST","SPONSOR","PAVILION","HALL","ROOM","BADGE"
}
TITLE_TERMS = {
  "SALES","DISTRICT","MANAGER","DIRECTOR","ENGINEER","COORDINATOR","PRODUCER",
  "TECHNICIAN","SPECIALIST","DESIGNER","DEVELOPER","MARKETING","ACCOUNT",
  "REPRESENTATIVE","PRESIDENT","VICE","VP","OFFICER","LEAD","STUDENT","PROFESSOR",
  "SUPPORT","OPERATIONS","OWNER","FOUNDER","CEO","CFO","COO","CTO"
}
COMPANY_HINT = {
  "INC","LLC","LTD","GMBH","BV","SA","AG","SARL","CORP","CO","COMPANY","GROUP",
  "HOLDINGS","STUDIOS","SYSTEMS","SOLUTIONS","TECHNOLOGIES","TECHNOLOGY",
  "PRODUCTIONS","PRODUCTION","THEATRE","THEATER","AUDIO","LIGHTING","INTERCOM",
  "COMMUNICATIONS","LABS","ENTERPRISES"
}

def crop_header(img: Image.Image) -> Image.Image:
    """Auto-crop the saturated (red) header band typical on event badges."""
    w, h = img.size
    scan_h = min(int(h*0.35), h)
    arr = np.asarray(img.convert("RGB"))[:scan_h, :, :]
    mx = arr.max(axis=2).astype(np.float32)
    mn = arr.min(axis=2).astype(np.float32)
    sat = np.divide(mx - mn, np.maximum(mx, 1.0), out=np.zeros_like(mx), where=mx>0)
    sat_row = sat.mean(axis=1)
    HIGH, LOW, RUN = 0.35, 0.18, max(40, h//30)
    boundary = int(h*0.15)
    for y in range(5, scan_h - RUN - 5, 2):
        if sat_row[max(0,y-3):y].mean() > HIGH and sat_row[y:y+RUN:2].mean() < LOW:
            boundary = y + RUN//2
            break
    top = max(0, boundary - 8)
    return img.crop((0, top, w, h))

def normalize(s: str) -> str:
    s = re.sub(r'([A-Za-z])1([A-Za-z])', r'\1I\2', s)     # fix I/1 confusion
    return re.sub(r'\s{2,}', ' ', s).strip()

def looks_name(tokens: List[str]) -> bool:
    if not (1 <= len(tokens) <= 3): return False
    if any(re.search(r'\d', t) for t in tokens): return False
    ok = 0
    for t in tokens:
        if re.match(r"^[A-Z][a-z'’.-]+$", t) or re.match(r"^[A-Z]{2,4}$", t):
            ok += 1
    return ok == len(tokens)

def title_case_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b([a-z])([a-z'’-]*)", lambda m: m.group(1).upper()+m.group(2), s)
    s = re.sub(r"\b(Mc|Mac)([a-z])", lambda m: m.group(1)+m.group(2).upper(), s)
    s = re.sub(r"\bO'([a-z])", lambda m: "O'"+m.group(1).upper(), s)
    return s

def company_case(s: str) -> str:
    parts = []
    for w in s.split():
        parts.append(w if re.match(r'^[A-Z]{2,}$', w) else w.capitalize())
    return ' '.join(parts)

def label_line(text: str):
    """
    Label a line with spaCy NER + keyword priors.
    Returns (label, score) where label ∈ {PERSON, ORG, EVENT, TITLE, OTHER}.
    """
    t = normalize(text)
    u = t.upper()
    if any(term in u.split() for term in EVENT_TERMS): return "EVENT", 0.95
    if any(term in u.split() for term in TITLE_TERMS): return "TITLE", 0.90
    doc = nlp(t)
    lbl, score = "OTHER", 0.0
    for ent in doc.ents:
        if ent.label_ == "PERSON": lbl, score = "PERSON", max(score, 0.85)
        elif ent.label_ == "ORG":  lbl, score = "ORG", max(score, 0.80)
    if lbl != "PERSON" and any(k in u.split() for k in COMPANY_HINT):
        return "ORG", max(score, 0.82)
    return lbl, score

def pick_fields(lines: List[Tuple[float,float,str]]) -> Tuple[str,str]:
    # classify lines
    labeled = []
    for y0,h,t in lines:
        lab, sc = label_line(t)
        labeled.append((y0,h,t,lab,sc))

    # NAME: biggest PERSON-looking line + optional near-below surname line
    persons = [l for l in labeled if l[3] == "PERSON" or looks_name(l[2].split())]
    persons.sort(key=lambda x: (x[1], x[4]), reverse=True)
    name1 = persons[0] if persons else None
    name_tokens = []
    if name1:
        name_tokens += name1[2].split()
        y0,h = name1[0], name1[1]
        near = [l for l in labeled if l[0] > y0 and (l[0]-y0) < h*3.0]
        near_names = [l for l in near if l[3] == "PERSON" or looks_name(l[2].split())]
        if near_names:
            name_tokens += near_names[0][2].split()
    full_name = title_case_name(' '.join(name_tokens).strip()) if name_tokens else ""

    # COMPANY: prefer ORG below the name; avoid EVENT/TITLE
    ref_y = (name1[0] if name1 else -1)
    below = [l for l in labeled if l[0] > ref_y and l[3] not in ("EVENT","TITLE")]
    above = [l for l in labeled if (name1 and l[0] < name1[0]) and l[3] not in ("EVENT","TITLE")]

    def pick_company(cands):
        cands.sort(key=lambda x: (-1 if x[3]=="ORG" else 0, -x[4], x[0]))
        for y0,h,t,lab,sc in cands:
            toks = [w for w in t.split() if w.lower() not in {w.lower() for w in name_tokens}]
            if not toks: continue
            s = ' '.join(toks)
            if lab == "ORG" or any(k in t.upper().split() for k in COMPANY_HINT) or re.search(r"^[A-Z&]{2,}$", t.replace(' ','&')):
                return company_case(s)
        return ""

    company = pick_company(below) or pick_company(above)
    return full_name, company

def run_rapid_ocr(img: Image.Image):
    """Run RapidOCR; return (full_text, lines[y0,h,text])."""
    result, _ = ocr(img)  # result: [ [ [x,y]... ], text, conf ], ...
    lines, texts = [], []
    for item in (result or []):
        text = item[1]
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = item[0]
        y0, y1m = min(y1,y2,y3,y4), max(y1,y2,y3,y4)
        h = y1m - y0
        t = normalize(text)
        if not t: continue
        texts.append(t); lines.append((y0, h, t))
    full_text = '\n'.join(t for t in texts)
    return full_text, lines

app = FastAPI(title="Badge OCR API")
# During testing allow all; later set allow_origins to your HubSpot domain.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.post("/ocr")
async def ocr_badge(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        img = Image.open(BytesIO(raw)).convert("RGB")
        img = crop_header(img)                # remove red header band
        text, lines = run_rapid_ocr(img)
        full, company = pick_fields(lines)

        # Optional handwriting fallback
        if ENABLE_HANDWRITING and eocr and (not full or not company):
            res = eocr.readtext(raw, detail=1, paragraph=False)
            lines2 = []
            for (bbox, t, conf) in res:
                y0 = min(p[1] for p in bbox); y1 = max(p[1] for p in bbox); h = y1 - y0
                t = normalize(t)
                if t: lines2.append((y0, h, t))
            full2, company2 = pick_fields(lines2)
            full = full or full2
            company = company or company2
            text += '\n' + '\n'.join(t for _,_,t in sorted(lines2, key=lambda x: x[0]))

        return JSONResponse({"ok": True, "fullName": full, "company": company, "text": text})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})
