# typhoon_ocr_to_excel.py
import os, json, re
from typing import List, Optional, Tuple
import pandas as pd
from bs4 import BeautifulSoup
import markdown as md
from dotenv import load_dotenv
from pathlib import Path
from typhoon_ocr import ocr_document
from rapidfuzz import process, fuzz

# -------------------------------------------------------------
# Load .env from the same directory as this script
# -------------------------------------------------------------
ENV_PATH = Path(__file__).with_name('.env')
load_dotenv(dotenv_path=ENV_PATH)

# -------------------- small utils --------------------
def _ensure_obj(result):
    if isinstance(result, dict):
        return result
    try:
        return json.loads(result)
    except Exception:
        return {"natural_text": str(result)}

def _norm_space(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def _clean_html(text: str) -> str:
    """strip simple HTML tags that might leak from OCR (e.g., <td>xxx</td>)"""
    return re.sub(r"<[^>]+>", " ", text or "")

def _strip_invisibles(s: str) -> str:
    """remove zero-width / BOM / normalize weird dashes"""
    if not s:
        return s
    s = s.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")
    # normalize various dashes to hyphen-minus
    s = (s.replace("\u2010", "-").replace("\u2011", "-")
           .replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-"))
    return s

def _merge_thai_linebreaks(s: str) -> str:
    """
    รวมคำไทยที่ถูกตัดด้วยขึ้นบรรทัดใหม่ให้กลับมาติดกัน
    - ไทย↔ไทย ข้าม \n หรือ \r\n จะติดกัน
    - ไทย↔อังกฤษ/ตัวเลขยังคงเว้นวรรค
    """
    if not s:
        return s
    th = r"\u0E00-\u0E7F"
    # ไทย + (space/newline) + ไทย -> ติดกัน
    s = re.sub(rf"([{th}])\s*[\r\n]+\s*([{th}])", r"\1\2", s)
    return s

def _format_th_en_spacing(s: str) -> str:
    """
    จัดช่องว่าง ไทย↔อังกฤษ/ตัวเลข และรอบวงเล็บ/ขีดกลาง/คอมมา
    - space ก่อน "(" และหลัง ")"
    - normalize " - " (ขีดกลางมีช่องว่างสองข้าง)
    - แยกไทยกับอังกฤษ/ตัวเลข
    - normalize comma และคอมมาก่อนปิดวงเล็บ
    """
    if not s:
        return s
    s = s.replace("\n", " ")

    # space before "(" and after ")"
    s = re.sub(r"([^\s(])\(", r"\1 (", s)
    s = re.sub(r"\)([^\s)\.,;:])", r") \1", s)

    # normalize hyphen spacing to " - "
    s = re.sub(r"\s*-\s*", " - ", s)

    # separate Thai <-> Latin/number (both directions)
    th = r"\u0E00-\u0E7F"
    latnum = r"A-Za-z0-9"
    s = re.sub(rf"([{th}])([{latnum}])", r"\1 \2", s)
    s = re.sub(rf"([{latnum}])([{th}])", r"\1 \2", s)

    # normalize commas
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r",\s*\)", ")", s)
    s = re.sub(r",\s*$", "", s)

    # collapse multiple spaces
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()

def _tighten_thai_words(s: str) -> str:
    """
    บีบช่องว่าง ‘ภายในวงเล็บ’ และเคสคำไทยที่มักเป็นคำเดี่ยวแต่โดนแยกด้วย space
    - (ไม่รวม มอเตอร์ไซค์) -> (ไม่รวมมอเตอร์ไซค์)
    - ต่อ เนื่อง -> ต่อเนื่อง
    - บีบช่องว่างเฉพาะ ‘ในวงเล็บ’ ระหว่างอักษรไทย
    """
    if not s:
        return s
    th = r"\u0E00-\u0E7F"

    # บีบคำไทยภายในวงเล็บให้ติดกัน แต่ยังเว้นวรรคไทย↔อังกฤษ/ตัวเลข
    def _fix_paren(m):
        inner = m.group(1)
        inner = re.sub(rf"([{th}])\s+([{th}])", r"\1\2", inner)        # ไทย+ไทย -> ติด
        inner = re.sub(rf"([{th}])\s+([A-Za-z0-9])", r"\1 \2", inner)  # ไทย↔EN/เลข -> 1 space
        inner = re.sub(rf"([A-Za-z0-9])\s+([{th}])", r"\1 \2", inner)
        inner = re.sub(r"[ ]{2,}", " ", inner)
        return f"({inner.strip()})"

    s = re.sub(r"\(\s*(.*?)\s*\)", _fix_paren, s)

    # ‘ไม่รวม’ + ไทย -> ติดกัน (ไม่รวมมอเตอร์ไซค์)
    s = re.sub(rf"(ไม่รวม)\s+([{th}])", r"\1\2", s)

    # คำเดี่ยวที่มักโดนแยก
    s = re.sub(r"ต่อ\s+เนื่อง", "ต่อเนื่อง", s)

    return s

def _thai_norm(s: str) -> str:
    """normalize เบา ๆ ก่อน matching"""
    s = _strip_invisibles(s)
    s = _merge_thai_linebreaks(s)   # รวมคำไทยที่ถูกตัดบรรทัดให้ติดกันก่อน
    s = _norm_space(s)
    s = _format_th_en_spacing(s)
    s = _tighten_thai_words(s)      # บีบช่องว่างในวงเล็บ/คำที่ควรติดกัน
    return s

def _tables_from_markdown(natural_text: str):
    html = md.markdown(natural_text, extensions=['tables'])
    soup = BeautifulSoup(html, 'lxml')
    tables = []
    for t in soup.find_all('table'):
        try:
            dfs = pd.read_html(str(t))
            tables.extend(dfs)
        except Exception:
            pass
    return tables

def _normalize_cols(df: pd.DataFrame):
    mapping = {}
    for c in df.columns:
        k = re.sub(r"[^A-Za-z0-9ก-๙\- ]", "", str(c)).strip().lower()
        if k in ["no", "no.","ลำดับ"]:
            mapping[c] = "No"
        elif "coverage" in k or "ความคุ้มครอง" in k:
            mapping[c] = "Coverage"
        elif "member" in k and ("co" in k or "copay" in k or "co-pay" in k):
            mapping[c] = "Member Co-Pay"
        elif "limit" in k or "ผลประโยชน์สูงสุด" in k:
            mapping[c] = "Limit"
        elif "balance" in k or "คงเหลือ" in k:
            mapping[c] = "Balance"
        else:
            mapping[c] = c
    return df.rename(columns=mapping)

# ---- number helpers (thai digits & extraction) ----
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

def _to_arabic_digits(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.translate(_THAI_DIGITS)

def _extract_numbers(s: str) -> list[str]:
    s = _to_arabic_digits(s)
    return re.findall(r"\d+", s)

# -------------------- template matching --------------------
def _load_templates(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"⚠️  Template file not found: {path} (skip template matching)")
        return []
    encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be"]
    text = ""
    for enc in encodings:
        try:
            text = p.read_text(encoding=enc)
            break
        except Exception:
            continue
    if not text:
        print(f"⚠️  Could not read {path} in utf-8/utf-16")
        return []
    templates = []
    for line in text.splitlines():
        t = _thai_norm(line)
        if t:
            templates.append(t)
    return templates

def _match_template(
    text: str,
    templates: List[str],
    threshold: int = 88,
    enforce_number_match: bool = True
) -> Tuple[str, float, bool]:
    """
    คืน (ข้อความที่เลือก, คะแนน, matched?)
    - ถ้าคะแนนถึง threshold และ enforce_number_match=True:
        * ตรวจตัวเลขใน raw กับ best_template ต้อง "เท่ากัน" (หลังแปลงเลขไทย->อารบิก)
        * ถ้าเลขต่างกัน -> บังคับไม่ match (คืน raw)
    """
    raw = _thai_norm(text)
    if not raw or not templates:
        return raw, 0.0, False

    best = process.extractOne(raw, templates, scorer=fuzz.token_sort_ratio)
    if not best:
        return raw, 0.0, False

    best_text, score, _ = best
    matched = score >= threshold

    if matched and enforce_number_match:
        nums_raw = set(_extract_numbers(raw))
        nums_tpl = set(_extract_numbers(best_text))
        if nums_raw and nums_tpl and nums_raw != nums_tpl:
            return raw, float(score), False

    return (best_text if matched else raw), float(score), matched

# -------------------- Policy/Card extraction (หน้าแรกเท่านั้น) --------------------
POLICY_LABELS = [
    r"Policy\s*No\.?", r"Policy\s*Number", r"Policy\s*ID",
    r"เลขที่\s*กรมธรรม์", r"เลขกรมธรรม์", r"กรมธรรม์\s*เลขที่",
    r"เลขที่\s*กรมธรรม์ประกันภัย", r"เลขที่\s*กรมธรรม์\s*ประกันภัย"
]
CARD_LABELS = [
    r"Card\s*No\.?", r"Card\s*Number",
    r"เลขที่\s*บัตร", r"หมายเลข\s*บัตร", r"หมายเลขบัตร", r"เลขบัตร"
]

# Card: ปกติและหลวม (ยอมคั่นแปลก ๆ/ไม่มีคั่น)
CARD_VALUE_STRICT = r"\d{5}-\d{3}-\d{9}"              # 14048-108-240005162
CARD_VALUE_LOOSE  = r"(\d{5})[^\d]{0,3}(\d{3})[^\d]{0,3}(\d{9})"

# Policy: ตัวอักษร/ตัวเลข/ขีด + optional _digits  e.g., IHA4001YC_13 หรือ 14048-108-240005162
POLICY_VALUE = r"[A-Z0-9\-]{5,}(?:_[0-9]+)?"

def _find_first(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1) if m else None

def _normalize_ocr_block(s: str) -> str:
    return _norm_space(_clean_html(_strip_invisibles(s)))

def _extract_policy_card(natural_text_first_page: str) -> Tuple[str, str]:
    """
    ดึง Policy/Card จาก 'หน้าแรกเท่านั้น' ให้เสถียร:
    1) หาแบบ label: strict
    2) label: loose (เฉพาะ Card)
    3) next-line strict/loose (ภายใน 3 บรรทัด)
    4) สแกนทั้งหน้าแบบ loose
    5) หาในตารางของหน้าแรกแบบ loose
    """
    text = _normalize_ocr_block(natural_text_first_page)

    # ---- Policy ----
    pol = _find_first(rf"(?:{'|'.join(POLICY_LABELS)})\s*[:：]?\s*({POLICY_VALUE})", text)
    if not pol:
        lines = [_normalize_ocr_block(x) for x in natural_text_first_page.splitlines()]
        for i, l in enumerate(lines):
            if re.search(rf"(?:{'|'.join(POLICY_LABELS)})", l, flags=re.IGNORECASE):
                for j in range(i+1, min(i+4, len(lines))):
                    v = _find_first(rf"({POLICY_VALUE})", lines[j])
                    if v:
                        pol = v
                        break
            if pol:
                break
    pol = (pol or "").strip()

    # ---- Card (หน้าแรกเท่านั้น, robust) ----
    # 1) label + strict
    card = _find_first(rf"(?:{'|'.join(CARD_LABELS)})\s*[:：]?\s*({CARD_VALUE_STRICT})", text)

    # 2) label + loose
    if not card:
        m = re.search(rf"(?:{'|'.join(CARD_LABELS)})\s*[:：]?\s*{CARD_VALUE_LOOSE}", text, flags=re.IGNORECASE)
        if m:
            card = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 3) next-line strict/loose
    if not card:
        lines = [_normalize_ocr_block(x) for x in natural_text_first_page.splitlines()]
        for i, l in enumerate(lines):
            if re.search(rf"(?:{'|'.join(CARD_LABELS)})", l, flags=re.IGNORECASE):
                for j in range(i+1, min(i+4, len(lines))):
                    v = _find_first(rf"({CARD_VALUE_STRICT})", lines[j])
                    if v:
                        card = v
                        break
                    m = re.search(CARD_VALUE_LOOSE, lines[j])
                    if m:
                        card = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                        break
            if card:
                break

    # 4) สแกนทั้งหน้าแบบ loose
    if not card:
        m = re.search(CARD_VALUE_LOOSE, text)
        if m:
            card = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 5) หาในตาราง (หน้าแรก)
    if not card:
        for df in _tables_from_markdown(natural_text_first_page):
            for val in df.astype(str).values.flatten():
                valc = _normalize_ocr_block(val)
                m = re.search(CARD_VALUE_LOOSE, valc)
                if m:
                    card = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    break
            if card:
                break

    return pol, (card or "").strip()

# -------------------- core helpers --------------------
def _resolve_pdf_path(pdf_arg: str) -> str:
    if os.path.exists(pdf_arg):
        return pdf_arg
    candidate = os.path.join("data", os.path.basename(pdf_arg))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"File not found: {pdf_arg} or {candidate}")

def _filter_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["No"].astype(str).str.match(r"(?i)^\s*(no\.?|ลำดับ)\s*$")]

def _only_numeric_no(df: pd.DataFrame) -> pd.DataFrame:
    """เก็บเฉพาะแถวที่ No เป็นตัวเลขจริง เพื่อตัด noise ช่วงข้ามหน้า"""
    def is_int(x):
        s = str(x).strip()
        return s.isdigit()
    return df[df["No"].map(is_int)]

def _filter_no_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure No is non-decreasing; drop rows where No jumps backward (cross-page noise)."""
    def safe_int(x):
        try:
            return int(str(x).strip())
        except:
            return None
    df = df.copy()
    df["__NoInt"] = df["No"].map(safe_int)

    cleaned = []
    last = -10**9
    for _, row in df.iterrows():
        n = row["__NoInt"]
        if n is None or n >= last:
            cleaned.append(row)
            if n is not None:
                last = n
        # else: drop
    out = pd.DataFrame(cleaned)
    return out.drop(columns=["__NoInt"], errors="ignore")

# -------------------- main --------------------
def extract_pdf_to_excel(pdf_path: str, out_path: Optional[str] = None,
                         base_url: str = None, api_key: str = None,
                         templates_path: Optional[str] = "template/master-data-wording.txt",
                         sim_threshold: int = 88,
                         enforce_number_match: bool = True):
    base_url = base_url or os.getenv("TYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key  = api_key  or os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY")

    pdf_path = _resolve_pdf_path(pdf_path)
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    if not out_path:
        out_dir = "xls"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{basename}.xlsx")
    else:
        if out_path.endswith(os.sep) or os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, f"{basename}.xlsx")
        else:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    templates = _load_templates(templates_path)

    rows: List[dict] = []
    policy, card = "", ""
    wrote_policy_card = False

    page = 0
    while True:
        try:
            res = ocr_document(
                pdf_or_image_path=pdf_path,
                task_type="structure",
                page_num=page,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception:
            if page == 0:
                raise
            break

        obj = _ensure_obj(res)
        md_text = obj.get("natural_text", "")

        if page == 0:
            policy, card = _extract_policy_card(md_text)
        # หน้าถัด ๆ ไปไม่ยุ่งกับ policy/card อีก (ป้องกันถูกทับ)

        for df in _tables_from_markdown(md_text):
            df = _normalize_cols(df)
            need = {"No","Coverage","Limit","Balance"}
            if len(need - set(df.columns)) != 0:
                continue

            if "Member Co-Pay" not in df.columns:
                df["Member Co-Pay"] = ""
            df = df[["No","Coverage","Member Co-Pay","Limit","Balance"]]

            # cleanup rows
            df = _filter_header_rows(df)
            df = _only_numeric_no(df)  # keep only numeric No

            first_row_this_table = True
            for _, r in df.iterrows():
                cov_raw = _thai_norm(str(r["Coverage"]))
                cov_best, score, matched = _match_template(
                    cov_raw, templates, threshold=sim_threshold, enforce_number_match=enforce_number_match
                )

                rows.append({
                    "Policy No": (policy if not wrote_policy_card and first_row_this_table else ""),
                    "Card No":   (card   if not wrote_policy_card and first_row_this_table else ""),
                    "No": _norm_space(r["No"]),
                    "Coverage": cov_best,                         # ถ้า match จะได้ข้อความตาม master
                    "Member Co-Pay": _norm_space(r["Member Co-Pay"]),
                    "Limit": _norm_space(r["Limit"]),
                    "Balance": _norm_space(r["Balance"]),
                    "Coverage (raw)": cov_raw,
                    "MatchScore": f"{score:.0f}",
                    "Matched": "TRUE" if matched else "FALSE",
                })
                first_row_this_table = False
            if not wrote_policy_card and len(rows) > 0:
                wrote_policy_card = True

        page += 1

    out_cols = ["Policy No","Card No","No","Coverage","Member Co-Pay","Limit","Balance","Coverage (raw)","MatchScore","Matched"]
    out = pd.DataFrame(rows, columns=out_cols)
    if out.empty:
        raise RuntimeError("ไม่พบตารางผลประโยชน์")

    # de-dup + sequence check
    out = out.drop_duplicates(subset=["No","Coverage","Member Co-Pay","Limit","Balance"], keep="first").reset_index(drop=True)
    out = _filter_no_sequence(out)

    out.to_excel(out_path, index=False)
    print(f"✅ Saved: {out_path}")

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="ไฟล์ PDF หรือชื่อไฟล์ใน ./data")
    ap.add_argument("--out", default="", help="path สำหรับ .xlsx หรือโฟลเดอร์ (เช่น xls/)")
    ap.add_argument("--base_url", default=os.getenv("TYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    ap.add_argument("--api_key",  default=os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--templates", default="template/master-data-wording.txt", help="ไฟล์ master wording")
    ap.add_argument("--sim", type=int, default=80, help="เกณฑ์ fuzzy match (0-100)")
    ap.add_argument(
        "--no_strict_numbers",
        action="store_true",
        help="ปิดการตรวจเลขในข้อความ (ค่าเริ่มต้น: ตรวจเลขให้ตรงกัน)"
    )
    args = ap.parse_args()

    extract_pdf_to_excel(
        args.pdf,
        args.out,
        base_url=args.base_url,
        api_key=args.api_key,
        templates_path=args.templates,
        sim_threshold=args.sim,
        enforce_number_match=not args.no_strict_numbers
    )
