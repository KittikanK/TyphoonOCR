import os, json, re
from typing import List, Optional, Tuple
import pandas as pd
from bs4 import BeautifulSoup
import markdown as md
from dotenv import load_dotenv
from pathlib import Path
from typhoon_ocr import ocr_document

# โหลด .env จากโฟลเดอร์เดียวกับสคริปต์เสมอ
ENV_PATH = Path(__file__).with_name('.env')
load_dotenv(dotenv_path=ENV_PATH)

# -------------------- helpers --------------------
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
        if k in ["no", "no."]:
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

# -------------------- policy/card extraction --------------------
POLICY_LABELS = [
    r"Policy\s*No\.?", r"Policy\s*Number", r"Policy\s*ID",
    r"เลขที่\s*กรมธรรม์", r"เลขกรมธรรม์", r"กรมธรรม์\s*เลขที่",
    r"เลขที่\s*กรมธรรม์ประกันภัย", r"เลขที่\s*กรมธรรม์\s*ประกันภัย"
]
CARD_LABELS = [
    r"Card\s*No\.?", r"Card\s*Number",
    r"เลขที่\s*บัตร", r"หมายเลข\s*บัตร", r"หมายเลขบัตร", r"เลขบัตร"
]

def _find_label_value_inline(text: str, label_patterns: List[str]) -> Optional[str]:
    for pat in label_patterns:
        m = re.search(rf"(?:{pat})\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9\-\/\s]{{4,}})", text, flags=re.IGNORECASE)
        if m:
            val = _norm_space(m.group(1))
            val = re.split(r"\s{2,}|\t", val)[0].strip()
            return val
    return None

def _find_label_value_next_line(lines: List[str], label_patterns: List[str]) -> Optional[str]:
    pats = [re.compile(p, re.IGNORECASE) for p in label_patterns]
    for i, line in enumerate(lines):
        s = _norm_space(line)
        if not s:
            continue
        if any(p.search(s) for p in pats):
            for j in range(i+1, min(i+4, len(lines))):
                nxt = _norm_space(lines[j])
                if not nxt:
                    continue
                m = re.search(r"([A-Za-z0-9][A-Za-z0-9\-\/\s]{4,})", nxt)
                if m:
                    val = _norm_space(m.group(1))
                    val = re.split(r"\s{2,}|\t", val)[0].strip()
                    return val
    return None

def _find_label_value_in_tables(dfs: List[pd.DataFrame], label_patterns: List[str]) -> Optional[str]:
    pats = [re.compile(p, re.IGNORECASE) for p in label_patterns]
    for df in dfs:
        try:
            sdf = df.copy()
            sdf.columns = [_norm_space(c) for c in sdf.columns]

            # label:value ในเซลล์เดียว
            for i in range(sdf.shape[0]):
                for j in range(sdf.shape[1]):
                    cell = _norm_space(sdf.iat[i, j])
                    if not cell:
                        continue
                    if any(p.search(cell) for p in pats):
                        parts = re.split(r"[:：]", cell, maxsplit=1)
                        if len(parts) == 2:
                            val = _norm_space(parts[1])
                            val = re.split(r"\s{2,}|\t", val)[0].strip()
                            m = re.search(r"([A-Za-z0-9][A-Za-z0-9\-\/\s]{4,})", val)
                            if m:
                                return _norm_space(m.group(1))

            if sdf.shape[1] == 2:
                for _, r in sdf.iterrows():
                    left = _norm_space(r.iloc[0]); right = _norm_space(r.iloc[1])
                    if any(p.search(left) for p in pats) and right:
                        m = re.search(r"([A-Za-z0-9][A-Za-z0-9\-\/\s]{4,})", right)
                        return _norm_space(m.group(1) if m else right)

            for i in range(sdf.shape[0]):
                for j in range(sdf.shape[1]):
                    cell = _norm_space(sdf.iat[i, j])
                    if any(p.search(cell) for p in pats):
                        if j+1 < sdf.shape[1]:
                            val = _norm_space(sdf.iat[i, j+1])
                            if val:
                                m = re.search(r"([A-Za-z0-9][A-Za-z0-9\-\/\s]{4,})", val)
                                return _norm_space(m.group(1) if m else val)
                        if i+1 < sdf.shape[0]:
                            val = _norm_space(sdf.iat[i+1, j])
                            if val:
                                m = re.search(r"([A-Za-z0-9][A-Za-z0-9\-\/\s]{4,})", val)
                                return _norm_space(m.group(1) if m else val)
        except Exception:
            continue
    return None

def _extract_policy_card(natural_text_first_page: str) -> Tuple[str, str]:
    text = _norm_space(natural_text_first_page)
    lines = [_norm_space(x) for x in natural_text_first_page.splitlines()]
    pol = _find_label_value_inline(text, POLICY_LABELS) or ""
    card = _find_label_value_inline(text, CARD_LABELS) or ""
    if not pol:
        pol = _find_label_value_next_line(lines, POLICY_LABELS) or pol
    if not card:
        card = _find_label_value_next_line(lines, CARD_LABELS) or card
    dfs = _tables_from_markdown(natural_text_first_page)
    if not pol:
        pol = _find_label_value_in_tables(dfs, POLICY_LABELS) or pol
    if not card:
        card = _find_label_value_in_tables(dfs, CARD_LABELS) or card
    return pol or "", card or ""

# -------------------- core --------------------
def _resolve_pdf_path(pdf_arg: str) -> str:
    """ถ้าไม่พบไฟล์ตามที่ส่งมา จะลองหาใน ./data/ ให้อัตโนมัติ"""
    if os.path.exists(pdf_arg):
        return pdf_arg
    candidate = os.path.join("data", os.path.basename(pdf_arg))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"ไม่พบไฟล์: {pdf_arg} หรือ {candidate}")

def extract_pdf_to_excel(pdf_path: str, out_path: Optional[str] = None,
                         base_url: str = None, api_key: str = None):
    base_url = base_url or os.getenv("TYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key  = api_key  or os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY")

    # input/output
    pdf_path = _resolve_pdf_path(pdf_path)
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    if out_path is None or out_path.strip() == "":
        out_dir = "xls"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{basename}.xlsx")
    else:
        if out_path.endswith(os.sep) or os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, f"{basename}.xlsx")
        else:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    page = 0
    policy, card = "", ""
    rows: List[dict] = []
    wrote_policy_card = False  # ใส่ Policy/Card แค่ครั้งแรกสุดเท่านั้น

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

        # ดึงทุกตารางในหน้านี้
        dfs = _tables_from_markdown(md_text)
        for df in dfs:
            df = _normalize_cols(df)
            need = {"No", "Coverage", "Limit", "Balance"}
            if len(need - set(df.columns)) != 0:
                continue

            if "Member Co-Pay" not in df.columns:
                df["Member Co-Pay"] = ""
            df = df[["No", "Coverage", "Member Co-Pay", "Limit", "Balance"]]

            # ตัด header ที่ OCR อาจคัดลอกซ้ำทุกหน้า
            df = df[~df["No"].astype(str).str.contains(r"^no\.?$", case=False, regex=True)]

            # เก็บทุกรอบ (รองรับตารางหลายหน้า) แล้วลบแถวซ้ำตอนท้าย
            first_row_this_table = True
            for _, r in df.iterrows():
                rows.append({
                    "Policy No": (policy if not wrote_policy_card and first_row_this_table else ""),
                    "Card No":   (card   if not wrote_policy_card and first_row_this_table else ""),
                    "No": _norm_space(r["No"]),
                    "Coverage": _norm_space(r["Coverage"]),
                    "Member Co-Pay": _norm_space(r["Member Co-Pay"]),
                    "Limit": _norm_space(r["Limit"]),
                    "Balance": _norm_space(r["Balance"]),
                })
                first_row_this_table = False
            if not wrote_policy_card and len(rows) > 0:
                wrote_policy_card = True

        page += 1

    # รวมและกันซ้ำ (ถ้ามีหน้าที่ซ้ำเนื้อหา)
    out = pd.DataFrame(rows, columns=["Policy No", "Card No", "No", "Coverage", "Member Co-Pay", "Limit", "Balance"])
    if out.empty:
        raise RuntimeError("ไม่พบตารางผลประโยชน์ (หัวคอลัมน์ No/Coverage/Limit/Balance)")
    out = out.drop_duplicates(subset=["No","Coverage","Member Co-Pay","Limit","Balance"], keep="first").reset_index(drop=True)

    out.to_excel(out_path, index=False)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="ไฟล์ PDF หรือชื่อไฟล์ในโฟลเดอร์ ./data")
    ap.add_argument("--out", default="", help="พาธไฟล์ .xlsx หรือโฟลเดอร์ (เช่น xls/)")
    ap.add_argument("--base_url", default=os.getenv("TYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    ap.add_argument("--api_key",  default=os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY"))
    args = ap.parse_args()
    extract_pdf_to_excel(args.pdf, args.out, base_url=args.base_url, api_key=args.api_key)
