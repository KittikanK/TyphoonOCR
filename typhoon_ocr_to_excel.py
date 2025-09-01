import os, json, re
from typing import List, Optional, Tuple
import pandas as pd
from bs4 import BeautifulSoup
import markdown as md
from dotenv import load_dotenv
from typhoon_ocr import ocr_document

# โหลดค่า ENV จากไฟล์ .env (ถ้ามี)
load_dotenv()

# ------------------------ Utilities ------------------------

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
    s = s.replace("\u00a0", " ")  # NBSP -> space
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

# ------------------------ Policy/Card extraction ------------------------

POLICY_LABELS = [
    r"Policy\s*No\.?", r"Policy\s*Number", r"Policy\s*ID",
    r"เลขที่\s*กรมธรรม์", r"เลขกรมธรรม์", r"กรมธรรม์\s*เลขที่",
    r"เลขที่\s*กรมธรรม์ประกันภัย", r"เลขที่\s*กรมธรรม์\s*ประกันภัย"
]
CARD_LABELS = [
    r"Card\s*No\.?", r"Card\s*Number",
    r"เลขที่\s*บัตร", r"หมายเลข\s*บัตร", r"หมายเลขบัตร", r"เลขบัตร"
]

# same-line value (label + value on one line)
def _find_label_value_inline(text: str, label_patterns: List[str]) -> Optional[str]:
    for pat in label_patterns:
        # อนุญาตเว้นวรรค/ขีด/ทับในค่า (อย่างน้อย 5 ตัวอักษรรวม)
        m = re.search(rf"(?:{pat})\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9\-\/\s]{{4,}})", text, flags=re.IGNORECASE)
        if m:
            val = _norm_space(m.group(1))
            # ตัดส่วนเกิน เช่น มีคอลัมน์อื่นติดมา (ช่องว่างต่อเนื่องหลายตัว/แท็บ)
            val = re.split(r"\s{2,}|\t", val)[0].strip()
            return val
    return None

# next-line value (label on one line, value on next non-empty line)
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

# table value (label/value in table cells; also handle "label: value" in one cell)
def _find_label_value_in_tables(dfs: List[pd.DataFrame], label_patterns: List[str]) -> Optional[str]:
    pats = [re.compile(p, re.IGNORECASE) for p in label_patterns]
    for df in dfs:
        try:
            sdf = df.copy()
            sdf.columns = [_norm_space(c) for c in sdf.columns]

            # 0) label:value อยู่ในเซลล์เดียว
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

            # 1) ถ้า 2 คอลัมน์ -> ซ้าย=label ขวา=value
            if sdf.shape[1] == 2:
                for _, r in sdf.iterrows():
                    left = _norm_space(r.iloc[0]); right = _norm_space(r.iloc[1])
                    if any(p.search(left) for p in pats) and right:
                        m = re.search(r"([A-Za-z0-9][A-Za-z0-9\-\/\s]{4,})", right)
                        return _norm_space(m.group(1) if m else right)

            # 2) >2 คอลัมน์ -> next col / next row
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

    # 1) inline patterns
    pol = _find_label_value_inline(text, POLICY_LABELS) or ""
    card = _find_label_value_inline(text, CARD_LABELS) or ""

    # 2) next-line patterns
    if not pol:
        pol = _find_label_value_next_line(lines, POLICY_LABELS) or pol
    if not card:
        card = _find_label_value_next_line(lines, CARD_LABELS) or card

    # 3) scan tables on first page
    dfs = _tables_from_markdown(natural_text_first_page)
    if not pol:
        pol = _find_label_value_in_tables(dfs, POLICY_LABELS) or pol
    if not card:
        card = _find_label_value_in_tables(dfs, CARD_LABELS) or card

    return pol or "", card or ""

# ------------------------ Main extraction ------------------------

def extract_pdf_to_excel(pdf_path: str, out_xlsx: str, base_url: str = None, api_key: str = None):
    # base_url/api_key: ถ้าไม่ส่งมา ใช้ค่าจาก ENV ที่ load จาก .env แล้ว
    base_url = base_url or os.getenv("TYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key  = api_key  or os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY")

    page = 0
    policy, card = "", ""
    rows: List[dict] = []

    while True:
        try:
            res = ocr_document(
                pdf_or_image_path=pdf_path,
                task_type="structure",   # ได้ HTML tables ที่อ่านง่ายด้วย pandas
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

        # Debug: เซฟหน้าแรกไว้ตรวจรูปแบบจริง (เปิดคอมเมนต์ได้ถ้าต้องการ)
        # if page == 0:
        #     with open("first_page_ocr.md", "w", encoding="utf-8") as f:
        #         f.write(md_text)

        if page == 0:
            policy, card = _extract_policy_card(md_text)

        dfs = _tables_from_markdown(md_text)
        for df in dfs:
            df = _normalize_cols(df)
            need = {"No", "Coverage", "Limit", "Balance"}
            if len(need - set(df.columns)) == 0:
                if "Member Co-Pay" not in df.columns:
                    df["Member Co-Pay"] = ""
                df = df[["No", "Coverage", "Member Co-Pay", "Limit", "Balance"]]

                def to_int(x):
                    try:
                        return int(str(x).strip().split()[0])
                    except Exception:
                        return None
                df["__no__"] = df["No"].map(to_int)
                sel = df[df["__no__"].notnull()].copy()
                if not sel.empty:
                    df = sel
                df.drop(columns=["__no__"], inplace=True, errors="ignore")

                first = True
                for _, r in df.iterrows():
                    rows.append({
                        "Policy No": policy if (page == 0 and first) else "",
                        "Card No":   card   if (page == 0 and first) else "",
                        "No": _norm_space(r["No"]),
                        "Coverage": _norm_space(r["Coverage"]),
                        "Member Co-Pay": _norm_space(r["Member Co-Pay"]),
                        "Limit": _norm_space(r["Limit"]),
                        "Balance": _norm_space(r["Balance"]),
                    })
                    first = False

        page += 1

    out = pd.DataFrame(rows, columns=["Policy No", "Card No", "No", "Coverage", "Member Co-Pay", "Limit", "Balance"])
    if out.empty:
        raise RuntimeError("ไม่พบตารางผลประโยชน์ (หัวคอลัมน์ No/Coverage/Limit/Balance)")
    out.to_excel(out_xlsx, index=False)
    print(f"✅ Saved: {out_xlsx}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="output.xlsx")
    # base_url/api_key ปล่อยว่างได้ จะดึงจาก .env อัตโนมัติ
    ap.add_argument("--base_url", default=os.getenv("TYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    ap.add_argument("--api_key", default=os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY"))
    args = ap.parse_args()
    extract_pdf_to_excel(args.pdf, args.out, base_url=args.base_url, api_key=args.api_key)
