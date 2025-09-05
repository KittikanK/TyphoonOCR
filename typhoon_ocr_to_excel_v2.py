
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
typhoon_ocr_to_excel_v2.py
--------------------------
Single script that:
- OCRs all pages of a PDF with Typhoon OCR ("structure")
- Extracts benefit tables split into:
    1) Member Co-Pay
    2) Insured Co-Pay (Major Medical / Inpatient Benefits with co-pay column)
- Writes one Excel with two sheets:
    - "Member Co-Pay": columns
        Policy No, Card No, No, Coverage, Member Co-Pay, Limit, Balance, Coverage (raw), MatchScore, Matched
    - "Insured Co-Pay": columns
        Policy No, Card No, No, Coverage, Insured Co-Pay, Limit, Balance, Coverage (raw), MatchScore, Matched
- No Summary sheet
- Policy No / Card No appear only in the first row of each sheet

Optional:
- Provide a --wording file (master-data-wording.txt) with canonical coverage terms (one per line).
  We'll fuzzy-match "Coverage" to the nearest canonical wording and report MatchScore + Matched (score>=80 by default).

Usage:
  python typhoon_ocr_to_excel_v2.py --pdf INPUT.pdf --out OUTPUT.xlsx --wording master-data-wording.txt
"""

import argparse
import os, re, shlex, subprocess
from typing import List, Tuple, Optional

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

try:
    from rapidfuzz import process, fuzz
except Exception:
    process = None
    fuzz = None

try:
    from typhoon_ocr import ocr_document
except Exception as e:
    raise RuntimeError("Please install typhoon-ocr and ensure Poppler is installed (pdfinfo, pdftoppm).") from e


# ---------------- Utilities ----------------
def pdf_num_pages(pdf_path: str) -> int:
    try:
        out = subprocess.run(f'pdfinfo {shlex.quote(pdf_path)}',
                             shell=True, capture_output=True, text=True, check=True).stdout
        m = re.search(r'^Pages:\s+(\d+)', out, re.M)
        return int(m.group(1)) if m else 1
    except Exception:
        try:
            from PyPDF2 import PdfReader
            return len(PdfReader(pdf_path).pages)
        except Exception:
            return 1


def ocr_all_pages_markdown(pdf_path: str, task_type: str = "structure") -> str:
    pages = pdf_num_pages(pdf_path)
    parts: List[str] = []
    for p in range(1, pages + 1):  # Typhoon OCR uses 1-based page_num
        md = ocr_document(pdf_path, task_type=task_type, page_num=p)
        parts.append(md or "")
    return "\n\n".join(parts)


# ---------------- Cleaning helpers ----------------
def _strip_trailing_punct(s: str) -> str:
    return re.sub(r'[\s,;:–—-]+$', '', (s or '').strip())


def _to_null_if_empty_or_na(s: str) -> str:
    txt = (s or "").strip()
    if txt == "" or txt in {"-", "—", "N/A", "n/a", "NA"}:
        return "Null"
    return txt


def _clean_text_for_cell(s: str) -> str:
    return _to_null_if_empty_or_na(_strip_trailing_punct(s))


# ---------------- Member Info ----------------
POLICY_RE = re.compile(r'Policy\s*No\.?\s*:\s*([A-Z0-9_\-]+)', re.I)
CARD_RE = re.compile(r'Card\s*No\.?\s*:\s*([0-9()\-\s]+)', re.I)


def extract_policy_card_from_html(html_or_md_text: str,
                                  fallback_filename: str = "") -> Tuple[Optional[str], Optional[str]]:
    text = html_or_md_text
    looks_like_html = "</table>" in text or "<table" in text or "<tr" in text
    if looks_like_html:
        soup = BeautifulSoup(text, "lxml")
    else:
        soup = BeautifulSoup(f"<pre>{text}</pre>", "lxml")

    blocks: List[str] = []
    tables = soup.find_all("table")
    if tables:
        for tbl in tables:
            t = tbl.get_text(" ", strip=True)
            if t:
                blocks.append(t)
            if re.search(r'(New Health Standard|Outpatient Benefits|Inpatient Benefits|Major Medical)', t, re.I):
                break
    else:
        blocks.append(soup.get_text(" ", strip=True))

    blob = " \n ".join(blocks)

    policy_no = None
    m = POLICY_RE.search(blob)
    if m:
        policy_no = m.group(1).strip()

    card_no = None
    m = CARD_RE.search(blob)
    if m:
        card_no = re.sub(r'\s+', ' ', m.group(1)).strip()

    if not card_no and fallback_filename:
        mf = re.search(r'(\d{5}-\d{3}-\d{9})', os.path.basename(fallback_filename))
        if mf:
            card_no = mf.group(1)

    return policy_no, card_no


# ---------------- Table extraction ----------------
def normalize_header(h: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (h or '').lower())


# Synonyms (normalized) incl. Thai and common OCR typos
NO_SYNS = ['no', 'no.', 'number', 'ลำดับ']
COV_SYNS = ['coverage', 'benefit', 'benefits', 'รายการความคุ้มครอง',
            'ความคุ้มครอง', 'รายการผลประโยชน์', 'ผลประโยชน์', 'coveragebenefit']
LIMIT_SYNS = ['limit', 'maximumlimit', 'maxlimit', 'suminsured', 'วงเงิน',
              'maximumbenefit', 'benefitlimit',
              # common typos:
              'limited', 'lim1t', 'limlt', 'limlted', 'limiit']
BAL_SYNS = ['balance', 'คงเหลือ', 'remaining', 'remain', 'ยอดคงเหลือ']


def is_member_header(txt: str) -> bool:
    t = normalize_header(txt)
    return ('membercopay' in t) or ('membercopayment' in t) or ('membercopa' in t) or ('memberco' in t and 'pay' in t)


def is_insured_header(txt: str) -> bool:
    t = normalize_header(txt)
    return ('insuredcopay' in t) or ('injuredcopay' in t) or ('insuredcopayment' in t) or ('insurancecopay' in t) or ('insuredcopa' in t)


def find_idx_any(headers_norm: List[str], names: List[str]) -> int:
    names_norm = [normalize_header(nm) for nm in names]
    for i, h in enumerate(headers_norm):
        for nm in names_norm:
            if nm and nm in h:
                return i
        if process and fuzz:
            scores = []
            for nm in names_norm:
                if not nm:
                    continue
                try:
                    scores.append(fuzz.WRatio(h, nm))
                except Exception:
                    pass
            if scores and max(scores) >= 90:
                return i
    return -1


def parse_html_tables(md_text: str, wording: List[str] = None, threshold: int = 80):
    looks_like_html = ("<table" in md_text) or ("</table>" in md_text)
    if not looks_like_html:
        return None, None

    soup = BeautifulSoup(md_text, "lxml")
    member_frames = []
    insured_frames = []

    for tbl in soup.find_all("table"):
        trs = tbl.find_all("tr")
        if not trs:
            continue

        header_cells = [th.get_text(" ", strip=True) for th in trs[0].find_all(["th", "td"])]
        headers_norm = [normalize_header(h) for h in header_cells]

        idx_no = find_idx_any(headers_norm, NO_SYNS)
        idx_cov = find_idx_any(headers_norm, COV_SYNS)
        idx_limit = find_idx_any(headers_norm, LIMIT_SYNS)
        idx_balance = find_idx_any(headers_norm, BAL_SYNS)

        idx_member_copay = -1
        idx_insured_copay = -1
        for i, _ in enumerate(headers_norm):
            if is_member_header(header_cells[i]):
                idx_member_copay = i
            if is_insured_header(header_cells[i]):
                idx_insured_copay = i
        if idx_member_copay == -1 and idx_insured_copay == -1:
            for i, h in enumerate(headers_norm):
                if 'copay' in h or 'copayment' in h or 'copayments' in h:
                    idx_member_copay = i
                    break

        # accept table if at least No + Coverage present
        has_no_cov = (idx_no != -1 and idx_cov != -1)
        has_member = has_no_cov and (idx_member_copay != -1)
        has_insured = has_no_cov and (idx_insured_copay != -1)
        allow_without_copay = False
        if not (has_member or has_insured) and has_no_cov:
            allow_without_copay = True

        if not (has_member or has_insured or allow_without_copay):
            continue

        rows = []
        for tr in trs[1:]:
            tds = tr.find_all(["td", "th"])
            if not tds:
                continue

            def pick(idx):
                if idx is None or idx < 0:
                    return ""
                try:
                    return tds[idx].get_text(" ", strip=True)
                except Exception:
                    return ""

            no = pick(idx_no)
            if not re.match(r'^\s*\d+\s*$', no or ""):
                continue

            cov_raw = pick(idx_cov)
            limit = pick(idx_limit)
            balance = pick(idx_balance)
            copay = pick(idx_member_copay if has_member else idx_insured_copay) if not allow_without_copay else ""

            matched_term = cov_raw
            score = 100
            matched = True
            if wording and process and fuzz:
                mm = process.extractOne(cov_raw, wording, scorer=fuzz.WRatio)
                if mm:
                    matched_term = mm[0]
                    score = int(mm[1])
                    matched = score >= threshold

            row = {
                "No": _clean_text_for_cell(no),
                "Coverage": _clean_text_for_cell(matched_term),
                ("Member Co-Pay" if (has_member or allow_without_copay) else "Insured Co-Pay"): _clean_text_for_cell(copay),
                "Limit": _clean_text_for_cell(limit),
                "Balance": _clean_text_for_cell(balance),
                "Coverage (raw)": _clean_text_for_cell(cov_raw),
                "MatchScore": score,
                "Matched": matched,
            }
            rows.append(row)

        if not rows:
            continue

        df = pd.DataFrame(rows)
        if has_member or allow_without_copay:
            want = ["No", "Coverage", "Member Co-Pay", "Limit", "Balance", "Coverage (raw)", "MatchScore", "Matched"]
            df = df.reindex(columns=want)
            member_frames.append(df)
        else:
            want = ["No", "Coverage", "Insured Co-Pay", "Limit", "Balance", "Coverage (raw)", "MatchScore", "Matched"]
            df = df.reindex(columns=want)
            insured_frames.append(df)

    df_member = pd.concat(member_frames, ignore_index=True) if member_frames else None
    df_insured = pd.concat(insured_frames, ignore_index=True) if insured_frames else None
    return df_member, df_insured


# ---------------- Runner ----------------
def run(pdf_path: str, out_xlsx: str, wording_file: str = None, threshold: int = 80,
        debug: bool = False, debugdir: str = None):
    md = ocr_all_pages_markdown(pdf_path, task_type="structure")

    policy_no, card_no = extract_policy_card_from_html(md, fallback_filename=pdf_path)

    # Debug (combined only)
    if debug:
        dbg_dir = debugdir if debugdir else (os.path.dirname(os.path.abspath(out_xlsx)) or ".")
        os.makedirs(dbg_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(out_xlsx))[0]
        with open(os.path.join(dbg_dir, f"{base}_debug.txt"), "w", encoding="utf-8") as fdbg:
            fdbg.write(md)

    # wording
    wording = None
    if wording_file and os.path.exists(wording_file):
        enc_trials = ["utf-8", "utf-16", "utf-16le", "utf-16be", "latin-1"]
        for enc in enc_trials:
            try:
                with open(wording_file, "r", encoding=enc) as f:
                    wording = [re.sub(r"[\s,;:–—-]+$", "", ln.strip()) for ln in f if ln.strip()]
                break
            except UnicodeDecodeError:
                continue
        if wording is None:
            raise UnicodeDecodeError("wording", b"", 0, 1,
                                     "Unsupported encoding. Tried: " + ", ".join(enc_trials))

    # extract
    df_member, df_insured = parse_html_tables(md, wording=wording, threshold=threshold)

    # add meta
    def add_meta(df: pd.DataFrame, policy, card, sheet_type: str) -> pd.DataFrame:
        if sheet_type == "member":
            cols = ["Policy No", "Card No", "No", "Coverage", "Member Co-Pay",
                    "Limit", "Balance", "Coverage (raw)", "MatchScore", "Matched"]
        else:
            cols = ["Policy No", "Card No", "No", "Coverage", "Insured Co-Pay",
                    "Limit", "Balance", "Coverage (raw)", "MatchScore", "Matched"]

        if df is None or df.empty:
            return pd.DataFrame(columns=cols)

        df = df.copy()
        df.insert(0, "Card No", "")
        df.insert(0, "Policy No", "")
        df.at[df.index[0], "Policy No"] = _clean_text_for_cell(policy or "")
        df.at[df.index[0], "Card No"] = _clean_text_for_cell(card or "")
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
        for c in cols:
            if c in ["MatchScore", "Matched"]:
                continue
            df[c] = df[c].map(_clean_text_for_cell)
        return df

    out_member = add_meta(df_member, policy_no, card_no, "member")
    out_insured = add_meta(df_insured, policy_no, card_no, "insured")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        out_member.to_excel(xw, sheet_name="Member Co-Pay", index=False)
        out_insured.to_excel(xw, sheet_name="Insured Co-Pay", index=False)


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Input PDF")
    ap.add_argument("--out", required=True, help="Output XLSX")
    ap.add_argument("--wording", help="Path to master-data-wording.txt", default=None)
    ap.add_argument("--threshold", type=int, default=80,
                    help="Fuzzy match threshold (0-100)")
    ap.add_argument("--debug", action="store_true",
                    help="Dump raw OCR (combined) alongside output")
    ap.add_argument("--debugdir",
                    help="Directory to write debug OCR files (default: alongside --out)")
    args = ap.parse_args()

    run(args.pdf, args.out, wording_file=args.wording,
        threshold=args.threshold, debug=args.debug, debugdir=args.debugdir)


if __name__ == "__main__":
    main()
