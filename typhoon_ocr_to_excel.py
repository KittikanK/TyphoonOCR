# typhoon_ocr_to_excel.py
"""
Extract benefit tables from PDF (Typhoon OCR) and export to Excel.

Features
- Extract Policy No / Card No (from the FIRST page only)
- Merge tables across pages; drop duplicate headers
- Keep only numeric "No" rows and enforce non-decreasing sequence
- Normalize Thai/EN spacing, hyphens, commas, and common OCR artefacts
- Fuzzy match 'Coverage' with master wording (template file)
  * Number guard: numbers must match (e.g., 5 vs 10 -> not replaced)
  * Negation guard: presence of "ไม่รวม" must agree (avoid false matches)
- Output columns: Policy No, Card No, No, Coverage, Member Co-Pay, Limit, Balance
  plus QC columns: Coverage (raw), MatchScore, Matched

CLI
  python typhoon_ocr_to_excel.py --pdf data/xxx.pdf --sim 80
  # relax number guard:
  python typhoon_ocr_to_excel.py --pdf data/xxx.pdf --sim 80 --no_strict_numbers
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import markdown as md
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rapidfuzz import fuzz, process
from typhoon_ocr import ocr_document


# =============================================================================
# Environment
# =============================================================================

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)


# =============================================================================
# Utilities - generic text cleanup
# =============================================================================

def _ensure_obj(result) -> dict:
    """Return result as dict (Typhoon may return dict or JSON string)."""
    if isinstance(result, dict):
        return result
    try:
        return json.loads(result)
    except Exception:
        return {"natural_text": str(result)}


def _norm_space(s: str) -> str:
    """Collapse multiple spaces; strip; convert NBSP -> space."""
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _clean_html(text: str) -> str:
    """Remove simple HTML tags that can leak from OCR (e.g., <td>...</td>)."""
    return re.sub(r"<[^>]+>", " ", text or "")


def _strip_invisibles(s: str) -> str:
    """Remove zero-width/BOM; normalize dashes to a simple hyphen."""
    if not s:
        return s
    s = (
        s.replace("\u200b", "")  # zero-width space
        .replace("\ufeff", "")   # BOM
        .replace("\u00a0", " ")  # NBSP
        .replace("\u2010", "-")  # hyphen variants
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return s


def _format_th_en_spacing(s: str) -> str:
    """
    Normalize Thai/EN spacing, parentheses, hyphen, and commas.
    Also join OCR-broken phrase "ไม่รวม มอเตอร์ไซด์/ไซค์" -> "ไม่รวมมอเตอร์ไซด์/ไซค์".
    """
    if not s:
        return s

    s = s.replace("\n", " ")

    # parentheses spacing
    s = re.sub(r"([^\s(])\(", r"\1 (", s)
    s = re.sub(r"\)([^\s)\.,;:])", r") \1", s)

    # hyphen spacing  -> " - "
    s = re.sub(r"\s*-\s*", " - ", s)

    # separate Thai <-> Latin/number (both directions)
    th = r"\u0E00-\u0E7F"
    latnum = r"A-Za-z0-9"
    s = re.sub(rf"([{th}])([{latnum}])", r"\1 \2", s)
    s = re.sub(rf"([{latnum}])([{th}])", r"\1 \2", s)

    # commas
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r",\s*\)", ")", s)
    s = re.sub(r",\s*$", "", s)

    # special join: "ไม่รวม มอเตอร์ไซด์/ไซค์"
    s = re.sub(r"ไม่\s*รวม\s+(มอเตอร์ไซ[ด์ค์])", r"ไม่รวม\1", s)
    s = re.sub(r"\(\s*ไม่รวม\s+(มอเตอร์ไซ[ด์ค์])\s*\)", r"(ไม่รวม\1)", s)

    # collapse spaces
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()


def _thai_norm(s: str) -> str:
    """Light normalization pipeline used before matching."""
    return _format_th_en_spacing(_norm_space(_strip_invisibles(s)))


# =============================================================================
# Tables (markdown -> html -> pandas)
# =============================================================================

def _tables_from_markdown(natural_text: str) -> List[pd.DataFrame]:
    """Convert markdown tables in Typhoon `natural_text` to pandas DataFrames."""
    html = md.markdown(natural_text, extensions=["tables"])
    soup = BeautifulSoup(html, "lxml")
    dfs: List[pd.DataFrame] = []
    for t in soup.find_all("table"):
        try:
            dfs.extend(pd.read_html(str(t)))
        except Exception:
            pass
    return dfs


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map column names to canonical ones."""
    mapping = {}
    for c in df.columns:
        key = re.sub(r"[^A-Za-z0-9ก-๙\- ]", "", str(c)).strip().lower()
        if key in {"no", "no.", "ลำดับ"}:
            mapping[c] = "No"
        elif "coverage" in key or "ความคุ้มครอง" in key:
            mapping[c] = "Coverage"
        elif "member" in key and ("co" in key or "copay" in key or "co-pay" in key):
            mapping[c] = "Member Co-Pay"
        elif "limit" in key or "ผลประโยชน์สูงสุด" in key:
            mapping[c] = "Limit"
        elif "balance" in key or "คงเหลือ" in key:
            mapping[c] = "Balance"
        else:
            mapping[c] = c
    return df.rename(columns=mapping)


# =============================================================================
# Matching helpers (numbers / templates / guards)
# =============================================================================

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def _to_arabic_digits(s: str) -> str:
    """Convert Thai digits -> Arabic digits."""
    return ("" if s is None else str(s)).translate(_THAI_DIGITS)


def _extract_numbers(s: str) -> List[str]:
    """Extract numbers (after converting Thai digits)."""
    return re.findall(r"\d+", _to_arabic_digits(s))


def _read_text_file_any_encoding(path: Path) -> str:
    """Read text file trying utf-8/utf-16 encodings."""
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def _load_templates(path: Optional[str]) -> List[str]:
    """Load master wording templates as normalized strings."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"⚠️ Template file not found: {path}")
        return []
    text = _read_text_file_any_encoding(p)
    if not text:
        print(f"⚠️ Could not read {path}")
        return []
    return [t for line in text.splitlines() if (t := _thai_norm(line))]


def _match_template(
    text: str,
    templates: List[str],
    threshold: int = 88,
    enforce_number_match: bool = True,
    enforce_negation_match: bool = True,
) -> Tuple[str, float, bool]:
    """
    Fuzzy-match OCR text with templates.

    Returns:
        (chosen_text, score, matched)
        - If matched:
            * Number guard: numbers set must be equal (if both sides contain numbers)
            * Negation guard: presence of "ไม่รวม" must agree (True/True or False/False)
    """
    raw = _thai_norm(text)
    if not raw or not templates:
        return raw, 0.0, False

    best = process.extractOne(raw, templates, scorer=fuzz.token_sort_ratio)
    if not best:
        return raw, 0.0, False

    best_text, score, _ = best
    matched = score >= threshold

    # Number guard
    if matched and enforce_number_match:
        nums_raw = set(_extract_numbers(raw))
        nums_tpl = set(_extract_numbers(best_text))
        if nums_raw and nums_tpl and nums_raw != nums_tpl:
            return raw, float(score), False

    # Negation guard ("ไม่รวม")
    if matched and enforce_negation_match:
        has_neg_raw = bool(re.search(r"ไม่\s*รวม", raw))
        has_neg_tpl = bool(re.search(r"ไม่\s*รวม", best_text))
        if has_neg_raw != has_neg_tpl:
            return raw, float(score), False

    return (best_text if matched else raw), float(score), matched


# =============================================================================
# Policy / Card extraction (FIRST page only)
# =============================================================================

POLICY_LABELS = [
    r"Policy\s*No\.?",
    r"Policy\s*Number",
    r"Policy\s*ID",
    r"เลขที่\s*กรมธรรม์",
    r"เลขกรมธรรม์",
    r"กรมธรรม์\s*เลขที่",
    r"เลขที่\s*กรมธรรม์ประกันภัย",
    r"เลขที่\s*กรมธรรม์\s*ประกันภัย",
]

CARD_LABELS = [
    r"Card\s*No\.?",
    r"Card\s*Number",
    r"เลขที่\s*บัตร",
    r"หมายเลข\s*บัตร",
    r"หมายเลขบัตร",
    r"เลขบัตร",
]

CARD_VALUE_STRICT = r"\d{5}-\d{3}-\d{9}"                     # 14048-108-240005162
CARD_VALUE_LOOSE = r"(\d{5})[^\d]{0,3}(\d{3})[^\d]{0,3}(\d{9})"  # tolerant
POLICY_VALUE = r"[A-Z0-9\-]{5,}(?:_[0-9]+)?"                 # IHA4001YC_13 / 14048-...


def _find_first(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _normalize_ocr_block(s: str) -> str:
    return _norm_space(_clean_html(_strip_invisibles(s)))


def _extract_policy_card(natural_text_first_page: str) -> Tuple[str, str]:
    """
    Robust extraction from page-0 only (avoid conflicting values on later pages).
    Strategy for Policy:
      - Label-line strict; else scan next 3 lines for value.
    Strategy for Card:
      - Label+strict -> Label+loose -> Next-line (strict/loose) -> Page scan (loose)
    """
    text = _normalize_ocr_block(natural_text_first_page)

    # Policy (label-line strict; else next lines)
    policy = _find_first(rf"(?:{'|'.join(POLICY_LABELS)})\s*[:：]?\s*({POLICY_VALUE})", text)
    if not policy:
        lines = [_normalize_ocr_block(x) for x in natural_text_first_page.splitlines()]
        for i, line in enumerate(lines):
            if re.search(rf"(?:{'|'.join(POLICY_LABELS)})", line, flags=re.IGNORECASE):
                for j in range(i + 1, min(i + 4, len(lines))):
                    if v := _find_first(rf"({POLICY_VALUE})", lines[j]):
                        policy = v
                        break
            if policy:
                break
    policy = (policy or "").strip()

    # Card (strict)
    card = _find_first(rf"(?:{'|'.join(CARD_LABELS)})\s*[:：]?\s*({CARD_VALUE_STRICT})", text)

    # Card (loose with label)
    if not card:
        if m := re.search(rf"(?:{'|'.join(CARD_LABELS)})\s*[:：]?\s*{CARD_VALUE_LOOSE}", text, re.I):
            card = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Card (next-line strict/loose)
    if not card:
        lines = [_normalize_ocr_block(x) for x in natural_text_first_page.splitlines()]
        for i, line in enumerate(lines):
            if re.search(rf"(?:{'|'.join(CARD_LABELS)})", line, re.I):
                for j in range(i + 1, min(i + 4, len(lines))):
                    if v := _find_first(rf"({CARD_VALUE_STRICT})", lines[j]):
                        card = v
                        break
                    m = re.search(CARD_VALUE_LOOSE, lines[j])
                    if m:
                        card = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                        break
            if card:
                break

    # Card (page scan loose)
    if not card:
        if m := re.search(CARD_VALUE_LOOSE, text):
            card = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    return policy, (card or "").strip()


# =============================================================================
# Row filtering / sanity checks
# =============================================================================

def _resolve_pdf_path(pdf_arg: str) -> str:
    """Accept absolute path or basename under ./data."""
    if os.path.exists(pdf_arg):
        return pdf_arg
    candidate = os.path.join("data", os.path.basename(pdf_arg))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"File not found: {pdf_arg} or {candidate}")


def _filter_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose 'No' cell looks like header text."""
    mask = ~df["No"].astype(str).str.match(r"(?i)^\s*(no\.?|ลำดับ)\s*$")
    return df[mask]


def _only_numeric_no(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose 'No' is an integer."""
    is_int = df["No"].map(lambda x: str(x).strip().isdigit())
    return df[is_int]


def _filter_no_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce non-decreasing sequence of No (drop rows that jump backwards)."""
    def to_int(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    df = df.copy()
    df["__NoInt"] = df["No"].map(to_int)

    cleaned = []
    last = -10**9
    for _, row in df.iterrows():
        n = row["__NoInt"]
        if n is None or n >= last:
            cleaned.append(row)
            if n is not None:
                last = n

    return pd.DataFrame(cleaned).drop(columns=["__NoInt"], errors="ignore")


# =============================================================================
# Core
# =============================================================================

def extract_pdf_to_excel(
    pdf_path: str,
    out_path: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    templates_path: Optional[str] = "template/master-data-wording.txt",
    sim_threshold: int = 88,
    enforce_number_match: bool = True,
) -> None:
    """OCR a PDF -> merge benefit tables -> export to Excel."""
    base_url = base_url or os.getenv("TYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = api_key or os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY")

    pdf_path = _resolve_pdf_path(pdf_path)
    basename = Path(pdf_path).stem

    # choose output path (default to ./xls/<basename>.xlsx)
    if not out_path:
        out_dir = Path("xls")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{basename}.xlsx")

    # load templates (normalized)
    templates = _load_templates(templates_path)

    rows: List[dict] = []
    policy_no, card_no = "", ""
    wrote_header_row_for_first_table = False

    page = 0
    while True:
        # --- OCR page ---
        try:
            res = ocr_document(
                pdf_or_image_path=pdf_path,
                task_type="structure",
                page_num=page,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception:
            # no more pages
            if page == 0:
                raise
            break

        obj = _ensure_obj(res)
        md_text = obj.get("natural_text", "")

        # extract policy/card only from FIRST page
        if page == 0:
            policy_no, card_no = _extract_policy_card(md_text)

        # --- parse all tables on this page ---
        for df in _tables_from_markdown(md_text):
            df = _normalize_cols(df)
            required = {"No", "Coverage", "Limit", "Balance"}
            if not required.issubset(df.columns):
                continue

            if "Member Co-Pay" not in df.columns:
                df["Member Co-Pay"] = ""

            df = df[["No", "Coverage", "Member Co-Pay", "Limit", "Balance"]]
            df = _filter_header_rows(df)
            df = _only_numeric_no(df)

            first_row_in_this_table = True
            for _, r in df.iterrows():
                cov_raw = _thai_norm(str(r["Coverage"]))
                cov_best, score, matched = _match_template(
                    cov_raw,
                    templates,
                    threshold=sim_threshold,
                    enforce_number_match=enforce_number_match,
                    enforce_negation_match=True,
                )

                rows.append(
                    {
                        "Policy No": policy_no if not wrote_header_row_for_first_table and first_row_in_this_table else "",
                        "Card No": card_no if not wrote_header_row_for_first_table and first_row_in_this_table else "",
                        "No": _norm_space(r["No"]),
                        "Coverage": cov_best,
                        "Member Co-Pay": _norm_space(r["Member Co-Pay"]),
                        "Limit": _norm_space(r["Limit"]),
                        "Balance": _norm_space(r["Balance"]),
                        "Coverage (raw)": cov_raw,
                        "MatchScore": f"{score:.0f}",
                        "Matched": "TRUE" if matched else "FALSE",
                    }
                )
                first_row_in_this_table = False

            if not wrote_header_row_for_first_table and rows:
                wrote_header_row_for_first_table = True

        page += 1

    # --- finalize ---
    out_cols = [
        "Policy No",
        "Card No",
        "No",
        "Coverage",
        "Member Co-Pay",
        "Limit",
        "Balance",
        "Coverage (raw)",
        "MatchScore",
        "Matched",
    ]

    out_df = pd.DataFrame(rows, columns=out_cols)
    if out_df.empty:
        raise RuntimeError("ไม่พบตารางผลประโยชน์")

    out_df = (
        out_df
        .drop_duplicates(subset=["No", "Coverage", "Member Co-Pay", "Limit", "Balance"], keep="first")
        .reset_index(drop=True)
    )
    out_df = _filter_no_sequence(out_df)
    out_df.to_excel(out_path, index=False)
    print(f"✅ Saved: {out_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract PDF benefits (Typhoon OCR) to Excel")
    parser.add_argument("--pdf", required=True, help="PDF path or basename under ./data")
    parser.add_argument("--out", default="", help="Output .xlsx path (or folder). Default: ./xls/<basename>.xlsx")
    parser.add_argument("--base_url", default=os.getenv("TYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api_key", default=os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--templates", default="template/master-data-wording.txt", help="Master wording file")
    parser.add_argument("--sim", type=int, default=88, help="Fuzzy-match threshold (0-100)")
    parser.add_argument(
        "--no_strict_numbers",
        action="store_true",
        help="Disable number guard (default: enabled)",
    )

    args = parser.parse_args()

    extract_pdf_to_excel(
        pdf_path=args.pdf,
        out_path=args.out,
        base_url=args.base_url,
        api_key=args.api_key,
        templates_path=args.templates,
        sim_threshold=args.sim,
        enforce_number_match=not args.no_strict_numbers,
    )
