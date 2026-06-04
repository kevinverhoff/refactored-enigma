"""
Download DLGF memos (2022-2026) and write a metadata manifest.

Output layout:
  docs/<year>/<filename>   -- downloaded files
  metadata.json            -- manifest with per-file metadata

Filename convention on the site:  YYMMDD-Author-Memo-Title.pdf
Parsed into: memo_date, author, doc_type (Memo/Attachment/Template/...), title.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.in.gov"

# Note: 2024 page uses a non-obvious slug "2024-memos2"
YEAR_PAGES: dict[int, str] = {
    2022: "/dlgf/memos-and-presentations/memos/2022-memos",
    2023: "/dlgf/memos-and-presentations/memos/2023-memos",
    2024: "/dlgf/memos-and-presentations/memos/2024-memos2",
    2025: "/dlgf/memos-and-presentations/memos/2025-memos",
    2026: "/dlgf/memos-and-presentations/memos/2026-memos",
}

DOC_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx"}
DOCS_DIR = Path("docs")
METADATA_FILE = Path("metadata.json")

# Polite delay between downloads (seconds)
DOWNLOAD_DELAY = 0.25

# Separates <author> from <title> in filenames like 221230-Wood-Memo-Title.pdf
_DOC_TYPE_RE = re.compile(
    r"^(.+?)-(Memo|Attachment|Attachments|Template|Templates|Forms?|Instructions?|FAQ|Resources?|Guidance|Update|Supplement|Report|Overview)",
    re.IGNORECASE,
)

_DATE_PREFIX_RE = re.compile(r"^(\d{6})-(.+)$")


def parse_filename(filename: str) -> dict:
    """
    Extract memo_date, author, doc_type, and title from a DLGF filename.

    Expected pattern:  YYMMDD-Author-DocType-Rest-of-title.ext
    Falls back gracefully for hash filenames or non-standard names.
    """
    stem = unquote(Path(filename).stem)
    result = {"memo_date": None, "author": None, "doc_type": None, "title": None}

    date_match = _DATE_PREFIX_RE.match(stem)
    if not date_match:
        result["title"] = stem.replace("-", " ")
        return result

    date_str, remainder = date_match.groups()
    try:
        result["memo_date"] = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    except Exception:
        pass

    type_match = _DOC_TYPE_RE.match(remainder)
    if type_match:
        result["author"] = type_match.group(1).replace("-", " ")
        result["doc_type"] = type_match.group(2).upper()
        result["title"] = remainder[type_match.end() + 1 :].replace("-", " ")
    else:
        result["title"] = remainder.replace("-", " ")

    return result


def _is_memo_file(url: str) -> bool:
    """
    Accept only files under /dlgf/files/202X-* folders.
    Filters out navigation reference PDFs (e.g. Townships-by-City.pdf).
    """
    path = urlparse(url).path
    parts = path.split("/")
    try:
        idx = parts.index("files")
        folder = parts[idx + 1] if len(parts) > idx + 1 else ""
        return bool(re.match(r"^202[2-6]", folder))
    except ValueError:
        return False


def collect_links(year: int, page_path: str) -> list[dict]:
    """Scrape one year page and return document metadata records."""
    url = BASE_URL + page_path
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    docs: list[dict] = []

    for a in soup.find_all("a", href=True):
        href: str = a["href"].strip()
        ext = Path(href.split("?")[0]).suffix.lower()
        if ext not in DOC_EXTENSIONS:
            continue

        full_url = urljoin(BASE_URL, href) if href.startswith("/") else href
        if not _is_memo_file(full_url):
            continue
        if full_url in seen:
            continue
        seen.add(full_url)

        filename = unquote(Path(urlparse(full_url).path).name)
        parsed = parse_filename(filename)

        docs.append(
            {
                "url": full_url,
                "filename": filename,
                "file_type": ext.lstrip("."),
                "source_year": year,
                "local_path": f"{year}/{filename}",
                **parsed,
                "downloaded": False,
                "download_error": None,
            }
        )

    return docs


def download_doc(doc: dict, base_dir: Path) -> dict:
    """Download one file; skips if already present. Returns updated doc dict."""
    dest = base_dir / doc["local_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        return {**doc, "downloaded": True, "skipped": True}

    try:
        resp = requests.get(doc["url"], timeout=60, stream=True)
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=16_384):
                fh.write(chunk)
        return {**doc, "downloaded": True, "skipped": False}
    except Exception as exc:
        return {**doc, "downloaded": False, "download_error": str(exc), "skipped": False}


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)

    # Collect links from all year pages
    all_docs: list[dict] = []
    print("Collecting document links...")
    for year, path in YEAR_PAGES.items():
        print(f"  {year} ", end="", flush=True)
        docs = collect_links(year, path)
        print(f"-> {len(docs)} documents")
        all_docs.extend(docs)

    # Deduplicate across years (some documents are cross-linked)
    seen_urls: set[str] = set()
    deduped: list[dict] = []
    for doc in all_docs:
        if doc["url"] not in seen_urls:
            seen_urls.add(doc["url"])
            deduped.append(doc)

    print(f"\nTotal unique documents: {len(deduped)}")

    # Download
    print("\nDownloading...")
    results: list[dict] = []
    for i, doc in enumerate(deduped, 1):
        result = download_doc(doc, DOCS_DIR)
        results.append(result)

        if result.get("skipped"):
            label = "SKIP"
        elif result["downloaded"]:
            label = "OK  "
        else:
            label = "ERR "

        print(f"  [{i:>4}/{len(deduped)}] {label}  {doc['filename'][:70]}")
        if not result.get("skipped"):
            time.sleep(DOWNLOAD_DELAY)

    # Write metadata manifest
    METADATA_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nMetadata written -> {METADATA_FILE}")

    ok = sum(1 for r in results if r["downloaded"])
    skipped = sum(1 for r in results if r.get("skipped"))
    err = sum(1 for r in results if not r["downloaded"])
    print(f"  Downloaded: {ok - skipped}  Already present: {skipped}  Errors: {err}")


if __name__ == "__main__":
    main()
