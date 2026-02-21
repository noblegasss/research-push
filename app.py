import html
import json
import os
import re
import smtplib
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4
from zoneinfo import ZoneInfo
from xml.etree import ElementTree as ET

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
try:
    import psycopg2
except Exception:
    psycopg2 = None
try:
    import feedparser
except Exception:
    feedparser = None
try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:
    streamlit_js_eval = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


ARXIV_API = "http://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works/https://doi.org/"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/"
PUBMED_ESEARCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EUROPEPMC_SEARCH_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
HIGH_IMPACT = {"nature", "science", "cell", "the lancet", "nejm", "jama", "bmj"}
STOPWORDS = {"the", "and", "for", "with", "from", "using", "study", "new", "that"}
FETCH_CACHE_TTL_SEC = 15 * 60
MAX_ABSTRACT_ENRICH = 24
ABSTRACT_ENRICH_WORKERS = 6
MAX_AI_ENHANCE = 6
MAX_CROSSREF_ABSTRACT_LOOKUPS = 8
MAX_DISPLAY_ABSTRACT_RECOVERY = 80
MAX_NO_ABSTRACT_AI = 20
MAX_NO_ABSTRACT_FULLTEXT = 20
JOURNAL_ALIASES = {
    "nature biotechnology": ["nat biotechnol"],
    "nature medicine": ["nat med"],
    "nature genetics": ["nat genet"],
    "nature neuroscience": ["nat neurosci"],
    "nature communications": ["nat commun"],
    "the lancet": ["lancet"],
    "the lancet oncology": ["lancet oncol"],
    "the lancet digital health": ["lancet digit health"],
    "the lancet neurology": ["lancet neurol"],
    "nejm": ["new england journal of medicine", "n engl j med"],
    "jama": ["journal of the american medical association"],
    "jama oncology": ["jama oncol"],
    "jama neurology": ["jama neurol"],
}
JOURNAL_RSS_FEEDS = {
    "nature": ["https://www.nature.com/nature.rss"],
    "nature biotechnology": ["https://www.nature.com/nbt.rss"],
    "nature medicine": ["https://www.nature.com/nm.rss"],
    "nature genetics": ["https://www.nature.com/ng.rss"],
    "nature neuroscience": ["https://www.nature.com/neuro.rss"],
    "nature communications": ["https://www.nature.com/ncomms.rss"],
}
JOURNAL_OPTIONS = [
    "Nature",
    "Nature Medicine",
    "Nature Biotechnology",
    "Nature Genetics",
    "Nature Neuroscience",
    "Nature Machine Intelligence",
    "Nature Communications",
    "Scientific Reports",
    "Science",
    "Science Translational Medicine",
    "Science Advances",
    "Science Robotics",
    "Science Immunology",
    "Cell",
    "Cell Reports",
    "Cell Metabolism",
    "Molecular Cell",
    "Cancer Cell",
    "The Lancet",
    "The Lancet Digital Health",
    "The Lancet Oncology",
    "The Lancet Neurology",
    "The Lancet Public Health",
    "NEJM",
    "JAMA",
    "JAMA Network Open",
    "JAMA Oncology",
    "JAMA Neurology",
    "BMJ",
    "BMJ Open",
    "eLife",
    "PLOS ONE",
    "PLOS Biology",
    "PLOS Medicine",
    "PNAS Nexus",
    "PNAS",
    "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "IEEE Transactions on Medical Imaging",
    "IEEE Transactions on Neural Networks and Learning Systems",
    "IEEE Journal of Biomedical and Health Informatics",
    "IEEE TPAMI",
    "Artificial Intelligence in Medicine",
    "Journal of Biomedical Informatics",
    "Bioinformatics",
    "Briefings in Bioinformatics",
    "Nucleic Acids Research",
    "Genome Biology",
    "Genome Research",
    "Journal of Clinical Oncology",
    "Annals of Oncology",
    "Blood",
    "Circulation",
    "European Heart Journal",
    "Radiology",
    "AJR American Journal of Roentgenology",
    "Medical Image Analysis",
    "NPJ Digital Medicine",
    "Frontiers in Immunology",
    "Frontiers in Oncology",
    "ICML",
    "NeurIPS",
    "CVPR",
    "ECCV",
    "ICCV",
    "MICCAI",
    "AAAI",
    "IJCAI",
    "ICLR",
    "ACL",
    "EMNLP",
    "NAACL",
    "COLING",
    "arXiv",
]
FIELD_OPTIONS = [
    "AI",
    "Machine Learning",
    "Statistics",
    "Healthcare",
    "Biology",
    "Medicine",
    "Oncology",
    "Cardiology",
    "Neuroscience",
    "Computer Vision",
    "NLP",
    "Bioinformatics",
]
COMMON_TIMEZONE_OPTIONS = [
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Phoenix",
    "America/Anchorage",
    "Pacific/Honolulu",
    "UTC",
    "Europe/London",
    "Europe/Paris",
    "Asia/Shanghai",
    "Asia/Tokyo",
]
SETTINGS_FILE = Path(".app_settings.json")
LAST_DIGEST_FILE = Path(".last_digest_state.json")
LOCAL_CACHE_FILE = Path(".local_cache.json")
BROWSER_SETTINGS_KEY = "research_digest_user_settings_v2"
ENV_PUBLIC_MODE = os.getenv("PUBLIC_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
IS_STREAMLIT_CLOUD = bool(os.getenv("STREAMLIT_SHARING_MODE")) or bool(os.getenv("STREAMLIT_CLOUD"))
# On Streamlit Community Cloud, default to public-safe mode unless explicitly overridden.
PUBLIC_MODE = ENV_PUBLIC_MODE or IS_STREAMLIT_CLOUD
# Server-side file persistence is OFF by default to avoid cross-user leakage.
# Enable only for trusted single-user/self-host deployments.
SERVER_PERSISTENCE = os.getenv("SERVER_PERSISTENCE", "0").strip().lower() in {"1", "true", "yes", "on"}
PERSIST_TO_DISK = (not PUBLIC_MODE) and SERVER_PERSISTENCE
AUTO_PUSH_MULTIUSER = os.getenv("AUTO_PUSH_MULTIUSER", "1").strip().lower() in {"1", "true", "yes", "on"}
AUTO_PUSH_DATABASE_URL = os.getenv("AUTO_PUSH_DATABASE_URL", "").strip()
AUTO_PUSH_DB_FILE = Path(os.getenv("AUTO_PUSH_DB_FILE", ".auto_push_subscriptions.sqlite3"))
AUTO_PUSH_BACKEND = "postgres" if AUTO_PUSH_DATABASE_URL else ("sqlite" if SERVER_PERSISTENCE else "")
AUTO_PUSH_ENABLED = AUTO_PUSH_MULTIUSER and bool(AUTO_PUSH_BACKEND)
SHOW_AUTO_PUSH_ADMIN_INFO = os.getenv("SHOW_AUTO_PUSH_ADMIN_INFO", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Paper:
    title: str
    authors: list[str]
    venue: str
    publication_date: str
    doi: str = ""
    pmid: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    url: str = ""


def L(lang: str, zh: str, en: str) -> str:
    return zh if (lang or "zh").lower().startswith("zh") else en


def now_utc() -> datetime:
    return datetime.now(UTC)


def now_local() -> datetime:
    tz_name = os.getenv("APP_TIMEZONE", "America/New_York").strip() or "America/New_York"
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return now_utc()


def parse_hhmm(text: str) -> tuple[int, int] | None:
    raw = (text or "").strip()
    m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", raw)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def normalize_hhmm(text: str) -> str:
    parsed = parse_hhmm(text)
    if parsed is None:
        return ""
    hh, mm = parsed
    return f"{hh:02d}:{mm:02d}"


def normalize_timezone(text: str) -> str:
    tz_name = (text or "").strip()
    if not tz_name:
        return ""
    try:
        ZoneInfo(tz_name)
        return tz_name
    except Exception:
        return ""


def window_start_date(days: int) -> datetime.date:
    return (now_local() - timedelta(days=max(0, int(days)))).date()


def in_day_window(dt: datetime | None, days: int) -> bool:
    if dt is None:
        return True
    cutoff_date = window_start_date(days)
    return dt.date() >= cutoff_date


def parse_date(text: str) -> datetime | None:
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(text, fmt)
            if fmt == "%Y":
                return datetime(dt.year, 1, 1, tzinfo=UTC)
            if fmt == "%Y-%m":
                return datetime(dt.year, dt.month, 1, tzinfo=UTC)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def parse_rss_datetime(text: str) -> datetime | None:
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def parse_pubmed_month(month: str) -> int:
    if not month:
        return 1
    m = month.strip().lower()
    mapping = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    if m.isdigit():
        v = int(m)
        return v if 1 <= v <= 12 else 1
    return mapping.get(m, 1)


def parse_pubmed_date_node(node: ET.Element | None) -> str:
    if node is None:
        return ""
    year = (node.findtext("Year", default="") or "").strip()
    month = (node.findtext("Month", default="") or "").strip()
    day = (node.findtext("Day", default="") or "").strip()
    if not year.isdigit():
        medline_date = (node.findtext("MedlineDate", default="") or "").strip()
        m = re.search(r"\b(19|20)\d{2}\b", medline_date)
        if not m:
            return ""
        year = m.group(0)
    y = int(year)
    m = parse_pubmed_month(month)
    d = int(day) if day.isdigit() and 1 <= int(day) <= 31 else 1
    try:
        return datetime(y, m, d).strftime("%Y-%m-%d")
    except ValueError:
        return datetime(y, 1, 1).strftime("%Y-%m-%d")


def normalize_str_list_input(value: Any) -> list[str]:
    parts: list[str] = []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        for item in value:
            s = str(item or "").strip()
            if not s:
                continue
            for tok in re.split(r"[,;\n]+", s):
                t = tok.strip()
                if t:
                    parts.append(t)
    out: list[str] = []
    for x in parts:
        if x not in out:
            out.append(x)
    return out


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def normalize_venue_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _journal_alias_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for canonical, aliases in JOURNAL_ALIASES.items():
        pairs.append((normalize_venue_text(canonical), normalize_venue_text(canonical)))
        for a in aliases:
            pairs.append((normalize_venue_text(a), normalize_venue_text(canonical)))
    # longest alias first to avoid parent journals swallowing subjournals
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def canonical_journal_name(raw: str) -> str:
    text = normalize_venue_text(raw)
    if not text:
        return ""
    for alias, canonical in _journal_alias_pairs():
        if alias and re.search(rf"\b{re.escape(alias)}\b", text):
            return canonical
    return text


def venue_matches_selected(venue: str, journals: list[str], strict: bool = True) -> bool:
    if not journals:
        return True
    venue_n = normalize_venue_text(venue)
    venue_canon = canonical_journal_name(venue_n)
    for j in journals:
        jn = normalize_venue_text(j)
        aliases = JOURNAL_ALIASES.get(jn, [])
        candidates = [jn] + [normalize_venue_text(a) for a in aliases]
        for c in candidates:
            if not c:
                continue
            if strict:
                # strict: canonical journal name must match exactly
                if venue_canon == canonical_journal_name(c):
                    return True
            else:
                # Relaxed direction: venue contains selected journal alias/name.
                if c in venue_n:
                    return True
    return False


def selected_rss_urls(journals: list[str]) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    for j in journals:
        jn = normalize_venue_text(j)
        candidates = [jn] + [normalize_venue_text(a) for a in JOURNAL_ALIASES.get(jn, [])]
        for c in candidates:
            for u in JOURNAL_RSS_FEEDS.get(c, []):
                pair = (jn, u)
                if pair not in urls:
                    urls.append(pair)
    return urls


def fetch_journal_rss(journals: list[str], days: int) -> list[Paper]:
    out: list[Paper] = []
    if not journals:
        return out
    cutoff = now_utc() - timedelta(days=days)
    feeds = selected_rss_urls(journals)
    for journal_norm, feed_url in feeds:
        feed_added = 0
        try:
            r = requests.get(
                feed_url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
                    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
                },
            )
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.text)
            # RSS 2.0
            for item in root.findall(".//item"):
                title = (item.findtext("title", default="") or "").strip()
                link = (item.findtext("link", default="") or "").strip()
                desc = clean_abstract((item.findtext("description", default="") or "").strip())
                pub = (item.findtext("pubDate", default="") or "").strip()
                dt = parse_rss_datetime(pub)
                if not in_day_window(dt, days):
                    continue
                date = dt.strftime("%Y-%m-%d") if dt else ""
                if not date:
                    # common RSS alternatives
                    alt_date = (
                        (item.findtext("{http://purl.org/dc/elements/1.1/date", default="") or "").strip()
                        or (item.findtext("{http://prismstandard.org/namespaces/basic/2.0/coverDate", default="") or "").strip()
                    )
                    if alt_date:
                        date = alt_date[:10]
                if not title:
                    continue
                out.append(
                    Paper(
                        title=title,
                        authors=[],
                        venue=journal_norm.title(),
                        publication_date=date,
                        abstract=desc,
                        url=link,
                    )
                )
                feed_added += 1
            # Atom (many publisher feeds use Atom `entry`)
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                title = (entry.findtext("{http://www.w3.org/2005/Atom}title", default="") or "").strip()
                link = ""
                link_node = entry.find("{http://www.w3.org/2005/Atom}link")
                if link_node is not None:
                    link = (link_node.attrib.get("href", "") or "").strip()
                summary = (
                    entry.findtext("{http://www.w3.org/2005/Atom}summary", default="")
                    or entry.findtext("{http://www.w3.org/2005/Atom}content", default="")
                    or ""
                ).strip()
                desc = clean_abstract(summary)
                pub = (
                    entry.findtext("{http://www.w3.org/2005/Atom}published", default="")
                    or entry.findtext("{http://www.w3.org/2005/Atom}updated", default="")
                    or entry.findtext("{http://purl.org/dc/elements/1.1/date", default="")
                    or ""
                ).strip()
                dt = parse_date(pub[:10]) if pub else None
                if not dt:
                    dt = parse_rss_datetime(pub)
                if not in_day_window(dt, days):
                    continue
                date = dt.strftime("%Y-%m-%d") if dt else (pub[:10] if pub else "")
                if not title:
                    continue
                out.append(
                    Paper(
                        title=title,
                        authors=[],
                        venue=journal_norm.title(),
                        publication_date=date,
                        abstract=desc,
                        url=link,
                    )
                )
                feed_added += 1
            # Fallback parser for feeds with uncommon namespace/structure.
            if feed_added == 0 and feedparser is not None:
                fp = feedparser.parse(r.text)
                for e in getattr(fp, "entries", [])[:200]:
                    title = (getattr(e, "title", "") or "").strip()
                    link = (getattr(e, "link", "") or "").strip()
                    desc = clean_abstract((getattr(e, "summary", "") or getattr(e, "description", "") or "").strip())
                    pub = (getattr(e, "published", "") or getattr(e, "updated", "") or "").strip()
                    dt = None
                    if getattr(e, "published_parsed", None):
                        try:
                            ts = datetime(*e.published_parsed[:6], tzinfo=UTC)
                            dt = ts
                        except Exception:
                            dt = None
                    if not dt and pub:
                        dt = parse_rss_datetime(pub) or parse_date(pub[:10])
                    if not in_day_window(dt, days):
                        continue
                    date = dt.strftime("%Y-%m-%d") if dt else (pub[:10] if pub else "")
                    if not title:
                        continue
                    out.append(
                        Paper(
                            title=title,
                            authors=[],
                            venue=journal_norm.title(),
                            publication_date=date,
                            abstract=desc,
                            url=link,
                        )
                    )
        except Exception:
            continue
    return out


def fetch_pubmed(
    query_terms: list[str],
    journals: list[str],
    days: int,
    strict_journal_only: bool = True,
    retmax_per_term: int = 40,
) -> list[Paper]:
    out: list[Paper] = []
    if not query_terms:
        return out
    cutoff = now_utc() - timedelta(days=days)
    seen_pmids: set[str] = set()

    for term in query_terms[:10]:
        try:
            esearch_params = {
                "db": "pubmed",
                "term": term,
                "retmode": "json",
                "retmax": retmax_per_term,
                "sort": "pub_date",
                "datetype": "pdat",
                "reldate": days,
            }
            sr = requests.get(PUBMED_ESEARCH_API, params=esearch_params, timeout=15)
            if sr.status_code != 200:
                continue
            idlist = sr.json().get("esearchresult", {}).get("idlist", []) or []
            ids = [str(x).strip() for x in idlist if str(x).strip() and str(x).strip() not in seen_pmids]
            if not ids:
                continue
            for pmid in ids:
                seen_pmids.add(pmid)
            fr = requests.get(
                PUBMED_EFETCH_API,
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
                timeout=20,
            )
            if fr.status_code != 200:
                continue
            root = ET.fromstring(fr.text)
            for article in root.findall(".//PubmedArticle"):
                pmid = (article.findtext(".//PMID", default="") or "").strip()
                if not pmid:
                    continue
                title_node = article.find(".//ArticleTitle")
                title = "".join(title_node.itertext()).strip() if title_node is not None else ""
                if not title:
                    continue
                abstract_parts: list[str] = []
                for abs_node in article.findall(".//Abstract/AbstractText"):
                    label = (abs_node.attrib.get("Label", "") or "").strip()
                    txt = "".join(abs_node.itertext()).strip()
                    if txt:
                        abstract_parts.append(f"{label}: {txt}" if label else txt)
                abstract = clean_abstract(" ".join(abstract_parts))
                venue = (
                    (article.findtext(".//MedlineJournalInfo/MedlineTA", default="") or "").strip()
                    or (article.findtext(".//Article/Journal/Title", default="") or "").strip()
                    or "PubMed"
                )

                date = ""
                date = parse_pubmed_date_node(article.find(".//Article/ArticleDate"))
                if not date:
                    date = parse_pubmed_date_node(article.find(".//Article/Journal/JournalIssue/PubDate"))
                if not date:
                    date = parse_pubmed_date_node(article.find(".//DateCompleted"))

                doi = ""
                for aid in article.findall(".//ArticleIdList/ArticleId"):
                    if (aid.attrib.get("IdType", "") or "").lower() == "doi":
                        doi = (aid.text or "").strip()
                        break

                authors: list[str] = []
                for a in article.findall(".//AuthorList/Author"):
                    fore = (a.findtext("ForeName", default="") or "").strip()
                    last = (a.findtext("LastName", default="") or "").strip()
                    full = f"{fore} {last}".strip()
                    if full:
                        authors.append(full)

                if journals and not venue_matches_selected(venue, journals, strict=strict_journal_only):
                    continue
                dt = parse_date(date)
                if not in_day_window(dt, days):
                    continue

                out.append(
                    Paper(
                        title=title,
                        authors=authors,
                        venue=venue,
                        publication_date=date,
                        doi=doi,
                        pmid=pmid,
                        abstract=abstract,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    )
                )
        except Exception:
            continue
    return out


def fetch_pubmed_abstract_by_pmid(pmid: str) -> str:
    if not pmid:
        return ""
    try:
        r = requests.get(
            PUBMED_EFETCH_API,
            params={"db": "pubmed", "id": pmid, "retmode": "xml"},
            timeout=15,
        )
        if r.status_code != 200:
            return ""
        root = ET.fromstring(r.text)
        parts: list[str] = []
        for abs_node in root.findall(".//Abstract/AbstractText"):
            label = (abs_node.attrib.get("Label", "") or "").strip()
            txt = "".join(abs_node.itertext()).strip()
            if txt:
                parts.append(f"{label}: {txt}" if label else txt)
        return clean_abstract(" ".join(parts))
    except Exception:
        return ""


def fetch_europepmc_abstract_by_doi(doi: str) -> str:
    if not doi:
        return ""
    try:
        query = f'DOI:"{doi}"'
        r = requests.get(
            EUROPEPMC_SEARCH_API,
            params={"query": query, "format": "json", "pageSize": 1},
            timeout=15,
        )
        if r.status_code != 200:
            return ""
        results = r.json().get("resultList", {}).get("result", []) or []
        if not results:
            return ""
        abstract = (results[0].get("abstractText") or "").strip()
        return clean_abstract(abstract)
    except Exception:
        return ""


def fetch_publisher_meta_abstract(url: str) -> str:
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser") if BeautifulSoup else None
        if soup is None:
            return ""
        candidates = [
            soup.find("meta", attrs={"name": "description"}),
            soup.find("meta", attrs={"property": "og:description"}),
            soup.find("meta", attrs={"name": "dc.description"}),
            soup.find("meta", attrs={"name": "summary"}),
            soup.find("meta", attrs={"property": "og:summary"}),
            soup.find("meta", attrs={"name": "twitter:description"}),
            soup.find("meta", attrs={"name": "citation_abstract"}),
            soup.find("meta", attrs={"name": "dc.Description"}),
        ]
        for node in candidates:
            if node and node.get("content"):
                txt = clean_abstract(str(node.get("content")))
                if len(txt) > 50:
                    return txt
        # Some publisher pages expose structured summary/description in JSON-LD.
        for node in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = (node.string or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            items = data if isinstance(data, list) else [data]
            for it in items:
                if not isinstance(it, dict):
                    continue
                for key in ("description", "summary", "abstract"):
                    val = clean_abstract(str(it.get(key) or ""))
                    if len(val) > 50:
                        return val
        return ""
    except Exception:
        return ""


def fetch_summary_from_fulltext(url: str) -> str:
    if not url or BeautifulSoup is None:
        return ""
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # Prefer section headings like "Summary" / "Abstract".
        heading = None
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "strong"]):
            t = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip().lower()
            if t in {"summary", "abstract"} or t.startswith("summary ") or t.startswith("abstract "):
                heading = tag
                break
        if heading is not None:
            chunks: list[str] = []
            for sib in heading.find_all_next():
                if sib == heading:
                    continue
                if getattr(sib, "name", "") in {"h1", "h2", "h3", "h4"}:
                    break
                if getattr(sib, "name", "") in {"p", "li"}:
                    txt = clean_abstract(sib.get_text(" ", strip=True))
                    if len(txt) > 30:
                        chunks.append(txt)
                    if sum(len(x) for x in chunks) > 1800:
                        break
            out = " ".join(chunks).strip()
            if len(out) > 80:
                return out[:2200]
        # Fallback: use main page text snippet as a pseudo-summary for OA pages.
        text = fetch_html_text(url, max_chars=8000)
        if len(text) > 120:
            # Keep an introductory slice instead of full text.
            return clean_abstract(text[:2200])
        return ""
    except Exception:
        return ""


def enrich_missing_abstracts(papers: list[Paper], max_enrich: int = 80) -> list[Paper]:
    def enrich_one(p: Paper) -> Paper:
        if p.abstract.strip():
            return p
        abstract = recover_missing_abstract_for_paper(p)
        if not abstract:
            return p
        return Paper(
            title=p.title,
            authors=p.authors,
            venue=p.venue,
            publication_date=p.publication_date,
            doi=p.doi,
            pmid=p.pmid,
            arxiv_id=p.arxiv_id,
            abstract=abstract,
            url=p.url,
        )

    if not papers:
        return []

    enriched: list[Paper] = list(papers)
    targets = [i for i, p in enumerate(papers) if not p.abstract.strip()][: max(0, max_enrich)]
    if not targets:
        return enriched

    workers = max(1, min(ABSTRACT_ENRICH_WORKERS, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {ex.submit(enrich_one, papers[i]): i for i in targets}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                enriched[idx] = fut.result()
            except Exception:
                enriched[idx] = papers[idx]
    return enriched


def recover_missing_abstract_for_paper(p: Paper) -> str:
    if p.abstract.strip():
        return p.abstract.strip()
    pid = paper_id(p)
    local_cache = st.session_state.get("local_cache", {})
    abs_cache = local_cache.get("abstract_by_id", {}) if isinstance(local_cache, dict) else {}
    if isinstance(abs_cache, dict):
        cached = str(abs_cache.get(pid, "")).strip()
        if cached:
            return cached
    abstract = ""
    if p.pmid:
        abstract = fetch_pubmed_abstract_by_pmid(p.pmid)
    if not abstract and p.doi:
        abstract = fetch_europepmc_abstract_by_doi(p.doi)
    if not abstract and (p.doi or p.arxiv_id or p.title):
        abstract = fetch_semantic_scholar_abstract(doi=p.doi, arxiv_id=p.arxiv_id, title=p.title)
    if not abstract and p.url:
        abstract = fetch_publisher_meta_abstract(p.url)
    if not abstract and p.url:
        abstract = fetch_summary_from_fulltext(p.url)
    abstract = abstract.strip()
    if abstract:
        if "local_cache" not in st.session_state or not isinstance(st.session_state.local_cache, dict):
            st.session_state.local_cache = {"abstract_by_id": {}, "llm_summary_cache": {}}
        st.session_state.local_cache.setdefault("abstract_by_id", {})[pid] = abstract
        st.session_state.local_cache_dirty = True
    return abstract


def enrich_digest_display_abstracts(digest: dict[str, Any], lang: str = "zh", max_cards: int = 80) -> None:
    cards: list[dict[str, Any]] = []
    for k in ("top_picks", "also_notable"):
        arr = digest.get(k, [])
        if isinstance(arr, list):
            cards.extend(arr)
    targets = [c for c in cards if not str(c.get("source_abstract", "")).strip()][: max(0, max_cards)]
    if not targets:
        return

    def enrich_card(c: dict[str, Any]) -> tuple[str, str]:
        pid = str(c.get("paper_id", ""))
        doi = pid[4:] if pid.startswith("doi:") else ""
        pmid = pid[5:] if pid.startswith("pmid:") else ""
        arxiv_id = pid[6:] if pid.startswith("arxiv:") else ""
        p = Paper(
            title=str(c.get("title", "")),
            authors=[],
            venue=str(c.get("venue", "")),
            publication_date=str(c.get("date", "")),
            doi=doi,
            pmid=pmid,
            arxiv_id=arxiv_id,
            abstract="",
            url=str(c.get("link", "")),
        )
        abs_text = recover_missing_abstract_for_paper(p)
        return pid, abs_text

    workers = max(1, min(ABSTRACT_ENRICH_WORKERS, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_card = {ex.submit(enrich_card, c): c for c in targets}
        for fut in as_completed(future_to_card):
            c = future_to_card[fut]
            try:
                _, abs_text = fut.result()
            except Exception:
                abs_text = ""
            if not abs_text:
                continue
            c["source_abstract"] = abs_text
            c["abstract_excerpt"] = abs_text
            c["evidence_note"] = "Based on abstract/metadata."
            # Refresh feed summary now that abstract is available.
            sc = c.get("scores", {"total": 0})
            p_tmp = Paper(
                title=str(c.get("title", "")),
                authors=[],
                venue=str(c.get("venue", "")),
                publication_date=str(c.get("date", "")),
                abstract=abs_text,
                url=str(c.get("link", "")),
            )
            c["ai_feed_summary"] = fallback_feed_summary(p_tmp, sc, lang=lang)


def clean_abstract(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text))).strip()


def paper_id(p: Paper) -> str:
    if p.doi:
        return f"doi:{p.doi}"
    if p.pmid:
        return f"pmid:{p.pmid}"
    if p.arxiv_id:
        return f"arxiv:{p.arxiv_id}"
    return f"title:{normalize_text(p.title)}"


def best_link(p: Paper) -> str:
    if p.doi:
        return f"https://doi.org/{p.doi}"
    if p.pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/"
    if p.url:
        return p.url
    if p.arxiv_id:
        return f"https://arxiv.org/abs/{p.arxiv_id}"
    return ""


def apply_proxy(link: str, proxy_prefix: str) -> str:
    if not link or not proxy_prefix:
        return link
    prefix = proxy_prefix.strip().rstrip("/")
    if not prefix:
        return link
    if link.startswith(prefix):
        return link
    return f"{prefix}/{link}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&display=swap');
        :root {
          --bg: #f2f6fb;
          --surface: #ffffff;
          --surface-soft: #f8fbff;
          --line: #d7e2ee;
          --text: #0f2438;
          --muted: #5c7288;
          --brand: #0a5ea8;
          --brand-2: #0d9d89;
          --brand-soft: #e8f2ff;
          --shadow: 0 16px 34px rgba(11, 28, 44, 0.08);
        }
        .stApp {
          background:
            radial-gradient(1200px 600px at -10% -10%, #dce9ff 0%, transparent 55%),
            radial-gradient(1200px 600px at 100% -20%, #dff8f0 0%, transparent 58%),
            linear-gradient(180deg, #f7faff 0%, #f2f6fb 100%),
            var(--bg);
          color: var(--text);
          font-family: "Sora", "Segoe UI", sans-serif;
        }
        .block-container {
          max-width: 1240px;
          padding-top: 2.8rem;
          padding-bottom: 2.4rem;
        }
        @media (max-width: 900px) {
          .block-container {
            padding-top: 3.6rem;
          }
        }
        .hero {
          padding: 1.05rem 1.12rem;
          border-radius: 16px;
          background:
            linear-gradient(120deg, #062a4b 0%, #0a5ea8 60%, #0d9d89 100%);
          color: #f8fbff;
          border: 1px solid #1f6797;
          margin-bottom: 1rem;
          box-shadow: 0 18px 34px rgba(6, 42, 75, 0.25);
        }
        .hero-title {
          margin: 0;
          font-size: 1.1rem;
          font-weight: 700;
          letter-spacing: 0.01em;
          color: #f4fbff;
        }
        .hero-sub {
          margin: 0.35rem 0 0 0;
          font-size: 0.82rem;
          color: rgba(236, 248, 255, 0.93);
        }
        .feed-title {
          font-size: 1.04rem;
          font-weight: 700;
          margin: 0;
          color: #11283f;
          line-height: 1.35;
        }
        .kpi {
          border-radius: 14px;
          padding: 0.8rem 0.95rem;
          background: var(--surface);
          border: 1px solid var(--line);
          box-shadow: 0 8px 22px rgba(12, 26, 44, 0.06);
        }
        .paper-card {
          border-radius: 15px;
          padding: 0.98rem 1.04rem;
          border: 1px solid var(--line);
          background: linear-gradient(180deg, var(--surface) 0%, var(--surface-soft) 100%);
          margin-bottom: 0.75rem;
          box-shadow: 0 8px 24px rgba(12, 26, 44, 0.07);
          transition: transform 120ms ease, box-shadow 120ms ease;
        }
        .paper-card:hover {
          transform: translateY(-1px);
          box-shadow: 0 14px 30px rgba(12, 26, 44, 0.11);
        }
        .meta {
          color: var(--muted);
          font-size: 0.8rem;
          margin-bottom: 0.55rem;
        }
        .pill {
          display: inline-block;
          border-radius: 999px;
          padding: 0.12rem 0.62rem;
          margin-right: 0.28rem;
          margin-top: 0.24rem;
          background: #edf5ff;
          color: #175182;
          font-size: 0.71rem;
          border: 1px solid #cfddf0;
          font-weight: 600;
        }
        .section-line {
          margin-top: 0.4rem;
          color: var(--text);
          font-size: 0.9rem;
          line-height: 1.56;
        }
        .settings-panel {
          border: 1px solid #d5e1ed;
          border-radius: 15px;
          background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
          padding: 0.9rem 1rem;
          margin-bottom: 0.9rem;
          box-shadow: 0 10px 24px rgba(12, 26, 44, 0.08);
        }
        .stTabs [data-baseweb="tab-list"] {
          gap: 8px;
          background: #e8f0f9;
          padding: 5px;
          border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
          border-radius: 10px;
          padding: 8px 14px;
          font-weight: 600;
          color: #183953 !important;
        }
        .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
          color: #244760 !important;
        }
        .stTabs [aria-selected="true"] {
          background: #ffffff !important;
          color: #0d4b7c !important;
          border: 1px solid #d1dfec !important;
          box-shadow: 0 4px 10px rgba(17, 42, 67, 0.08);
        }
        .stExpander {
          border: 1px solid #d7e3ef !important;
          border-radius: 12px !important;
          background: #ffffff;
          box-shadow: 0 6px 16px rgba(16, 35, 54, 0.06);
        }
        .stExpander summary, .stExpander summary * {
          color: #16324a !important;
        }
        .stButton > button {
          border-radius: 10px;
          border: 1px solid #c8d9eb;
          background: linear-gradient(180deg, #ffffff 0%, #f4f9ff 100%);
          color: #123956;
          font-weight: 600;
          min-height: 2.55rem;
          box-shadow: 0 4px 12px rgba(17, 42, 67, 0.08);
        }
        .stButton > button:hover {
          border-color: #89afd5;
          color: #0f3053;
          transform: translateY(-1px);
        }
        .stStatus {
          border-radius: 12px !important;
          border: 1px solid #d6e2ee !important;
          box-shadow: var(--shadow);
        }
        .toolbar-title {
          margin: 0.45rem 0 0.35rem 0;
          color: var(--muted);
          font-weight: 600;
          font-size: 0.82rem;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }
        .value-high { color: #166534; font-weight: 700; }
        .value-mid { color: #854d0e; font-weight: 700; }
        .value-low { color: #991b1b; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fetch_arxiv(keywords: list[str], days: int) -> list[Paper]:
    out: list[Paper] = []
    cutoff = now_utc() - timedelta(days=days)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for kw in keywords:
        try:
            r = requests.get(
                ARXIV_API,
                params={"search_query": f"all:{kw}", "start": 0, "max_results": 20, "sortBy": "submittedDate", "sortOrder": "descending"},
                timeout=20,
            )
            r.raise_for_status()
            root = ET.fromstring(r.text)
            for e in root.findall("atom:entry", ns):
                title = (e.findtext("atom:title", default="", namespaces=ns) or "").strip()
                abstract = (e.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                published = (e.findtext("atom:published", default="", namespaces=ns) or "").strip()[:10]
                dt = parse_date(published)
                if not in_day_window(dt, days):
                    continue
                url = (e.findtext("atom:id", default="", namespaces=ns) or "").strip()
                authors = [(a.findtext("atom:name", default="", namespaces=ns) or "").strip() for a in e.findall("atom:author", ns)]
                out.append(Paper(title=title, authors=[x for x in authors if x], venue="arXiv", publication_date=published, arxiv_id=url.split("/")[-1], abstract=abstract, url=url))
        except Exception:
            continue
    return out


def crossref_date(item: dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "created", "issued"):
        parts = (item.get(key) or {}).get("date-parts", [])
        if parts and parts[0]:
            y = parts[0][0]
            m = parts[0][1] if len(parts[0]) > 1 else 1
            d = parts[0][2] if len(parts[0]) > 2 else 1
            try:
                return datetime(y, m, d).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return ""


def fetch_crossref(keywords: list[str], journals: list[str], days: int, strict_journal_only: bool = True) -> list[Paper]:
    out: list[Paper] = []
    start_date = window_start_date(days)
    openalex_cache: dict[str, str] = {}
    external_abstract_lookups = 0
    for kw in keywords:
        try:
            r = requests.get(
                CROSSREF_API,
                params={"query": kw, "filter": f"from-pub-date:{start_date.isoformat()}", "rows": 30, "sort": "published", "order": "desc"},
                timeout=20,
            )
            r.raise_for_status()
            for it in r.json().get("message", {}).get("items", []):
                title = ((it.get("title") or [""])[0] or "").strip()
                if not title:
                    continue
                venue = ((it.get("container-title") or [""])[0] or "").strip() or "Unknown Venue"
                if journals and not venue_matches_selected(venue, journals, strict=strict_journal_only):
                    continue
                date = crossref_date(it)
                dt = parse_date(date)
                if not in_day_window(dt, days):
                    continue
                authors = []
                for a in it.get("author", []) or []:
                    full = f"{(a.get('given') or '').strip()} {(a.get('family') or '').strip()}".strip()
                    if full:
                        authors.append(full)
                doi = (it.get("DOI") or "").strip()
                abstract = clean_abstract(it.get("abstract") or "")
                if not abstract and doi and external_abstract_lookups < MAX_CROSSREF_ABSTRACT_LOOKUPS:
                    if doi in openalex_cache:
                        abstract = openalex_cache[doi]
                    else:
                        abstract = fetch_openalex_abstract_by_doi(doi)
                        openalex_cache[doi] = abstract
                        external_abstract_lookups += 1
                if not abstract and external_abstract_lookups < MAX_CROSSREF_ABSTRACT_LOOKUPS:
                    abstract = fetch_semantic_scholar_abstract(doi=doi, title=title)
                    external_abstract_lookups += 1
                out.append(
                    Paper(
                        title=title,
                        authors=authors,
                        venue=venue,
                        publication_date=date,
                        doi=doi,
                        abstract=abstract,
                        url=(it.get("URL") or "").strip(),
                    )
                )
        except Exception:
            continue
    return out


def fetch_crossref_by_journals(journals: list[str], days: int, strict_journal_only: bool = True) -> list[Paper]:
    out: list[Paper] = []
    start_date = window_start_date(days)
    openalex_cache: dict[str, str] = {}
    external_abstract_lookups = 0
    for journal in journals[:30]:
        j = (journal or "").strip()
        if not j:
            continue
        try:
            r = requests.get(
                CROSSREF_API,
                params={
                    "query.container-title": j,
                    "filter": f"from-pub-date:{start_date.isoformat()}",
                    "rows": 20,
                    "sort": "published",
                    "order": "desc",
                },
                timeout=20,
            )
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", []) or []
            for it in items:
                title = ((it.get("title") or [""])[0] or "").strip()
                if not title:
                    continue
                venue = ((it.get("container-title") or [""])[0] or "").strip() or "Unknown Venue"
                if journals and not venue_matches_selected(venue, journals, strict=strict_journal_only):
                    continue
                date = crossref_date(it)
                dt = parse_date(date)
                if not in_day_window(dt, days):
                    continue
                authors: list[str] = []
                for a in it.get("author", []) or []:
                    full = f"{(a.get('given') or '').strip()} {(a.get('family') or '').strip()}".strip()
                    if full:
                        authors.append(full)
                doi = (it.get("DOI") or "").strip()
                abstract = clean_abstract(it.get("abstract") or "")
                if not abstract and doi and external_abstract_lookups < MAX_CROSSREF_ABSTRACT_LOOKUPS:
                    if doi in openalex_cache:
                        abstract = openalex_cache[doi]
                    else:
                        abstract = fetch_openalex_abstract_by_doi(doi)
                        openalex_cache[doi] = abstract
                        external_abstract_lookups += 1
                if not abstract and external_abstract_lookups < MAX_CROSSREF_ABSTRACT_LOOKUPS:
                    abstract = fetch_semantic_scholar_abstract(doi=doi, title=title)
                    external_abstract_lookups += 1
                out.append(
                    Paper(
                        title=title,
                        authors=authors,
                        venue=venue,
                        publication_date=date,
                        doi=doi,
                        abstract=abstract,
                        url=(it.get("URL") or "").strip(),
                    )
                )
        except Exception:
            continue
    return out


def fetch_openalex_abstract_by_doi(doi: str) -> str:
    if not doi:
        return ""
    try:
        safe_doi = quote(doi, safe="")
        r = requests.get(f"{OPENALEX_API}{safe_doi}", timeout=12)
        if r.status_code != 200:
            return ""
        data = r.json()
        inv = data.get("abstract_inverted_index", {})
        if not isinstance(inv, dict) or not inv:
            return ""
        pos_to_word: dict[int, str] = {}
        for word, positions in inv.items():
            if not isinstance(positions, list):
                continue
            for pos in positions:
                if isinstance(pos, int):
                    pos_to_word[pos] = word
        if not pos_to_word:
            return ""
        text = " ".join(pos_to_word[i] for i in sorted(pos_to_word.keys()))
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def fetch_semantic_scholar_abstract(doi: str = "", arxiv_id: str = "", title: str = "") -> str:
    try_ids = []
    if doi:
        try_ids.append(f"DOI:{doi}")
    if arxiv_id:
        try_ids.append(f"ARXIV:{arxiv_id}")
    for pid in try_ids:
        try:
            r = requests.get(
                f"{SEMANTIC_SCHOLAR_API}{quote(pid, safe='')}",
                params={"fields": "abstract"},
                timeout=12,
            )
            if r.status_code == 200:
                abstract = (r.json().get("abstract") or "").strip()
                if abstract:
                    return re.sub(r"\s+", " ", abstract)
        except Exception:
            continue
    if title:
        try:
            r = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": title, "limit": 1, "fields": "title,abstract"},
                timeout=12,
            )
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    abstract = (data[0].get("abstract") or "").strip()
                    if abstract:
                        return re.sub(r"\s+", " ", abstract)
        except Exception:
            return ""
    return ""


def fetch_pdf_text(pdf_url: str, max_pages: int = 8, max_chars: int = 40000) -> str:
    if PdfReader is None:
        return ""
    try:
        r = requests.get(pdf_url, timeout=20)
        if r.status_code != 200 or not r.content:
            return ""
        import io

        reader = PdfReader(io.BytesIO(r.content))
        parts: list[str] = []
        for page in reader.pages[:max_pages]:
            txt = (page.extract_text() or "").strip()
            if txt:
                parts.append(txt)
            if sum(len(p) for p in parts) >= max_chars:
                break
        text = "\n\n".join(parts)
        return text[:max_chars]
    except Exception:
        return ""


def fetch_html_text(url: str, max_chars: int = 40000) -> str:
    if BeautifulSoup is None:
        return ""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        candidates = []
        for sel in ["article", "main", "section"]:
            for node in soup.select(sel):
                txt = node.get_text(" ", strip=True)
                if len(txt) > 200:
                    candidates.append(txt)
        if not candidates:
            txt = soup.get_text(" ", strip=True)
            candidates = [txt]
        best = max(candidates, key=len) if candidates else ""
        best = re.sub(r"\s+", " ", best).strip()
        return best[:max_chars]
    except Exception:
        return ""


def fetch_paper_content(paper: Paper) -> tuple[str, str]:
    if paper.arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
        txt = fetch_pdf_text(pdf_url)
        if txt:
            return txt, "arXiv PDF"
    link = best_link(paper)
    if link and "arxiv.org/abs/" in link:
        arxiv_id = link.rstrip("/").split("/")[-1]
        txt = fetch_pdf_text(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        if txt:
            return txt, "arXiv PDF"
    if link:
        txt = fetch_html_text(link)
        if txt:
            return txt, "Publisher HTML"
    return "", ""


def merge(a: Paper, b: Paper) -> Paper:
    return Paper(a.title or b.title, a.authors or b.authors, a.venue or b.venue, a.publication_date or b.publication_date, a.doi or b.doi, a.pmid or b.pmid, a.arxiv_id or b.arxiv_id, a.abstract or b.abstract, best_link(a) or best_link(b))


def dedupe(papers: list[Paper]) -> list[Paper]:
    by_id: dict[str, Paper] = {}
    by_fallback: dict[str, Paper] = {}
    for p in papers:
        ids = []
        if p.doi:
            ids.append(f"doi:{p.doi.lower()}")
        if p.pmid:
            ids.append(f"pmid:{p.pmid.lower()}")
        if p.arxiv_id:
            ids.append(f"arxiv:{p.arxiv_id.lower()}")
        if ids:
            hit = next((i for i in ids if i in by_id), None)
            if hit:
                merged = merge(by_id[hit], p)
                for i in ids:
                    by_id[i] = merged
            else:
                for i in ids:
                    by_id[i] = p
            continue
        year = (p.publication_date or "0000")[:4]
        first = p.authors[0] if p.authors else "unknown"
        key = f"{normalize_text(p.title)}|{normalize_text(first)}|{year}"
        by_fallback[key] = merge(by_fallback[key], p) if key in by_fallback else p
    return list({id(v): v for v in list(by_id.values()) + list(by_fallback.values())}.values())


def match_count(text: str, terms: list[str]) -> int:
    low = text.lower()
    return sum(1 for t in terms if t.lower() in low)


def passes(p: Paper, prefs: dict[str, Any]) -> bool:
    text = f"{p.title} {p.abstract}".lower()
    if any(t.lower() in text for t in prefs.get("exclude_keywords", [])):
        return False
    journals = prefs.get("journals", [])
    # Keep fields as discovery/ranking hints; only explicit keywords are hard filters.
    keywords = prefs.get("keywords", [])
    if journals and not venue_matches_selected(
        p.venue or "",
        journals,
        strict=bool(prefs.get("strict_journal_only", True)),
    ):
        return False
    # Journal subscriptions are treated as primary scope. When journals are selected,
    # keywords become ranking signals instead of hard gates.
    if (not journals) and keywords and not any(k.lower() in text for k in keywords):
        return False
    dt = parse_date(p.publication_date)
    if not in_day_window(dt, int(prefs.get("date_range_days", 14))):
        return False
    return True


def score(p: Paper, prefs: dict[str, Any]) -> dict[str, int]:
    text = f"{p.title} {p.abstract}".lower()
    kw = prefs.get("keywords", [])
    fields = prefs.get("fields", [])
    rel = min(100, (0 if not kw else round(60 * match_count(text, kw) / len(kw))) + (0 if not fields else round(20 * match_count(text, fields) / len(fields))) + (20 if prefs.get("journals") and venue_matches_selected((p.venue or ""), prefs["journals"], strict=bool(prefs.get("strict_journal_only", True))) else 10 if not prefs.get("journals") else 0) - (10 if not p.abstract else 0))
    cues = ["novel", "new", "first", "we propose", "benchmark", "dataset", "state-of-the-art"]
    nov = min(100, 35 + min(40, 8 * sum(1 for c in cues if c in text)) + (10 if (parse_date(p.publication_date) and (now_utc() - parse_date(p.publication_date)).days <= 30) else 0) + (5 if p.venue.lower() == "arxiv" else 0))
    if not p.abstract:
        nov = min(nov, 45)
    rig = 25 if not p.abstract else min(100, 30 + min(25, 6 * sum(1 for c in ["randomized", "systematic review", "meta-analysis", "external validation", "prospective"] if c in text)) + min(20, 4 * sum(1 for c in ["confidence interval", "p-value", "cross-validation", "ablation"] if c in text)) + (15 if re.search(r"\\bn\\s*=\\s*\\d+\\b|\\b\\d+\\s+participants\\b|\\bcohort\\b", text) else 0))
    imp = round(0.4 * rel + 0.3 * nov + 0.3 * rig + (10 if any(v in (p.venue or "").lower() for v in HIGH_IMPACT) else 0) - (10 if not p.abstract else 0))
    rel, nov, rig, imp = max(0, rel), max(0, nov), max(0, rig), max(0, min(100, imp))
    w = prefs.get("ranking_weights", {"relevance": 0.35, "novelty": 0.25, "rigor": 0.25, "impact": 0.15})
    s = sum(float(w.get(k, 0)) for k in ("relevance", "novelty", "rigor", "impact")) or 1.0
    total = round((float(w.get("relevance", 0.35)) / s) * rel + (float(w.get("novelty", 0.25)) / s) * nov + (float(w.get("rigor", 0.25)) / s) * rig + (float(w.get("impact", 0.15)) / s) * imp)
    return {"relevance": rel, "novelty": nov, "rigor": rig, "impact": imp, "total": total}


def value_label(total: int, lang: str = "zh") -> str:
    if total >= 75:
        return L(lang, "高价值（建议优先读）", "High value (read first)")
    if total >= 60:
        return L(lang, "中等价值（可按需读）", "Medium value (read as needed)")
    return L(lang, "低价值（可延后）", "Low value (defer)")


def intro_method_value(p: Paper, sc: dict[str, int], prefs: dict[str, Any]) -> dict[str, str]:
    lang = prefs.get("language", "zh")
    matched = [k for k in prefs.get("keywords", []) if k.lower() in f"{p.title} {p.abstract}".lower()]
    hit = L(lang, f"命中关键词“{matched[0]}”", f"matched keyword '{matched[0]}'") if matched else L(lang, "与主题相关", "relevant to your topic")
    intro = L(
        lang,
        f"这篇论文聚焦于《{p.title}》相关问题，{hit}，发表于 {p.venue or 'Unknown Venue'}。",
        f"This paper focuses on '{p.title}', {hit}, published in {p.venue or 'Unknown Venue'}.",
    )
    if p.abstract:
        low = p.abstract.lower()
        if "randomized" in low:
            method = L(lang, "方法上采用随机对照设计，重点验证核心假设的因果效应。", "Uses a randomized controlled design to test causal effects of the core hypothesis.")
        elif "meta-analysis" in low or "systematic review" in low:
            method = L(lang, "方法上采用系统综述或Meta分析，对已有证据进行整合。", "Uses a systematic review or meta-analysis to synthesize existing evidence.")
        elif "benchmark" in low:
            method = L(lang, "方法上以基准测试为主，对比不同方案在同一任务下表现。", "Uses benchmark evaluation to compare methods on the same task.")
        elif "cohort" in low:
            method = L(lang, "方法上使用队列数据进行关联或效果评估。", "Uses cohort data for association or effect estimation.")
        else:
            method = L(lang, "方法细节来自摘要，属于实证或实验验证导向。", "Method details are abstract-based and indicate an empirical/experimental validation path.")
    else:
        method = L(lang, "摘要缺失，方法细节无法确认。", "Abstract unavailable; method details cannot be confirmed.")
    value = L(
        lang,
        "基于题名与摘要判断：具备进一步阅读价值。",
        "Based on title and abstract: worth further reading.",
    )
    return {"intro": intro, "method_summary": method, "value_assessment": value}


def detailed_sections(p: Paper, sc: dict[str, int], lang: str = "zh") -> dict[str, str]:
    if not p.abstract:
        return {
            "methods_detailed": L(
                lang,
                "摘要缺失，无法确认具体实验流程、数据设置和评估协议；仅能推断为该主题的常见研究范式。",
                "Abstract unavailable; exact experimental pipeline, data setup, and evaluation protocol cannot be confirmed.",
            ),
            "main_conclusion": L(
                lang,
                "当前仅能基于标题与元数据判断主题相关性，结论暂不确定。",
                "At this stage, only topic relevance can be inferred from title/metadata; conclusions remain uncertain.",
            ),
            "future_direction": L(
                lang,
                "建议补读正文后关注：样本规模、外部验证、消融实验与可复现性细节。",
                "After full-text reading, focus on sample size, external validation, ablations, and reproducibility details.",
            ),
        }
    low = p.abstract.lower()
    if "randomized" in low:
        methods = L(lang, "采用随机对照设计进行干预/对比，核心在于通过分组控制偏差并评估因果效应。", "Uses a randomized controlled intervention/comparison design to reduce bias and estimate causal effects.")
    elif "meta-analysis" in low or "systematic review" in low:
        methods = L(lang, "采用系统综述/Meta分析整合既有研究，方法重点是检索策略、纳排标准与异质性处理。", "Uses a systematic review/meta-analysis, focusing on search strategy, inclusion criteria, and heterogeneity handling.")
    elif "benchmark" in low or "state-of-the-art" in low:
        methods = L(lang, "以基准测试为核心，对多个方法在统一任务和指标下进行性能比较，并强调结果可比性。", "Centers on benchmark testing across methods under unified tasks/metrics for comparable performance evaluation.")
    elif "cohort" in low:
        methods = L(lang, "基于队列数据开展关联或效果分析，关键在协变量控制、分层分析与稳健性检验。", "Uses cohort data for association/effect analysis with covariate control, subgrouping, and robustness checks.")
    else:
        methods = L(lang, "基于摘要可见信息，属于实证研究路径：提出问题、给出方法并通过实验或数据分析验证。", "From available abstract evidence, this follows an empirical path: problem framing, method proposal, and validation via experiments/data analysis.")

    if any(k in low for k in ["improve", "outperform", "effective", "significant", "better"]):
        conclusion = L(lang, "摘要显示该方法在目标任务上有正向表现或统计意义，但具体增益幅度需以正文结果表格为准。", "The abstract suggests positive or statistically meaningful results on the target task, but exact gains should be verified in full-text result tables.")
    else:
        conclusion = L(lang, "摘要未给出充分量化结果，结论应视为初步信号，需结合正文实验细节进一步确认。", "The abstract lacks sufficient quantitative detail; treat conclusions as preliminary until full-text experiment details are reviewed.")

    future = L(lang, "后续建议关注跨数据集泛化、外部验证、真实场景部署效果以及失败案例分析。", "Next, focus on cross-dataset generalization, external validation, real-world deployment outcomes, and failure-case analysis.")
    return {
        "methods_detailed": methods,
        "main_conclusion": conclusion,
        "future_direction": future,
    }


def abstract_snippet(text: str, max_chars: int = 480, lang: str = "zh") -> str:
    if not text:
        return L(lang, "暂无可用摘要。", "Abstract unavailable.")
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars]
    # try to end at sentence boundary
    for sep in ("。", ".", ";", "；"):
        idx = cut.rfind(sep)
        if idx > 120:
            return cut[: idx + 1]
    return cut + "..."


def infer_paper_type(title: str, abstract: str) -> str:
    text = f"{title} {abstract}".lower()
    if any(k in text for k in ["systematic review", "meta-analysis", "review", "survey"]):
        return "Review"
    if any(k in text for k in ["protocol", "study protocol"]):
        return "Protocol"
    if any(k in text for k in ["case report", "case series"]):
        return "Case Report"
    if any(k in text for k in ["randomized", "trial", "cohort", "retrospective", "prospective"]):
        return "Original Research"
    if any(k in text for k in ["benchmark", "dataset", "method", "framework"]):
        return "Method / Benchmark"
    return "Research Article"


def paper_type_label(paper_type: str, lang: str = "zh") -> str:
    labels = {
        "Review": L(lang, "综述", "Review"),
        "Protocol": L(lang, "研究方案", "Protocol"),
        "Case Report": L(lang, "病例报告", "Case Report"),
        "Original Research": L(lang, "原创研究", "Original Research"),
        "Method / Benchmark": L(lang, "方法/基准", "Method / Benchmark"),
        "Research Article": L(lang, "研究论文", "Research Article"),
    }
    return labels.get(paper_type, paper_type)


def load_saved_settings(defaults: dict[str, Any]) -> dict[str, Any]:
    if not PERSIST_TO_DISK:
        return defaults
    if not SETTINGS_FILE.exists():
        return defaults
    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        merged = defaults.copy()
        merged.update(raw if isinstance(raw, dict) else {})
        return merged
    except Exception:
        return defaults


def save_settings(data: dict[str, Any]) -> tuple[bool, str]:
    lang = data.get("language", "zh")
    if not PERSIST_TO_DISK:
        return True, L(lang, "会话模式：设置仅保留在当前会话。", "Session mode: settings are session-only.")
    try:
        SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, L(lang, f"设置已保存到 {SETTINGS_FILE}", f"Settings saved to {SETTINGS_FILE}")
    except Exception as exc:
        return False, L(lang, f"保存失败：{exc}", f"Save failed: {exc}")


def load_last_digest_state() -> dict[str, Any]:
    if not PERSIST_TO_DISK:
        return {}
    if not LAST_DIGEST_FILE.exists():
        return {}
    try:
        raw = json.loads(LAST_DIGEST_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def save_last_digest_state(data: dict[str, Any]) -> None:
    if not PERSIST_TO_DISK:
        return
    try:
        LAST_DIGEST_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_local_cache() -> dict[str, Any]:
    if not PERSIST_TO_DISK:
        return {"abstract_by_id": {}, "llm_summary_cache": {}}
    if not LOCAL_CACHE_FILE.exists():
        return {"abstract_by_id": {}, "llm_summary_cache": {}}
    try:
        raw = json.loads(LOCAL_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {"abstract_by_id": {}, "llm_summary_cache": {}}
        return {
            "abstract_by_id": raw.get("abstract_by_id", {}) if isinstance(raw.get("abstract_by_id", {}), dict) else {},
            "llm_summary_cache": raw.get("llm_summary_cache", {}) if isinstance(raw.get("llm_summary_cache", {}), dict) else {},
        }
    except Exception:
        return {"abstract_by_id": {}, "llm_summary_cache": {}}


def save_local_cache(data: dict[str, Any]) -> None:
    if not PERSIST_TO_DISK:
        return
    try:
        out = dict(data)
        abs_map = out.get("abstract_by_id", {})
        llm_map = out.get("llm_summary_cache", {})
        if isinstance(abs_map, dict) and len(abs_map) > 3000:
            keys = list(abs_map.keys())[-3000:]
            out["abstract_by_id"] = {k: abs_map[k] for k in keys}
        if isinstance(llm_map, dict) and len(llm_map) > 1000:
            keys = list(llm_map.keys())[-1000:]
            out["llm_summary_cache"] = {k: llm_map[k] for k in keys}
        LOCAL_CACHE_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_browser_settings(defaults: dict[str, Any]) -> dict[str, Any]:
    if streamlit_js_eval is None:
        return {}
    try:
        raw = streamlit_js_eval(
            js_expressions=f"localStorage.getItem('{BROWSER_SETTINGS_KEY}')",
            key="browser_settings_get",
            want_output=True,
        )
        if not raw or not isinstance(raw, str):
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        allowed = set(defaults.keys())
        out = {k: v for k, v in parsed.items() if k in allowed}
        out["journals"] = normalize_str_list_input(out.get("journals", []))
        out["fields"] = normalize_str_list_input(out.get("fields", []))
        return out
    except Exception:
        return {}


def save_browser_settings(data: dict[str, Any]) -> None:
    if streamlit_js_eval is None:
        return
    try:
        payload = json.dumps(data, ensure_ascii=False)
        js = f"(function(){{localStorage.setItem('{BROWSER_SETTINGS_KEY}', {json.dumps(payload)}); return true;}})()"
        streamlit_js_eval(
            js_expressions=js,
            key=f"browser_settings_set_{abs(hash(payload)) % 1000000000}",
            want_output=True,
        )
    except Exception:
        return


def clear_browser_settings() -> None:
    if streamlit_js_eval is None:
        return
    try:
        streamlit_js_eval(
            js_expressions=f"localStorage.removeItem('{BROWSER_SETTINGS_KEY}')",
            key="browser_settings_clear",
            want_output=False,
        )
    except Exception:
        return


def normalize_push_schedule(value: str) -> str:
    schedule = str(value or "daily").strip().lower()
    if schedule == "weekly (monday)":
        return "weekly_monday"
    if schedule in {"daily", "weekly_monday", "custom"}:
        return schedule
    return "daily"


def push_days_by_schedule(schedule: str, custom_days: int) -> int:
    if schedule == "daily":
        return 1
    if schedule == "weekly_monday":
        return 7
    return max(1, int(custom_days or 1))


def build_runtime_prefs_from_settings(settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    def parse_setting_tokens(value: Any) -> list[str]:
        if isinstance(value, list):
            return normalize_str_list_input(value)
        return parse_csv(str(value or ""))

    selected_fields = normalize_str_list_input(settings.get("fields", []))
    selected_journals = normalize_str_list_input(settings.get("journals", []))
    custom_fields = str(settings.get("custom_fields", ""))
    custom_journals = str(settings.get("custom_journals", ""))
    push_schedule = normalize_push_schedule(str(settings.get("push_schedule", "daily")))
    custom_days = int(settings.get("custom_days", 7))
    if selected_fields:
        fields = list(dict.fromkeys(selected_fields))
    else:
        fields = list(dict.fromkeys(parse_csv(custom_fields)))
    if selected_journals:
        journals = list(dict.fromkeys(selected_journals))
    else:
        journals = list(dict.fromkeys(parse_csv(custom_journals)))
    keywords = parse_setting_tokens(settings.get("keywords", ""))
    exclude = parse_setting_tokens(settings.get("exclude_keywords", "survey"))
    days = push_days_by_schedule(push_schedule, custom_days)
    prefs = {
        "language": str(settings.get("language", "zh")),
        "fields": fields,
        "journals": journals,
        "strict_journal_only": bool(settings.get("strict_journal_only", True)),
        "keywords": keywords,
        "exclude_keywords": exclude,
        "date_range_days": days,
        "max_papers": int(settings.get("max_papers", 0)),
        "reading_level": "mixed",
        "ranking_weights": {"relevance": 0.35, "novelty": 0.25, "rigor": 0.25, "impact": 0.15},
    }
    return prefs, {
        "push_schedule": push_schedule,
        "custom_days": custom_days,
        "daily_push_time": normalize_hhmm(str(settings.get("daily_push_time", "09:00"))) or "09:00",
        "push_timezone": normalize_timezone(
            str(settings.get("push_timezone", os.getenv("APP_TIMEZONE", "America/New_York")))
        )
        or "America/New_York",
    }


def auto_push_backend_reason() -> str:
    if not AUTO_PUSH_MULTIUSER:
        return "AUTO_PUSH_MULTIUSER is disabled."
    if AUTO_PUSH_BACKEND == "postgres" and psycopg2 is None:
        return "AUTO_PUSH_DATABASE_URL is set but psycopg2 is not installed."
    if not AUTO_PUSH_BACKEND:
        return "No backend configured. Set AUTO_PUSH_DATABASE_URL or SERVER_PERSISTENCE=1."
    return ""


def init_auto_push_db() -> None:
    if not AUTO_PUSH_ENABLED:
        return
    if AUTO_PUSH_BACKEND == "postgres":
        if psycopg2 is None:
            return
        try:
            with psycopg2.connect(AUTO_PUSH_DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS auto_push_subscriptions (
                            subscriber_id TEXT PRIMARY KEY,
                            enabled INTEGER NOT NULL DEFAULT 0,
                            schedule TEXT NOT NULL,
                            custom_days INTEGER NOT NULL DEFAULT 7,
                            push_time TEXT NOT NULL,
                            timezone TEXT NOT NULL,
                            settings_json TEXT NOT NULL,
                            last_run_local_date TEXT NOT NULL DEFAULT '',
                            last_error TEXT NOT NULL DEFAULT '',
                            updated_at_utc TEXT NOT NULL
                        )
                        """
                    )
            return
        except Exception:
            return
    if AUTO_PUSH_BACKEND != "sqlite":
        return
    try:
        with sqlite3.connect(AUTO_PUSH_DB_FILE) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auto_push_subscriptions (
                    subscriber_id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    schedule TEXT NOT NULL,
                    custom_days INTEGER NOT NULL DEFAULT 7,
                    push_time TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    last_run_local_date TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
    except Exception:
        return


def upsert_auto_push_subscription(subscriber_id: str, settings: dict[str, Any], enabled: bool) -> tuple[bool, str]:
    if not AUTO_PUSH_ENABLED:
        return False, auto_push_backend_reason() or "Auto-push persistence is disabled."
    sid = (subscriber_id or "").strip()
    if not sid:
        return False, "Missing subscriber id."
    init_auto_push_db()
    prefs, plan = build_runtime_prefs_from_settings(settings)
    packed_settings = dict(settings)
    packed_settings["derived_prefs"] = prefs
    params = (
        sid,
        1 if enabled else 0,
        plan["push_schedule"],
        int(plan["custom_days"]),
        plan["daily_push_time"],
        plan["push_timezone"],
        json.dumps(packed_settings, ensure_ascii=False),
        now_utc().isoformat(),
    )
    if AUTO_PUSH_BACKEND == "postgres":
        if psycopg2 is None:
            return False, "psycopg2 is not installed."
        try:
            with psycopg2.connect(AUTO_PUSH_DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO auto_push_subscriptions
                        (subscriber_id, enabled, schedule, custom_days, push_time, timezone, settings_json, last_run_local_date, last_error, updated_at_utc)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, '', '', %s)
                        ON CONFLICT(subscriber_id) DO UPDATE SET
                            enabled=EXCLUDED.enabled,
                            schedule=EXCLUDED.schedule,
                            custom_days=EXCLUDED.custom_days,
                            push_time=EXCLUDED.push_time,
                            timezone=EXCLUDED.timezone,
                            settings_json=EXCLUDED.settings_json,
                            last_run_local_date=CASE
                                WHEN auto_push_subscriptions.schedule <> EXCLUDED.schedule
                                  OR auto_push_subscriptions.custom_days <> EXCLUDED.custom_days
                                  OR auto_push_subscriptions.push_time <> EXCLUDED.push_time
                                  OR auto_push_subscriptions.timezone <> EXCLUDED.timezone
                                THEN ''
                                ELSE auto_push_subscriptions.last_run_local_date
                            END,
                            updated_at_utc=EXCLUDED.updated_at_utc
                        """,
                        params,
                    )
            return True, "ok"
        except Exception as exc:
            return False, str(exc)
    if AUTO_PUSH_BACKEND != "sqlite":
        return False, auto_push_backend_reason() or "Auto-push backend unavailable."
    try:
        with sqlite3.connect(AUTO_PUSH_DB_FILE) as conn:
            conn.execute(
                """
                INSERT INTO auto_push_subscriptions
                (subscriber_id, enabled, schedule, custom_days, push_time, timezone, settings_json, last_run_local_date, last_error, updated_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, '', '', ?)
                ON CONFLICT(subscriber_id) DO UPDATE SET
                    enabled=excluded.enabled,
                    schedule=excluded.schedule,
                    custom_days=excluded.custom_days,
                    push_time=excluded.push_time,
                    timezone=excluded.timezone,
                    settings_json=excluded.settings_json,
                    last_run_local_date=CASE
                        WHEN auto_push_subscriptions.schedule <> excluded.schedule
                          OR auto_push_subscriptions.custom_days <> excluded.custom_days
                          OR auto_push_subscriptions.push_time <> excluded.push_time
                          OR auto_push_subscriptions.timezone <> excluded.timezone
                        THEN ''
                        ELSE auto_push_subscriptions.last_run_local_date
                    END,
                    updated_at_utc=excluded.updated_at_utc
                """,
                params,
            )
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def list_due_auto_push_subscriptions(now_dt: datetime | None = None) -> list[dict[str, Any]]:
    if not AUTO_PUSH_ENABLED:
        return []
    init_auto_push_db()
    now_dt = now_dt or now_utc()
    rows: list[dict[str, Any]] = []
    if AUTO_PUSH_BACKEND == "postgres":
        if psycopg2 is None:
            return []
        try:
            with psycopg2.connect(AUTO_PUSH_DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT subscriber_id, schedule, custom_days, push_time, timezone, settings_json, last_run_local_date
                        FROM auto_push_subscriptions
                        WHERE enabled = 1
                        """
                    )
                    for row in cur.fetchall():
                        rows.append(
                            {
                                "subscriber_id": row[0],
                                "schedule": row[1],
                                "custom_days": row[2],
                                "push_time": row[3],
                                "timezone": row[4],
                                "settings_json": row[5],
                                "last_run_local_date": row[6],
                            }
                        )
        except Exception:
            return []
    elif AUTO_PUSH_BACKEND == "sqlite":
        try:
            with sqlite3.connect(AUTO_PUSH_DB_FILE) as conn:
                conn.row_factory = sqlite3.Row
                raw_rows = conn.execute(
                    """
                    SELECT subscriber_id, schedule, custom_days, push_time, timezone, settings_json, last_run_local_date
                    FROM auto_push_subscriptions
                    WHERE enabled = 1
                    """
                ).fetchall()
                rows = [dict(r) for r in raw_rows]
        except Exception:
            return []
    else:
        return []

    due: list[dict[str, Any]] = []
    for row in rows:
        tz_name = normalize_timezone(str(row["timezone"])) or "America/New_York"
        try:
            local_dt = now_dt.astimezone(ZoneInfo(tz_name))
        except Exception:
            local_dt = now_dt
        schedule = normalize_push_schedule(str(row["schedule"]))
        push_time = normalize_hhmm(str(row["push_time"])) or "09:00"
        try:
            push_hh, push_mm = push_time.split(":")
            push_minutes = int(push_hh) * 60 + int(push_mm)
        except Exception:
            push_minutes = 9 * 60
        now_minutes = local_dt.hour * 60 + local_dt.minute
        # Scheduler jitter-safe rule:
        # trigger once on the first run at/after target time in local day.
        if now_minutes < push_minutes:
            continue
        if schedule == "weekly_monday" and local_dt.weekday() != 0:
            continue
        local_date = local_dt.date().isoformat()
        if (row["last_run_local_date"] or "") == local_date:
            continue
        try:
            settings = json.loads(row["settings_json"])
            if not isinstance(settings, dict):
                continue
        except Exception:
            continue
        due.append(
            {
                "subscriber_id": str(row["subscriber_id"]),
                "settings": settings,
                "local_date": local_date,
                "local_time": push_time,
                "timezone": tz_name,
            }
        )
    return due


def mark_auto_push_run(subscriber_id: str, local_date: str | None, error: str = "") -> None:
    if not AUTO_PUSH_ENABLED:
        return
    err = (error or "")[:500]
    now_text = now_utc().isoformat()
    if AUTO_PUSH_BACKEND == "postgres":
        if psycopg2 is None:
            return
        try:
            with psycopg2.connect(AUTO_PUSH_DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    if local_date:
                        cur.execute(
                            """
                            UPDATE auto_push_subscriptions
                            SET last_run_local_date = %s, last_error = %s, updated_at_utc = %s
                            WHERE subscriber_id = %s
                            """,
                            (local_date, err, now_text, subscriber_id),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE auto_push_subscriptions
                            SET last_error = %s, updated_at_utc = %s
                            WHERE subscriber_id = %s
                            """,
                            (err, now_text, subscriber_id),
                        )
            return
        except Exception:
            return
    if AUTO_PUSH_BACKEND != "sqlite":
        return
    try:
        with sqlite3.connect(AUTO_PUSH_DB_FILE) as conn:
            if local_date:
                conn.execute(
                    """
                    UPDATE auto_push_subscriptions
                    SET last_run_local_date = ?, last_error = ?, updated_at_utc = ?
                    WHERE subscriber_id = ?
                    """,
                    (local_date, err, now_text, subscriber_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE auto_push_subscriptions
                    SET last_error = ?, updated_at_utc = ?
                    WHERE subscriber_id = ?
                    """,
                    (err, now_text, subscriber_id),
                )
    except Exception:
        return


def fallback_feed_summary(p: Paper, sc: dict[str, int], lang: str = "zh") -> str:
    if p.abstract:
        abs_text = abstract_snippet(p.abstract, max_chars=180, lang=lang)
        return L(lang, f"摘要要点：{abs_text}", f"Key points: {abs_text}")
    return L(
        lang,
        "摘要缺失。当前仅能基于元数据判断主题相关性。",
        "Abstract unavailable. Topic relevance can be inferred only from metadata.",
    )


def llm_enhance_summary(
    p: Paper,
    sc: dict[str, int],
    prefs: dict[str, Any],
    api_key: str,
    model: str,
    full_text: str = "",
    content_source: str = "",
) -> dict[str, str] | None:
    if not api_key.strip():
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    payload = {
        "title": p.title,
        "venue": p.venue,
        "date": p.publication_date,
        "abstract": p.abstract,
        "content_source": content_source,
        "paper_content": full_text[:35000] if full_text else "",
        "scores": sc,
        "keywords": prefs.get("keywords", []),
        "fields": prefs.get("fields", []),
    }
    lang = prefs.get("language", "zh")
    if (lang or "zh").startswith("zh"):
        prompt = (
            "你是科研助手。仅根据输入metadata生成简洁中文JSON。"
            "必须输出键：methods_detailed, main_conclusion, future_direction, value_assessment, ai_feed_summary。"
            "如果 paper_content 非空，必须优先依据论文正文内容总结；仅在正文缺失时回退到摘要/元数据。"
            "不要编造实验结果；信息不足时明确不确定性。"
        )
    else:
        prompt = (
            "You are a research assistant. Generate concise ENGLISH JSON strictly from provided metadata."
            "Required keys: methods_detailed, main_conclusion, future_direction, value_assessment, ai_feed_summary."
            "If paper_content is present, you must prioritize full-text evidence; only fall back to abstract/metadata when full text is unavailable."
            "Do not invent results; explicitly state uncertainty when evidence is limited."
        )
    try:
        client = OpenAI(api_key=api_key.strip())
        resp = client.responses.create(
            model=model.strip() or "gpt-4.1-mini",
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        data = json.loads(text)
        if all(k in data for k in ("methods_detailed", "main_conclusion", "future_direction", "value_assessment", "ai_feed_summary")):
            return {
                "methods_detailed": str(data["methods_detailed"]).strip(),
                "main_conclusion": str(data["main_conclusion"]).strip(),
                "future_direction": str(data["future_direction"]).strip(),
                "value_assessment": str(data["value_assessment"]).strip(),
                "ai_feed_summary": str(data["ai_feed_summary"]).strip(),
            }
    except Exception:
        return None
    return None


def ai_select_worth_reading(
    cards: list[dict[str, Any]],
    api_key: str,
    model: str,
    desired_count: int,
) -> list[dict[str, Any]]:
    if not cards:
        return []
    if not api_key.strip():
        return []
    try:
        from openai import OpenAI
    except Exception:
        return []

    compact_cards = []
    for c in cards:
        compact_cards.append(
            {
                "paper_id": c.get("paper_id", ""),
                "title": c.get("title", ""),
                "venue": c.get("venue", ""),
                "date": c.get("date", ""),
                "intro": c.get("intro", ""),
                "method_summary": c.get("method_summary", ""),
                "evidence_note": c.get("evidence_note", ""),
                "abstract_excerpt": c.get("abstract_excerpt", ""),
                "source_abstract": c.get("source_abstract", ""),
                "link": c.get("link", ""),
            }
        )
    lang = cards[0].get("language", "zh") if cards else "zh"
    if (lang or "zh").startswith("zh"):
        prompt = (
            "你是论文编辑。仅根据给定列表选择最值得读的论文。"
            "输出严格JSON：{\"selected_paper_ids\":[...]}。"
            "优先高价值、方法更可靠、对实践更有用的论文。"
            "不得输出列表外ID。"
        )
    else:
        prompt = (
            "You are an academic editor. Select the most worth-reading papers only from the provided list."
            "Output strict JSON: {\"selected_paper_ids\":[...]}."
            "Prioritize high value, stronger methodological reliability, and practical usefulness."
            "Never output IDs not in the list."
        )
    try:
        client = OpenAI(api_key=api_key.strip())
        resp = client.responses.create(
            model=model.strip() or "gpt-4.1-mini",
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"desired_count": max(1, desired_count), "cards": compact_cards},
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.1,
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        data = json.loads(text)
        selected_ids = data.get("selected_paper_ids", [])
        if not isinstance(selected_ids, list):
            return []
        selected_ids = [str(x) for x in selected_ids]
        id_to_card = {c.get("paper_id", ""): c for c in cards}
        out = [id_to_card[i] for i in selected_ids if i in id_to_card]
        if not out:
            return []
        return out[: max(1, desired_count)]
    except Exception:
        return []


def build_daily_push_text(digest: dict[str, Any], lang: str = "zh") -> dict[str, str]:
    top = digest.get("top_picks", [])
    also = digest.get("also_notable", [])
    all_papers = list(top) + list(also)
    total = len(all_papers)
    if total == 0:
        return {
            "today_new_summary": L(lang, "今日未发现符合订阅条件的新论文。", "No papers matched your subscription today."),
            "worth_reading_summary": L(lang, "暂无推荐必读论文，可适当放宽关键词或期刊范围。", "No must-read papers found; consider broadening keywords or journal scope."),
        }
    today_lines: list[str] = []
    for i, p in enumerate(all_papers, start=1):
        link = str(p.get("link", "")).strip()
        line = L(
            lang,
            f"{i}. {p['title']}（{p['venue']}，{p['date']}）",
            f"{i}. {p['title']} ({p['venue']}, {p['date']})",
        )
        if link:
            line = line + L(lang, f" | 链接：{link}", f" | Link: {link}")
        today_lines.append(
            line
        )
    worth = []
    for p in top[:3]:
        worth.append(
            L(
                lang,
                f"{p['title']}：{p.get('value_assessment', '')}",
                f"{p['title']}: {p.get('value_assessment', '')}",
            )
        )
    return {
        "today_new_summary": (L(lang, f"今日共筛出 {total} 篇新论文。重点如下：\n", f"Found {total} new papers today. Highlights:\n") + "\n".join(today_lines)),
        "worth_reading_summary": L(lang, "值得优先阅读：\n", "Worth reading first:\n") + "\n".join(worth),
    }


def build_worth_summary_from_cards(cards: list[dict[str, Any]], lang: str = "zh", show_ai_summary: bool = True) -> str:
    if not cards:
        return L(lang, "暂无 AI 推荐必读论文。", "No AI-selected must-read papers.")
    lines = []
    for i, c in enumerate(cards[:8], start=1):
        lines.append(f"{i}. {c['title']}")
        lines.append(L(lang, f"   期刊/日期：{c.get('venue','')} | {c.get('date','')}", f"   Venue/Date: {c.get('venue','')} | {c.get('date','')}"))
        lines.append(L(lang, f"   类型：{paper_type_label(c.get('paper_type','Research Article'), lang)}", f"   Type: {paper_type_label(c.get('paper_type','Research Article'), lang)}"))
        reason = c.get("ai_feed_summary", "") if show_ai_summary else c.get("why_it_matters", "")
        reason_label = L(lang, "AI推荐理由", "AI reason") if show_ai_summary else L(lang, "推荐理由", "Reason")
        lines.append(L(lang, f"   {reason_label}：{reason}", f"   {reason_label}: {reason}"))
        lines.append(L(lang, f"   链接：{c.get('link','')}", f"   Link: {c.get('link','')}"))
        lines.append("")
    return L(lang, "值得优先阅读：\n", "Worth reading first:\n") + "\n".join(lines)


def post_webhook(webhook_url: str, payload: dict[str, Any], lang: str = "zh") -> tuple[bool, str]:
    if not webhook_url.strip():
        return False, L(lang, "未填写 webhook URL", "Webhook URL is empty.")
    try:
        url = webhook_url.strip()
        # Slack Incoming Webhook expects a Slack-shaped payload (e.g., "text").
        if "hooks.slack.com/services/" in url:
            date = payload.get("date", "")
            worth = str(payload.get("worth_reading_summary", ""))
            digest = payload.get("digest", {}) if isinstance(payload.get("digest"), dict) else {}
            today_papers: list[dict[str, Any]] = []
            if isinstance(digest.get("top_picks"), list):
                today_papers.extend(digest.get("top_picks", []))
            if isinstance(digest.get("also_notable"), list):
                today_papers.extend(digest.get("also_notable", []))
            paper_lines: list[str] = []
            for i, p in enumerate(today_papers, start=1):
                title = str(p.get("title", "")).strip()
                venue = str(p.get("venue", "")).strip()
                date = str(p.get("date", "")).strip()
                link = str(p.get("link", "")).strip()
                meta = ", ".join([x for x in [venue, date] if x])
                head = f"{i}. {title}" + (f" ({meta})" if meta else "")
                paper_lines.append(head + (f" - {link}" if link else ""))
            if not paper_lines:
                paper_lines.append(L(lang, "（无论文）", "(No papers)"))
            text = "\n".join(
                [
                    f"Research Digest {date}",
                    "",
                    L(lang, "今日论文（含链接）：", "Today's papers (with links):"),
                    "\n".join(paper_lines),
                    "",
                    L(lang, "值得读：", "Worth reading:"),
                    worth,
                ]
            )
            slack_payload = {
                "text": text[:35000],  # keep below Slack payload limits
            }
            resp = requests.post(url, json=slack_payload, timeout=15)
        else:
            resp = requests.post(url, json=payload, timeout=15)
        if 200 <= resp.status_code < 300:
            body = (resp.text or "").strip()
            if body and len(body) <= 120:
                return True, L(lang, f"推送成功（HTTP {resp.status_code}，响应：{body}）", f"Push succeeded (HTTP {resp.status_code}, response: {body})")
            return True, L(lang, f"推送成功（HTTP {resp.status_code}）", f"Push succeeded (HTTP {resp.status_code})")
        body = (resp.text or "").strip()
        body_part = f"，响应：{body[:180]}" if body else ""
        return False, L(lang, f"推送失败（HTTP {resp.status_code}{body_part}）", f"Push failed (HTTP {resp.status_code}{', response: ' + body[:180] if body else ''})")
    except Exception as exc:
        return False, L(lang, f"推送异常：{exc}", f"Push error: {exc}")


def format_email_body(digest: dict[str, Any], push_text: dict[str, str], lang: str = "zh") -> str:
    lines = []
    header = digest.get("digest_header", {})
    stats = header.get("stats", {})
    lines.append(L(lang, "Research Digest 每日推送", "Research Digest Daily Update"))
    lines.append(L(lang, f"日期：{now_utc().strftime('%Y-%m-%d')}", f"Date: {now_utc().strftime('%Y-%m-%d')}"))
    lines.append(L(lang, f"覆盖范围：{header.get('coverage', '')}", f"Coverage: {header.get('coverage', '')}"))
    lines.append(
        L(
            lang,
            f"统计：抓取 {stats.get('fetched_count', 0)} | 去重后 {stats.get('deduplicated_count', 0)} | 入选 {stats.get('selected_count', 0)}",
            f"Stats: fetched {stats.get('fetched_count', 0)} | deduped {stats.get('deduplicated_count', 0)} | selected {stats.get('selected_count', 0)}",
        )
    )
    lines.append("")
    lines.append(L(lang, "【今日新论文总结】", "[Today's New Paper Summary]"))
    lines.append(push_text.get("today_new_summary", ""))
    lines.append("")
    lines.append(L(lang, "【值得读总结】", "[Worth Reading Summary]"))
    lines.append(push_text.get("worth_reading_summary", ""))
    lines.append("")
    lines.append(L(lang, "【优先推荐】", "[Top Picks]"))
    for i, p in enumerate(digest.get("top_picks", []), start=1):
        lines.append(f"{i}. {p.get('title', '')}")
        lines.append(L(lang, f"   期刊/日期：{p.get('venue', '')} | {p.get('date', '')}", f"   Venue/Date: {p.get('venue', '')} | {p.get('date', '')}"))
        lines.append(L(lang, f"   价值判断：{p.get('value_assessment', '')}", f"   Value: {p.get('value_assessment', '')}"))
        lines.append(L(lang, f"   链接：{p.get('link', '')}", f"   Link: {p.get('link', '')}"))
    return "\n".join(lines)


def send_email_digest(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
    subject: str,
    body: str,
    lang: str = "zh",
) -> tuple[bool, str]:
    if not smtp_host.strip() or not smtp_user.strip() or not smtp_password.strip() or not to_email.strip():
        return False, L(lang, "邮件配置不完整（SMTP host/user/password/to）", "SMTP config incomplete (host/user/password/to)")
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg.set_content(body)
        with smtplib.SMTP(smtp_host.strip(), smtp_port, timeout=20) as server:
            server.starttls()
            server.login(smtp_user.strip(), smtp_password.strip())
            server.send_message(msg)
        return True, L(lang, "邮件发送成功", "Email sent successfully")
    except Exception as exc:
        return False, L(lang, f"邮件发送失败：{exc}", f"Email send failed: {exc}")


def card(item: dict[str, Any], prefs: dict[str, Any]) -> dict[str, Any]:
    p: Paper = item["paper"]
    sc = item["scores"]
    lang = prefs.get("language", "zh")
    text = f"{p.title} {p.abstract}".lower()
    matched = [k for k in prefs.get("keywords", []) if k.lower() in text]
    why = (
        L(lang, f"与关键词“{matched[0]}”直接相关，适合纳入本期必读。", f"Directly aligned with keyword '{matched[0]}', suitable for priority reading.")
        if matched
        else L(lang, "匹配你的关键词与期刊偏好，值得优先阅读。", "Matches your keywords and journal preferences; worth prioritizing.")
    )
    short_title = re.sub(r"\s+", " ", p.title)[:16]
    t1 = (L(lang, "关注主题：", "Focus: ") + short_title)[:25]
    t2 = (
        L(lang, "方法线索较明确，可快读", "Method signal is clear; quick read")
        if p.abstract and any(k in p.abstract.lower() for k in ["randomized", "trial", "cohort", "review", "benchmark"])
        else (L(lang, "方法信息有限，需看全文", "Method detail limited; read full text") if p.abstract else L(lang, "摘要缺失，方法待确认", "Abstract missing; method pending confirmation"))
    )
    t3 = L(lang, "与订阅关键词直接相关", "Directly related to subscribed keywords") if p.abstract else L(lang, "仅依据标题与元数据判断", "Judged from title and metadata only")
    strengths = []
    if sc["relevance"] >= 70:
        strengths.append(L(lang, "与订阅主题匹配度高", "High alignment with your subscribed topics"))
    if sc["novelty"] >= 70:
        strengths.append(L(lang, "新意信号较强", "Strong novelty signals"))
    if sc["rigor"] >= 60:
        strengths.append(L(lang, "方法与验证信息相对充分", "Method and validation signals are relatively sufficient"))
    strengths.append(L(lang, "提供摘要便于快速初筛", "Abstract is available for quick triage") if p.abstract else L(lang, "元数据完整，便于定位原文", "Metadata is complete for locating the paper"))
    limitations = []
    if not p.abstract:
        limitations.append(L(lang, "摘要未说明/无法判断", "Not stated in abstract / cannot determine"))
    if sc["rigor"] < 50:
        limitations.append(L(lang, "研究设计细节不足", "Insufficient study-design detail"))
    if not p.authors:
        limitations.append(L(lang, "作者信息不完整", "Incomplete author information"))
    if not p.publication_date:
        limitations.append(L(lang, "发表日期不完整", "Incomplete publication date"))
    tags = [k for k in prefs.get("keywords", []) if k.lower() in text]
    if p.venue:
        tags.append(p.venue)
    if p.arxiv_id:
        tags.append("preprint")
    if "review" in text or "survey" in text:
        tags.append("review")
    out_tags: list[str] = []
    for t in tags:
        if t not in out_tags:
            out_tags.append(t)
    while len(out_tags) < 3:
        out_tags.append(L(lang, "基于元数据", "metadata-based"))
    fields = prefs.get("fields", [])
    who = (
        L(lang, f"{'、'.join(fields[:3])}方向研究者与产品决策者。", f"Researchers and product decision-makers in {', '.join(fields[:3])}.")
        if fields
        else L(lang, "关注该主题的一线研究者与工程实践者。", "Researchers and practitioners focused on this topic.")
    )
    iv = intro_method_value(p, sc, prefs)
    ds = detailed_sections(p, sc, lang=lang)
    abs_excerpt = p.abstract.strip() if p.abstract.strip() else L(lang, "摘要不可用。", "Abstract unavailable.")
    feed_summary = fallback_feed_summary(p, sc, lang=lang)
    paper_type = infer_paper_type(p.title, p.abstract)
    return {
        "paper_id": paper_id(p),
        "title": p.title,
        "venue": p.venue or "Unknown Venue",
        "date": p.publication_date or "1900-01-01",
        "link": best_link(p),
        "why_it_matters": why,
        "key_takeaways": [t1, t2[:25], t3[:25]],
        "methods_in_one_line": L(
            lang,
            "摘要未提供，具体方法无法判断。" if not p.abstract else ("采用随机对照设计评估核心假设。" if "randomized" in p.abstract.lower() else "采用系统综述/Meta分析整合既有证据。" if ("systematic review" in p.abstract.lower() or "meta-analysis" in p.abstract.lower()) else "通过基准测试比较方法性能。" if "benchmark" in p.abstract.lower() else "基于摘要可见的方法描述进行实证分析。"),
            "Method cannot be confirmed because abstract is unavailable." if not p.abstract else ("Uses a randomized controlled design to evaluate the core hypothesis." if "randomized" in p.abstract.lower() else "Uses a systematic review/meta-analysis to synthesize evidence." if ("systematic review" in p.abstract.lower() or "meta-analysis" in p.abstract.lower()) else "Uses benchmark experiments to compare method performance." if "benchmark" in p.abstract.lower() else "Performs empirical analysis based on abstract-level method description."),
        ),
        "strengths": strengths[:4],
        "limitations": (limitations[:3] or [L(lang, "摘要未说明/无法判断", "Not stated in abstract / cannot determine")]),
        "who_should_read": who,
        "intro": iv["intro"],
        "method_summary": iv["method_summary"],
        "methods_detailed": ds["methods_detailed"],
        "main_conclusion": ds["main_conclusion"],
        "future_direction": ds["future_direction"],
        "value_assessment": iv["value_assessment"],
        "value_label": value_label(sc["total"], lang),
        "abstract_excerpt": abs_excerpt,
        "ai_feed_summary": feed_summary,
        "paper_type": paper_type,
        "language": lang,
        "scores": sc,
        "tags": out_tags[:6],
        "evidence_note": "Based on abstract/metadata." if p.abstract else "Abstract unavailable; summary is tentative.",
        "source_abstract": p.abstract,
        "source_content": "",
        "content_source_label": "",
    }


def build_digest(prefs: dict[str, Any], candidates: list[Paper]) -> dict[str, Any]:
    lang = prefs.get("language", "zh")
    deduped = dedupe(candidates)
    filtered = [p for p in deduped if passes(p, prefs)]
    filter_mode = "strict"
    if not filtered and deduped:
        # Fallback 1: keep journal/date/exclude, drop keyword/field hard constraint
        relaxed = []
        for p in deduped:
            text = f"{p.title} {p.abstract}".lower()
            if any(t.lower() in text for t in prefs.get("exclude_keywords", [])):
                continue
            if prefs.get("journals") and not venue_matches_selected(
                p.venue or "",
                prefs.get("journals", []),
                strict=bool(prefs.get("strict_journal_only", True)),
            ):
                continue
            dt = parse_date(p.publication_date)
            if not in_day_window(dt, int(prefs.get("date_range_days", 14))):
                continue
            relaxed.append(p)
        if relaxed:
            filtered = relaxed
            filter_mode = "relaxed_no_keyword"
    if not filtered and deduped:
        # Fallback 2: date-only fallback to avoid empty feed.
        # IMPORTANT: if journals are selected, never break journal scope.
        filtered = []
        for p in deduped:
            if prefs.get("journals") and not venue_matches_selected(
                p.venue or "",
                prefs.get("journals", []),
                strict=bool(prefs.get("strict_journal_only", True)),
            ):
                continue
            dt = parse_date(p.publication_date)
            if not in_day_window(dt, int(prefs.get("date_range_days", 14))):
                continue
            filtered.append(p)
        if filtered:
            filter_mode = "date_only_fallback"
    scored = [{"paper": p, "scores": score(p, prefs)} for p in filtered]
    scored.sort(key=lambda x: x["scores"]["total"], reverse=True)
    max_papers = int(prefs.get("max_papers", 0))
    picked = scored if max_papers <= 0 else scored[:max_papers]
    # Journal diversity guard: when multiple journals are selected, avoid single-journal domination.
    selected_journals = normalize_str_list_input(prefs.get("journals", []))
    if selected_journals and len(selected_journals) > 1 and scored:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in scored:
            p = item["paper"]
            assigned = None
            for j in selected_journals:
                if venue_matches_selected(p.venue or "", [j], strict=bool(prefs.get("strict_journal_only", True))):
                    assigned = j
                    break
            if not assigned:
                assigned = p.venue or "Unknown Venue"
            grouped.setdefault(assigned, []).append(item)
        balanced: list[dict[str, Any]] = []
        # first pass: pick one from each journal bucket (score-sorted already)
        for j in selected_journals:
            bucket = grouped.get(j, [])
            if bucket:
                balanced.append(bucket.pop(0))
        # second pass: fill remaining by global score order
        used_ids = {id(x["paper"]) for x in balanced}
        for item in scored:
            if id(item["paper"]) in used_ids:
                continue
            balanced.append(item)
        picked = balanced if max_papers <= 0 else balanced[:max_papers]
    n = len(picked)
    top_n = n if n <= 3 else 3 if n <= 5 else 5
    top, also = picked[:top_n], picked[top_n:]
    selected = top + also
    vocab: dict[str, int] = {}
    preprint = 0
    with_abs = 0
    method_signal = 0
    clinical_signal = 0
    for x in selected:
        p = x["paper"]
        if p.venue.lower() == "arxiv":
            preprint += 1
        if p.abstract:
            with_abs += 1
            al = p.abstract.lower()
            if any(k in al for k in ["randomized", "cohort", "meta-analysis", "systematic review", "ablation", "benchmark"]):
                method_signal += 1
            if any(k in al for k in ["patient", "clinical", "trial", "hospital", "treatment"]):
                clinical_signal += 1
        for tok in re.findall(r"[a-zA-Z]{4,}", p.title.lower()):
            if tok not in STOPWORDS:
                vocab[tok] = vocab.get(tok, 0) + 1
    top_terms = "、".join(k for k, _ in sorted(vocab.items(), key=lambda i: i[1], reverse=True)[:3]) or L(lang, "暂无高频术语", "No dominant terms")
    n_selected = len(selected)
    preprint_pct = round(100 * preprint / n_selected) if n_selected else 0
    abs_pct = round(100 * with_abs / n_selected) if n_selected else 0
    method_pct = round(100 * method_signal / n_selected) if n_selected else 0
    clinical_pct = round(100 * clinical_signal / n_selected) if n_selected else 0
    trends = [
        (
            L(
                lang,
                f"本期共入选 {n_selected} 篇，主题高频词为 {top_terms}。从题名分布看，研究焦点集中在这些关键词对应的问题簇，适合围绕它们做连续追踪。",
                f"This issue selected {n_selected} papers. Top topic terms: {top_terms}. The title distribution suggests concentrated focus areas suitable for continuous tracking.",
            )
        ),
        (
            L(
                lang,
                f"来源结构上，预印本占比约 {preprint_pct}%，有摘要论文占比约 {abs_pct}%。这意味着你能较快看到前沿进展，同时多数条目具备可用于初筛的文本证据。",
                f"Source mix: preprints {preprint_pct}%, papers with abstract {abs_pct}%. This supports both fast frontier tracking and practical first-pass screening.",
            )
        ),
        (
            L(
                lang,
                f"方法信号（随机/队列/Meta/benchmark 等）覆盖约 {method_pct}%，临床相关信号覆盖约 {clinical_pct}%。建议优先阅读同时具备方法信号和明确任务边界的论文，以提升投入产出比。",
                f"Method signals (randomized/cohort/meta/benchmark, etc.) appear in about {method_pct}%, and clinical signals in about {clinical_pct}%. Prioritize papers with clear method signals and task boundaries.",
            )
        ),
    ]
    return {
        "digest_header": {
            "coverage": L(lang, f"过去 {int(prefs.get('date_range_days', 14))} 天", f"Past {int(prefs.get('date_range_days', 14))} days"),
            "stats": {"fetched_count": len(candidates), "deduplicated_count": len(deduped), "selected_count": len(selected)},
            "trends": trends,
            "subscription_suggestion": L(
                lang,
                "若想提高命中率，可新增2-3个更具体方法关键词，并补充1-2本核心期刊。",
                "To improve hit rate, add 2-3 method-specific keywords and 1-2 core journals.",
            ),
            "diagnostics": {
                "raw_count": len(candidates),
                "deduped_count": len(deduped),
                "filtered_count": len(filtered),
                "selected_count": len(selected),
                "filter_mode": filter_mode,
            },
        },
        "top_picks": [card(x, prefs) for x in top],
        "also_notable": [card(x, prefs) for x in also],
    }


def render_one_card(
    c: dict[str, Any],
    i: int,
    section_title: str,
    proxy_prefix: str,
    compact: bool,
    show_ai_summary: bool = True,
) -> None:
    lang = c.get("language", "zh")
    abstract_label = L(lang, "摘要", "Abstract")
    ai_label = L(lang, "AI摘要", "AI Feed Summary")
    tags = "".join([f"<span class='pill'>{t}</span>" for t in c.get("tags", [])[:6]])
    full_abstract = str(c.get("source_abstract", "") or "").strip() or str(c.get("abstract_excerpt", "")).strip()
    ai_line = (
        f"<div class=\"section-line\"><strong>{ai_label}:</strong> {c.get('ai_feed_summary', '')}</div>"
        if show_ai_summary
        else ""
    )
    if compact:
        body = f"""
          {ai_line}
          <div>{tags}</div>
        """
    else:
        body = f"""
          {ai_line}
          <div>{tags}</div>
        """
    st.markdown(
        f"""
        <div class="paper-card">
          <p class="feed-title">{i}. {c["title"]}</p>
          <div class="meta">{c["venue"]} | {c["date"]} | {c["paper_id"]} | {paper_type_label(c.get("paper_type","Research Article"), lang)}</div>
          {body}
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(L(lang, "摘要（点击展开）", "Abstract (click to expand)"), expanded=False):
        if full_abstract:
            st.write(full_abstract)
        else:
            st.write(L(lang, "摘要不可用。", "Abstract unavailable."))
    url = apply_proxy(c.get("link", ""), proxy_prefix)
    if url:
        st.link_button(L(lang, "查看论文", "Open Paper"), url)


def render_cards(
    section_title: str,
    cards: list[dict[str, Any]],
    proxy_prefix: str,
    layout_mode: str,
    lang: str = "zh",
    show_ai_summary: bool = True,
) -> None:
    if not cards:
        st.info(L(lang, "该分区暂无论文。", "No papers in this section."))
        return
    if layout_mode == "board2":
        cols = st.columns(2)
        for i, c in enumerate(cards, start=1):
            with cols[(i - 1) % 2]:
                render_one_card(c, i, section_title, proxy_prefix, compact=False, show_ai_summary=show_ai_summary)
        return
    if layout_mode == "board3":
        cols = st.columns(3)
        for i, c in enumerate(cards, start=1):
            with cols[(i - 1) % 3]:
                render_one_card(c, i, section_title, proxy_prefix, compact=False, show_ai_summary=show_ai_summary)
        return
    compact = layout_mode == "compact"
    for i, c in enumerate(cards, start=1):
        render_one_card(c, i, section_title, proxy_prefix, compact=compact, show_ai_summary=show_ai_summary)


def render_cards_grouped(
    section_title: str,
    cards: list[dict[str, Any]],
    proxy_prefix: str,
    layout_mode: str,
    lang: str = "zh",
    show_ai_summary: bool = True,
) -> None:
    if not cards:
        st.info(L(lang, "该分区暂无论文。", "No papers in this section."))
        return
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in cards:
        journal = str(c.get("venue", "")).strip() or L(lang, "未知期刊", "Unknown Venue")
        groups.setdefault(journal, []).append(c)
    # Show larger journal groups first, then by journal name.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0].lower()))
    for journal_name, gcards in ordered:
        with st.expander(f"{journal_name} ({len(gcards)})", expanded=True):
            render_cards(
                section_title=section_title,
                cards=gcards,
                proxy_prefix=proxy_prefix,
                layout_mode=layout_mode,
                lang=lang,
                show_ai_summary=show_ai_summary,
            )


def parse_csv(text: str) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in re.split(r"[,，;\n]+", text) if x.strip()]


def get_backend_openai_api_key() -> str:
    session_key = str(st.session_state.get("session_openai_api_key", "")).strip()
    if session_key:
        return session_key
    if PUBLIC_MODE:
        return ""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except StreamlitSecretNotFoundError:
        key = ""
    return (key or os.getenv("OPENAI_API_KEY", "")).strip()


def get_backend_smtp_config() -> tuple[str, int, str, str]:
    host = ""
    port = 587
    user = ""
    password = ""
    if not PUBLIC_MODE:
        try:
            host = str(st.secrets.get("SMTP_HOST", "") or "")
            port = int(st.secrets.get("SMTP_PORT", 587) or 587)
            user = str(st.secrets.get("SMTP_USER", "") or "")
            password = str(st.secrets.get("SMTP_PASSWORD", "") or "")
        except StreamlitSecretNotFoundError:
            pass
        except Exception:
            pass
    host = host or os.getenv("SMTP_HOST", "")
    try:
        port = int(os.getenv("SMTP_PORT", str(port)) or port)
    except Exception:
        port = 587
    user = user or os.getenv("SMTP_USER", "")
    password = password or os.getenv("SMTP_PASSWORD", "")
    return host.strip(), int(port), user.strip(), password.strip()


def fetch_candidates_once(prefs: dict[str, Any], days: int, strict_journal_only: bool) -> tuple[list[Paper], dict[str, int]]:
    kws = prefs.get("keywords", [])
    fields = prefs.get("fields", [])
    journals = prefs.get("journals", [])

    # If no topic terms are provided, fetch by selected journals directly.
    query_terms = kws + fields
    if not query_terms:
        query_terms = journals[:]
    journal_only_mode = (not kws and not fields and bool(journals))

    # arXiv should be keyword/field-driven, not journal-name-driven.
    # If user only selected journals (no topic terms), do not crawl arXiv by journal names.
    arxiv_terms = kws + fields
    if not arxiv_terms:
        arxiv_terms = []
    # If user did not select arXiv at all, disable arXiv crawl.
    if not any("arxiv" in j.lower() for j in journals):
        arxiv_terms = []

    cache_key = json.dumps(
        {
            "days": days,
            "strict": strict_journal_only,
            "kws": kws,
            "fields": fields,
            "journals": journals,
            "qterms": query_terms,
            "arxiv_terms": arxiv_terms,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    now_ts = datetime.now(UTC).timestamp()
    fetch_cache = st.session_state.setdefault("fetch_cache", {})
    cache_item = fetch_cache.get(cache_key)
    if cache_item and now_ts - float(cache_item.get("ts", 0)) <= FETCH_CACHE_TTL_SEC and cache_item.get("papers"):
        cached_diag = dict(cache_item["diag"])
        cached_diag["cache_hit"] = 1
        return cache_item["papers"], cached_diag

    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_arxiv = ex.submit(fetch_arxiv, arxiv_terms, days)
        if journal_only_mode:
            fut_crossref = ex.submit(fetch_crossref_by_journals, journals, days, strict_journal_only)
            pubmed_terms = [f"\"{j}\"[journal]" for j in journals[:10]]
            fut_pubmed = ex.submit(fetch_pubmed, pubmed_terms, journals, days, strict_journal_only)
            fut_crossref_journal = None
        else:
            fut_crossref = ex.submit(fetch_crossref, query_terms, journals, days, strict_journal_only)
            fut_pubmed = ex.submit(fetch_pubmed, query_terms, journals, days, strict_journal_only)
            # Always add a journal-driven pull to avoid missing same-day papers
            # when keyword queries are too narrow.
            fut_crossref_journal = ex.submit(fetch_crossref_by_journals, journals, days, strict_journal_only) if journals else None
        fut_rss = ex.submit(fetch_journal_rss, journals, days)
        arxiv_results = fut_arxiv.result()
        crossref_results = fut_crossref.result()
        crossref_journal_results = fut_crossref_journal.result() if fut_crossref_journal else []
        rss_results = fut_rss.result()
        pubmed_results = fut_pubmed.result()
    combined = arxiv_results + crossref_results + crossref_journal_results + rss_results + pubmed_results

    # Journal-only multi-select guard: if only one journal is hit, try to backfill missing journals.
    journal_backfill = 0
    if journal_only_mode and len(journals) > 1:
        missing: list[str] = []
        for j in journals:
            has_j = any(
                venue_matches_selected(p.venue or "", [j], strict=bool(strict_journal_only))
                for p in combined
            )
            if not has_j:
                missing.append(j)
        for j in missing[:8]:
            # Backfill must stay strict; relaxed matching causes false positives
            # (e.g., journals containing generic words like "Nature").
            extra = fetch_crossref_by_journals([j], days=days, strict_journal_only=True)
            extra = [p for p in extra if venue_matches_selected(p.venue or "", [j], strict=True)]
            if extra:
                combined.extend(extra)
                journal_backfill += len(extra)
    combined = enrich_missing_abstracts(combined, max_enrich=MAX_ABSTRACT_ENRICH)
    diag = {
        "arxiv": len(arxiv_results),
        "crossref": len(crossref_results) + len(crossref_journal_results),
        "rss": len(rss_results),
        "pubmed": len(pubmed_results),
        "total_raw": len(arxiv_results) + len(crossref_results) + len(crossref_journal_results) + len(rss_results) + len(pubmed_results),
        "journal_backfill": journal_backfill,
        "cache_hit": 0,
    }
    # Cache only non-empty fetch results to avoid "empty cache lock" for 15 minutes.
    if combined:
        fetch_cache[cache_key] = {"ts": now_ts, "papers": combined, "diag": diag}
    # If upstream sources temporarily return empty, fallback to stale cache for same query.
    if not combined and cache_item and cache_item.get("papers"):
        stale_diag = dict(cache_item.get("diag", {}))
        stale_diag["cache_hit"] = 2
        stale_diag["stale_cache_used"] = 1
        return cache_item.get("papers", []), stale_diag
    # Trim cache size to avoid unbounded growth.
    if len(fetch_cache) > 20:
        oldest_key = min(fetch_cache.keys(), key=lambda k: float(fetch_cache[k].get("ts", 0)))
        fetch_cache.pop(oldest_key, None)
    return combined, diag


def fetch_candidates(prefs: dict[str, Any]) -> tuple[list[Paper], str, dict[str, int], dict[str, Any]]:
    days = int(prefs.get("date_range_days", 14))
    strict = bool(prefs.get("strict_journal_only", True))
    lang = prefs.get("language", "zh")
    journals = list(prefs.get("journals", []))

    papers, diag = fetch_candidates_once(prefs, days=days, strict_journal_only=strict)
    if papers:
        return papers, L(lang, f"抓取完成：窗口 {days} 天，严格期刊匹配={'开' if strict else '关'}。", f"Fetch complete: {days}-day window, strict journal match={'on' if strict else 'off'}."), diag, {
            "effective_days": days,
            "effective_strict_journal_only": strict,
        }

    # Same-day index lag fallback: keep strict journal scope, widen only slightly.
    if days == 1 and strict and journals:
        papers_3d, diag_3d = fetch_candidates_once(prefs, days=3, strict_journal_only=True)
        if papers_3d:
            return papers_3d, L(
                lang,
                "当天未命中，已启用近3天索引滞后补抓（仍严格限制在所选期刊）。",
                "No same-day hit; enabled 3-day index-lag fallback (still strict to selected journals).",
            ), diag_3d, {
                "effective_days": 3,
                "effective_strict_journal_only": True,
            }

    # If strict journal matching is enabled, do not auto-relax journal matching.
    if strict:
        # Fallback: RSS-only pull for selected journals within the same day window.
        if journals:
            rss_only = fetch_journal_rss(journals, days)
            if rss_only:
                selected_set = {canonical_journal_name(j) for j in journals if j}
                rss_set = {canonical_journal_name(p.venue) for p in rss_only if p.venue}
                # Avoid single-source bias in multi-journal strict mode.
                if len(selected_set) > 1 and len(rss_set) <= 1:
                    return [], L(
                        lang,
                        "严格匹配下 API 未命中，且 RSS 仅覆盖单一期刊，已跳过该回退以避免偏置。",
                        "Strict mode API returned no hit, and RSS covered only one journal; skipped fallback to avoid bias.",
                    ), diag if "diag" in locals() else {"arxiv": 0, "crossref": 0, "rss": 0, "pubmed": 0, "total_raw": 0}, {
                        "effective_days": days,
                        "effective_strict_journal_only": True,
                    }
                diag_rss = dict(diag if "diag" in locals() else {"arxiv": 0, "crossref": 0, "rss": 0, "pubmed": 0, "total_raw": 0})
                diag_rss["rss"] = int(diag_rss.get("rss", 0)) + len(rss_only)
                diag_rss["total_raw"] = int(diag_rss.get("total_raw", 0)) + len(rss_only)
                return rss_only, L(
                    lang,
                    "严格匹配未命中 API 结果，已使用期刊 RSS 当天回退。",
                    "Strict matching missed API results; used same-day journal RSS fallback.",
                ), diag_rss, {
                    "effective_days": days,
                    "effective_strict_journal_only": True,
                }
        return [], L(lang, "严格期刊匹配未命中：已按你的设置保持严格模式（未自动放宽）。", "No hit under strict journal matching: strict mode is kept as configured (no auto-relax)."), diag if "diag" in locals() else {"arxiv": 0, "crossref": 0, "rss": 0, "pubmed": 0, "total_raw": 0}, {
            "effective_days": days,
            "effective_strict_journal_only": True,
        }

    return [], L(lang, "未抓到论文：请检查网络、期刊名拼写，或关闭 Daily Mode。", "No papers fetched: check network, journal spellings, or disable Daily mode."), diag if "diag" in locals() else {"arxiv": 0, "crossref": 0, "rss": 0, "pubmed": 0, "total_raw": 0}, {
        "effective_days": days,
        "effective_strict_journal_only": strict,
    }


def main() -> None:
    st.set_page_config(page_title="Research Digest Engine", layout="wide")
    inject_styles()
    init_auto_push_db()
    current_lang = st.session_state.get("saved_settings", {}).get("language", "zh")
    feed_title = L(current_lang, "今日论文", "Research Feed")
    st.markdown(
        f"""
        <div class="hero">
          <h3 class="hero-title">{feed_title}</h3>
          <p class="hero-sub">{L(current_lang, "以最少噪音，持续跟踪高价值研究进展。", "Track high-value research with minimal noise.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_settings = {
        "language": "zh",
        "fields": [],
        "custom_fields": "",
        "journals": [],
        "custom_journals": "",
        "strict_journal_only": True,
        "keywords": "",
        "exclude_keywords": "survey",
        "proxy_prefix": "",
        "push_schedule": "daily",
        "custom_days": 7,
        "max_papers": 0,
        "reading_level": "mixed",
        "wr": 0.35,
        "wn": 0.25,
        "wg": 0.25,
        "wi": 0.15,
        "layout_mode": "board2",
        "enable_webhook_push": False,
        "webhook_url": "",
        "email_to": "",
        "auto_send_email": False,
        "enable_auto_push": False,
        "daily_push_time": "09:00",
        "push_timezone": os.getenv("APP_TIMEZONE", "America/New_York").strip() or "America/New_York",
        "subscriber_id": "",
        "use_api": False,
        "api_model": "gpt-4.1-mini",
        "deep_read_mode": False,
        "deep_read_limit": 5,
        "ai_pick_worth": True,
        "worth_count": 4,
        "auto_refresh_on_load": False,
    }
    if "saved_settings" not in st.session_state:
        merged = load_saved_settings(default_settings)
        browser_override = load_browser_settings(default_settings)
        if browser_override:
            merged.update(browser_override)
        if not str(merged.get("subscriber_id", "")).strip():
            merged["subscriber_id"] = uuid4().hex
        merged["journals"] = normalize_str_list_input(merged.get("journals", []))
        merged["fields"] = normalize_str_list_input(merged.get("fields", []))
        smtp_host_init, smtp_port_init, smtp_user_init, smtp_password_init = get_backend_smtp_config()
        smtp_ready_init = all([smtp_host_init, smtp_user_init, smtp_password_init])
        if not smtp_ready_init:
            merged["auto_send_email"] = False
        st.session_state.saved_settings = merged
    if "browser_settings_try_count" not in st.session_state:
        st.session_state.browser_settings_try_count = 0
    if "browser_settings_synced" not in st.session_state:
        st.session_state.browser_settings_synced = False
    if "session_openai_api_key" not in st.session_state:
        st.session_state.session_openai_api_key = ""
    if "local_cache" not in st.session_state:
        st.session_state.local_cache = load_local_cache()
    if "local_cache_dirty" not in st.session_state:
        st.session_state.local_cache_dirty = False
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "last_digest" not in st.session_state:
        # Session-only result state: do not restore server-shared last digest.
        st.session_state.last_digest = None
        st.session_state.last_push_text = None
        st.session_state.last_today_cards = None
        st.session_state.last_worth_cards = None
        st.session_state.last_fetch_note = ""
        st.session_state.last_fetch_diag = {}
    if "last_push_text" not in st.session_state:
        st.session_state.last_push_text = None
    if "last_today_cards" not in st.session_state:
        st.session_state.last_today_cards = None
    if "last_worth_cards" not in st.session_state:
        st.session_state.last_worth_cards = None
    if "last_fetch_note" not in st.session_state:
        st.session_state.last_fetch_note = ""
    if "last_fetch_diag" not in st.session_state:
        st.session_state.last_fetch_diag = {}
    if "last_auto_email_sig" not in st.session_state:
        st.session_state.last_auto_email_sig = ""
    if (
        streamlit_js_eval is not None
        and not st.session_state.get("browser_settings_synced", False)
        and int(st.session_state.get("browser_settings_try_count", 0)) < 5
    ):
        st.session_state.browser_settings_try_count = int(st.session_state.get("browser_settings_try_count", 0)) + 1
        browser_override_runtime = load_browser_settings(default_settings)
        if browser_override_runtime:
            merged_runtime = dict(st.session_state.saved_settings)
            merged_runtime.update(browser_override_runtime)
            merged_runtime["journals"] = normalize_str_list_input(merged_runtime.get("journals", []))
            merged_runtime["fields"] = normalize_str_list_input(merged_runtime.get("fields", []))
            if not str(merged_runtime.get("subscriber_id", "")).strip():
                merged_runtime["subscriber_id"] = uuid4().hex
            st.session_state.saved_settings = merged_runtime
            st.session_state.browser_settings_synced = True
            st.rerun()
        elif int(st.session_state.get("browser_settings_try_count", 0)) >= 5:
            st.session_state.browser_settings_synced = True
    # Safety unlock: avoid persistent disabled UI if previous run ended abnormally.
    if st.session_state.get("is_generating", False):
        st.session_state.is_generating = False
    s = st.session_state.saved_settings

    @st.dialog(L(st.session_state.saved_settings.get("language", "zh"), "设置", "Settings"), width="large")
    def settings_modal() -> None:
        cur = st.session_state.saved_settings
        busy = bool(st.session_state.get("is_generating", False))
        smtp_host_b, smtp_port_b, smtp_user_b, smtp_password_b = get_backend_smtp_config()
        smtp_ready_b = all([smtp_host_b, smtp_user_b, smtp_password_b])
        lang_opt = cur.get("language", "zh")
        lang_labels = [L(lang_opt, "中文", "Chinese"), "English"]
        language = st.selectbox(L(lang_opt, "语言", "Language"), lang_labels, index=0 if lang_opt == "zh" else 1)
        ui_lang = "zh" if language == lang_labels[0] else "en"

        selected_fields = normalize_str_list_input(cur.get("fields", []))
        custom_fields = str(cur.get("custom_fields", ""))
        selected_journals = normalize_str_list_input(cur.get("journals", []))
        custom_journals = str(cur.get("custom_journals", ""))
        strict_journal_only = bool(cur.get("strict_journal_only", True))
        keywords = str(cur.get("keywords", ""))
        exclude = str(cur.get("exclude_keywords", "survey"))
        push_schedule = normalize_push_schedule(str(cur.get("push_schedule", "daily")))
        custom_days = int(cur.get("custom_days", 7))
        max_papers = int(cur.get("max_papers", 0))
        saved_layout = str(cur.get("layout_mode", "board2"))
        proxy_prefix = str(cur.get("proxy_prefix", ""))
        enable_auto_push = bool(cur.get("enable_auto_push", False))
        daily_push_time = str(cur.get("daily_push_time", "09:00"))
        saved_tz = normalize_timezone(str(cur.get("push_timezone", os.getenv("APP_TIMEZONE", "America/New_York")))) or "America/New_York"
        enable_webhook_push = bool(cur.get("enable_webhook_push", False))
        webhook_url = str(cur.get("webhook_url", ""))
        email_to = str(cur.get("email_to", ""))
        auto_send_email = bool(cur.get("auto_send_email", False))
        use_api = bool(cur.get("use_api", False))
        session_api_key = str(st.session_state.get("session_openai_api_key", ""))
        api_model = str(cur.get("api_model", "gpt-4.1-mini"))
        deep_read_mode = bool(cur.get("deep_read_mode", False))
        deep_read_limit = int(cur.get("deep_read_limit", 5))
        worth_count = int(cur.get("worth_count", 4))
        auto_refresh_on_load = bool(cur.get("auto_refresh_on_load", False))

        schedule_options = {
            "daily": L(ui_lang, "每天", "Daily"),
            "weekly_monday": L(ui_lang, "每周一", "Weekly (Monday)"),
            "custom": L(ui_lang, "自定义", "Custom"),
        }
        schedule_keys = ["daily", "weekly_monday", "custom"]
        layout_options = {
            "expanded": L(ui_lang, "舒展单列", "Expanded List"),
            "compact": L(ui_lang, "紧凑单列", "Compact List"),
            "board2": L(ui_lang, "双列看板", "Board (2 columns)"),
            "board3": L(ui_lang, "三列看板", "Board (3 columns)"),
        }
        legacy_layout = {"舒展单列": "expanded", "紧凑单列": "compact", "Board-2列": "board2", "Board-3列": "board3"}
        saved_layout = legacy_layout.get(saved_layout, saved_layout)
        if saved_layout not in layout_options:
            saved_layout = "board2"
        layout_keys = ["expanded", "compact", "board2", "board3"]

        tab_general, tab_delivery, tab_ai = st.tabs(
            [L(ui_lang, "基础", "General"), L(ui_lang, "推送", "Delivery"), L(ui_lang, "AI", "AI")]
        )
        with tab_general:
            st.caption(L(ui_lang, "订阅范围与展示策略", "Scope and display strategy"))
            g1, g2 = st.columns(2)
            with g1:
                selected_fields = st.multiselect(L(ui_lang, "领域", "Fields"), FIELD_OPTIONS, default=selected_fields)
                custom_fields = st.text_input(L(ui_lang, "自定义领域", "Custom Fields"), custom_fields)
                selected_journals = st.multiselect(L(ui_lang, "期刊", "Journals"), JOURNAL_OPTIONS, default=selected_journals)
                custom_journals = st.text_input(L(ui_lang, "自定义期刊", "Custom Journals"), custom_journals)
                strict_journal_only = st.toggle(L(ui_lang, "严格期刊匹配", "Strict journal match"), value=strict_journal_only)
            with g2:
                keywords = st.text_input(L(ui_lang, "关键词", "Keywords"), keywords)
                exclude = st.text_input(L(ui_lang, "排除关键词", "Exclude Keywords"), exclude)
                schedule_label = st.selectbox(
                    L(ui_lang, "推送周期", "Push Schedule"),
                    [schedule_options[k] for k in schedule_keys],
                    index=schedule_keys.index(push_schedule),
                )
                push_schedule = schedule_keys[[schedule_options[k] for k in schedule_keys].index(schedule_label)]
                custom_days = st.slider(L(ui_lang, "自定义天数", "Custom range days"), 1, 60, custom_days, 1, disabled=push_schedule != "custom")
                max_papers = st.slider(L(ui_lang, "最多论文数（0=全部）", "Max Papers (0=All)"), 0, 50, max_papers, 1)
                layout_label = st.selectbox(
                    L(ui_lang, "信息流布局", "Feed Layout"),
                    [layout_options[k] for k in layout_keys],
                    index=layout_keys.index(saved_layout),
                )
                layout_mode = layout_keys[[layout_options[k] for k in layout_keys].index(layout_label)]
                proxy_prefix = st.text_input(L(ui_lang, "学校代理前缀", "Institution Proxy Prefix"), proxy_prefix)

        with tab_delivery:
            st.caption(L(ui_lang, "自动推送与投递渠道", "Automation and delivery channels"))
            d1, d2 = st.columns(2)
            with d1:
                enable_auto_push = st.toggle(
                    L(ui_lang, "启用自动推送订阅", "Enable Auto Push Subscription"),
                    value=enable_auto_push,
                    help=L(ui_lang, "开启后可被后台定时任务按用户独立推送。", "When enabled, backend scheduler can push per user."),
                    disabled=not AUTO_PUSH_ENABLED,
                )
                if not AUTO_PUSH_ENABLED:
                    reason = auto_push_backend_reason()
                    st.caption(
                        L(
                            ui_lang,
                            f"自动推送订阅存储未启用：{reason or '请配置 AUTO_PUSH_DATABASE_URL 或 SERVER_PERSISTENCE=1。'}",
                            f"Auto-push subscription storage is disabled: {reason or 'Configure AUTO_PUSH_DATABASE_URL or SERVER_PERSISTENCE=1.'}",
                        )
                    )
                daily_push_time = st.text_input(
                    L(ui_lang, "每日推送时间（HH:MM）", "Daily Push Time (HH:MM)"),
                    value=daily_push_time,
                    help=L(ui_lang, "用于外部定时任务（cron/云调度）。", "Used by external scheduler (cron/cloud scheduler)."),
                )
                tz_options = list(COMMON_TIMEZONE_OPTIONS)
                if saved_tz not in tz_options:
                    tz_options = [saved_tz] + tz_options
                push_timezone = st.selectbox(
                    L(ui_lang, "推送时区（IANA）", "Push Timezone (IANA)"),
                    tz_options,
                    index=tz_options.index(saved_tz),
                    help=L(ui_lang, "选择推送时区。", "Select a timezone for auto push."),
                )
            with d2:
                enable_webhook_push = st.toggle(L(ui_lang, "启用 Webhook 推送", "Enable Webhook Push"), value=enable_webhook_push)
                webhook_url = st.text_input(L(ui_lang, "Webhook 地址", "Webhook URL"), webhook_url, disabled=not enable_webhook_push)
                email_to = st.text_input(L(ui_lang, "收件邮箱", "Email To"), email_to)
                auto_send_email = st.toggle(
                    L(ui_lang, "自动发送邮件", "Auto send email"),
                    value=auto_send_email,
                    disabled=not smtp_ready_b,
                )
                st.caption(L(ui_lang, "邮箱发送由后台 SMTP 配置统一管理。", "Email delivery uses backend SMTP configuration."))
                if not smtp_ready_b:
                    st.caption(
                        L(
                            ui_lang,
                            "后台 SMTP 未配置（SMTP_HOST/PORT/USER/PASSWORD），自动发邮件不可用。",
                            "Backend SMTP is not configured (SMTP_HOST/PORT/USER/PASSWORD), auto-email is unavailable.",
                        )
                    )

        with tab_ai:
            st.caption(L(ui_lang, "模型与内容深读策略", "Model and deep-reading strategy"))
            a1, a2 = st.columns(2)
            with a1:
                use_api = st.toggle(L(ui_lang, "启用 ChatGPT API", "Use ChatGPT API"), value=use_api)
                session_api_key = st.text_input(
                    L(ui_lang, "会话 API Key", "Session API Key"),
                    value=session_api_key,
                    type="password",
                    help=L(
                        ui_lang,
                        "会话有效；关闭应用后需重新填写。推荐生产环境用 Secrets。",
                        "Session-only; you need to re-enter after restart. Use Secrets for production.",
                    ),
                )
                api_model = st.text_input(L(ui_lang, "模型", "Model"), api_model, disabled=not use_api)
            with a2:
                deep_read_mode = st.toggle(L(ui_lang, "可访问时读取全文", "AI read full text when possible"), value=deep_read_mode, disabled=not use_api)
                deep_read_limit = st.slider(L(ui_lang, "全文阅读篇数", "Full-text read count"), 1, 10, deep_read_limit, 1, disabled=not use_api)
                worth_count = st.slider(L(ui_lang, "值得读数量", "Worth Reading count"), 2, 8, worth_count, 1)
                auto_refresh_on_load = st.toggle(L(ui_lang, "自动刷新（加载即运行）", "Auto refresh with saved settings"), value=auto_refresh_on_load)

        cs, cc = st.columns([1, 1])
        with cs:
            if st.button(L(ui_lang, "保存设置", "Save Settings"), type="primary", disabled=busy):
                prev_webhook_url = str(cur.get("webhook_url", "")).strip()
                input_webhook_url = webhook_url.strip()
                daily_push_time_norm = normalize_hhmm(daily_push_time)
                if not daily_push_time_norm:
                    st.error(L(ui_lang, "每日推送时间格式无效，请使用 HH:MM（24小时制）。", "Invalid daily push time. Use HH:MM in 24-hour format."))
                    return
                push_timezone_norm = normalize_timezone(push_timezone)
                if not push_timezone_norm:
                    st.error(L(ui_lang, "时区无效，请填写 IANA 时区名（如 Asia/Shanghai）。", "Invalid timezone. Use an IANA timezone name (for example Asia/Shanghai)."))
                    return
                # Keep existing webhook URL unless user explicitly replaces it.
                webhook_url_final = input_webhook_url or prev_webhook_url
                if enable_webhook_push and not webhook_url_final:
                    st.error(L(ui_lang, "已启用 Webhook 推送，请填写 Webhook URL。", "Webhook push is enabled. Please provide a Webhook URL."))
                    return
                if enable_auto_push:
                    has_webhook_target = bool(enable_webhook_push and webhook_url_final)
                    has_email_target = bool(auto_send_email and email_to.strip())
                    if not has_webhook_target and not has_email_target:
                        st.error(
                            L(
                                ui_lang,
                                "已启用自动推送订阅，但没有可用推送目标。请配置 Webhook 或开启自动邮件并填写收件邮箱。",
                                "Auto push is enabled but no delivery target is configured. Set webhook or auto-email recipient.",
                            )
                        )
                        return
                new_settings = {
                    "language": "zh" if language == lang_labels[0] else "en",
                    "fields": normalize_str_list_input(selected_fields),
                    "custom_fields": custom_fields,
                    "journals": normalize_str_list_input(selected_journals),
                    "custom_journals": custom_journals,
                    "strict_journal_only": strict_journal_only,
                    "keywords": keywords,
                    "exclude_keywords": exclude,
                    "proxy_prefix": proxy_prefix,
                    "push_schedule": push_schedule,
                    "custom_days": custom_days,
                    "max_papers": max_papers,
                    "reading_level": "mixed",
                    "wr": 0.35,
                    "wn": 0.25,
                    "wg": 0.25,
                    "wi": 0.15,
                    "layout_mode": layout_mode,
                    "enable_webhook_push": bool(enable_webhook_push and bool(webhook_url_final)),
                    "webhook_url": webhook_url_final,
                    "email_to": email_to,
                    "auto_send_email": bool(auto_send_email and smtp_ready_b),
                    "enable_auto_push": bool(enable_auto_push and AUTO_PUSH_ENABLED),
                    "daily_push_time": daily_push_time_norm,
                    "push_timezone": push_timezone_norm,
                    "subscriber_id": str(cur.get("subscriber_id", "")).strip() or uuid4().hex,
                    "use_api": use_api,
                    "api_model": api_model,
                    "deep_read_mode": deep_read_mode,
                    "deep_read_limit": deep_read_limit,
                    "ai_pick_worth": True,
                    "worth_count": worth_count,
                    "auto_refresh_on_load": auto_refresh_on_load,
                }
                ok, msg = save_settings(new_settings)
                if ok:
                    requested_auto_push = bool(new_settings.get("enable_auto_push", False))
                    sub_ok = True
                    sub_msg = ""
                    if requested_auto_push:
                        sub_ok, sub_msg = upsert_auto_push_subscription(
                            new_settings["subscriber_id"],
                            new_settings,
                            enabled=True,
                        )
                        if not sub_ok:
                            # Keep normal settings, but do not mark auto-push as enabled when remote persistence failed.
                            new_settings["enable_auto_push"] = False
                            _fallback_ok, _fallback_msg = save_settings(new_settings)
                    st.session_state.session_openai_api_key = session_api_key.strip()
                    st.session_state.saved_settings = new_settings
                    save_browser_settings(new_settings)
                    # Settings changed: clear previous digest to avoid showing stale results.
                    st.session_state.last_digest = None
                    st.session_state.last_push_text = None
                    st.session_state.last_today_cards = None
                    st.session_state.last_worth_cards = None
                    st.session_state.last_fetch_note = ""
                    st.session_state.last_fetch_diag = {}
                    if requested_auto_push and not sub_ok:
                        st.warning(
                            L(
                                ui_lang,
                                f"{msg} 其他设置已保存，但自动推送订阅保存失败：{sub_msg}",
                                f"{msg} Other settings were saved, but auto-push subscription failed: {sub_msg}",
                            )
                        )
                    elif new_settings.get("enable_auto_push", False):
                        st.success(
                            L(
                                ui_lang,
                                f"{msg} 自动推送订阅已更新（用户ID: {new_settings['subscriber_id'][:8]}）。",
                                f"{msg} Auto-push subscription updated (user id: {new_settings['subscriber_id'][:8]}).",
                            )
                        )
                    else:
                        st.success(msg)
                else:
                    st.error(msg)
        with cc:
            if st.button(L(lang_opt, "关闭", "Close"), disabled=busy):
                st.rerun()
        st.download_button(
            L(ui_lang, "下载当前 User Preferences", "Download User Preferences"),
            data=json.dumps(cur, ensure_ascii=False, indent=2),
            file_name="user_prefs.json",
            mime="application/json",
            disabled=busy,
        )
        tz_for_cron = normalize_timezone(str(cur.get("push_timezone", os.getenv("APP_TIMEZONE", "America/New_York")))) or "America/New_York"
        if SHOW_AUTO_PUSH_ADMIN_INFO:
            hhmm_for_cron = normalize_hhmm(str(cur.get("daily_push_time", "09:00"))) or "09:00"
            hh, mm = hhmm_for_cron.split(":")
            if AUTO_PUSH_ENABLED:
                if AUTO_PUSH_BACKEND == "postgres":
                    cron_line = "python daily_push.py --all-due"
                    st.caption(
                        L(
                            ui_lang,
                            "多用户自动推送（推荐在 GitHub Actions/云调度中每5分钟运行一次）",
                            "Multi-user auto push (recommended: run every 5 minutes in GitHub Actions/cloud scheduler)",
                        )
                    )
                else:
                    cron_line = f"* * * * * cd \"{Path.cwd()}\" && .venv/bin/python daily_push.py --all-due"
                    st.caption(
                        L(
                            ui_lang,
                            "多用户自动推送（推荐每分钟扫描一次到点用户）",
                            "Multi-user auto push (recommended: check due users every minute)",
                        )
                    )
            else:
                cron_line = (
                    f"CRON_TZ={tz_for_cron} {int(mm)} {int(hh)} * * * "
                    f"cd \"{Path.cwd()}\" && .venv/bin/python daily_push.py --prefs user_prefs.json"
                )
                st.caption(L(ui_lang, "每日自动推送（单用户示例）", "Daily auto push (single-user example)"))
            st.code(cron_line, language="bash")
            st.caption(
                L(
                    ui_lang,
                    f"你的订阅用户ID：{str(cur.get('subscriber_id', ''))[:12]}",
                    f"Your subscription user id: {str(cur.get('subscriber_id', ''))[:12]}",
                )
            )
        if st.button(L(ui_lang, "清空抓取缓存", "Clear Fetch Cache"), disabled=busy):
            # Clear runtime/persistent fetch caches only; keep user settings intact.
            st.session_state.fetch_cache = {}
            st.session_state.local_cache = {"abstract_by_id": {}, "llm_summary_cache": {}}
            st.session_state.llm_summary_cache = {}
            st.session_state.local_cache_dirty = True
            st.success(L(ui_lang, "已清空抓取缓存（设置保持不变）。", "Fetch cache cleared (settings unchanged)."))
            st.rerun()

    @st.dialog(L(st.session_state.saved_settings.get("language", "zh"), "使用说明", "How to Use"), width="large")
    def guide_modal() -> None:
        glang = st.session_state.saved_settings.get("language", "zh")
        st.markdown(
            L(
                glang,
                "【快速开始】\n"
                "访问地址：https://research-push.streamlit.app/\n"
                "1. 点 ⚙ 设置期刊、关键词和推送方式。\n"
                "2. 点“生成今日Digest”抓取并生成摘要。\n"
                "3. 在「今日Feed / 值得读 / 趋势洞察」查看结果。\n"
                "\n"
                "【推送到 Slack】\n"
                "1. 在 Slack 创建 Incoming Webhook（hooks.slack.com/services/...）。\n"
                "2. 在设置里开启“启用 Webhook 推送”，填入 Webhook URL 并保存。\n"
                "3. 生成 Digest 后，点击“推送到 Webhook”。\n"
                "\n"
                "【Slack 常见问题】\n"
                "- URL 为空：请先在设置保存 Webhook URL。\n"
                "- 推送无消息：检查 URL 是否有效、频道权限是否允许机器人发言。\n"
                "- 值得读为空：需开启 ChatGPT API 并填写会话 API Key。\n"
                "\n"
                "【邮件说明】\n"
                "- 前端只需要填收件邮箱；SMTP 由后台配置。\n"
                "\n"
                "【隐私说明】\n"
                "- 会话模式下 API Key 仅会话有效，不会保存到服务器。",
                "[Quick Start]\n"
                "App URL: https://research-push.streamlit.app/\n"
                "1. Click ⚙ to set journals, keywords, and delivery options.\n"
                "2. Click 'Generate Today's Digest'.\n"
                "3. Read results in Today Feed / Worth Reading / Insights.\n"
                "\n"
                "[Push to Slack]\n"
                "1. Create a Slack Incoming Webhook (hooks.slack.com/services/...).\n"
                "2. Enable 'Webhook Push' in Settings, paste the Webhook URL, and save.\n"
                "3. After digest generation, click 'Push to Webhook'.\n"
                "\n"
                "[Slack Troubleshooting]\n"
                "- Empty URL: save a valid Webhook URL in Settings first.\n"
                "- No message delivered: verify webhook validity and channel bot permissions.\n"
                "- Empty Worth Reading: enable ChatGPT API and set a session API key.\n"
                "\n"
                "[Email]\n"
                "- Frontend only needs recipient email; SMTP is configured on backend.\n"
                "\n"
                "[Privacy]\n"
                "- In session mode, API key is session-only and never persisted on server.",
            ).replace("\n", "  \n")
        )
        if st.button(L(glang, "关闭", "Close")):
            st.rerun()

    top_left, top_spacer, top_right_l, top_right_r = st.columns([4, 6, 1.2, 1.2], gap="small")
    with top_left:
        run = st.button(
            L(s.get("language", "zh"), "生成今日Digest", "Generate Today's Digest"),
            type="primary",
            key="generate_main_btn",
            use_container_width=True,
            disabled=bool(st.session_state.get("is_generating", False)),
        )
    with top_right_l:
        open_guide = st.button(
            "ⓘ",
            help=L(s.get("language", "zh"), "使用说明", "Guide"),
            key="guide_btn",
            use_container_width=True,
            disabled=bool(st.session_state.get("is_generating", False)),
        )
    with top_right_r:
        open_settings = st.button(
            "⚙",
            help=L(s.get("language", "zh"), "设置", "Settings"),
            key="settings_toggle_btn",
            use_container_width=True,
            disabled=bool(st.session_state.get("is_generating", False)),
        )
    if open_settings:
        # Keep current in-session settings to avoid dialog auto-close from extra reruns.
        settings_modal()
    if open_guide:
        guide_modal()

    s = st.session_state.saved_settings
    lang = s.get("language", "zh")
    selected_fields = normalize_str_list_input(s.get("fields", []))
    custom_fields = s.get("custom_fields", "")
    selected_journals = normalize_str_list_input(s.get("journals", []))
    custom_journals = s.get("custom_journals", "")
    strict_journal_only = bool(s.get("strict_journal_only", True))
    keywords = s.get("keywords", "")
    exclude = s.get("exclude_keywords", "survey")
    push_schedule = str(s.get("push_schedule", "daily")).lower()
    if push_schedule == "weekly (monday)":
        push_schedule = "weekly_monday"
    custom_days = int(s.get("custom_days", 7))
    max_papers = int(s.get("max_papers", 0))
    layout_mode = str(s.get("layout_mode", "board2"))
    legacy_layout = {"舒展单列": "expanded", "紧凑单列": "compact", "Board-2列": "board2", "Board-3列": "board3"}
    layout_mode = legacy_layout.get(layout_mode, layout_mode)
    enable_webhook_push = bool(s.get("enable_webhook_push", False))
    proxy_prefix = s.get("proxy_prefix", "")
    webhook_url = s.get("webhook_url", "")
    email_to = s.get("email_to", "")
    smtp_host, smtp_port, smtp_user, smtp_password = get_backend_smtp_config()
    smtp_ready = all([smtp_host, smtp_user, smtp_password])
    auto_send_email = bool(s.get("auto_send_email", False))
    use_api = bool(s.get("use_api", False))
    api_model = s.get("api_model", "gpt-4.1-mini")
    deep_read_mode = bool(s.get("deep_read_mode", False))
    deep_read_limit = int(s.get("deep_read_limit", 5))
    worth_count = int(s.get("worth_count", 4))
    auto_refresh_on_load = bool(s.get("auto_refresh_on_load", False))

    # If multiselect has values, treat it as the source of truth and ignore custom text leftovers.
    # This avoids hidden stale entries (e.g., old custom journals) from polluting selection.
    if selected_fields:
        fields = list(dict.fromkeys(selected_fields))
    else:
        fields = list(dict.fromkeys(parse_csv(custom_fields)))
    if selected_journals:
        journals = list(dict.fromkeys(selected_journals))
    else:
        journals = list(dict.fromkeys(parse_csv(custom_journals)))
    keywords = parse_csv(keywords)
    exclude = parse_csv(exclude)
    api_key = get_backend_openai_api_key()
    if use_api and not api_key:
        st.warning(
            L(
                lang,
                "请在设置中填写会话 API Key；未填写时 AI 功能不可用。"
                if not PERSIST_TO_DISK
                else "未检测到 OPENAI_API_KEY，AI相关功能将回退到非AI模式。",
                "Set a Session API Key in Settings; without it AI features are unavailable."
                if not PERSIST_TO_DISK
                else "OPENAI_API_KEY not found. AI features will fall back to non-AI mode.",
            )
        )

    if push_schedule == "daily":
        days = 1
    elif push_schedule == "weekly_monday":
        days = 7
    else:
        days = int(custom_days)
    is_monday = now_utc().weekday() == 0

    prefs = {
        "language": lang,
        "fields": fields,
        "journals": journals,
        "strict_journal_only": strict_journal_only,
        "keywords": keywords,
        "exclude_keywords": exclude,
        "date_range_days": days,
        "max_papers": max_papers,
        "reading_level": "mixed",
        "ranking_weights": {"relevance": 0.35, "novelty": 0.25, "rigor": 0.25, "impact": 0.15},
    }
    prefs_sig = json.dumps(prefs, ensure_ascii=False, sort_keys=True)

    if auto_refresh_on_load and "auto_ran_once" not in st.session_state:
        if push_schedule == "weekly_monday" and not is_monday:
            run = False
        else:
            run = True
        st.session_state.auto_ran_once = True

    if run:
        st.session_state.is_generating = True
        try:
            if push_schedule == "weekly_monday" and not is_monday:
                st.info(L(lang, "当前是周一推模式：仅周一自动刷新。你也可以手动点按钮强制生成。", "Weekly mode is active: auto refresh runs on Mondays only. You can still trigger manually."))
            if not journals and not fields and not keywords:
                st.error(L(lang, "请至少选择一个 Journal，或填写 Fields/Keywords。", "Please select at least one journal or provide fields/keywords."))
                st.stop()
            with st.status(L(lang, "正在生成 Digest...", "Generating digest..."), expanded=True) as run_status:
                run_status.write(L(lang, "开始抓取候选论文...", "Start fetching candidate papers..."))
                papers, fetch_note, fetch_diag, effective_filter = fetch_candidates(prefs)
            prefs_runtime = dict(prefs)
            prefs_runtime["date_range_days"] = int(effective_filter.get("effective_days", prefs["date_range_days"]))
            prefs_runtime["strict_journal_only"] = bool(
                effective_filter.get("effective_strict_journal_only", prefs["strict_journal_only"])
            )
            run_status.write(fetch_note)
            run_status.write(
                L(
                    lang,
                    f"当前生效期刊：{len(journals)} 个（{', '.join(journals[:5])}{'...' if len(journals) > 5 else ''}）",
                    f"Active journals: {len(journals)} ({', '.join(journals[:5])}{'...' if len(journals) > 5 else ''})",
                )
            )
            run_status.write(
                L(
                    lang,
                    f"生效期刊完整列表：{', '.join(journals)}",
                    f"Active journal list: {', '.join(journals)}",
                )
            )
            run_status.write(
                L(
                    lang,
                    f"抓取明细：Crossref {fetch_diag.get('crossref',0)} | PubMed {fetch_diag.get('pubmed',0)} | RSS {fetch_diag.get('rss',0)} | arXiv {fetch_diag.get('arxiv',0)} | Raw {fetch_diag.get('total_raw',0)}",
                    f"Fetch breakdown: Crossref {fetch_diag.get('crossref',0)} | PubMed {fetch_diag.get('pubmed',0)} | RSS {fetch_diag.get('rss',0)} | arXiv {fetch_diag.get('arxiv',0)} | Raw {fetch_diag.get('total_raw',0)}",
                )
            )
            if papers:
                venue_counts: dict[str, int] = {}
                for p in papers:
                    v = (p.venue or "Unknown Venue").strip() or "Unknown Venue"
                    venue_counts[v] = venue_counts.get(v, 0) + 1
                top_venues = ", ".join(
                    [f"{k}:{v}" for k, v in sorted(venue_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]]
                )
                run_status.write(
                    L(
                        lang,
                        f"期刊命中分布：{top_venues}",
                        f"Journal hit distribution: {top_venues}",
                    )
                )
            if int(fetch_diag.get("journal_backfill", 0)) > 0:
                run_status.write(
                    L(
                        lang,
                        f"多期刊补抓命中：{fetch_diag.get('journal_backfill',0)}",
                        f"Multi-journal backfill hits: {fetch_diag.get('journal_backfill',0)}",
                    )
                )
            if int(fetch_diag.get("cache_hit", 0)) == 1:
                run_status.write(L(lang, "命中缓存（15分钟）", "Cache hit (15 minutes)"))
            elif int(fetch_diag.get("cache_hit", 0)) == 2:
                run_status.write(L(lang, "上游暂时无结果，已使用历史缓存", "Upstream temporarily empty; used stale cache"))
            if not papers:
                run_status.update(label=L(lang, "生成失败", "Generation failed"), state="error")
                st.error(
                    L(
                        lang,
                        "未抓取到候选论文。建议：1) 关闭 Daily Mode 2) 放宽期刊匹配 3) 增加时间窗口到7-14天。",
                        "No papers fetched. Try: 1) turn off Daily mode 2) relax journal match 3) extend to 7-14 days.",
                    )
                )
                st.caption(
                    L(
                        lang,
                        f"失败明细：{fetch_note} | Crossref {fetch_diag.get('crossref',0)} | PubMed {fetch_diag.get('pubmed',0)} | RSS {fetch_diag.get('rss',0)} | arXiv {fetch_diag.get('arxiv',0)}",
                        f"Failure details: {fetch_note} | Crossref {fetch_diag.get('crossref',0)} | PubMed {fetch_diag.get('pubmed',0)} | RSS {fetch_diag.get('rss',0)} | arXiv {fetch_diag.get('arxiv',0)}",
                    )
                )
                st.stop()
            run_status.write(L(lang, "正在构建 Digest 与 AI 总结...", "Building digest and AI summaries..."))
            digest = build_digest(prefs_runtime, papers)
            # Second-pass recovery: prioritize abstract completeness for displayed cards.
            enrich_digest_display_abstracts(digest, lang=lang, max_cards=MAX_DISPLAY_ABSTRACT_RECOVERY)
            if use_api and api_key.strip():
                local_cache = st.session_state.get("local_cache", {})
                persistent_llm_cache = local_cache.get("llm_summary_cache", {}) if isinstance(local_cache, dict) else {}
                llm_cache = st.session_state.setdefault("llm_summary_cache", persistent_llm_cache if isinstance(persistent_llm_cache, dict) else {})
                all_targets: list[tuple[str, int, dict[str, Any]]] = []
                for gk in ("top_picks", "also_notable"):
                    for idx, card_obj in enumerate(digest[gk]):
                        all_targets.append((gk, idx, card_obj))
                top_targets = sorted(
                    all_targets, key=lambda x: int(x[2].get("scores", {}).get("total", 0)), reverse=True
                )[:MAX_AI_ENHANCE]
                no_abs_targets = [
                    t for t in all_targets if not str(t[2].get("source_abstract", "")).strip()
                ][:MAX_NO_ABSTRACT_AI]
                ai_targets: list[tuple[str, int, dict[str, Any]]] = []
                seen_ids: set[str] = set()
                for t in top_targets + no_abs_targets:
                    pid = str(t[2].get("paper_id", ""))
                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    ai_targets.append(t)
                deep_read_counter = 0
                no_abs_fulltext_counter = 0
                for _group_key, _idx, c in ai_targets:
                    p = Paper(
                        title=c["title"],
                        authors=[],
                        venue=c["venue"],
                        publication_date=c["date"],
                        doi=c["paper_id"][4:] if c["paper_id"].startswith("doi:") else "",
                        pmid=c["paper_id"][5:] if c["paper_id"].startswith("pmid:") else "",
                        arxiv_id=c["paper_id"][6:] if c["paper_id"].startswith("arxiv:") else "",
                        abstract=c.get("source_abstract", ""),
                        url=c["link"],
                    )
                    full_text = ""
                    content_source = ""
                    no_abs = not str(c.get("source_abstract", "")).strip()
                    should_try_fulltext = False
                    if no_abs and no_abs_fulltext_counter < MAX_NO_ABSTRACT_FULLTEXT:
                        should_try_fulltext = True
                    elif deep_read_mode and deep_read_counter < deep_read_limit:
                        should_try_fulltext = True
                    if should_try_fulltext:
                        full_text, content_source = fetch_paper_content(p)
                        if full_text:
                            c["source_content"] = full_text
                            c["content_source_label"] = content_source
                            c["evidence_note"] = f"Based on full paper content ({content_source}) + metadata."
                            if no_abs:
                                no_abs_fulltext_counter += 1
                            else:
                                deep_read_counter += 1
                    llm_key = json.dumps(
                        {
                            "paper_id": c.get("paper_id", ""),
                            "model": api_model,
                            "lang": lang,
                            "deep": bool(full_text),
                            "content_source": content_source,
                            "abstract": c.get("source_abstract", ""),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    enhanced = llm_cache.get(llm_key)
                    if enhanced is None:
                        enhanced = llm_enhance_summary(
                            p,
                            c["scores"],
                            prefs_runtime,
                            api_key,
                            api_model,
                            full_text=full_text,
                            content_source=content_source,
                        )
                        if enhanced:
                            llm_cache[llm_key] = enhanced
                            if "local_cache" not in st.session_state or not isinstance(st.session_state.local_cache, dict):
                                st.session_state.local_cache = {"abstract_by_id": {}, "llm_summary_cache": {}}
                            st.session_state.local_cache.setdefault("llm_summary_cache", {})[llm_key] = enhanced
                            st.session_state.local_cache_dirty = True
                    if enhanced:
                        if enhanced.get("methods_detailed"):
                            c["methods_detailed"] = enhanced["methods_detailed"]
                        if enhanced.get("main_conclusion"):
                            c["main_conclusion"] = enhanced["main_conclusion"]
                        if enhanced.get("future_direction"):
                            c["future_direction"] = enhanced["future_direction"]
                        if enhanced.get("ai_feed_summary"):
                            c["ai_feed_summary"] = enhanced["ai_feed_summary"]
                        c["value_assessment"] = enhanced["value_assessment"]

            push_text = build_daily_push_text(digest, lang=lang)
            today_cards = digest["top_picks"] + digest["also_notable"]
            worth_cards = ai_select_worth_reading(
                cards=today_cards,
                api_key=api_key,
                model=api_model,
                desired_count=worth_count,
            )
            if not use_api:
                st.warning(
                    L(
                        lang,
                        "Worth Reading 需要在设置中开启 ChatGPT API。",
                        "Worth Reading requires enabling ChatGPT API in Settings.",
                    )
                )
            elif not api_key.strip():
                st.warning(
                    L(
                        lang,
                        "Worth Reading 需要会话 API Key 才能由 AI 生成。"
                        if not PERSIST_TO_DISK
                        else "Worth Reading 需要 OPENAI_API_KEY 才能由 AI 生成。",
                        "Worth Reading requires a Session API Key to be generated by AI."
                        if not PERSIST_TO_DISK
                        else "Worth Reading requires OPENAI_API_KEY to be generated by AI.",
                    )
                )
            elif today_cards and not worth_cards:
                st.warning(
                    L(
                        lang,
                        "AI 值得读未生成结果：可能是模型调用失败、超时或配额限制，请稍后重试。",
                        "AI Worth Reading returned no result: possible model failure, timeout, or quota/rate limit. Please retry.",
                    )
                )
            show_ai_summary = bool(use_api and api_key.strip())
            push_text["worth_reading_summary"] = build_worth_summary_from_cards(
                worth_cards, lang=lang, show_ai_summary=show_ai_summary
            )
            st.session_state.last_digest = digest
            st.session_state.last_push_text = push_text
            st.session_state.last_today_cards = today_cards
            st.session_state.last_worth_cards = worth_cards
            st.session_state.last_fetch_note = fetch_note
            st.session_state.last_fetch_diag = fetch_diag
            st.session_state.last_digest_prefs_sig = prefs_sig
            if st.session_state.get("local_cache_dirty", False):
                save_local_cache(st.session_state.get("local_cache", {}))
                st.session_state.local_cache_dirty = False
            run_status.update(label=L(lang, "Digest 生成完成", "Digest ready"), state="complete")
        finally:
            st.session_state.is_generating = False

    if st.session_state.get("last_digest") and st.session_state.get("last_digest_prefs_sig", "") == prefs_sig:
        digest = st.session_state.last_digest
        push_text = st.session_state.last_push_text or {"today_new_summary": "", "worth_reading_summary": ""}
        today_cards = st.session_state.last_today_cards or []
        worth_cards = st.session_state.last_worth_cards or []
        fetch_note = st.session_state.get("last_fetch_note", "")
        fetch_diag = st.session_state.get("last_fetch_diag", {}) or {}

        # Run details are already shown in the generation status panel to avoid duplicate messaging.

        header = digest["digest_header"]
        k1, k2, k3 = st.columns([1, 1, 1], gap="small")
        k1.markdown(f"<div class='kpi'><strong>{L(lang,'覆盖范围','Coverage')}</strong><br>{header['coverage']}</div>", unsafe_allow_html=True)
        k2.markdown(
            f"<div class='kpi'><strong>{L(lang,'抓取/去重','Fetched/Deduped')}</strong><br>{header['stats']['fetched_count']} / {header['stats']['deduplicated_count']}</div>",
            unsafe_allow_html=True,
        )
        k3.markdown(f"<div class='kpi'><strong>{L(lang,'入选论文','Selected')}</strong><br>{header['stats']['selected_count']}</div>", unsafe_allow_html=True)
        diag = header.get("diagnostics", {})
        if diag.get("filter_mode") and diag.get("filter_mode") != "strict":
            st.info(L(lang, f"已自动放宽过滤模式：{diag.get('filter_mode')}。", f"Auto-relaxed filter mode: {diag.get('filter_mode')}."))
        if header["stats"]["selected_count"] == 0:
            st.warning(
                L(
                    lang,
                    "有抓取结果但无展示：通常是筛选过严。"
                    f" Raw={diag.get('raw_count',0)} -> 去重={diag.get('deduped_count',0)} -> 过滤后={diag.get('filtered_count',0)}。",
                    "Fetched but nothing displayed: filters are likely too strict. "
                    f"Raw={diag.get('raw_count',0)} -> Deduped={diag.get('deduped_count',0)} -> Filtered={diag.get('filtered_count',0)}.",
                )
            )

        email_body = format_email_body(digest, push_text, lang=lang)
        digest_sig = json.dumps(digest, ensure_ascii=False, sort_keys=True)
        st.markdown(f"<p class='toolbar-title'>{L(lang, '推送与发送', 'Delivery')}</p>", unsafe_allow_html=True)
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            push_clicked = st.button(
                L(lang, "推送到 Webhook", "Push to Webhook"),
                disabled=bool(st.session_state.get("is_generating", False)) or (not enable_webhook_push),
            )
            if push_clicked:
                if not webhook_url.strip():
                    st.error(L(lang, "Webhook URL 为空，请在 ⚙ Settings 中填写 Webhook URL 并点击 Save Settings。", "Webhook URL is empty. Fill it in ⚙ Settings and click Save Settings."))
                else:
                    ok, msg = post_webhook(
                        webhook_url,
                        {
                            "date": now_utc().strftime("%Y-%m-%d"),
                            "today_new_summary": push_text["today_new_summary"],
                            "worth_reading_summary": push_text["worth_reading_summary"],
                            "digest": digest,
                        },
                        lang=lang,
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
        with action_col2:
            send_clicked = st.button(
                L(lang, "发送到邮箱", "Send Email"),
                disabled=bool(st.session_state.get("is_generating", False)) or (not smtp_ready),
            )
            if send_clicked:
                ok, msg = send_email_digest(
                    smtp_host=smtp_host,
                    smtp_port=int(smtp_port),
                    smtp_user=smtp_user,
                    smtp_password=smtp_password,
                    to_email=email_to,
                    subject=f"Research Digest {now_utc().strftime('%Y-%m-%d')}",
                    body=email_body,
                    lang=lang,
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        if not smtp_ready:
            st.caption(L(lang, "后台未配置 SMTP，邮箱发送暂不可用。", "Backend SMTP is not configured, so email delivery is unavailable."))
        if not enable_webhook_push:
            st.caption(L(lang, "Webhook 未启用，可在设置中开启。", "Webhook is disabled. Enable it in Settings."))
        if auto_send_email:
            if st.session_state.get("last_auto_email_sig", "") == digest_sig:
                pass
            elif smtp_ready and email_to.strip():
                ok, msg = send_email_digest(
                    smtp_host=smtp_host,
                    smtp_port=int(smtp_port),
                    smtp_user=smtp_user,
                    smtp_password=smtp_password,
                    to_email=email_to,
                    subject=f"Research Digest {now_utc().strftime('%Y-%m-%d')}",
                    body=email_body,
                    lang=lang,
                )
                if ok:
                    st.session_state.last_auto_email_sig = digest_sig
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning(L(lang, "自动发邮件已开启，但收件邮箱或后台 SMTP 配置不完整。", "Auto email is enabled, but recipient email or backend SMTP config is incomplete."))

        tab_today, tab_worth, tab_insights = st.tabs(
            [L(lang, "今日Feed", "Today Feed"), L(lang, "值得读", "Worth Reading"), L(lang, "趋势洞察", "Insights")]
        )
        with tab_today:
            st.subheader(L(lang, "今日Feed", "Today Feed"))
            with st.expander(L(lang, "今日新论文总结（可收起）", "Today's Summary (collapsible)"), expanded=True):
                st.markdown(push_text["today_new_summary"].replace("\n", "  \n"))
            render_cards_grouped(
                L(lang, "今日论文流", "Today Feed Papers"),
                today_cards,
                proxy_prefix,
                layout_mode,
                lang=lang,
                show_ai_summary=bool(use_api and api_key.strip()),
            )
        with tab_worth:
            st.subheader(L(lang, "值得读", "Worth Reading"))
            if use_api and api_key.strip():
                st.caption(L(lang, "Worth Reading 完全由 AI 基于论文内容分析生成。", "Worth Reading is fully generated by AI analysis of paper content."))
            with st.expander(L(lang, "Worth Reading Summary（可收起）", "Worth Reading Summary (collapsible)"), expanded=True):
                st.markdown(push_text["worth_reading_summary"].replace("\n", "  \n"))
            render_cards_grouped(
                L(lang, "值得读论文", "Worth Reading Papers"),
                worth_cards,
                proxy_prefix,
                layout_mode,
                lang=lang,
                show_ai_summary=bool(use_api and api_key.strip()),
            )
        with tab_insights:
            st.subheader(L(lang, "趋势洞察", "Insights"))
            for idx, t in enumerate(header["trends"], start=1):
                st.markdown(f"**{L(lang,'洞察','Insight')} {idx}**")
                st.write(t)
            st.write(f"{L(lang,'订阅建议','Subscription suggestion')}: {header['subscription_suggestion']}")
            digest_json = json.dumps(digest, ensure_ascii=False, indent=2)
            with st.expander(L(lang, "查看/下载结构化 JSON", "View/Download Structured JSON")):
                st.code(digest_json, language="json")
                st.download_button(L(lang, "下载 Digest JSON", "Download Digest JSON"), data=digest_json, file_name="research_digest.json", mime="application/json")
    elif st.session_state.get("last_digest"):
        st.info(L(lang, "检测到设置已更新，请重新生成 Digest。", "Settings changed. Please generate a new digest."))

if __name__ == "__main__":
    main()
