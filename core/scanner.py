"""Content safety scanner — prompt injection & exfiltration defense."""

import hashlib
import json
import re
import shutil
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader as _PdfReader
from filelock import FileLock

import config


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class Threat:
    pattern_name: str
    category: str
    matched_text: str
    location: str
    severity: str
    confidence: float


@dataclass
class ScanResult:
    is_safe: bool
    threats: list[Threat] = field(default_factory=list)
    source_file: str = ""
    scan_type: str = ""
    chunks_scanned: int = 0
    llm_escalations: int = 0


@dataclass
class BackendResult:
    threats: list[Threat] = field(default_factory=list)
    suspicion_score: float = 0.0


class ContentBlockedError(Exception):
    def __init__(self, message: str, report_path: Path):
        super().__init__(message)
        self.report_path = report_path


# ── Zero-Width Characters ────────────────────────────────────────────────

_ZERO_WIDTH = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad]"
)
_MULTI_SPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]")


# ── Normalization ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def normalize_text_super_cleaned(text: str) -> str:
    text = normalize_text(text)
    text = _NON_ALNUM.sub("", text)
    return text.lower()


# ── File Hashing ─────────────────────────────────────────────────────────

def compute_file_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Rule Loading ─────────────────────────────────────────────────────────

def load_rules() -> list[dict]:
    with open(config.SCANNER_RULES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rules", [])


def load_rules_version() -> str:
    with open(config.SCANNER_RULES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pattern_version", "0.0")


# ── Scanner Backends ─────────────────────────────────────────────────────

class ScannerBackend:
    def scan(self, text: str, source: str, location: str, scope: str = "document") -> BackendResult:
        raise NotImplementedError


def _make_super_cleaned_pattern(pattern: str) -> str:
    """Convert a regex pattern to match super-cleaned (no-space, alphanumeric-only) text.

    Removes literal spaces and optional-space constructs from the pattern so it
    can match against text where all non-alphanumeric chars have been stripped.
    """
    # Remove literal spaces inside the pattern (not inside character classes)
    # and collapse optional-space groups like '( )?' or '\s*'
    p = pattern
    p = re.sub(r" \?", "", p)          # remove optional spaces ` ?`
    p = re.sub(r"\\s[*+?]?", "", p)    # remove \s \s* \s+ \s?
    p = re.sub(r" ", "", p)             # remove remaining literal spaces
    return p


class RegexBackend(ScannerBackend):
    def __init__(self):
        self._rules = load_rules()
        self._compiled: list[dict] = []
        for rule in self._rules:
            sc_pattern = _make_super_cleaned_pattern(rule["pattern"])
            self._compiled.append({
                **rule,
                "_re": re.compile(rule["pattern"], re.IGNORECASE),
                "_re_sc": re.compile(sc_pattern, re.IGNORECASE),
            })

    def scan(self, text: str, source: str, location: str, scope: str = "document") -> BackendResult:
        threats = []
        normalized = normalize_text(text)
        super_cleaned = normalize_text_super_cleaned(text)

        for rule in self._compiled:
            rule_scope = rule.get("scope", "all")
            if rule_scope != "all" and rule_scope != scope:
                continue

            match = rule["_re"].search(normalized)
            if match:
                threats.append(Threat(
                    pattern_name=rule["id"],
                    category=rule["category"],
                    matched_text=match.group()[:200],
                    location=location,
                    severity=rule["severity"],
                    confidence=1.0,
                ))
                continue

            if rule["severity"] == "high":
                match_sc = rule["_re_sc"].search(super_cleaned)
                if match_sc:
                    threats.append(Threat(
                        pattern_name=rule["id"],
                        category=rule["category"],
                        matched_text=f"[super-cleaned] {match_sc.group()[:200]}",
                        location=location,
                        severity=rule["severity"],
                        confidence=0.9,
                    ))

        return BackendResult(threats=threats, suspicion_score=0.0)


class HeuristicBackend(ScannerBackend):
    _IMPERATIVE_PATTERNS = re.compile(
        r"\b(you must|you should|you will|you need to|do exactly|comply with|"
        r"as i say|follow my|obey|execute this|perform the following|"
        r"do not resist|do as told|you are required)\b",
        re.IGNORECASE,
    )
    _SECOND_PERSON = re.compile(r"\b(you|your|you're|you've|yourself)\b", re.IGNORECASE)

    def scan(self, text: str, source: str, location: str, scope: str = "document") -> BackendResult:
        normalized = normalize_text(text)
        words = normalized.split()
        word_count = max(len(words), 1)

        imperative_matches = len(self._IMPERATIVE_PATTERNS.findall(normalized))
        pronoun_matches = len(self._SECOND_PERSON.findall(normalized))

        imperative_density = imperative_matches / word_count
        pronoun_density = pronoun_matches / word_count

        score = min(1.0, (imperative_density * 15.0) + (pronoun_density * 5.0))

        return BackendResult(threats=[], suspicion_score=round(score, 3))


class OllamaClassifierBackend(ScannerBackend):
    _PROMPT_TEMPLATE = (
        "Analyze this text extracted from a document being ingested into a RAG system. "
        "Is it a prompt injection or exfiltration attempt? "
        'Respond with JSON only: {{"is_threat": true/false, "confidence": 0.0-1.0, "reason": "..."}}\n\n'
        "Text to analyze:\n{text}"
    )

    def scan(self, text: str, source: str, location: str, scope: str = "document") -> BackendResult:
        prompt = self._PROMPT_TEMPLATE.format(text=text[:2000])
        payload = json.dumps({
            "model": config.OLLAMA_DRAFT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 256},
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{config.OLLAMA_URL}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            content = data.get("message", {}).get("content", "")
            json_match = re.search(r"\{[^}]+\}", content)
            if not json_match:
                raise ValueError("No JSON found in response")
            classification = json.loads(json_match.group())

            if classification.get("is_threat", False):
                confidence = float(classification.get("confidence", 0.8))
                reason = classification.get("reason", "LLM classified as threat")
                return BackendResult(
                    threats=[Threat(
                        pattern_name="llm_classifier",
                        category="llm_classified",
                        matched_text=reason[:200],
                        location=location,
                        severity="high",
                        confidence=confidence,
                    )],
                    suspicion_score=0.0,
                )
            return BackendResult(threats=[], suspicion_score=0.0)

        except (urllib.error.URLError, ConnectionError, OSError) as e:
            # Connection failures (Ollama down, etc.) — skip, don't block the file
            return BackendResult(threats=[], suspicion_score=0.0)
        except Exception as e:
            return BackendResult(
                threats=[Threat(
                    pattern_name="llm_classifier_error",
                    category="llm_classified",
                    matched_text=f"LLM classifier parse failure — manual review required: {str(e)[:100]}",
                    location=location,
                    severity="low",
                    confidence=0.3,
                )],
                suspicion_score=0.0,
            )


# ── Cached backend singletons ─────────────────────────────────────────
_regex_backend: RegexBackend | None = None
_heuristic_backend: HeuristicBackend | None = None
_ollama_backend: OllamaClassifierBackend | None = None


def _get_regex_backend() -> RegexBackend:
    global _regex_backend
    if _regex_backend is None:
        _regex_backend = RegexBackend()
    return _regex_backend


def _get_heuristic_backend() -> HeuristicBackend:
    global _heuristic_backend
    if _heuristic_backend is None:
        _heuristic_backend = HeuristicBackend()
    return _heuristic_backend


def _get_ollama_backend() -> OllamaClassifierBackend:
    global _ollama_backend
    if _ollama_backend is None:
        _ollama_backend = OllamaClassifierBackend()
    return _ollama_backend


def scan_text(
    text: str,
    source: str,
    location: str,
    regex_only: bool = False,
    scope: str = "document",
    _llm_budget: int | None = None,
) -> ScanResult:
    """Run the scanner pipeline on a text chunk.

    _llm_budget: remaining LLM escalations allowed for this file.
        None means use the global max. 0 or negative means skip LLM.
    """
    regex_result = _get_regex_backend().scan(text, source, location, scope=scope)

    all_threats = list(regex_result.threats)

    has_high = any(t.severity == "high" for t in regex_result.threats)
    if has_high or regex_only:
        return ScanResult(
            is_safe=len(all_threats) == 0,
            threats=all_threats,
            source_file=source,
            scan_type="content" if not regex_only else "retrieval",
            chunks_scanned=1,
            llm_escalations=0,
        )

    heuristic_result = _get_heuristic_backend().scan(text, source, location, scope=scope)

    llm_escalations = 0
    budget = _llm_budget if _llm_budget is not None else config.SCANNER_MAX_LLM_ESCALATIONS

    if (
        config.SCANNER_LLM_ESCALATION
        and heuristic_result.suspicion_score > config.SCANNER_SUSPICION_THRESHOLD
        and budget > 0
    ):
        llm_result = _get_ollama_backend().scan(text, source, location, scope=scope)
        all_threats.extend(llm_result.threats)
        llm_escalations = 1

    return ScanResult(
        is_safe=len(all_threats) == 0,
        threats=all_threats,
        source_file=source,
        scan_type="content",
        chunks_scanned=1,
        llm_escalations=llm_escalations,
    )


# ── PDF Structure / Metadata ─────────────────────────────────────────────

_DANGEROUS_PDF_KEYS = {"/JS", "/JavaScript", "/OpenAction", "/AA", "/EmbeddedFile", "/Launch"}


def scan_structure(pdf_path: Path) -> ScanResult:
    threats = []

    try:
        reader = _PdfReader(str(pdf_path))
    except Exception as e:
        threats.append(Threat(
            pattern_name="structure_open_error",
            category="pdf_structure",
            matched_text=f"Failed to open PDF: {str(e)[:200]}",
            location="structure:open",
            severity="high",
            confidence=1.0,
        ))
        return ScanResult(is_safe=False, threats=threats, source_file=pdf_path.name,
                          scan_type="structural", chunks_scanned=0, llm_escalations=0)

    if reader.is_encrypted:
        threats.append(Threat(
            pattern_name="structure_encrypted",
            category="pdf_structure",
            matched_text="PDF is encrypted/password-protected — cannot verify content safety",
            location="structure:encryption",
            severity="high",
            confidence=1.0,
        ))

    try:
        # Walk all objects in the PDF for dangerous keys
        from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject
        visited = set()

        def _scan_obj(obj, path="root"):
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)
            if isinstance(obj, DictionaryObject):
                for key_obj, val in obj.items():
                    key_str = str(key_obj)
                    for dkey in _DANGEROUS_PDF_KEYS:
                        if key_str == dkey:
                            threats.append(Threat(
                                pattern_name=f"structure_{dkey.strip('/').lower()}",
                                category="pdf_active_content",
                                matched_text=f"PDF contains {dkey} at {path}",
                                location=f"structure:{path}",
                                severity="high",
                                confidence=1.0,
                            ))
                    _scan_obj(val, f"{path}/{key_str}")
            elif isinstance(obj, ArrayObject):
                for i, item in enumerate(obj):
                    _scan_obj(item, f"{path}[{i}]")
            elif isinstance(obj, IndirectObject):
                try:
                    _scan_obj(obj.get_object(), path)
                except Exception:
                    pass

        if not reader.is_encrypted:
            for page in reader.pages:
                _scan_obj(page.get_object(), "page")
            if reader.trailer:
                _scan_obj(reader.trailer, "trailer")
    except Exception:
        pass

    return ScanResult(
        is_safe=len(threats) == 0,
        threats=threats,
        source_file=pdf_path.name,
        scan_type="structural",
        chunks_scanned=0,
        llm_escalations=0,
    )


_METADATA_FIELDS = ["author", "title", "subject", "keywords", "creator", "producer"]


def extract_pdf_metadata(pdf_path: Path) -> dict:
    reader = _PdfReader(str(pdf_path))
    meta = reader.metadata or {}
    result = {}
    for k in _METADATA_FIELDS:
        # pypdf metadata uses keys like /Author, /Title etc.
        pypdf_key = f"/{k[0].upper()}{k[1:]}"
        result[k] = str(meta.get(pypdf_key, "")) if meta.get(pypdf_key) else ""
    return result


def scan_metadata(meta: dict, source: str) -> ScanResult:
    regex_backend = _get_regex_backend()
    threats = []

    for field_name, value in meta.items():
        if not value or not isinstance(value, str):
            continue
        result = regex_backend.scan(value, source, f"metadata:{field_name}", scope="document")
        threats.extend(result.threats)

    return ScanResult(
        is_safe=len(threats) == 0,
        threats=threats,
        source_file=source,
        scan_type="metadata",
        chunks_scanned=0,
        llm_escalations=0,
    )


# ── Report Generation ─────────────────────────────────────────────────────

def generate_report(results: list[ScanResult], source_file: str, file_hash: str = "") -> Path:
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = re.sub(r"[^\w.\-]", "_", source_file)
    report_path = config.REPORTS_DIR / f"{timestamp}_{safe_name}.md"

    all_threats = []
    total_chunks = 0
    total_llm = 0
    scan_types = set()
    for r in results:
        all_threats.extend(r.threats)
        total_chunks += r.chunks_scanned
        total_llm += r.llm_escalations
        scan_types.add(r.scan_type)

    is_safe = all(r.is_safe for r in results)
    result_str = "PASSED" if is_safe else "BLOCKED"

    try:
        version = load_rules_version()
    except Exception:
        version = "unknown"

    lines = [
        "# Content Safety Report",
        "",
        f"- **File:** {source_file}",
        f"- **Date:** {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- **SHA-256:** {file_hash}",
        f"- **Result:** {result_str}",
        f"- **Pattern Version:** {version}",
        "",
    ]

    if all_threats:
        lines.append("## Threats Found")
        lines.append("")
        for i, t in enumerate(all_threats, 1):
            lines.append(f"### {i}. {t.pattern_name} — {t.severity.upper()} (confidence: {t.confidence})")
            lines.append(f"- **Location:** {t.location}")
            lines.append(f"- **Category:** {t.category}")
            lines.append(f"- **Matched:** \"{t.matched_text}\"")
            lines.append("")

    high_count = sum(1 for t in all_threats if t.severity == "high")
    med_count = sum(1 for t in all_threats if t.severity == "medium")

    lines.extend([
        "## Scan Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Scan types | {', '.join(sorted(scan_types))} |",
        f"| Chunks scanned | {total_chunks} |",
        f"| LLM escalations | {total_llm} |",
        f"| Total threats | {len(all_threats)} |",
        f"| High severity | {high_count} |",
        f"| Medium severity | {med_count} |",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ── Scan History ──────────────────────────────────────────────────────────

def _history_lock():
    return FileLock(str(config.SCAN_HISTORY_PATH) + ".lock")


def load_scan_history() -> dict:
    with _history_lock():
        if not config.SCAN_HISTORY_PATH.exists():
            return {}
        try:
            return json.loads(config.SCAN_HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return {}


def update_scan_history(file_hash: str, filename: str, result: str, pattern_version: str, report: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _history_lock():
        history = {}
        if config.SCAN_HISTORY_PATH.exists():
            try:
                history = json.loads(config.SCAN_HISTORY_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                history = {}

        history[file_hash] = {
            "filename": filename,
            "first_seen": history.get(file_hash, {}).get("first_seen", now),
            "last_seen": now,
            "result": result,
            "pattern_version": pattern_version,
            "report": report,
        }

        config.SCAN_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ── Quarantine ────────────────────────────────────────────────────────────

def clear_quarantine() -> int:
    """Delete all quarantined files and remove blocked entries from scan history.
    Returns the number of files removed."""
    count = 0
    for f in config.QUARANTINE_DIR.iterdir():
        if f.is_file():
            f.unlink()
            count += 1
    # Remove blocked entries from scan history
    with _history_lock():
        if config.SCAN_HISTORY_PATH.exists():
            try:
                history = json.loads(config.SCAN_HISTORY_PATH.read_text(encoding="utf-8"))
                history = {k: v for k, v in history.items() if v.get("result") != "blocked"}
                config.SCAN_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
            except (json.JSONDecodeError, IOError):
                pass
    return count


def quarantine_file(file_path: Path) -> Path:
    dest = config.QUARANTINE_DIR / file_path.name
    if dest.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        counter = 1
        while dest.exists():
            dest = config.QUARANTINE_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
    shutil.move(str(file_path), str(dest))
    return dest


# ── Allowlist ─────────────────────────────────────────────────────────────

def load_allowlist() -> set:
    """Load allowlisted file hashes."""
    if not config.SCANNER_ALLOWLIST_PATH.exists():
        return set()
    try:
        data = json.loads(config.SCANNER_ALLOWLIST_PATH.read_text(encoding="utf-8"))
        return set(data.get("entries", {}).keys())
    except (json.JSONDecodeError, IOError):
        return set()


def add_to_allowlist(file_hash: str, filename: str, reason: str) -> None:
    """Add a file hash to the scanner allowlist."""
    lock = FileLock(str(config.SCANNER_ALLOWLIST_PATH) + ".lock")
    with lock:
        data = {"description": "Pre-approved file hashes that skip scanning", "entries": {}}
        if config.SCANNER_ALLOWLIST_PATH.exists():
            try:
                data = json.loads(config.SCANNER_ALLOWLIST_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass
        data.setdefault("entries", {})[file_hash] = {
            "filename": filename,
            "added": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "reason": reason,
        }
        config.SCANNER_ALLOWLIST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def remove_from_allowlist(file_hash: str) -> bool:
    """Remove a hash from the allowlist. Returns True if found and removed."""
    lock = FileLock(str(config.SCANNER_ALLOWLIST_PATH) + ".lock")
    with lock:
        if not config.SCANNER_ALLOWLIST_PATH.exists():
            return False
        try:
            data = json.loads(config.SCANNER_ALLOWLIST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return False
        if file_hash in data.get("entries", {}):
            del data["entries"][file_hash]
            config.SCANNER_ALLOWLIST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return True
        return False


def quarantine_release(file_hash: str) -> Path:
    """Release a file from quarantine: restore, allowlist, and return path."""
    history = load_scan_history()
    if file_hash not in history:
        raise ValueError(f"Hash {file_hash[:16]}... not found in scan history")

    entry = history[file_hash]
    filename = entry["filename"]

    quarantine_path = config.QUARANTINE_DIR / filename
    if not quarantine_path.exists():
        for f in config.QUARANTINE_DIR.iterdir():
            if f.stem.startswith(Path(filename).stem):
                quarantine_path = f
                break
    if not quarantine_path.exists():
        raise FileNotFoundError(f"File not found in quarantine: {filename}")

    from config import PDF_DIR, DATA_DIR
    if quarantine_path.suffix.lower() == ".pdf":
        dest = PDF_DIR / filename
    else:
        dest = DATA_DIR / "images" / filename

    shutil.move(str(quarantine_path), str(dest))
    add_to_allowlist(file_hash, filename, "Released from quarantine by user")
    update_scan_history(file_hash, filename, "released", load_rules_version())
    return dest


# ── OCR Divergence Check ──────────────────────────────────────────────────

def check_ocr_caption_divergence(ocr: str, caption: str) -> Threat | None:
    ocr_len = len(ocr)
    caption_len = max(len(caption), 1)

    if (
        ocr_len > config.SCANNER_OCR_DIVERGENCE_MIN_CHARS
        and ocr_len > caption_len * config.SCANNER_OCR_DIVERGENCE_RATIO
    ):
        return Threat(
            pattern_name="ocr_caption_divergence",
            category="ocr_divergence",
            matched_text=f"OCR length ({ocr_len}) is {ocr_len / caption_len:.1f}x caption length ({caption_len})",
            location="image:ocr_divergence",
            severity="medium",
            confidence=0.7,
        )
    return None
