"""Tests for scanner data structures, normalization, and rule loading."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def test_threat_dataclass():
    from core.scanner import Threat
    t = Threat(
        pattern_name="role_hijack_01",
        category="role_hijack",
        matched_text="ignore previous instructions",
        location="chunk:3",
        severity="high",
        confidence=1.0,
    )
    assert t.pattern_name == "role_hijack_01"
    assert t.severity == "high"


def test_scan_result_dataclass():
    from core.scanner import ScanResult
    r = ScanResult(
        is_safe=True,
        threats=[],
        source_file="test.pdf",
        scan_type="content",
        chunks_scanned=5,
        llm_escalations=0,
    )
    assert r.is_safe is True
    assert r.chunks_scanned == 5


def test_backend_result_dataclass():
    from core.scanner import BackendResult
    br = BackendResult(threats=[], suspicion_score=0.3)
    assert br.suspicion_score == 0.3


def test_content_blocked_error():
    from core.scanner import ContentBlockedError
    err = ContentBlockedError("blocked", Path("reports/test.md"))
    assert err.report_path == Path("reports/test.md")
    assert str(err) == "blocked"


def test_normalize_text_strips_zero_width():
    from core.scanner import normalize_text
    text = "ignore\u200b previous\ufeff instructions"
    result = normalize_text(text)
    assert "\u200b" not in result
    assert "\ufeff" not in result
    assert "ignore" in result
    assert "instructions" in result


def test_normalize_text_nfkc():
    from core.scanner import normalize_text
    text = "\uff21\uff22\uff23"
    result = normalize_text(text)
    assert result == "ABC"


def test_normalize_text_collapses_whitespace():
    from core.scanner import normalize_text
    text = "hello    world\t\n  test"
    result = normalize_text(text)
    assert result == "hello world test"


def test_normalize_text_super_cleaned():
    from core.scanner import normalize_text_super_cleaned
    assert normalize_text_super_cleaned("I g n o r e") == "ignore"
    assert normalize_text_super_cleaned("I.g.n.o.r.e") == "ignore"
    assert normalize_text_super_cleaned("ignore previous!!! instructions???") == "ignorepreviousinstructions"


def test_compute_file_hash(tmp_path):
    from core.scanner import compute_file_hash
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h1 = compute_file_hash(f)
    assert len(h1) == 64
    f2 = tmp_path / "test2.txt"
    f2.write_text("hello world")
    assert compute_file_hash(f2) == h1
    f3 = tmp_path / "test3.txt"
    f3.write_text("different")
    assert compute_file_hash(f3) != h1


def test_load_rules():
    from core.scanner import load_rules
    rules = load_rules()
    assert len(rules) > 0
    assert rules[0]["id"] == "role_hijack_01"
    assert "pattern" in rules[0]
    assert "scope" in rules[0]


def test_load_rules_pattern_version():
    from core.scanner import load_rules_version
    version = load_rules_version()
    assert version == "1.0"
