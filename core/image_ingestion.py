"""Image ingestion pipeline: describe via Florence-2 (local) → embed → store."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

import config
from config import SCANNER_DRY_RUN
from core.embedder import embed_passages
from core.db import get_or_create_table
from core.scanner import (
    compute_file_hash,
    load_scan_history,
    update_scan_history,
    load_allowlist,
    load_rules_version,
    scan_text,
    check_ocr_caption_divergence,
    generate_report,
    quarantine_file,
    ContentBlockedError,
    ScanResult,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


@dataclass
class DescriptionResult:
    caption: str
    ocr_text: str
    combined: str
FLORENCE_MODEL_ID = "microsoft/Florence-2-large"

_florence_model = None
_florence_processor = None


def _load_florence():
    """Load Florence-2 model and processor. Cached after first call."""
    global _florence_model, _florence_processor
    if _florence_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _florence_processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_ID, trust_remote_code=True
        )
        _florence_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_ID,
            trust_remote_code=True,
            dtype=torch.float16,
            attn_implementation="eager",
        ).to(device)

    return _florence_model, _florence_processor


def _run_florence(image: Image.Image, task: str, text_input: str = "") -> str:
    """Run a Florence-2 task on an image and return the parsed text."""
    model, processor = _load_florence()
    device = next(model.parameters()).device

    prompt = task if not text_input else task + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    # Move each tensor to device with appropriate dtype
    inputs = {
        k: v.to(device, torch.float16) if v.dtype.is_floating_point else v.to(device)
        for k, v in inputs.items()
    }

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        use_cache=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
    return result.get(task, "")


def describe_image(image_path: Path) -> DescriptionResult:
    """Generate a detailed description of an image using Florence-2.

    Combines detailed captioning + OCR for maximum text extraction.
    Returns a DescriptionResult with separate caption, ocr_text, and combined fields.
    """
    image = Image.open(image_path).convert("RGB")

    # Detailed caption — describes visual content
    caption = _run_florence(image, "<MORE_DETAILED_CAPTION>")

    # OCR — extracts all visible text
    ocr_text = _run_florence(image, "<OCR>")

    parts = []
    if caption:
        parts.append(f"Description: {caption}")
    if ocr_text:
        parts.append(f"Extracted text: {ocr_text}")

    return DescriptionResult(
        caption=caption or "",
        ocr_text=ocr_text or "",
        combined="\n".join(parts),
    )


def _image_chunk_id(filename: str) -> str:
    """Deterministic ID for an image chunk."""
    raw = f"image:{filename}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def is_image_ingested(image_path: Path) -> bool:
    """Check if an image has already been ingested."""
    table = get_or_create_table()
    try:
        df = table.to_pandas()
        if df.empty:
            return False
        return image_path.name in df["source_pdf"].values
    except Exception:
        return False


def ingest_image(image_path: Path, force: bool = False) -> int:
    """Describe image via Florence-2 → scan → embed description → store in LanceDB.

    Raises ContentBlockedError if the scanner blocks the image (unless SCANNER_DRY_RUN).
    Returns 1 if ingested, 0 if skipped.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {image_path.suffix}")

    if not force and is_image_ingested(image_path):
        return 0

    # Get description from Florence-2
    description = describe_image(image_path)
    if not description.combined:
        return 0

    # ── Scanner gate ─────────────────────────────────────────────────────
    file_hash = compute_file_hash(image_path)

    allowlist = load_allowlist()
    if file_hash not in allowlist:
        history = load_scan_history()
        current_version = load_rules_version()
        cached = history.get(file_hash, {})
        skip_scan = (
            cached.get("pattern_version") == current_version
            and cached.get("result") == "passed"
        )

        if not skip_scan:
            scan_results = []

            # OCR divergence check
            divergence_threat = check_ocr_caption_divergence(
                description.ocr_text, description.caption
            )
            if divergence_threat:
                divergence_result = ScanResult(
                    is_safe=False,
                    threats=[divergence_threat],
                    source_file=image_path.name,
                    scan_type="ocr_divergence",
                    chunks_scanned=0,
                    llm_escalations=0,
                )
                scan_results.append(divergence_result)

            # Content scan on combined description
            content_result = scan_text(
                description.combined,
                source=image_path.name,
                location="image:description",
                scope="document",
            )
            scan_results.append(content_result)

            overall_safe = all(r.is_safe for r in scan_results)
            result_str = "passed" if overall_safe else "blocked"

            if not overall_safe:
                report_path = generate_report(scan_results, image_path.name, file_hash)
                update_scan_history(file_hash, image_path.name, result_str, current_version, str(report_path))

                if not config.SCANNER_DRY_RUN:
                    quarantine_file(image_path)
                    raise ContentBlockedError(
                        f"Image blocked by content scanner: {image_path.name}",
                        report_path=report_path,
                    )
            else:
                report_path = generate_report(scan_results, image_path.name, file_hash)
                update_scan_history(file_hash, image_path.name, result_str, current_version, str(report_path))

    # ── Embed + store pipeline ────────────────────────────────────────────

    # Prepend image filename for context
    text = f"[Image: {image_path.name}]\n{description.combined}"

    # Embed
    vectors = embed_passages([text])

    # Store
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "id": _image_chunk_id(image_path.name),
        "vector": vectors[0],
        "text": text,
        "source_pdf": image_path.name,  # reuse field for source tracking
        "page_start": 0,
        "page_end": 0,
        "chunk_index": 0,
        "section_hint": f"image:{image_path.stem}",
        "ingested_at": now,
        "safety_flag": "",
        "source_type": "image",
    }

    table = get_or_create_table()
    table.add([record])
    return 1


def ingest_images_dir(images_dir: Path, force: bool = False) -> dict:
    """Ingest all images from a directory.

    Returns dict with counts: {ingested, skipped, errors}.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {images_dir}")

    results = {"ingested": 0, "skipped": 0, "errors": []}

    image_files = [
        f for f in sorted(images_dir.iterdir())
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for img in image_files:
        try:
            count = ingest_image(img, force=force)
            if count:
                results["ingested"] += 1
            else:
                results["skipped"] += 1
        except Exception as e:
            results["errors"].append(f"{img.name}: {e}")

    return results
