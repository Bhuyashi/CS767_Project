"""Integration test: load X-ray JPGs, run BioViL-T inference, print a report.

No MIMIC metadata required — works directly on a folder of images.

Usage (from repo root):
    # All images, single-image mode
    python code/study2/scripts/test_inference.py

    # Point to a specific folder
    python code/study2/scripts/test_inference.py --xray-dir /path/to/images

    # Temporal mode: one image is the prior, all others are conditioned on it
    python code/study2/scripts/test_inference.py \\
        --prior code/study2/xray/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg

    # Save the printed report
    python code/study2/scripts/test_inference.py --output-report results/report.txt

    # GPU
    python code/study2/scripts/test_inference.py --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging
from study2.core.model import BioVilTInferenceEngine

logger = logging.getLogger(__name__)

_DEFAULT_XRAY_DIR = Path(__file__).resolve().parents[1] / "xray"

# Prompts used in the report. Each entry is (column_key, display_label, text).
_PROMPTS: list[tuple[str, str]] = [
    ("no_acute_abnormality",   "No acute cardiopulmonary abnormality"),
    ("normal_heart_size",      "Normal heart size"),
    ("clear_lungs",            "Clear lungs bilaterally"),
    ("pleural_effusion",       "Pleural effusion"),
    ("pulmonary_edema",        "Pulmonary edema"),
    ("cardiomegaly",           "Cardiomegaly"),
    ("pneumonia",              "Pneumonia / consolidation"),
    ("atelectasis",            "Atelectasis / collapse"),
    ("pneumothorax",           "Pneumothorax"),
    ("interstitial_opacities", "Interstitial opacities / fibrosis"),
    ("hilar_enlargement",      "Hilar enlargement / lymphadenopathy"),
    ("support_devices",        "Support devices or lines present"),
]

_TEMPORAL_PROMPTS: list[tuple[str, str]] = [
    ("worsening",  "Worsening findings compared to prior"),
    ("stable",     "Stable findings compared to prior"),
    ("improving",  "Improving findings compared to prior"),
    ("no_change",  "No significant change compared to prior"),
]

_BAR_WIDTH = 28


def _bar(score: float, lo: float, hi: float) -> str:
    span = hi - lo if hi > lo else 1.0
    frac = max(0.0, min(1.0, (score - lo) / span))
    filled = round(frac * _BAR_WIDTH)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _render_section(
    title: str,
    prompts: list[tuple[str, str]],
    scores: dict[str, float],
    lo: float,
    hi: float,
) -> list[str]:
    lines = [f"  {title}", f"  {'─' * 68}"]
    ranked = sorted(
        [(key, label, scores[key]) for key, label in prompts if key in scores],
        key=lambda t: t[2],
        reverse=True,
    )
    for _, label, score in ranked:
        bar = _bar(score, lo, hi)
        lines.append(f"  {label:<42}  {score:>7.4f}  {bar}")
    return lines


def _format_report(
    image_path: Path,
    prior_path: Path | None,
    scores: dict[str, float],
) -> str:
    sep = "═" * 72
    lo = min(scores.values())
    hi = max(scores.values())

    lines: list[str] = [sep]
    lines.append(f"  IMAGE : {image_path.name}")
    if prior_path:
        lines.append(f"  PRIOR : {prior_path.name}")
        lines.append(f"  MODE  : Temporal (prior-conditioned)")
    else:
        lines.append(f"  MODE  : Single-image")
    lines.append(sep)
    lines.append(f"  {'FINDING / CONDITION':<42}  {'SCORE':>7}  {'BAR (relative)'}")
    lines.append(f"  {'─' * 42}  {'─' * 7}  {'─' * _BAR_WIDTH}")
    lines.extend(
        _render_section("Clinical Findings", _PROMPTS, scores, lo, hi)[1:]
    )

    temporal_scores = {k: scores[k] for k, _ in _TEMPORAL_PROMPTS if k in scores}
    if temporal_scores:
        lines.append("")
        lines.extend(
            _render_section("Change Assessment (temporal prompts)", _TEMPORAL_PROMPTS, scores, lo, hi)
        )

    lines.append("")
    all_clinical = [(k, s) for k, _ in _PROMPTS if (k, s := scores.get(k)) is not None]
    if all_clinical:
        top_key, top_score = max(all_clinical, key=lambda t: t[1])
        top_label = dict(_PROMPTS)[top_key]
        lines.append(f"  ▶ Highest-scoring finding : {top_label}  ({top_score:.4f})")
    if temporal_scores:
        top_t_key = max(temporal_scores, key=temporal_scores.get)
        top_t_label = dict(_TEMPORAL_PROMPTS)[top_t_key]
        lines.append(f"  ▶ Most likely change       : {top_t_label}  ({temporal_scores[top_t_key]:.4f})")
    lines.append(sep)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BioViL-T inference test: score X-ray images against clinical text prompts."
    )
    parser.add_argument(
        "--xray-dir",
        type=Path,
        default=_DEFAULT_XRAY_DIR,
        help="Directory containing .jpg / .jpeg X-ray images.",
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=None,
        help=(
            "Path to a prior X-ray image. All other images will be temporally "
            "conditioned on this prior."
        ),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to save the text report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Torch device string, e.g. "cpu", "cuda", "cuda:0".',
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_study_logging(study_name="study2_test", level=getattr(logging, args.log_level))

    if not args.xray_dir.exists():
        logger.error("xray-dir not found: %s", args.xray_dir)
        sys.exit(1)

    images = sorted(args.xray_dir.glob("*.jpg")) + sorted(args.xray_dir.glob("*.jpeg"))
    if not images:
        logger.error("No .jpg/.jpeg files found in %s", args.xray_dir)
        sys.exit(1)

    prior_path = args.prior
    if prior_path:
        if not prior_path.exists():
            logger.error("Prior image not found: %s", prior_path)
            sys.exit(1)
        images = [img for img in images if img.resolve() != prior_path.resolve()]
        logger.info("Temporal mode — prior: %s", prior_path.name)

    logger.info("%d image(s) to process in %s", len(images), args.xray_dir)

    logger.info("Loading BioViL-T model on device: %s", args.device)
    engine = BioVilTInferenceEngine(device=args.device)

    all_prompt_labels = _PROMPTS + _TEMPORAL_PROMPTS
    all_texts = [label for _, label in all_prompt_labels]
    all_keys  = [key   for key, _ in all_prompt_labels]

    header = textwrap.dedent("""\

        ╔══════════════════════════════════════════════════════════════════════╗
        ║            BioViL-T Chest X-Ray Inference Report                   ║
        ║      Model : microsoft/BiomedVLP-BioViL-T                          ║
        ║      Scores: cosine similarity (image embedding ↔ text embedding)  ║
        ╚══════════════════════════════════════════════════════════════════════╝
    """)

    report_parts: list[str] = [header]

    for i, image_path in enumerate(images, start=1):
        logger.info("Processing image %d/%d: %s", i, len(images), image_path.name)
        try:
            similarities = engine.get_similarities(
                image_path=image_path,
                texts=all_texts,
                prior_path=prior_path,
            )
        except Exception as exc:
            logger.error("Inference failed for %s: %s", image_path.name, exc)
            continue

        scores = dict(zip(all_keys, similarities))
        section = _format_report(image_path, prior_path, scores)
        report_parts.append(section)
        report_parts.append("")

    full_report = "\n".join(report_parts)
    print(full_report)

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(full_report, encoding="utf-8")
        logger.info("Report saved: %s", args.output_report)


if __name__ == "__main__":
    run()
