from pathlib import Path
import argparse

import pydicom


def extract_dicom_metadata_to_txt(dcm_path: str) -> Path:
    """Read a DICOM and save all header metadata as a TXT file nearby."""
    dcm_file = Path(dcm_path)
    if not dcm_file.exists():
        raise FileNotFoundError(f"DICOM file not found: {dcm_file}")
    if dcm_file.suffix.lower() != ".dcm":
        raise ValueError(f"Expected a .dcm file, got: {dcm_file}")

    ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
    txt_path = dcm_file.with_suffix(".txt")

    with txt_path.open("w", encoding="utf-8") as out:
        out.write(f"DICOM File: {dcm_file}\n")
        out.write("=" * 80 + "\n")
        out.write(str(ds))
        out.write("\n")

    return txt_path


def process_directory_recursively(start_dir: str) -> tuple[int, int]:
    """Find all .dcm files under start_dir and write metadata .txt beside each."""
    root = Path(start_dir)
    if not root.is_absolute():
        cwd_candidate = Path.cwd() / root
        script_candidate = Path(__file__).resolve().parent / root
        root = cwd_candidate if cwd_candidate.exists() else script_candidate
    root = root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root}")

    dcm_files = list(root.rglob("*.dcm"))
    success_count = 0
    failure_count = 0

    for dcm_file in dcm_files:
        try:
            output_path = extract_dicom_metadata_to_txt(str(dcm_file))
            success_count += 1
            print(f"[OK] {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[ERROR] {dcm_file}: {exc}")

    return success_count, failure_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively extract metadata from all DICOM (.dcm) files under a directory."
    )
    parser.add_argument(
        "start_dir",
        nargs="?",
        default=r"../data/MIMIC-CXR/files",
        help="Starting directory to search recursively for .dcm files.",
    )

    args = parser.parse_args()
    success_count, failure_count = process_directory_recursively(args.start_dir)
    print(
        f"Done. Metadata files written: {success_count}. "
        f"Failed files: {failure_count}."
    )


if __name__ == "__main__":
    main()
