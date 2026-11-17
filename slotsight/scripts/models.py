"""List, install and choose SlotSight's detection models from a terminal.

The dashboard's Configure dialog does the same; both use the registry in
slotsight/detectors.py and download into slotsight/models/.

Usage (from the repository root):
    python slotsight/scripts/models.py list
    python slotsight/scripts/models.py install huggingface/rtdetr-v2-r18
    python slotsight/scripts/models.py use huggingface/rtdetr-v2-r18
    python slotsight/scripts/models.py install --all      # prepare an offline machine

A provider's Python package is only needed to run its models, not to download
them: `pip install transformers` (Hugging Face) or `pip install rfdetr` (RF-DETR).
Ultralytics models need `pip install ultralytics`; the OpenCV model needs nothing extra.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as slotsight  # noqa: E402
import detectors  # noqa: E402


def print_table() -> None:
    active = slotsight.configured_model_id()
    rows = slotsight.models.catalogue(active)
    print(f"{'':2}{'id':28}{'provider':28}{'licence':12}{'size':>9}  status")
    for row in rows:
        if not row["installed"]:
            status = "not installed"
        elif not row["package_available"]:
            status = f"installed; needs: pip install {row['pip']}"
        else:
            status = "installed"
        marker = "* " if row["active"] else "  "
        size = f"{row['size_mb']:.0f} MB" if row["size_mb"] else "local"
        print(f"{marker}{row['id']:28}{row['provider']:28}{row['licence']:12}{size:>9}  {status}")
    source = " (from SLOTSIGHT_MODEL)" if slotsight.model_overridden() else ""
    print(f"\n* = active{source}. Models are saved in {slotsight.MODELS_DIR}")


def install(model_ids: list[str]) -> int:
    failed = 0
    for model_id in model_ids:
        spec = detectors.spec_for(model_id, slotsight.MODELS_DIR)
        if detectors.is_installed(spec, slotsight.MODELS_DIR):
            print(f"{spec.name}: already installed")
            continue
        print(f"{spec.name}: downloading {spec.size_mb:.0f} MB ({spec.licence})")
        last = [-1]

        def progress(done: int, total: int) -> None:
            percent = int(done * 100 / total) if total else 0
            if percent // 5 != last[0]:
                last[0] = percent // 5
                print(f"\r  {percent:3d}%", end="", flush=True)

        try:
            slotsight.models.install(model_id, progress)
            print(f"\r  done, saved to {slotsight.MODELS_DIR}")
        except detectors.DetectorError as error:
            print(f"\r  failed: {error}", file=sys.stderr)
            failed += 1
            continue
        if not detectors.package_available(spec.provider):
            print(f"  To run it: pip install {detectors.PROVIDERS[spec.provider].pip}")
    return 1 if failed else 0


def use(model_id: str) -> int:
    model_id = detectors.normalize_model_id(model_id)
    try:
        spec = detectors.spec_for(model_id, slotsight.MODELS_DIR)
    except detectors.DetectorError as error:
        print(error, file=sys.stderr)
        return 1
    if not detectors.is_installed(spec, slotsight.MODELS_DIR):
        print(f"{spec.name} is not installed. Run: python slotsight/scripts/models.py install {model_id}", file=sys.stderr)
        return 1
    try:
        slotsight.save_settings(model=model_id)
    except slotsight.ConfigReadError:
        print(f"{slotsight.SETTINGS_PATH} can't be read right now (open elsewhere, or a typo?). Nothing was saved.", file=sys.stderr)
        return 1
    print(f"Active model: {spec.name} ({model_id}). A running dashboard picks it up on its next frame.")
    if not detectors.package_available(spec.provider):
        print(f"It needs its Python package first: pip install {detectors.PROVIDERS[spec.provider].pip}")
    if slotsight.model_overridden():
        print("Note: SLOTSIGHT_MODEL is set and takes precedence over this choice.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show every model, its licence, size and whether it is installed")
    install_parser = commands.add_parser("install", help="download one or more models into slotsight/models/")
    install_parser.add_argument("ids", nargs="*", help="model ids from `list`")
    install_parser.add_argument("--all", action="store_true", help="install every model (about 1 GB)")
    use_parser = commands.add_parser("use", help="make an installed model the active one (writes settings.txt)")
    use_parser.add_argument("id")
    args = parser.parse_args()

    if args.command == "list":
        print_table()
        return 0
    if args.command == "install":
        ids = list(detectors.MODELS) if args.all else args.ids
        if not ids:
            parser.error("give at least one model id, or --all")
        unknown = [model_id for model_id in ids if model_id not in detectors.MODELS]
        if unknown:
            print(f"Unknown model(s): {', '.join(unknown)}. See: python slotsight/scripts/models.py list", file=sys.stderr)
            return 1
        return install(ids)
    return use(args.id)


if __name__ == "__main__":
    raise SystemExit(main())
