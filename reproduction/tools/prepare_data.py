"""Prepare only the four author files needed for Achilles position decoding."""

import argparse
import logging
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from repro_io import ROOT, read_manifest, verify_file  # noqa: E402

logger = logging.getLogger(__name__)


def prepare(source_directory=None):
    destination = ROOT / "data/rat"
    destination.mkdir(parents=True, exist_ok=True)
    for name, metadata in read_manifest().items():
        assert "url" in metadata
        target = destination / name
        if target.exists():
            verify_file(target, metadata)
            logger.info("Verified existing file: %s", name)
            continue
        partial = target.with_suffix(target.suffix + ".part")
        if source_directory is not None:
            source = Path(source_directory) / name
            verify_file(source, metadata)
            shutil.copyfile(source, partial)
        else:
            logger.info("Downloading %s", metadata["url"])
            with urllib.request.urlopen(metadata["url"], timeout=60) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
        verify_file(partial, metadata)
        partial.replace(target)
        logger.info("Prepared and verified: %s", name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path)
    prepare(parser.parse_args().source_directory)
