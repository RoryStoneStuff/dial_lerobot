"""Fetch the exact upstream mesh files and reproduce our local joint limits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def fetch_assets(destination):
    manifest = json.loads((ROOT / "assets/robot_assets.json").read_text())
    destination = Path(destination).resolve()
    xml = destination / manifest["source_xml"]

    def fetch(entry):
        target = destination / entry["path"]
        actual = digest(target)
        if actual == entry["sha256"] or (
            target == xml and actual == manifest["patched_xml_sha256"]
        ):
            return
        url = (f"https://raw.githubusercontent.com/{manifest['repository']}/"
               f"{manifest['revision']}/{quote(entry['path'])}")
        with urlopen(Request(url, headers={"User-Agent": "dial-lerobot-assets"}), timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {entry['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, manifest["files"]))
    if digest(xml) != manifest["patched_xml_sha256"]:
        subprocess.run(["git", "apply", "--no-index", str(ROOT / "assets/robot_source.patch")],
                       cwd=destination, check=True)
    if digest(xml) != manifest["patched_xml_sha256"]:
        raise ValueError("Patched robot XML does not match the model used locally")
    print(f"Verified {len(manifest['files'])} robot files at {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "external/lerobot-legged-zoo")
    fetch_assets(parser.parse_args().destination)
