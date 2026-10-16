"""Fetch the two official UNSW-NB15 partition files and verify them against SHA256SUMS.

The files are the *official* training/testing partitions published by UNSW Canberra
(https://research.unsw.edu.au/projects/unsw-nb15-dataset). The university's server is the
primary source; if it is reachable from your network, prefer it and just run `sha256sum -c`.
The mirrors below are GitHub copies that were byte-identical to each other when this project was built.
"""
import hashlib, sys, urllib.request
from pathlib import Path
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
GH = "https://raw.githubusercontent.com"
SOURCES = {
    "UNSW_NB15_training-set.csv": [f"{GH}/shahariar-shibli/Feature-Selection-using-Metaheuristic-Approaches-for-Intrusion-Detection-System/main/datasets/UNSW_NB15_training-set.csv",
                                   f"{GH}/Ausommet/PacketGuard/main/Test-Files/UNSW_NB15_training-set.csv"],
    "UNSW_NB15_testing-set.csv":  [f"{GH}/shahariar-shibli/Feature-Selection-using-Metaheuristic-Approaches-for-Intrusion-Detection-System/main/datasets/UNSW_NB15_testing-set.csv"],
}
def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
expected = dict(reversed(l.split()) for l in (RAW / "SHA256SUMS").read_text().splitlines() if l.strip())
for name, urls in SOURCES.items():
    dest = RAW / name
    if dest.exists() and sha(dest) == expected[name]:
        print("ok (already present)", name); continue
    for u in urls:
        try:
            print("downloading", u); urllib.request.urlretrieve(u, dest)
            if sha(dest) == expected[name]: print("ok", name); break
            print("  checksum mismatch, trying next source")
        except Exception as e:
            print("  failed:", e)
    else:
        sys.exit(f"could not obtain a verified copy of {name}")
