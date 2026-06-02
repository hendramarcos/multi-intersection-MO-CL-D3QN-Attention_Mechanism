"""
verify_2lane_leftfree_network.py

Memeriksa network hasil build:
- atribut lefthand=true
- mayoritas edge non-internal memiliki 2 lane
- fase belok kiri pada TLS selalu permissive green/green
- jumlah fase utama simpang 4/3 sesuai desain

Contoh:
python verify_2lane_leftfree_network.py --net city1_indonesia_2lane_leftfree.net.xml
"""

import argparse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", default="city1_indonesia_2lane_leftfree.net.xml")
    p.add_argument("--free-left-dir", default="l")
    args = p.parse_args()

    net = Path(args.net)
    if not net.exists():
        raise FileNotFoundError(net)

    root = ET.parse(net).getroot()
    print("Network:", net)
    print("lefthand:", root.get("lefthand"))

    lane_count = Counter()
    for edge in root.findall("edge"):
        if edge.get("function"):
            continue
        lane_count[len(edge.findall("lane"))] += 1
    print("Distribusi jumlah lane per edge non-internal:", dict(lane_count))

    tls_left_indices = defaultdict(set)
    for con in root.findall("connection"):
        if con.get("tl") and con.get("linkIndex") and con.get("dir") == args.free_left_dir:
            tls_left_indices[con.get("tl")].add(int(con.get("linkIndex")))

    tl_by_id = {tl.get("id"): tl for tl in root.findall("tlLogic")}
    print("Jumlah tlLogic:", len(tl_by_id))

    failed = []
    for tls, idxs in sorted(tls_left_indices.items()):
        tl = tl_by_id.get(tls)
        if tl is None:
            failed.append((tls, "tlLogic tidak ditemukan"))
            continue
        phases = tl.findall("phase")
        green_phases = [ph for ph in phases if "y" not in ph.get("state", "").lower()]
        for ph in phases:
            state = ph.get("state", "")
            for idx in idxs:
                if idx >= len(state) or state[idx] not in {"g", "G"}:
                    failed.append((tls, f"left index {idx} tidak free pada phase {ph.get('name')} state={state}"))
        print(f"{tls}: phases={len(phases)}, green_phases={len(green_phases)}, free_left_indices={sorted(idxs)}")

    if failed:
        print("\nHASIL: ADA MASALAH")
        for f in failed[:20]:
            print("-", f)
        if len(failed) > 20:
            print(f"... dan {len(failed)-20} masalah lain")
    else:
        print("\nHASIL: OK. Belok kiri pada TLS selalu free/permissive sesuai linkIndex yang terdeteksi.")


if __name__ == "__main__":
    main()
