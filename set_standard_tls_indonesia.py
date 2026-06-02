"""
set_standard_tls_indonesia.py

Mengubah pengaturan traffic light seluruh persimpangan pada network SUMO ke konteks:
- Indonesia / left-hand traffic / setir kanan
- Simpang 4: 4 fasa pelayanan hijau + 4 transisi kuning
- Simpang 3: 3 fasa pelayanan hijau + 3 transisi kuning

Standar durasi default:
- Green duration  : 30 detik
- Yellow duration : 3 detik

Catatan:
Dalam file SUMO, fase kuning ditulis sebagai phase tambahan. Jadi simpang 4 akan memiliki
8 phase di tlLogic: 4 green service phases dan 4 yellow transition phases. Simpang 3 akan
memiliki 6 phase: 3 green service phases dan 3 yellow transition phases.

Cara pakai:
python set_standard_tls_indonesia.py --input city1_indonesia.net.xml --output city1_indonesia_tls_standard.net.xml

Jika ingin durasi lain:
python set_standard_tls_indonesia.py --input city1_indonesia.net.xml --output city1_indonesia_tls_standard.net.xml --green 35 --yellow 3
"""

import argparse
import copy
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def strip_ns(tag: str) -> str:
    return tag.split('}', 1)[-1] if '}' in tag else tag


def indent(elem, level=0):
    """Pretty print XML for Python versions without ET.indent compatibility assumptions."""
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def lane_to_edge(lane_id: str) -> str:
    """Mengubah id lane SUMO menjadi id edge. Contoh gneE12_0 -> gneE12."""
    return re.sub(r"_\d+$", "", lane_id)


def get_tls_junction_approaches(root):
    """
    Membaca incLanes pada junction traffic_light untuk menentukan jumlah lengan pendekat.
    Output: {tls_id: [edge_id_1, edge_id_2, ...]}
    """
    approaches = {}
    for j in root.findall("junction"):
        if j.get("type") == "traffic_light":
            tls_id = j.get("id")
            inc_lanes = j.get("incLanes", "").split()
            ordered_edges = []
            for lane in inc_lanes:
                edge = lane_to_edge(lane)
                if edge not in ordered_edges:
                    ordered_edges.append(edge)
            if len(ordered_edges) >= 3:
                approaches[tls_id] = ordered_edges
    return approaches


def get_tls_connections(root):
    """
    Membaca semua connection yang dikontrol traffic light.
    Output: {tls_id: [(linkIndex, from_edge), ...]}
    """
    conns = defaultdict(list)
    for c in root.findall("connection"):
        tls = c.get("tl")
        idx = c.get("linkIndex")
        frm = c.get("from")
        if tls is not None and idx is not None and frm is not None:
            conns[tls].append((int(idx), frm))
    return conns


def make_green_state(num_links, controlled_indices):
    state = ["r"] * num_links
    for idx in controlled_indices:
        if 0 <= idx < num_links:
            state[idx] = "G"
    return "".join(state)


def make_yellow_state(green_state):
    return "".join("y" if ch in ("G", "g") else "r" for ch in green_state)


def build_tllogic(tls_id, approach_edges, tls_connections, green_duration, yellow_duration):
    """
    Membuat tlLogic standar.
    Setiap approach mendapat 1 green service phase.
    """
    if not tls_connections:
        return None, []

    max_link_idx = max(idx for idx, _ in tls_connections)
    num_links = max_link_idx + 1
    by_from = defaultdict(list)
    for idx, frm in tls_connections:
        by_from[frm].append(idx)

    # Ambil hanya 3 atau 4 approach. Jika >4, ambil 4 terbesar/terurut dari junction.
    # Jika 3, menjadi simpang 3. Jika 4 atau lebih, menjadi simpang 4.
    ordered = [e for e in approach_edges if e in by_from]
    if len(ordered) >= 4:
        selected = ordered[:4]
        intersection_type = "simpang_4"
    elif len(ordered) == 3:
        selected = ordered[:3]
        intersection_type = "simpang_3"
    else:
        # Fallback dari connection jika incLanes tidak lengkap
        fallback = list(by_from.keys())
        if len(fallback) >= 4:
            selected = fallback[:4]
            intersection_type = "simpang_4"
        elif len(fallback) == 3:
            selected = fallback[:3]
            intersection_type = "simpang_3"
        else:
            return None, []

    tl = ET.Element("tlLogic", {
        "id": tls_id,
        "type": "static",
        "programID": "indonesia_standard",
        "offset": "0",
    })

    phase_rows = []
    for phase_no, edge in enumerate(selected, start=1):
        green_state = make_green_state(num_links, by_from[edge])
        yellow_state = make_yellow_state(green_state)
        tl.append(ET.Element("phase", {
            "duration": str(green_duration),
            "state": green_state,
            "name": f"P{phase_no}_{edge}_green",
        }))
        tl.append(ET.Element("phase", {
            "duration": str(yellow_duration),
            "state": yellow_state,
            "name": f"P{phase_no}_{edge}_yellow",
        }))
        phase_rows.append({
            "tls_id": tls_id,
            "intersection_type": intersection_type,
            "phase_no": phase_no,
            "approach_edge": edge,
            "green_duration": green_duration,
            "yellow_duration": yellow_duration,
            "green_state": green_state,
            "yellow_state": yellow_state,
        })

    return tl, phase_rows


def remove_existing_tllogics(root):
    for tl in list(root.findall("tlLogic")):
        root.remove(tl)


def insert_tllogics_after_locations(root, tllogics):
    """
    SUMO net biasanya menaruh tlLogic setelah location. Fungsi ini menempatkan kembali tlLogic
    setelah elemen location agar struktur tetap rapi.
    """
    children = list(root)
    insert_idx = 0
    for i, child in enumerate(children):
        if strip_ns(child.tag) == "location":
            insert_idx = i + 1
            break
    for offset, tl in enumerate(tllogics):
        root.insert(insert_idx + offset, tl)


def write_phase_report(report_rows, report_path):
    import csv
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tls_id", "intersection_type", "phase_no", "approach_edge", "green_duration", "yellow_duration", "green_state", "yellow_state"]
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input SUMO net.xml")
    parser.add_argument("--output", required=True, help="Output SUMO net.xml dengan TLS standar")
    parser.add_argument("--green", type=int, default=30, help="Durasi hijau standar per fase, detik")
    parser.add_argument("--yellow", type=int, default=3, help="Durasi kuning standar per fase, detik")
    parser.add_argument("--report", default="tls_phase_report.csv", help="Output laporan fase CSV")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    tree = ET.parse(in_path)
    root = tree.getroot()

    # Konteks Indonesia: left-hand traffic, kendaraan berjalan di lajur kiri.
    root.set("lefthand", "true")

    approaches = get_tls_junction_approaches(root)
    conns = get_tls_connections(root)

    new_tllogics = []
    report_rows = []
    for tls_id in sorted(conns.keys()):
        tl, rows = build_tllogic(
            tls_id=tls_id,
            approach_edges=approaches.get(tls_id, []),
            tls_connections=conns[tls_id],
            green_duration=args.green,
            yellow_duration=args.yellow,
        )
        if tl is not None:
            new_tllogics.append(tl)
            report_rows.extend(rows)

    if not new_tllogics:
        raise RuntimeError("Tidak ada traffic light yang berhasil dibuat. Periksa network dan connection tl/linkIndex.")

    remove_existing_tllogics(root)
    insert_tllogics_after_locations(root, new_tllogics)

    try:
        ET.indent(tree, space="    ")
    except Exception:
        indent(root)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    write_phase_report(report_rows, args.report)

    count_3 = sum(1 for tl in new_tllogics if len(tl.findall("phase")) == 6)
    count_4 = sum(1 for tl in new_tllogics if len(tl.findall("phase")) == 8)

    print("Selesai mengubah traffic light.")
    print(f"Input     : {in_path}")
    print(f"Output    : {out_path}")
    print(f"Report    : {args.report}")
    print(f"TLS total : {len(new_tllogics)}")
    print(f"Simpang 4 : {count_4} traffic light, masing-masing 4 fasa hijau + 4 kuning")
    print(f"Simpang 3 : {count_3} traffic light, masing-masing 3 fasa hijau + 3 kuning")
    print(f"Durasi    : green={args.green}s, yellow={args.yellow}s")


if __name__ == "__main__":
    main()
