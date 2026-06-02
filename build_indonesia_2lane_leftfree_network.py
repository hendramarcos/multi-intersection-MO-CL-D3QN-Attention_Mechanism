"""
build_indonesia_2lane_leftfree_network.py

Membangun ulang network SUMO multi-simpang untuk konteks Indonesia:
1. Left-hand traffic / stir kanan: kendaraan berjalan di lajur kiri.
2. Semua ruas jalan dibuat 2 lajur per arah.
3. Fase belok kiri pada setiap traffic light dibuat belok kiri langsung/permissive.
4. Simpang 4 menggunakan 4 fase utama; simpang 3 menggunakan 3 fase utama.
5. Durasi standar: hijau 30 detik, kuning 3 detik.

Mengapa harus rebuild dengan netconvert?
File .net.xml adalah compiled network. Mengubah jumlah lajur secara manual langsung
pada .net.xml tidak aman karena harus menyesuaikan internal lanes, junction, dan connection.
Script ini menggunakan netconvert untuk mengekstrak plain network, mengubah numLanes=2,
kemudian membangun ulang network dan menulis ulang program traffic light.

Contoh:
python build_indonesia_2lane_leftfree_network.py --input city1_original.net.xml --output city1_indonesia_2lane_leftfree.net.xml

Jika ingin mempertahankan connection asli, tambahkan --use-original-connections.
Namun untuk perubahan 2 lajur, mode default tanpa connection asli lebih disarankan agar
netconvert membangun ulang koneksi lajur secara konsisten.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def find_netconvert() -> str:
    exe = shutil.which("netconvert")
    if exe:
        return exe
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        cand = Path(sumo_home) / "bin" / ("netconvert.exe" if os.name == "nt" else "netconvert")
        if cand.exists():
            return str(cand)
    raise RuntimeError(
        "netconvert tidak ditemukan. Pastikan SUMO sudah terinstal, SUMO_HOME sudah diset, "
        "atau folder SUMO/bin sudah masuk PATH."
    )


def run(cmd):
    print("Menjalankan:", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError("Perintah gagal: " + " ".join(map(str, cmd)))


def force_two_lanes(edge_file: Path, lanes: int = 2, allow: str = "passenger motorcycle"):
    tree = ET.parse(edge_file)
    root = tree.getroot()
    changed = 0
    for edge in root.findall("edge"):
        # plain edge biasanya tidak memuat internal edge, tetapi skip jika ada function khusus
        if edge.get("function"):
            continue
        edge.set("numLanes", str(lanes))
        edge.set("allow", allow)
        changed += 1
    ET.indent(root, space="    ")
    tree.write(edge_file, encoding="UTF-8", xml_declaration=True)
    print(f"Jumlah edge yang diubah menjadi {lanes} lajur: {changed}")


def insert_tllogics_after_location(root, tl_logics):
    # hapus tlLogic lama
    for old in list(root.findall("tlLogic")):
        root.remove(old)

    children = list(root)
    insert_idx = 0
    for idx, child in enumerate(children):
        if child.tag == "location":
            insert_idx = idx + 1
            break
    for i, tl in enumerate(tl_logics):
        root.insert(insert_idx + i, tl)


def unique_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def edge_from_lane_id(lane_id: str) -> str:
    # lane id biasanya edge_0 atau edge_1. Edge id dapat mengandung underscore,
    # sehingga pisahkan dari underscore terakhir.
    if "_" not in lane_id:
        return lane_id
    return lane_id.rsplit("_", 1)[0]


def build_left_turn_free_tls(net_file: Path, output_file: Path, green: int, yellow: int,
                             free_dir: str, free_state: str, report_file: Path):
    tree = ET.parse(net_file)
    root = tree.getroot()
    root.set("lefthand", "true")

    junctions = {j.get("id"): j for j in root.findall("junction")}
    tls_connections = defaultdict(list)

    for con in root.findall("connection"):
        tl = con.get("tl")
        link_index = con.get("linkIndex")
        if tl is not None and link_index is not None:
            try:
                int(link_index)
            except Exception:
                continue
            tls_connections[tl].append(con)

    tl_logics = []
    report_rows = []

    for tl_id, cons in sorted(tls_connections.items()):
        max_idx = max(int(c.get("linkIndex")) for c in cons)
        state_len = max_idx + 1
        left_indices = sorted({int(c.get("linkIndex")) for c in cons if c.get("dir") == free_dir})

        # Connection non-belok-kiri dikelompokkan berdasarkan approach/incoming edge.
        by_from = defaultdict(list)
        for c in cons:
            idx = int(c.get("linkIndex"))
            if idx in left_indices:
                continue
            by_from[c.get("from")].append(idx)

        # Urutan approach mengikuti incLanes junction jika tersedia.
        junction = junctions.get(tl_id)
        approach_order = []
        if junction is not None and junction.get("incLanes"):
            approach_order = unique_order(edge_from_lane_id(x) for x in junction.get("incLanes").split())
            approach_order = [e for e in approach_order if e in by_from]
        for e in sorted(by_from.keys()):
            if e not in approach_order:
                approach_order.append(e)

        approach_count = len(approach_order)
        if approach_count >= 4:
            phase_count = 4
            junction_type = "simpang_4"
        elif approach_count == 3:
            phase_count = 3
            junction_type = "simpang_3"
        else:
            phase_count = max(1, approach_count)
            junction_type = f"simpang_{approach_count}"

        groups = []
        for phase_i in range(phase_count):
            # Jika approach lebih dari jumlah phase, gabungkan secara round-robin.
            edge_group = approach_order[phase_i::phase_count]
            idxs = []
            for e in edge_group:
                idxs.extend(by_from[e])
            groups.append((edge_group, sorted(set(idxs))))

        tl_logic = ET.Element("tlLogic", {
            "id": tl_id,
            "type": "static",
            "programID": "indonesia_2lane_leftfree",
            "offset": "0",
        })

        for phase_no, (edge_group, served_indices) in enumerate(groups, start=1):
            green_state = ["r"] * state_len
            for idx in left_indices:
                green_state[idx] = free_state
            for idx in served_indices:
                green_state[idx] = "G"
            ET.SubElement(tl_logic, "phase", {
                "duration": str(green),
                "state": "".join(green_state),
                "name": f"P{phase_no}_{junction_type}_green_" + "_".join(edge_group),
            })

            yellow_state = ["r"] * state_len
            for idx in left_indices:
                yellow_state[idx] = free_state
            for idx in served_indices:
                yellow_state[idx] = "y"
            ET.SubElement(tl_logic, "phase", {
                "duration": str(yellow),
                "state": "".join(yellow_state),
                "name": f"P{phase_no}_{junction_type}_yellow_" + "_".join(edge_group),
            })

        tl_logics.append(tl_logic)
        report_rows.append({
            "tls_id": tl_id,
            "junction_type": junction_type,
            "approach_count": approach_count,
            "phase_count": phase_count,
            "green_duration": green,
            "yellow_duration": yellow,
            "free_left_dir": free_dir,
            "free_left_state": free_state,
            "left_turn_link_indices": " ".join(map(str, left_indices)),
            "approaches": " ".join(approach_order),
            "link_count": state_len,
        })

    insert_tllogics_after_location(root, tl_logics)
    ET.indent(root, space="    ")
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)

    with open(report_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "tls_id", "junction_type", "approach_count", "phase_count",
            "green_duration", "yellow_duration", "free_left_dir", "free_left_state",
            "left_turn_link_indices", "approaches", "link_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Network final ditulis ke: {output_file}")
    print(f"Laporan TLS ditulis ke: {report_file}")
    print(f"Jumlah traffic light diubah: {len(report_rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="city1_original.net.xml", help="Input compiled .net.xml")
    parser.add_argument("--output", default="city1_indonesia_2lane_leftfree.net.xml", help="Output final .net.xml")
    parser.add_argument("--plain-prefix", default="plain/city1_plain", help="Prefix plain network sementara")
    parser.add_argument("--lanes", type=int, default=2, help="Jumlah lajur per edge/arah")
    parser.add_argument("--green", type=int, default=30, help="Durasi hijau standar")
    parser.add_argument("--yellow", type=int, default=3, help="Durasi kuning standar")
    parser.add_argument("--free-left-dir", default="l", help="Direction code SUMO untuk belok kiri")
    parser.add_argument("--free-left-state", default="g", choices=["g", "G"], help="g=permissive/yield, G=priority green")
    parser.add_argument("--report", default="tls_2lane_leftfree_report.csv")
    parser.add_argument("--intermediate", default="city1_indonesia_2lane_raw.net.xml")
    parser.add_argument("--use-original-connections", action="store_true", help="Gunakan connection asli saat rebuild. Tidak direkomendasikan untuk perubahan 2 lajur.")
    args = parser.parse_args()

    input_net = Path(args.input)
    if not input_net.exists():
        raise FileNotFoundError(input_net)

    netconvert = find_netconvert()
    plain_prefix = Path(args.plain_prefix)
    plain_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract plain files dari compiled network.
    run([netconvert, "-s", str(input_net), "--plain-output-prefix", str(plain_prefix)])

    nod = Path(str(plain_prefix) + ".nod.xml")
    edg = Path(str(plain_prefix) + ".edg.xml")
    con = Path(str(plain_prefix) + ".con.xml")
    typ = Path(str(plain_prefix) + ".typ.xml")

    if not edg.exists() or not nod.exists():
        raise FileNotFoundError("Plain node/edge file tidak terbentuk dari netconvert.")

    # 2. Ubah edge menjadi 2 lajur dan izinkan mobil+motor.
    force_two_lanes(edg, lanes=args.lanes)

    # 3. Rebuild network dengan konteks left-hand traffic.
    cmd = [
        netconvert,
        "-n", str(nod),
        "-e", str(edg),
        "--lefthand", "true",
        "--tls.default-type", "static",
        "--no-turnarounds", "true",
        "--junctions.corner-detail", "5",
        "-o", str(args.intermediate),
    ]
    if args.use_original_connections and con.exists():
        cmd += ["-x", str(con)]
    if typ.exists():
        cmd += ["-t", str(typ)]
    run(cmd)

    # 4. Tulis ulang traffic light: 4/3 fase dan belok kiri langsung.
    build_left_turn_free_tls(
        net_file=Path(args.intermediate),
        output_file=Path(args.output),
        green=args.green,
        yellow=args.yellow,
        free_dir=args.free_left_dir,
        free_state=args.free_left_state,
        report_file=Path(args.report),
    )


if __name__ == "__main__":
    main()
