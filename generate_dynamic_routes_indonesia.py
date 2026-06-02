"""
generate_dynamic_routes_indonesia.py

Membuat rute dinamis untuk jaringan multi-simpang dengan random trips.
Kendaraan yang dibangkitkan:
- mobil/passenger car
- motor/motorcycle

Cara pakai:
python generate_dynamic_routes_indonesia.py --net city1_indonesia_2lane_leftfree.net.xml --end 3600 --car-period 1.5 --motor-period 2.0 --allow-fringe

Output utama:
- city1_dynamic_mixed.rou.xml
"""

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def run(cmd):
    print("Menjalankan:", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Perintah gagal: {' '.join(map(str, cmd))}")


def generate_one(random_trips, net, additional, begin, end, period, prefix, vclass, vtype, trip_file, route_file, seed, allow_fringe):
    attrs = f'type="{vtype}" departLane="best" departSpeed="max" departPos="random"'
    cmd = [
        sys.executable, str(random_trips),
        "-n", str(net),
        "-o", str(trip_file),
        "-r", str(route_file),
        "-b", str(begin),
        "-e", str(end),
        "-p", str(period),
        "--seed", str(seed),
        "--prefix", prefix,
        "-c", vclass,
        "--trip-attributes", attrs,
        "--additional-files", str(additional),
        "--min-distance", "80",
        "--fringe-factor", "5",
        "--validate",
    ]
    if allow_fringe:
        cmd.append("--allow-fringe")
    run(cmd)


def merge_routes(files, output):
    vehicles = []
    for file in files:
        root = ET.parse(file).getroot()
        for elem in root:
            if elem.tag in {"vehicle", "flow", "trip"}:
                vehicles.append(elem)
    def depart_value(elem):
        if elem.tag == "flow":
            return float(elem.get("begin", 0))
        return float(elem.get("depart", 0))
    vehicles.sort(key=depart_value)

    out_root = ET.Element("routes")
    comment = ET.Comment("Mixed dynamic routes: cars and motorcycles for Indonesia left-hand traffic scenario")
    out_root.append(comment)
    for v in vehicles:
        out_root.append(v)
    ET.indent(out_root, space="    ")
    ET.ElementTree(out_root).write(output, encoding="UTF-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="city1_indonesia_2lane_leftfree.net.xml")
    parser.add_argument("--additional", default="vtypes_indonesia.add.xml")
    parser.add_argument("--random-trips", default="randomTrips.py")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=3600)
    parser.add_argument("--car-period", type=float, default=1.5)
    parser.add_argument("--motor-period", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-fringe", action="store_true")
    parser.add_argument("--output", default="city1_dynamic_mixed.rou.xml")
    args = parser.parse_args()

    net = Path(args.net)
    additional = Path(args.additional)
    random_trips = Path(args.random_trips)
    if not net.exists():
        raise FileNotFoundError(net)
    if not additional.exists():
        raise FileNotFoundError(additional)
    if not random_trips.exists():
        raise FileNotFoundError(random_trips)

    generate_one(random_trips, net, additional, args.begin, args.end, args.car_period,
                 "car_", "passenger", "car", "trips_car.trips.xml", "routes_car.rou.xml",
                 args.seed, args.allow_fringe)
    generate_one(random_trips, net, additional, args.begin, args.end, args.motor_period,
                 "motor_", "motorcycle", "motorcycle", "trips_motor.trips.xml", "routes_motor.rou.xml",
                 args.seed + 999, args.allow_fringe)
    merge_routes(["routes_car.rou.xml", "routes_motor.rou.xml"], args.output)
    print(f"Selesai membuat mixed route: {args.output}")


if __name__ == "__main__":
    main()
