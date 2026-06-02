"""
convert_to_indonesia_network.py

Mengonversi/menandai jaringan SUMO menjadi konteks Indonesia:
- left-hand traffic: kendaraan berjalan di sisi kiri jalan
- right-hand steering: konteks pengemudi/stir di kanan

Catatan:
1. Jika netconvert tersedia, script mencoba membangun ulang network dengan opsi --lefthand.
2. Jika netconvert tidak tersedia atau gagal, script tetap menambahkan atribut lefthand="true"
   pada root <net>. Ini sudah cukup agar SUMO mengenali network sebagai left-hand traffic.

Cara pakai:
python convert_to_indonesia_network.py --input city1_original.net.xml --output city1_indonesia.net.xml
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def set_lefthand_attr(input_file: Path, output_file: Path):
    text = input_file.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<net\b([^>]*)>", text)
    if not m:
        raise RuntimeError("Root <net> tidak ditemukan pada file network.")
    tag = m.group(0)
    if "lefthand=" in tag:
        new_tag = re.sub(r'lefthand="[^"]*"', 'lefthand="true"', tag)
    else:
        new_tag = tag[:-1] + ' lefthand="true">'
    text = text[:m.start()] + new_tag + text[m.end():]
    comment = "<!-- Indonesia context: left-hand traffic and right-hand steering. -->\n"
    if "Indonesia context" not in text:
        text = text.replace("\n\n<!-- generated", "\n\n" + comment + "\n<!-- generated", 1)
    output_file.write_text(text, encoding="utf-8")


def try_netconvert(input_file: Path, output_file: Path) -> bool:
    netconvert = shutil.which("netconvert")
    if not netconvert:
        return False
    cmd = [netconvert, "-s", str(input_file), "--lefthand", "true", "-o", str(output_file)]
    print("Menjalankan:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and output_file.exists():
        # Pastikan atribut tetap ada
        set_lefthand_attr(output_file, output_file)
        return True
    print("netconvert gagal, fallback ke patch XML.")
    if proc.stderr:
        print(proc.stderr)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="city1_original.net.xml")
    parser.add_argument("--output", default="city1_indonesia.net.xml")
    parser.add_argument("--no-netconvert", action="store_true")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    if not input_file.exists():
        raise FileNotFoundError(input_file)

    ok = False
    if not args.no_netconvert:
        ok = try_netconvert(input_file, output_file)
    if not ok:
        set_lefthand_attr(input_file, output_file)

    print(f"Selesai: {output_file}")
    print("Network ditandai sebagai lefthand=true untuk konteks Indonesia.")


if __name__ == "__main__":
    main()
