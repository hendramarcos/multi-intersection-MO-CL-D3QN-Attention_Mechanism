# Project Multi-Intersection MO-CL-D3QN

Project ini menyiapkan jaringan multi-simpang SUMO untuk konteks Indonesia:

- **Left-hand traffic / stir kanan**: kendaraan berjalan di sisi kiri jalan.
- **2 lajur per arah** pada setiap edge/ruas jalan.
- **Belok kiri langsung** pada setiap persimpangan bersinyal.
- **Simpang 4** menggunakan 4 fase utama.
- **Simpang 3** menggunakan 3 fase utama.
- Durasi standar: **hijau 30 detik** dan **kuning 3 detik**.
- Rute dinamis memakai **mobil** dan **motor** melalui random trips.
- Training mendukung **Multi-Objective Reward + Curriculum Learning** dan studi ablasi.

## Catatan penting

Perubahan jumlah lajur dari 1 menjadi 2 **tidak aman dilakukan dengan edit manual langsung pada file `.net.xml`**, karena file `.net.xml` adalah compiled network yang memiliki internal lanes, connection, dan TLS link index. Oleh karena itu, project ini memakai `netconvert` untuk membangun ulang jaringan secara benar.

## 1. Build network 2 lajur + belok kiri langsung

```bash
python build_indonesia_2lane_leftfree_network.py --input city1_original.net.xml --output city1_indonesia_2lane_leftfree.net.xml --lanes 2 --green 30 --yellow 3 --free-left-state g
```

Keterangan:

- `--lanes 2` membuat setiap edge menjadi 2 lajur per arah.
- `--free-left-state g` membuat belok kiri langsung sebagai permissive green/yield.
- Gunakan `--free-left-state G` bila ingin belok kiri selalu prioritas penuh.

Output:

- `city1_indonesia_2lane_leftfree.net.xml`
- `tls_2lane_leftfree_report.csv`
- `city1_indonesia_2lane_raw.net.xml`
- folder `plain/`

## 2. Generate route dinamis mobil dan motor

```bash
python generate_dynamic_routes_indonesia.py --net city1_indonesia_2lane_leftfree.net.xml --end 3600 --car-period 1.5 --motor-period 2.0 --allow-fringe
```

Output:

- `city1_dynamic_mixed.rou.xml`

## 3. Cek jaringan di SUMO GUI

```bash
sumo-gui -c city1_indonesia_2lane_leftfree.sumocfg
```

Di SUMO GUI, aktifkan tampilan lane/connection untuk memeriksa:

- jalan sudah 2 lajur;
- kendaraan berada pada sisi kiri;
- koneksi belok kiri tetap hijau/permissive pada seluruh fase.

## 4. Training model utama

```bash
python train_multiagent_mo_cl_d3qn_indonesia.py --episodes 120 --variant full_mo_cl_d3qn --sumocfg city1_indonesia_2lane_leftfree.sumocfg
```

## 5. Studi ablasi lengkap

```bash
python run_ablation_study_multi_indonesia.py --episodes 120 --sumocfg city1_indonesia_2lane_leftfree.sumocfg
```

Uji cepat:

```bash
python run_ablation_study_multi_indonesia.py --episodes 5 --max-steps 500 --sumocfg city1_indonesia_2lane_leftfree.sumocfg
```

## 6. Varian studi ablasi

| Varian | Deskripsi |
|---|---|
| `full_mo_cl_d3qn` | Model utama: D3QN + Multi-Objective Reward + Curriculum Learning |
| `ablation_no_cl` | Tanpa Curriculum Learning; reward multi-objective langsung penuh sejak awal |
| `ablation_single_objective` | Tanpa Multi-Objective Reward; reward hanya waiting time dan queue length |

## 7. Curriculum Learning

| Stage | Episode | Objective |
|---|---:|---|
| Stage 1 | 1–40 | Waiting time + queue length |
| Stage 2 | 41–80 | Waiting time + queue length + throughput |
| Stage 3 | 81–120 | Waiting time + queue length + throughput + fuel consumption |

## 8. Metrik evaluasi

- Average Waiting Time
- Average Queue Length
- Throughput
- Fuel Consumption
- Average Speed
- Average Travel Time
- Cumulative Reward
- Training Loss

## 9. Pipeline Windows

Jalankan:

```bat
run_full_pipeline_2lane_leftfree.bat
```

## 10. Verifikasi network

Setelah network dibangun, jalankan:

```bash
python verify_2lane_leftfree_network.py --net city1_indonesia_2lane_leftfree.net.xml
```

Script ini memeriksa atribut `lefthand`, distribusi jumlah lajur, serta memastikan link belok kiri pada TLS selalu berada pada status hijau/permissive.

## 11. Attention Mechanism untuk metode Multi-Agent Attention MO-CL-D3QN.

Isi program
File	Fungsi
train_multiagent_attention_mo_cl_d3qn_indonesia.py	Training model Attention MO-CL-D3QN
run_ablation_attention_indonesia.py	Studi ablasi lengkap termasuk ablasi tanpa attention
deploy_attention_model_gui_indonesia.py	Implementasi model attention ke SUMO GUI

Konsep yang ditambahkan

Setiap persimpangan sekarang tidak hanya membaca kondisi lokalnya sendiri, tetapi juga membaca kondisi persimpangan tetangga. Attention mechanism menghitung bobot kepentingan setiap tetangga, sehingga agent dapat menentukan fase lampu berdasarkan:

antrean lokal;
waiting time lokal;
kondisi persimpangan tetangga;
throughput jaringan;
fuel consumption;
reward curriculum learning.

Dengan demikian, persimpangan menjadi terkoordinasi, bukan hanya bekerja sendiri-sendiri.

## 12. Cara menjalankan training model attention

```bash
python train_multiagent_attention_mo_cl_d3qn_indonesia.py --episodes 120 --variant full_attention_mo_cl_d3qn --sumocfg city1_indonesia_2lane_leftfree.sumocfg
```

Uji cepat:
```bash
python train_multiagent_attention_mo_cl_d3qn_indonesia.py --episodes 5 --max-steps 500 --variant full_attention_mo_cl_d3qn
```

## Studi ablasi attention lengkap

```bash
python run_ablation_attention_indonesia.py --episodes 120 --sumocfg city1_indonesia_2lane_leftfree.sumocfg
```

Varian yang diuji:

| Varian                                | Tujuan                                                                |
| ------------------------------------- | --------------------------------------------------------------------- |
| `full_attention_mo_cl_d3qn`           | Model utama: Attention + Multi-Objective Reward + Curriculum Learning |
| `attention_ablation_no_cl`            | Menguji kontribusi Curriculum Learning                                |
| `attention_ablation_single_objective` | Menguji kontribusi Multi-Objective Reward                             |
| `ablation_no_attention`               | Menguji kontribusi Attention Mechanism                                |

