# Attention Mechanism untuk Multi-Agent MO-CL-D3QN

Paket ini menambahkan program koordinasi antar-persimpangan menggunakan **Attention Mechanism**.

## Konsep Metode

Setiap persimpangan diperlakukan sebagai agent. Agent tidak hanya membaca kondisi lokal, tetapi juga membaca kondisi persimpangan tetangga. Attention mechanism menghitung bobot kepentingan setiap tetangga sehingga agent dapat menentukan fase hijau berdasarkan:

1. kondisi lokal persimpangan;
2. kondisi antrean dan waiting time tetangga;
3. throughput jaringan;
4. fuel consumption;
5. objective curriculum learning.

Dengan mekanisme ini, durasi lampu hijau menjadi adaptif. Jika agent memilih fase yang sama berulang pada beberapa decision interval, maka fase hijau diperpanjang secara dinamis.

## File

- `train_multiagent_attention_mo_cl_d3qn_indonesia.py`
- `run_ablation_attention_indonesia.py`
- `deploy_attention_model_gui_indonesia.py`

Letakkan semua file ini pada folder project yang sama dengan:

- `train_multiagent_mo_cl_d3qn_indonesia.py`
- `city1_indonesia_2lane_leftfree.sumocfg`
- `city1_indonesia_2lane_leftfree.net.xml`
- `city1_dynamic_mixed.rou.xml`

## Training Model Attention

```bash
python train_multiagent_attention_mo_cl_d3qn_indonesia.py --episodes 120 --variant full_attention_mo_cl_d3qn --sumocfg city1_indonesia_2lane_leftfree.sumocfg
```

Uji cepat:

```bash
python train_multiagent_attention_mo_cl_d3qn_indonesia.py --episodes 5 --max-steps 500 --variant full_attention_mo_cl_d3qn
```

## Studi Ablasi Lengkap

```bash
python run_ablation_attention_indonesia.py --episodes 120 --sumocfg city1_indonesia_2lane_leftfree.sumocfg
```

Uji cepat:

```bash
python run_ablation_attention_indonesia.py --episodes 5 --max-steps 500
```

## Varian Ablasi

| Varian | Tujuan |
|---|---|
| `full_attention_mo_cl_d3qn` | Model utama: Attention + Multi-Objective Reward + Curriculum Learning |
| `attention_ablation_no_cl` | Menguji kontribusi Curriculum Learning |
| `attention_ablation_single_objective` | Menguji kontribusi Multi-Objective Reward |
| `ablation_no_attention` | Menguji kontribusi Attention Mechanism |

## Implementasi Model ke SUMO GUI

```bash
python deploy_attention_model_gui_indonesia.py --model outputs_attention_indonesia/full_attention_mo_cl_d3qn/best_model.pt --gui --wait-enter --keep-open
```

## Output Penting

```text
outputs_attention_indonesia/
  full_attention_mo_cl_d3qn/
    best_model.pt
    last_model.pt
    training_metrics.csv
    attention_weights.csv
    neighbor_map.csv
  plots/
  attention_ablation_last20_summary.csv
```

## Metrik Evaluasi

- Average Waiting Time
- Average Queue Length
- Throughput
- Fuel Consumption
- Average Speed
- Average Travel Time
- Cumulative Reward
- Training Loss
- Attention Weights antar-persimpangan
