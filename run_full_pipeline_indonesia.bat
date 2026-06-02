@echo off
REM Pipeline lengkap Indonesia: lajur kiri/stir kanan, TLS standar, random trips mobil+motor, ablation study.
python convert_to_indonesia_network.py --input city1_original.net.xml --output city1_indonesia.net.xml
python set_standard_tls_indonesia.py --input city1_indonesia.net.xml --output city1_indonesia_tls_standard.net.xml --green 30 --yellow 3 --report tls_phase_report.csv
python generate_dynamic_routes_indonesia.py --net city1_indonesia_tls_standard.net.xml --end 3600 --car-period 1.5 --motor-period 2.0 --allow-fringe
sumo-gui -c city1_indonesia_tls_standard.sumocfg
python run_ablation_study_multi_indonesia.py --episodes 120 --sumocfg city1_indonesia_tls_standard.sumocfg
python plot_results_multi_indonesia.py --input-dir outputs_multi_indonesia_ablation
pause
