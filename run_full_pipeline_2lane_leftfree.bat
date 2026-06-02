@echo off
setlocal

echo ============================================================
echo 1. Build network Indonesia: 2 lajur + belok kiri langsung
echo ============================================================
python build_indonesia_2lane_leftfree_network.py --input city1_original.net.xml --output city1_indonesia_2lane_leftfree.net.xml --lanes 2 --green 30 --yellow 3 --free-left-state g
if errorlevel 1 goto failed

echo ============================================================
echo 2. Generate route dinamis mobil dan motor
echo ============================================================
python generate_dynamic_routes_indonesia.py --net city1_indonesia_2lane_leftfree.net.xml --end 3600 --car-period 1.5 --motor-period 2.0 --allow-fringe
if errorlevel 1 goto failed

echo ============================================================
echo 3. Uji SUMO GUI
echo ============================================================
sumo-gui -c city1_indonesia_2lane_leftfree.sumocfg

echo ============================================================
echo 4. Jalankan studi ablasi penuh
echo ============================================================
python run_ablation_study_multi_indonesia.py --episodes 120 --sumocfg city1_indonesia_2lane_leftfree.sumocfg
if errorlevel 1 goto failed

echo Selesai.
goto end

:failed
echo Pipeline gagal. Periksa pesan error di atas.

:end
endlocal
