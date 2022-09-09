#ulimit -Sn 20000 # file handle issues
#ulimit -n 20000

rm ../exe/*.*

cd ..
python -m zipfile -c exe/tridither.zip src/*.py

cd src
zip -ur ../exe/tridither.zip ditherer -x "ditherer/__pycache__/*"
zip -ur ../exe/tridither.zip shapizer -x "shapizer/__pycache__/*"
zip -ur ../exe/tridither.zip tester -x "tester/__pycache__/*"
cp requirements.txt ../exe

cd ../exe
python3 -m venv .trid_env
chmod 755 .trid_env/bin/activate
.trid_env/bin/activate
python3 -m pip install -r requirements.txt

#echo "python3 tridither.zip exhaustive -i grey_31_1600_hexrev.png -g grey_31_1600_hexrev.npy -s hexagon_rev" > ./run.sh
echo "python3 tridither.zip ml -i david_hex.png,grey_31_1600_hex.png,david_hex_tiny.png -g david_hex.npy,grey_31_1600_hex.npy,david_hex_tiny.npy -s hexagon" > ./run.sh
chmod 755 run.sh
