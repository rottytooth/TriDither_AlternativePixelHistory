rm exe/*.*

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

echo "python3 tridither.zip exhaustive -i grey_49_1600_hex.png -g grey_49_1600_hex.npy -s hexagon" > run.sh