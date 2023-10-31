pip install pandas
python3 generate_data_files.py 
pip install scikit-learn
pip install --upgrade pip
pip install scikit-image
pip install matplotlib
pip3 install torch torchvision torchaudio
pip install keras==2.11.0 tensorflow tensorboardX
pip install --upgrade protobuf
cp /usr/local/lib/python3.8/dist-packages/google/protobuf/internal/builder.py  ./
pip install protobuf==3.19.4
cp ./builder.py /usr/local/lib/python3.8/dist-packages/google/protobuf/internal/
pip install opencv-python
apt update
apt-get install libgl1
pip install albumentations
rm ./builder.py 
