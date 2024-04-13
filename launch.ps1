cd spada_32_threads
mkdir build
cd build
cmake ..
cmake --build .

cd ..\..\spada_64_threads
mkdir build
cd build
cmake ..
cmake --build .

cd ..\..\

pip install -r requirements.txt