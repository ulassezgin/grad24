1. Download the repository zip file.
2. Extract the repository and be sure you are in the correct directory on the terminal.
3. command "./launch.ps1"
4. command "python spada.py <mtx_file> <boolean_flag>

Note: mtx_file is the mtx_file name which is in "mtx_files" folder, and boolean_flag is "false" if all nnz values are 1, otherwise "true".

Example_1: python spada.py ca-CondMat.mtx false

Example_2: python spada.py bayer10.mtx true

NOTE: Be sure CUDA is installed successfully and supported.

NOTE: This repository is to execute on Windows Machine.
