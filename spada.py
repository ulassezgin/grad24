import ctypes 
from ctypes import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sys

def plot_spada(coo_matrix_a, coo_matrix_b, csr_matrix_instance, exec_list, mtx_file_name):
    # substract 1 from the row and col indices
    for i in range(coo_matrix_a.nnz):
        coo_matrix_a.row[i] -= 1
        coo_matrix_a.col[i] -= 1


    for i in range(coo_matrix_b.nnz):
        coo_matrix_b.row[i] -= 1
        coo_matrix_b.col[i] -= 1


    sparse_matrix_res = sp.csr_matrix((csr_matrix_instance.values[:csr_matrix_instance.nnz], csr_matrix_instance.colInd[:csr_matrix_instance.nnz], csr_matrix_instance.rowPtr[:csr_matrix_instance.rows+1]), shape=(csr_matrix_instance.rows, csr_matrix_instance.cols))
    sparse_matrix_a = sp.coo_matrix((coo_matrix_a.values[:coo_matrix_a.nnz], (coo_matrix_a.row[:coo_matrix_a.nnz], coo_matrix_a.col[:coo_matrix_a.nnz])), shape=(coo_matrix_a.rows, coo_matrix_a.cols)).tocsr()
    sparse_matrix_b = sp.coo_matrix((coo_matrix_b.values[:coo_matrix_b.nnz], (coo_matrix_b.row[:coo_matrix_b.nnz], coo_matrix_b.col[:coo_matrix_b.nnz])), shape=(coo_matrix_b.rows, coo_matrix_b.cols)).tocsr()

    plt.figure()
    plt.suptitle("Execution Results for " + mtx_file_name)

    plt.subplot(221)
    plt.title("Matrix A")
    plt.spy(sparse_matrix_a, markersize=1)
    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels

    plt.subplot(222)
    plt.title("Matrix B")
    plt.spy(sparse_matrix_b, markersize=1)
    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels

    plt.subplot(223)
    plt.title("Result")
    plt.spy(sparse_matrix_res, markersize=1)
    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels

    block_sizes = ["64 x 1", "32 x 2", "16 x 4", "8 x 8", "4 x 16", "2 x 32", "1 x 64", "32 x 1", "16 x 2", "8 x 4", "4 x 8", "2 x 16", "1 x 32"]
    plt.subplot(224)
    plt.title("Execution Time")
    plt.bar(block_sizes, exec_list)
    plt.xticks(rotation=45)
    plt.show()

class csr_matrix(Structure):
    _fields_ = [("rowPtr", POINTER(c_int)),
                ("colInd", POINTER(c_int)),
                ("values", POINTER(c_float)),
                ("rows", c_int),
                ("cols", c_int),
                ("nnz", c_int),
                ("nnzRow", c_int)]
    

class coo_matrix(Structure):
    _fields_ = [("row", POINTER(c_int)),
                ("col", POINTER(c_int)),
                ("values", POINTER(c_float)),
                ("rows", c_int),
                ("cols", c_int),
                ("nnz", c_int)]

if len(sys.argv) < 3:
    print("Usage: python spada.py <mtx_file> <boolean_flag>")
    sys.exit(1)

mtx_file_name = sys.argv[1]
mtx_file_path = "./mtx_files/" + sys.argv[1]

if(sys.argv[2].lower() == "true"):
    is_one_mtx = True
elif(sys.argv[2].lower() == "false"):
    is_one_mtx = False
else:
    print("Usage: python spada.py <mtx_file> <boolean_flag>")
    sys.exit(1)


lib64 = ctypes.CDLL('./spada_64_threads/build/Debug/spadalib64.dll')
lib32 = ctypes.CDLL('./spada_32_threads/build/Debug/spadalib32.dll')

spada_1_64 = lib64.spada_1_64
spada_2_32 = lib64.spada_2_32
spada_4_16 = lib64.spada_4_16
spada_8_8 = lib64.spada_8_8
spada_16_4 = lib64.spada_16_4
spada_32_2 = lib64.spada_32_2
spada_64_1 = lib64.spada_64_1
free_mem_coo_64 = lib64.freeMemCOO
free_mem_csr_64 = lib64.freeMemCSR

spada_1_64.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_1_64.restype = c_float
spada_2_32.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_2_32.restype = c_float
spada_4_16.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_4_16.restype = c_float
spada_8_8.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_8_8.restype = c_float
spada_16_4.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_16_4.restype = c_float
spada_32_2.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_32_2.restype = c_float
spada_64_1.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_64_1.restype = c_float
free_mem_coo_64.argtypes = [POINTER(coo_matrix)]
free_mem_csr_64.argtypes = [POINTER(csr_matrix)]

spada_1_32 = lib32.spada_1_32
spada_2_16 = lib32.spada_2_16
spada_4_8 = lib32.spada_4_8
spada_8_4 = lib32.spada_8_4
spada_16_2 = lib32.spada_16_2
spada_32_1 = lib32.spada_32_1
free_mem_coo_32 = lib32.freeMemCOO
free_mem_csr_32 = lib32.freeMemCSR


spada_1_32.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_1_32.restype = c_float
spada_2_16.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_2_16.restype = c_float
spada_4_8.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_4_8.restype = c_float
spada_8_4.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_8_4.restype = c_float
spada_16_2.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_16_2.restype = c_float
spada_32_1.argtypes = [c_bool, type(ctypes.create_string_buffer(b"", 256)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(coo_matrix)), POINTER(POINTER(csr_matrix))]
spada_32_1.restype = c_float
free_mem_coo_32.argtypes = [POINTER(coo_matrix)]
free_mem_csr_32.argtypes = [POINTER(csr_matrix)]

#define csr_matrix 
csr_matrix_instance = csr_matrix()
csr_matrix_p = pointer(csr_matrix_instance)

coo_matrix_a = coo_matrix()
coo_matrix_p_a = pointer(coo_matrix_a)

coo_matrix_b = coo_matrix()
coo_matrix_p_b = pointer(coo_matrix_b)
#create ctype string buffer





file_name = ctypes.create_string_buffer(mtx_file_path.encode(), 256)

exec_time_list = []

print("Executing SPADA with block size 64x1")
exec_time_list.append(spada_64_1(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_64(coo_matrix_p_a)
free_mem_coo_64(coo_matrix_p_b)
free_mem_csr_64(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 32x2")
exec_time_list.append(spada_32_2(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_64(coo_matrix_p_a)
free_mem_coo_64(coo_matrix_p_b)
free_mem_csr_64(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 16x4")
exec_time_list.append(spada_16_4(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_64(coo_matrix_p_a)
free_mem_coo_64(coo_matrix_p_b)
free_mem_csr_64(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 8x8")
exec_time_list.append(spada_8_8(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_64(coo_matrix_p_a)
free_mem_coo_64(coo_matrix_p_b)
free_mem_csr_64(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 4x16")
exec_time_list.append(spada_4_16(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_64(coo_matrix_p_a)
free_mem_coo_64(coo_matrix_p_b)
free_mem_csr_64(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 2x32")
exec_time_list.append(spada_2_32(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_64(coo_matrix_p_a)
free_mem_coo_64(coo_matrix_p_b)
free_mem_csr_64(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 1x64")
exec_time_list.append(spada_1_64(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_64(coo_matrix_p_a)
free_mem_coo_64(coo_matrix_p_b)
free_mem_csr_64(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 32x1")
exec_time_list.append(spada_32_1(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_32(coo_matrix_p_a)
free_mem_coo_32(coo_matrix_p_b)
free_mem_csr_32(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 16x2")
exec_time_list.append(spada_16_2(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_32(coo_matrix_p_a)
free_mem_coo_32(coo_matrix_p_b)
free_mem_csr_32(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 8x4")
exec_time_list.append(spada_8_4(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_32(coo_matrix_p_a)
free_mem_coo_32(coo_matrix_p_b)
free_mem_csr_32(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 4x8")
exec_time_list.append(spada_4_8(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_32(coo_matrix_p_a)
free_mem_coo_32(coo_matrix_p_b)
free_mem_csr_32(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 2x16")
exec_time_list.append(spada_2_16(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))
free_mem_coo_32(coo_matrix_p_a)
free_mem_coo_32(coo_matrix_p_b)
free_mem_csr_32(csr_matrix_p)
coo_matrix_p_a = pointer(coo_matrix_a)
coo_matrix_p_b = pointer(coo_matrix_b)
csr_matrix_p = pointer(csr_matrix_instance)
print("Executing SPADA with block size 1x32")
exec_time_list.append(spada_1_32(is_one_mtx, file_name, byref(coo_matrix_p_a), byref(coo_matrix_p_b), byref(csr_matrix_p)))

plot_spada(coo_matrix_a, coo_matrix_b, csr_matrix_instance, exec_time_list, mtx_file_name)
