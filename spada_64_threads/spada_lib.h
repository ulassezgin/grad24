#ifndef SPADA_LIB_H
#define SPADA_LIB_H

#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <sstream>
#include <time.h>

struct CSRMatrix
{
    int* rowPtr;
    int* colInd;
    float* values;
    int rows;
    int cols;
    int nnz;
    int nnzRow;
};

struct COOMatrix
{
    int* row;
    int* col;
    float* values;
    int rows;
    int cols;
    int nnz;
};
#define DELLEXPORT extern "C" __declspec(dllexport)

using namespace std;

// Declare the functions here
__global__ void preciseSizePredictionKernel(const CSRMatrix* A, const CSRMatrix* B, int* rowPtrC, int* nnzC);
__global__ void shrinkResultMatrix(CSRMatrix* C);
__global__ void rowPtrRecalculation(CSRMatrix* C, int* rowPtrC);
__global__ void createResultMatrix(const CSRMatrix* C, int* rowPtrC, int* colIndC, float* valuesC);
void transpose(const COOMatrix& A, COOMatrix* B);
void readFile(string txtPath, COOMatrix* theMatrix, bool isWeight);
void convertCOOToCSR(COOMatrix* cooMatrix, CSRMatrix* csrMatrix);
void convertCSRToCOO(CSRMatrix* csrMatrix, COOMatrix* cooMatrix);
void copyCSRMatrix(const CSRMatrix& A, CSRMatrix* B);
DELLEXPORT void freeMemCSR(CSRMatrix* ptr);
DELLEXPORT void freeMemCOO(COOMatrix* ptr);
#endif