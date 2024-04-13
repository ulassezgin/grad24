#include "spada_lib.h"

#define DELLEXPORT extern "C" __declspec(dllexport)

__global__  void preciseSizePredictionKernel(const CSRMatrix* A, const CSRMatrix* B, int* rowPtrC, int* nnzC) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < A->rows) 
	{
		rowPtrC[row] = 0;
		for(int i = A->rowPtr[row]; i < A->rowPtr[row + 1]; i++)
		{
			int idxA = A->colInd[i];
			int nnzTemp = B->rowPtr[idxA+1] - B->rowPtr[idxA];
			atomicAdd(nnzC, nnzTemp);
			rowPtrC[row + 1] += nnzTemp;
		}
    }
}

__global__ void shrinkResultMatrix(CSRMatrix* C)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < C->rows)
	{
		for(int i = C->rowPtr[row]; i < C->rowPtr[row + 1]; i++)
		{
			if(C->colInd[i] == -1)
			{
				atomicAdd(&C->nnz, -1);
			}
		}
	}
}

__global__ void rowPtrRecalculation(CSRMatrix* C, int* rowPtrC)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < C->rows)
	{

		for(int i = C->rowPtr[row]; i < C->rowPtr[row + 1]; i++)
		{
			if(C->colInd[i] != -1)
			{
				rowPtrC[row + 1]++;
			}
		}
	}
}

__global__ void createResultMatrix(const CSRMatrix* C, int* rowPtrC, int* colIndC, float* valuesC)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < C->rows)
	{

		int newIndex = 0;
		for(int i = C->rowPtr[row]; i < C->rowPtr[row + 1]; i++)
		{
			if(C->colInd[i] != -1)
			{
				colIndC[rowPtrC[row] + newIndex] = C->colInd[i];
				valuesC[rowPtrC[row] + newIndex] = C->values[i];
				newIndex++;
			}
		}
	}
}

void transpose(const COOMatrix& A, COOMatrix* B)
{
	B->rows = A.cols;
	B->cols = A.rows;
	B->nnz = A.nnz;
	B->row = new int[B->nnz];
	B->col = new int[B->nnz];
	B->values = new float[B->nnz];
	for (int i = 0; i < A.nnz; i++)
	{
		B->row[i] = A.col[i];
		B->col[i] = A.row[i];
		B->values[i] = A.values[i];
	}
}

void readFile(string txtPath, COOMatrix* theMatrix, bool isWeight)
{
	string theReader;
	ifstream file(txtPath);
	getline(file, theReader);
	istringstream ss(theReader);
	ss >> theMatrix->rows >> theMatrix->cols >> theMatrix->nnz;
	theMatrix->row = new int[theMatrix->nnz];
	theMatrix->col = new int[theMatrix->nnz];
	theMatrix->values = new float[theMatrix->nnz];
	int i = 0;
	while (getline(file, theReader))
	{
		istringstream ss(theReader);
		ss >> theMatrix->row[i] >> theMatrix->col[i];
		if (isWeight)
		{
			ss >> theMatrix->values[i];
		}
		else
		{
			theMatrix->values[i] = 1;
		}
		i++;
	}
	file.close();
}

void convertCOOToCSR(COOMatrix* cooMatrix, CSRMatrix* csrMatrix)
{
	csrMatrix->rows = cooMatrix->rows;
	csrMatrix->cols = cooMatrix->cols;
	csrMatrix->nnz = cooMatrix->nnz;
	csrMatrix->nnzRow = 0;
	csrMatrix->rowPtr = new int[csrMatrix->rows + 1];
	csrMatrix->colInd = new int[csrMatrix->nnz];
	csrMatrix->values = new float[csrMatrix->nnz];
	for (int i = 0; i < csrMatrix->rows + 1; i++)
	{
		csrMatrix->rowPtr[i] = 0;
	}
	for (int i = 0; i < csrMatrix->nnz; i++)
	{
		csrMatrix->rowPtr[cooMatrix->row[i]]++;
	}
	for (int i = 0; i < csrMatrix->rows; i++)
	{
		if (csrMatrix->rowPtr[i + 1] - csrMatrix->rowPtr[i] > 0)
		{
			csrMatrix->nnzRow++;
		}
	}
	for (int i = 0; i < csrMatrix->rows; i++)
	{
		csrMatrix->rowPtr[i + 1] += csrMatrix->rowPtr[i];
	}
	int* temp = new int[csrMatrix->rows];
	for (int i = 0; i < csrMatrix->rows; i++)
	{
		temp[i] = 0;
	}
	for (int i = 0; i < csrMatrix->nnz; i++)
	{
		int row = cooMatrix->row[i] - 1;
		int dest = csrMatrix->rowPtr[row] + temp[row];
		csrMatrix->colInd[dest] = cooMatrix->col[i] - 1;
		csrMatrix->values[dest] = cooMatrix->values[i];
		temp[row]++;
	}
	delete[] temp;
}

void convertCSRToCOO(CSRMatrix* csrMatrix, COOMatrix* cooMatrix)
{
	cooMatrix->rows = csrMatrix->rows;
	cooMatrix->cols = csrMatrix->cols;
	cooMatrix->nnz = csrMatrix->nnz;
	cooMatrix->row = new int[cooMatrix->nnz];
	cooMatrix->col = new int[cooMatrix->nnz];
	cooMatrix->values = new float[cooMatrix->nnz];
	int index = 0;
	for (int i = 0; i < csrMatrix->rows; i++)
	{
		for (int j = csrMatrix->rowPtr[i]; j < csrMatrix->rowPtr[i + 1]; j++)
		{
			cooMatrix->row[index] = i + 1;
			cooMatrix->col[index] = csrMatrix->colInd[j] + 1;
			cooMatrix->values[index] = csrMatrix->values[j];
			index++;
		}
	}
}

void copyCSRMatrix(const CSRMatrix& A, CSRMatrix* B)
{
	B->rows = A.rows;
	B->cols = A.cols;
	B->nnz = A.nnz;
	B->nnzRow = A.nnzRow;
	B->rowPtr = new int[B->rows + 1];
	B->colInd = new int[B->nnz];
	B->values = new float[B->nnz];
	for(int i = 0; i < B->rows + 1; i++)
	{
		B->rowPtr[i] = A.rowPtr[i];
	}
	for(int i = 0; i < B->nnz; i++)
	{
		B->colInd[i] = A.colInd[i];
		B->values[i] = A.values[i];
	}
}

DELLEXPORT void freeMemCSR(CSRMatrix* ptr)
{
	if(ptr == NULL)
	{
		return;
	}
	delete[] ptr->rowPtr;
	delete[] ptr->colInd;
	delete[] ptr->values;
}

DELLEXPORT void freeMemCOO(COOMatrix* ptr)
{
	if(ptr == NULL)
	{
		return;
	}
	delete[] ptr->row;
	delete[] ptr->col;
	delete[] ptr->values;
}