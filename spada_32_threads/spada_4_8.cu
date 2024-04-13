#include "./spada_lib.h"
#define WINDOW_HEIGHT 4
#define WINDOW_WIDTH 8


#define DELLEXPORT extern "C" __declspec(dllexport)
using namespace std;


__global__  void spgemmByAdaptiveWindow_4_8(const CSRMatrix* A, const CSRMatrix* B, CSRMatrix* C)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < A->rows)
	{
		for(int i = 0; i < ceil((float)A->rows / WINDOW_HEIGHT); i++)
		{
			 if (row + i * WINDOW_HEIGHT >= A->rows) 
			 {
			 	break;
			 }
			__shared__ int indexHolder[WINDOW_HEIGHT];
			for(int j = 0; j < WINDOW_HEIGHT; j++)
			{
				indexHolder[j] = 0;
			}

			for(int j = 0; j < ceil((float)A->cols / WINDOW_WIDTH); j++)
			{
				int index = A->rowPtr[row + i * WINDOW_HEIGHT] + col + j * WINDOW_WIDTH;

				if (index >= A->rowPtr[row + i * WINDOW_HEIGHT + 1]) 
				{
					break;
				}
		
				int idxA = A->colInd[index];
				 if (idxA >= A->cols) 
				 {
				 	break;
				 }
				for(int k = B->rowPtr[idxA]; k < B->rowPtr[idxA+1]; k++)
				{
					int idxC = C->rowPtr[row + i * WINDOW_HEIGHT] + indexHolder[row];
					indexHolder[row]++;
					 if(idxC >= C->nnz)
					 {
					 	break;
					 }

					atomicAdd(&C->values[idxC], A->values[index] * B->values[k]);
					C->colInd[idxC] = B->colInd[k];
				}


			}
		}
	}
}

DELLEXPORT float spada_4_8(const bool isOne, const char* fileName, COOMatrix** h_coo_a, COOMatrix** h_coo_b, CSRMatrix** h_csr_c)
{

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	try {
		readFile(fileName, *h_coo_a, isOne);
	}
	catch (exception& e) {
		cout << "An error occurred: " << e.what() << endl;
	}


	CSRMatrix csrA, csrB, csrC;

	transpose(**h_coo_a, *h_coo_b);
	convertCOOToCSR(*h_coo_a, &csrA);
	convertCOOToCSR(*h_coo_b, &csrB);

	CSRMatrix* d_A, *d_B, *d_C;

	cudaMalloc((void**)&d_A, sizeof(CSRMatrix));
	cudaMalloc((void**)&d_B, sizeof(CSRMatrix));
	cudaMalloc((void**)&d_C, sizeof(CSRMatrix));

	int* d_A_rowPtr, *d_A_colInd, *d_B_rowPtr, *d_B_colInd, *d_C_rowPtr, *d_C_colInd;
	float* d_A_values, *d_B_values, *d_C_values;

	cudaMalloc((void**)&d_A_rowPtr, (csrA.rows + 1) * sizeof(int));
	cudaMalloc((void**)&d_A_colInd, csrA.nnz * sizeof(int));
	cudaMalloc((void**)&d_A_values, csrA.nnz * sizeof(float));
	cudaMemcpy(d_A_rowPtr, csrA.rowPtr, (csrA.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A_colInd, csrA.colInd, csrA.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A_values, csrA.values, csrA.nnz * sizeof(float), cudaMemcpyHostToDevice);

	CSRMatrix* d_A_struct = new CSRMatrix;
	d_A_struct->rowPtr = d_A_rowPtr;
	d_A_struct->colInd = d_A_colInd;
	d_A_struct->values = d_A_values;
	d_A_struct->rows = csrA.rows;
	d_A_struct->cols = csrA.cols;
	d_A_struct->nnz = csrA.nnz;
	d_A_struct->nnzRow = csrA.nnzRow;

	cudaMemcpy(d_A, d_A_struct, sizeof(CSRMatrix), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_B_rowPtr, (csrB.rows + 1) * sizeof(int));
	cudaMalloc((void**)&d_B_colInd, csrB.nnz * sizeof(int));
	cudaMalloc((void**)&d_B_values, csrB.nnz * sizeof(float));
	cudaMemcpy(d_B_rowPtr, csrB.rowPtr, (csrB.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_colInd, csrB.colInd, csrB.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_values, csrB.values, csrB.nnz * sizeof(float), cudaMemcpyHostToDevice);
	
	CSRMatrix* d_B_struct = new CSRMatrix;
	d_B_struct->rowPtr = d_B_rowPtr;
	d_B_struct->colInd = d_B_colInd;
	d_B_struct->values = d_B_values;
	d_B_struct->rows = csrB.rows;
	d_B_struct->cols = csrB.cols;
	d_B_struct->nnz = csrB.nnz;
	d_B_struct->nnzRow = csrB.nnzRow;
	
	cudaMemcpy(d_B, d_B_struct, sizeof(CSRMatrix), cudaMemcpyHostToDevice);


	csrC.rows = csrA.rows;
	csrC.cols = csrB.cols;
	csrC.rowPtr = new int[csrC.rows + 1];
	csrC.nnz = 0;
	csrC.nnzRow = 0;

	int* d_rowPtrC, *d_nnzC;
	cudaMalloc((void**)&d_rowPtrC, (csrC.rows + 1) * sizeof(int));
	cudaMalloc((void**)&d_nnzC, sizeof(int));
	cudaMemset(d_nnzC, 0, sizeof(int));
	
	cudaMemset(d_rowPtrC, 0, (csrC.rows + 1) * sizeof(int));
	
	
	
	int THREADS = 1024;
	int BLOCKS = ceil((float)csrC.rows / THREADS);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	preciseSizePredictionKernel<<<BLOCKS, THREADS>>>(d_A, d_B, d_rowPtrC, d_nnzC);

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Time for precise size prediction: " << milliseconds << "ms" << endl;
	

	cudaMemcpy(csrC.rowPtr, d_rowPtrC, (csrC.rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&csrC.nnz, d_nnzC, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_rowPtrC);
	cudaFree(d_nnzC);

	for(int i = 1; i < csrC.rows + 1; i++)
	{
		csrC.rowPtr[i] = csrC.rowPtr[i] + csrC.rowPtr[i-1];
	}	

	csrC.colInd = new int[csrC.nnz];
	csrC.values = new float[csrC.nnz];

	for(int i = 0; i < csrC.nnz; i++)
	{
		csrC.values[i] = 0;
		csrC.colInd[i] = -1;
	}


	cudaMalloc((void**)&d_C_rowPtr, (csrC.rows + 1) * sizeof(int));
	cudaMalloc((void**)&d_C_colInd, csrC.nnz * sizeof(int));
	cudaMalloc((void**)&d_C_values, csrC.nnz * sizeof(float));
	cudaMemcpy(d_C_rowPtr, csrC.rowPtr, (csrC.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_colInd, csrC.colInd, csrC.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_values, csrC.values, csrC.nnz * sizeof(float), cudaMemcpyHostToDevice);

	CSRMatrix* d_C_struct = new CSRMatrix;
	d_C_struct->rowPtr = d_C_rowPtr;
	d_C_struct->colInd = d_C_colInd;
	d_C_struct->values = d_C_values;
	d_C_struct->rows = csrC.rows;
	d_C_struct->cols = csrC.cols;
	d_C_struct->nnz = csrC.nnz;
	d_C_struct->nnzRow = csrC.nnzRow;

	cudaMemcpy(d_C, d_C_struct, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
	

	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	dim3 dimGrid(1, 1);
	dim3 dimBlock(WINDOW_HEIGHT, WINDOW_WIDTH);
	cudaEventRecord(start);
	spgemmByAdaptiveWindow_4_8<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float kernelTime = 0;
	cudaEventElapsedTime(&kernelTime, start, stop);
	cout << "Time for kernel execution: " << kernelTime << "ms" << endl;


	cudaEventRecord(start);
	shrinkResultMatrix<<<BLOCKS, THREADS>>>(d_C);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Time for shrinking result matrix: " << milliseconds << "ms" << endl;

	int* d_result_rowPtr, *d_result_colInd;
	float* d_result_values;
	

	int resultNnz;
	cudaMemcpy(&resultNnz, &d_C->nnz, sizeof(int), cudaMemcpyDeviceToHost);
	int* temp_row;
	temp_row = new int[csrC.rows + 1];


	cudaMalloc((void**)&d_result_rowPtr, (csrC.rows + 1) * sizeof(int));
	cudaMalloc((void**)&d_result_colInd, resultNnz * sizeof(int));
	cudaMalloc((void**)&d_result_values, resultNnz * sizeof(float));
	cudaMemset(d_result_rowPtr, 0, (csrC.rows + 1) * sizeof(int));

	cudaEventRecord(start);
	rowPtrRecalculation<<<BLOCKS, THREADS>>>(d_C, d_result_rowPtr);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Time for recalculating rowPtr: " << milliseconds << "ms" << endl;


	cudaMemcpy(temp_row, d_result_rowPtr, (csrC.rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 1; i < csrC.rows + 1; i++)
	{
		temp_row[i] = temp_row[i] + temp_row[i-1];
	}
	cudaMemcpy(d_result_rowPtr, temp_row, (csrC.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	createResultMatrix<<<BLOCKS, THREADS>>>(d_C, d_result_rowPtr, d_result_colInd, d_result_values);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Time for creating result matrix: " << milliseconds << "ms" << endl;


	cudaMemcpy(csrC.rowPtr, d_result_rowPtr, (csrC.rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);

	delete[] csrC.colInd;
	delete[] csrC.values;
	csrC.colInd = new int[resultNnz];
	csrC.values = new float[resultNnz];

	cudaMemcpy(csrC.colInd, d_result_colInd, resultNnz * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(csrC.values, d_result_values, resultNnz * sizeof(float), cudaMemcpyDeviceToHost);
	csrC.nnz = resultNnz;
	
	copyCSRMatrix(csrC, *h_csr_c);

	cout << endl;

	cudaFree(d_A_rowPtr);
	cudaFree(d_A_colInd);
	cudaFree(d_A_values);
	cudaFree(d_B_rowPtr);
	cudaFree(d_B_colInd);
	cudaFree(d_B_values);
	cudaFree(d_C_rowPtr);
	cudaFree(d_C_colInd);
	cudaFree(d_C_values);
	cudaFree(d_result_rowPtr);
	cudaFree(d_result_colInd);
	cudaFree(d_result_values);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	delete[] csrA.rowPtr;
	delete[] csrA.colInd;
	delete[] csrA.values;
	delete[] csrB.rowPtr;
	delete[] csrB.colInd;
	delete[] csrB.values;
	delete[] csrC.rowPtr;
	delete[] csrC.colInd;
	delete[] csrC.values;
	delete[] temp_row;

	delete d_A_struct;
	delete d_B_struct;
	delete d_C_struct;

	return kernelTime;
}
