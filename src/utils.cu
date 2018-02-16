#ifndef __cuANN_utils__
#define __cuANN_utils__

#include <cublas_v2.h>
#include <stdexcept>
#include <thrust/gather.h>
#include "utils.h"

namespace cuANN {
	void multiplyMatrix(const float* A, const float* B, float* result, const int rowsA, const int colsA, const int colsB) {
		const float alpha = 1.0, beta = 0.0;
		
		cublasHandle_t handle;
		if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
			throw std::runtime_error("Cannot create cuBLAS handle.");
		}

		cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			rowsA, colsB, colsA,
			&alpha,
			A, rowsA,
			B, colsA,			
			&beta,
			result, rowsA
		);

		cublasDestroy(handle);

		if (status != CUBLAS_STATUS_SUCCESS) {
			throw std::runtime_error("Cannot perform matrix multiplication.");
		}
	}

	void radixSortMatrix(const thrust::device_vector<float> &matrix, const int rows, const int cols, thrust::device_vector<unsigned> &sortedPermutationIndexes) {
		thrust::device_vector<float> dColumn(rows);
		thrust::sequence(sortedPermutationIndexes.begin(), sortedPermutationIndexes.end());

		for (int i = (cols - 1); i >= 0; i--)
		{
			thrust::gather(
				sortedPermutationIndexes.begin(), sortedPermutationIndexes.end(),
				matrix.begin() + rows * i, dColumn.begin()
			);
			thrust::stable_sort_by_key(dColumn.begin(), dColumn.end(), sortedPermutationIndexes.begin());
		}
	}

	__global__ void addVectorFromMatrix(float* matrix, const float* vector, const int rowsA, const int colsA) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;

		if (row < rowsA && col < colsA) {
			matrix[rowsA * col + row] += vector[col];
		}
	}

	__global__ void divideMatrixByScalar(float* matrix, const int scalar, const int rowsA, const int colsA) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;

		if (row < rowsA && col < colsA) {
			matrix[rowsA * col + row] /= scalar;
		}
	}

	__global__ void floorMatrix(float* matrix, const int rowsA, const int colsA) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;

		if (row < rowsA && col < colsA) {
			matrix[rowsA * col + row] = std::floor(matrix[rowsA * col + row]);
		}
	}

	__global__ void convertMatrixFromRawMajorToColumnMajor(const float* srcMatrix, float* destMatrix, const int rows, const int cols) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		
		if (row < rows && col < cols) {
			int rowMajorIdx = cols * row + col;
			int colMajorIdx = rows * col + row;
			destMatrix[colMajorIdx] = srcMatrix[rowMajorIdx];
		}
	}

	__global__ void copyGivenRowsFromMatrix(
		const float* srcMatrix, float* destMatrix,
		const int srcRows, const int destRows, const int cols,
		const unsigned* rowIdxs
	) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;

		if (col < cols && row < destRows) {
			destMatrix[col * destRows + row] = srcMatrix[col * srcRows + rowIdxs[row]];
		}
	}

	// concatenate 2 matrices placing the 2nd under the 1st
	__global__ void concatenateMatricesBelow(
		const float* firstMatrix, const float* secondMatrix,
		const int firstRows, const int secondRows, const int cols,
		float* destMatrix
	){
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int totalRows = firstRows * secondRows;

		if (row < totalRows && col < cols) {
			float value = 0.0;
			if(row < firstRows) {
				value = firstMatrix[firstRows * col + row];
			}
			if(row >= firstRows && row < secondRows) {
				value = secondMatrix[secondRows * col + row];
			}

			destMatrix[totalRows * col + row] = value;
		}
	}

	__global__ void getActualBinIdxs(
		int* binIdxsCandidates,
		const float* projectedQueries,
		const float* binCodes,
		int Q, int k, int binsNumber
	) {
		int queryId = blockIdx.x * blockDim.x + threadIdx.x;
		int i;
		bool isValidIdx;

		if(queryId < Q && binIdxsCandidates[queryId] >= 0) {
			i = 0;
			isValidIdx = true;
			while(i < k && isValidIdx) {
				if(projectedQueries[i * Q + queryId] != binCodes[i * binsNumber + binIdxsCandidates[queryId]]) {
					isValidIdx = false;
				}
			}

			if (!isValidIdx) {
				binIdxsCandidates[queryId] = -1;
			}
		}
	}
}

#endif // !__cuANN_utils__