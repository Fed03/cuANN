#ifndef __cuANN_utils__
#define __cuANN_utils__

#include <cublas_v2.h>
#include <stdexcept>
#include <thrust/gather.h>
#include "commons.h"
#include "utils.h"

namespace cuANN {
	void multiplyMatrix(const float* A, const float* B, float* result, const int rowsA, const int colsA, const int colsB) {
		const float alpha = 1.0, beta = 0.0;
		
		cublasHandle_t handle;
		if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
			throw std::runtime_error("Cannot create cuBLAS handle.");
		}

//		cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//			rowsA, colsB, colsA,
//			&alpha,
//			A, rowsA,
//			B, colsA,
//			&beta,
//			result, rowsA
//		);

		cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			colsB, rowsA, colsA,
			&alpha,
			B, colsB,
			A, colsA,
			&beta,
			result, colsB
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
			matrix[colsA * row + col] += vector[col];
		}
	}

	__global__ void divideMatrixByScalar(float* matrix, const int scalar, const int rowsA, const int colsA) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;

		if (row < rowsA && col < colsA) {
			matrix[colsA * row + col] /= scalar;
		}
	}

	__global__ void floorMatrix(float* matrix, const int rowsA, const int colsA) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;

		if (row < rowsA && col < colsA) {
			matrix[colsA * row + col] = std::floor(matrix[colsA * row + col]);
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
		int totalRows = firstRows + secondRows;

		if (row < totalRows && col < cols) {
			float value = 0.0;
			int rowIdx = row % firstRows;
			if(row < firstRows) {
				value = firstMatrix[firstRows * col + rowIdx];
			}
			if(row >= firstRows) {
				value = secondMatrix[secondRows * col + rowIdx];
			}

			destMatrix[totalRows * col + row] = value;
		}
	}

	__global__ void getActualBinIdxs(
		int* binIdxsCandidates,
		const size_t* queryHashes,
		const size_t* binCodes,
		int Q, int binsNumber
	) {
		int queryId = blockIdx.x * blockDim.x + threadIdx.x;

		if(queryId < Q && binIdxsCandidates[queryId] >= 0) {
			if (queryHashes[queryId] != binCodes[binIdxsCandidates[queryId]]) {
				binIdxsCandidates[queryId] = -1;
			}
		}
	}

	__global__ void calcSquaredDistances(
		const float* A,
		const float* B,
		int cols,
		const unsigned* rowIdxsA,
		const unsigned* rowIdxsB,
		unsigned distancesNumber,
		float* result
	) {
		__shared__ float distances[BLOCK_SIZE_STRIDE_X][BLOCK_SIZE_STRIDE_Y];

		int distanceIdx = blockDim.x * blockIdx.x + threadIdx.x;
		if (distanceIdx < distancesNumber) {
			int ARowIdx = rowIdxsA[distanceIdx];
			int BRowIdx = rowIdxsB[distanceIdx];

			float distance = 0.0;
			for (int strideIdx = threadIdx.y; strideIdx < cols; strideIdx += BLOCK_SIZE_STRIDE_Y) {
				distance += powf(A[cols * ARowIdx + strideIdx] - B[cols * BRowIdx + strideIdx], 2);
			}

			distances[threadIdx.x][threadIdx.y] = distance;
		}
		__syncthreads();

		if (threadIdx.y < 4) {
			distances[threadIdx.x][threadIdx.y] += distances[threadIdx.x][threadIdx.y + 4];
		}
		__syncthreads();

		if (threadIdx.y < 2) {
			distances[threadIdx.x][threadIdx.y] += distances[threadIdx.x][threadIdx.y + 2];
		}
		__syncthreads();

		if (threadIdx.y == 0) {
			result[distanceIdx] = distances[threadIdx.x][0] + distances[threadIdx.x][1];
		}
	}

	__device__ void hashRange(const float* iteratorBegin, const float* iteratorEnd, size_t& result) {
		size_t seed = 0;
		while(iteratorBegin != iteratorEnd) {
			seed ^= static_cast<int>(*iteratorBegin) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			++iteratorBegin;
		}
		result = seed;
	}

	__global__ void hashMatrixRows(const float* matrix, const int rows, const int cols, size_t* hashes) {
		int row = blockIdx.x * blockDim.x + threadIdx.x;

		if(row < rows) {
			size_t hash;
			hashRange(matrix + cols * row, matrix + cols * (row + 1), hash);
			hashes[row] = hash;
		}
	}

}

#endif // !__cuANN_utils__
