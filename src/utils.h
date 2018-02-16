#ifndef __cuANN_UTILS_H_
#define __cuANN_UTILS_H_

#include <thrust/device_vector.h>
#include <thrust/functional.h>

namespace cuANN {
	void multiplyMatrix(const float* A, const float* B, float* result, const int rowsA, const int colsA, const int colsB);

	void radixSortMatrix(const thrust::device_vector<float> &matrix, const int rows, const int cols, thrust::device_vector<unsigned> &sortedPermutationIndexes);


	struct getOrDefault : public thrust::binary_function<const float, const bool, bool> {
		__host__ __device__
		bool operator()(const float actual, const bool defaultValue) {
			return (actual == 0) ? defaultValue : true;
		}
	};

	struct isTrue {
		__host__ __device__
		bool operator()(const bool value) {
			return value;
		}
	};

	__global__ void addVectorFromMatrix(float* matrix, const float* vector, const int rowsA, const int colsA);

	__global__ void divideMatrixByScalar(float* matrix, const int scalar, const int rowsA, const int colsA);

	__global__ void floorMatrix(float* matrix, const int rowsA, const int colsA);

	__global__ void convertMatrixFromRawMajorToColumnMajor(
		const float* srcMatrix, float* destMatrix, const int rows, const int cols
	);

	__global__ void copyGivenRowsFromMatrix(
		const float* srcMatrix, float* destMatrix,
		const int srcRows, const int destRows, const int cols,
		const unsigned* rowIdxs
	);

	// concatenate 2 matrices placing the 2nd under the 1st
	__global__ void concatenateMatricesBelow(
		const float* firstMatrix, const float* secondMatrix,
		const int firstRows, const int secondRows, const int cols,
		float* destMatrix
	);

	__global__ void getActualBinIdxs(
		int* binIdxsCandidates,
		const float* projectedQueries,
		const float* binCodes,
		int Q, int k, int binsNumber
	);
}


#endif /* __cuANN_UTILS_H_ */