#ifndef __cuANN_UTILS_H_
#define __cuANN_UTILS_H_

#include <thrust/device_vector.h>
#include <thrust/functional.h>

namespace cuANN {
	void multiplyMatrix(const float* A, const float* B, float* result, const int rowsA, const int colsA, const int colsB);

	struct isTrue {
		__host__ __device__
		bool operator()(const bool value) {
			return value;
		}
	};

	struct isDifferentFromZero {
		__host__ __device__
		bool operator()(const size_t value) {
			return value != 0;
		}
	};

	__global__ void addVectorFromMatrix(float* matrix, const float* vector, const int rowsA, const int colsA);

	__global__ void divideMatrixByScalar(float* matrix, const int scalar, const int rowsA, const int colsA);

	__global__ void floorMatrix(float* matrix, const int rowsA, const int colsA);

	__global__ void getActualBinIdxs(
		int* binIdxsCandidates,
		const size_t* queryHashes,
		const size_t* binCodes,
		int Q, int binsNumber
	);

	__global__ void calcSquaredDistances(
		const float* A,
		const float* B,
		int cols,
		const unsigned* rowIdxsA,
		const unsigned* rowIdxsB,
		unsigned distancesNumber,
		float* result
	);

	__global__ void hashMatrixRows(const float* matrix, const int rows, const int cols, size_t* hashes);

	__device__ void hashRange(const float* iteratorBegin, const float* iteratorEnd, size_t& result);

}


#endif /* __cuANN_UTILS_H_ */
