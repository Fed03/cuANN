#ifndef __cuANN_HASHTABLE_H_
#define __cuANN_HASHTABLE_H_

#include <curand.h>
#include "commons.h"
#include "ThrustQueryResult.h"

namespace cuANN {
	class HashTable
	{
	public:
		HashTable(int k, int d, float w);

		~HashTable();

		void freeMemory();

		void allocateProjectionMemory();

		void generateProjection(curandGenerator_t* normalGen, curandGenerator_t* uniformGen);

		void hashDataset(const float* dataset, const int N);

		ThrustQueryResult* query(const float* queries, const int Q);

	private:
		int k;
		int d;
		float w;
		int N;

		float *projectionsMatrix;
		float *offsetVector;

		unsigned binsNumber;
		unsigned *sortedMappingIdxs;
		unsigned *binSizes;
		unsigned *binStartingIndexes;
		size_t *binCodes;

		void freeProjectionMemory();

		void freeBinsMemory();

		void allocateBinsMemory();

		void calcBins(const ThrustFloatV& dProjectedMatrix);

		ThrustBoolV areRowsDifferentFromTheOneAbove(const ThrustFloatV& matrix, const ThrustUnsignedV& dSortedPermutationIndx);

		ThrustUnsignedV computeStartingIndices(const ThrustBoolV& diff);

		ThrustUnsignedV computeBinSizes(const ThrustUnsignedV& startingIndices);

		ThrustUnsignedV originalDatasetIdxsFromStartingIdxs(const ThrustUnsignedV& startingIndices, const ThrustUnsignedV& dSortedPermutationIndx);

		thrust::device_vector<size_t> extractBinsCode(
			const thrust::device_vector<size_t>& hashes,
			const ThrustUnsignedV& startingIndices
		);

		void projectMatrix(const float* dataset, const int N, ThrustFloatV& dProjectedMatrix);
	};
}

#endif /* __cuANN_HASHTABLE_H_ */
