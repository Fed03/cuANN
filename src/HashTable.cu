#ifndef __cuANN_HashTable__
#define __cuANN_HashTable__

#include <thrust/gather.h>
#include "HashTable.h"
#include "QueryBinCalculator.h"
#include "utils.h"

namespace cuANN {
	HashTable::HashTable(int k, int d, float w) {
		this->k = k;
		this->d = d;
		this->w = w;
		this->N = binsNumber = 0;
		projectionsMatrix = offsetVector = binCodes = 0;
		binSizes = binStartingIndexes = sortedMappingIdxs = 0;
	}

	HashTable::~HashTable() {
		freeMemory();
	}

	void HashTable::allocateProjectionMemory() {
		freeProjectionMemory();

		projectionsMatrix = (float *)malloc(k * d * sizeof(float));
		offsetVector = (float *)malloc(k * sizeof(float));
		if (!(projectionsMatrix && offsetVector))
		{
			throw std::runtime_error("Cannot allocate projections memory");
		}
	}

	void HashTable::allocateBinsMemory() {
		freeBinsMemory();

		binSizes = (unsigned *) malloc(binsNumber * sizeof(unsigned));
		binStartingIndexes = (unsigned *) malloc(binsNumber * sizeof(unsigned));
		sortedMappingIdxs = (unsigned *) malloc(N * sizeof(unsigned));
		binCodes = (float *) malloc(binsNumber * k * sizeof(float));

		if (!(binSizes && binStartingIndexes && sortedMappingIdxs && binCodes))
		{
			throw std::runtime_error("Cannot allocate bins memory");
		}
	}

	void HashTable::freeMemory() {
		freeProjectionMemory();
		freeBinsMemory();
	}

	void HashTable::freeProjectionMemory() {
		if (projectionsMatrix)
		{
			free(projectionsMatrix);
		}
		if (offsetVector)
		{
			free(offsetVector);
		}

		projectionsMatrix = offsetVector = 0;
	}

	void HashTable::freeBinsMemory() {
		if (binSizes)
		{
			free(binSizes);
		}
		if (binStartingIndexes)
		{
			free(binStartingIndexes);
		}
		if (binCodes)
		{
			free(binCodes);
		}
		if (sortedMappingIdxs)
		{
			free(sortedMappingIdxs);
		}
		binCodes = 0;
		binSizes = binStartingIndexes = sortedMappingIdxs = 0;
	}

	void HashTable::generateProjection(curandGenerator_t* normalGen, curandGenerator_t* uniformGen) {
		std::cout << "allocating proj on device. params k: " << k << " d: " << d << std::endl;
		ThrustFloatV dProjections(k * d);
		ThrustFloatV dOffsetVector(k);


		curandGenerateNormal(*normalGen, thrust::raw_pointer_cast(dProjections.data()), k * d, 0, 1);
		curandGenerateUniform(*uniformGen, thrust::raw_pointer_cast(dOffsetVector.data()), k);

		thrust::transform(dOffsetVector.begin(), dOffsetVector.end(),
			thrust::make_constant_iterator(w), dOffsetVector.begin(),
			thrust::multiplies<float>());

		thrust::copy(dOffsetVector.begin(), dOffsetVector.end(), offsetVector);
		thrust::copy(dProjections.begin(), dProjections.end(), projectionsMatrix);
	}

	void HashTable::hashDataset(const float* dataset, const int N) {
		ThrustFloatV dProjectedMatrix(N * k);
		projectMatrix(dataset, N, dProjectedMatrix);
		calcBins(dProjectedMatrix);
	}

	ThrustQueryResult* HashTable::query(const float* queries, const int Q) {
		ThrustFloatV dProjectedQueries(Q * k);
		projectMatrix(queries, Q, dProjectedQueries);

		auto dQueriesBinIdxs = QueryBinCalculator::getBinsForProjectedQueries(
			dProjectedQueries,
			Q, k, binsNumber,
			binCodes
		);

		ThrustHIntV queriesBinIdxs(dQueriesBinIdxs);
		ThrustHUnsignedV resultIdxsForQueriesSizes(Q, 0);
		ThrustHUnsignedV resultIdxsForQueriesStartingIdxs(Q, 0);
		unsigned totalSize = 0;
		for (int query = 0; query < Q; ++query) {
			resultIdxsForQueriesStartingIdxs[query] = totalSize;
			if (queriesBinIdxs[query] != -1) {
				resultIdxsForQueriesSizes[query] = binSizes[queriesBinIdxs[query]];
				totalSize += resultIdxsForQueriesSizes[query];
			}
		}

		ThrustHUnsignedV resultIdxsForQueries(totalSize);
		for (int query = 0; query < Q; ++query) {
			if (queriesBinIdxs[query] != -1) {
				thrust::copy_n(
					sortedMappingIdxs + binStartingIndexes[queriesBinIdxs[query]],
					resultIdxsForQueriesSizes[query],
					resultIdxsForQueries.begin() + resultIdxsForQueriesStartingIdxs[query]
				);
			}
		}

		return new ThrustQueryResult(
			resultIdxsForQueriesStartingIdxs,
			resultIdxsForQueriesSizes,
			resultIdxsForQueries,
			Q, totalSize
		);
	}

	void HashTable::calcBins(const ThrustFloatV& dProjectedMatrix) {
		ThrustUnsignedV dSortedPermutationIndx(N);

		radixSortMatrix(dProjectedMatrix, N, k, dSortedPermutationIndx);
		auto diff = areRowsDifferentFromTheOneAbove(dProjectedMatrix, dSortedPermutationIndx);

		binsNumber = thrust::count(diff.begin(), diff.end(), true);
		auto dBinStartingIndexes = computeStartingIndices(diff);
		auto dBinSizes = computeBinSizes(dBinStartingIndexes);
		auto dBinCodes = extractBinsCode(dProjectedMatrix, dBinStartingIndexes, dSortedPermutationIndx);

		allocateBinsMemory();

		thrust::copy(dSortedPermutationIndx.begin(), dSortedPermutationIndx.end(), sortedMappingIdxs);
		thrust::copy(dBinStartingIndexes.begin(), dBinStartingIndexes.end(), binStartingIndexes);
		thrust::copy(dBinSizes.begin(), dBinSizes.end(), binSizes);
		thrust::copy(dBinCodes.begin(), dBinCodes.end(), binCodes);
	}

	ThrustFloatV HashTable::extractBinsCode(
		const ThrustFloatV& dProjectedMatrix,
		const ThrustUnsignedV& startingIndices,
		const ThrustUnsignedV& dSortedPermutationIndx
	) {
		ThrustFloatV codes(binsNumber * k);
		auto originalIdxs = originalDatasetIdxsFromStartingIdxs(startingIndices, dSortedPermutationIndx);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((k + dimBlock.x - 1)/dimBlock.x, (binsNumber + dimBlock.y - 1)/dimBlock.y);
		copyGivenRowsFromMatrix<<<dimGrid, dimBlock>>>(
			thrust::raw_pointer_cast(dProjectedMatrix.data()),
			thrust::raw_pointer_cast(codes.data()),
			N, binsNumber, k,
			thrust::raw_pointer_cast(originalIdxs.data())
		);

		return codes;
	}

	ThrustUnsignedV HashTable::originalDatasetIdxsFromStartingIdxs(
		const ThrustUnsignedV& startingIndices,
		const ThrustUnsignedV& dSortedPermutationIndx
	) {
		ThrustUnsignedV originalIdxs(startingIndices.size());

		thrust::gather(startingIndices.begin(), startingIndices.end(), dSortedPermutationIndx.begin(), originalIdxs.begin());

		return originalIdxs;
	}

	ThrustUnsignedV HashTable::computeBinSizes(const ThrustUnsignedV& startingIndices) {
		ThrustUnsignedV sizes(binsNumber);

		thrust::adjacent_difference(
			startingIndices.begin() + 1, startingIndices.end(),
			sizes.begin()
		);

		sizes[binsNumber - 1] = N - startingIndices.back();

		return sizes;
	}

	ThrustUnsignedV HashTable::computeStartingIndices(const ThrustBoolV& diff) {
		ThrustUnsignedV startingIndices(binsNumber, 0);

		ThrustUnsignedV mapping(N);
		thrust::sequence(mapping.begin(), mapping.end());

		thrust::copy_if(mapping.begin(), mapping.end(), diff.begin(), startingIndices.begin(), isTrue());

		return startingIndices;
	}

	ThrustBoolV HashTable::areRowsDifferentFromTheOneAbove(const ThrustFloatV& matrix, const ThrustUnsignedV& dSortedPermutationIndx){
		ThrustBoolV rowsDiff(N);
		ThrustFloatV dColumn(N);

		thrust::fill(rowsDiff.begin(), rowsDiff.end(), false);

		for (int col = 0; col < k; col++)
		{
			thrust::gather(
				dSortedPermutationIndx.begin(), dSortedPermutationIndx.end(),
				matrix.begin() + N * col, dColumn.begin()
			);
			thrust::adjacent_difference(dColumn.begin(), dColumn.end(), dColumn.begin());
			thrust::transform(
				dColumn.begin(), dColumn.end(), rowsDiff.begin(),
				rowsDiff.begin(), getOrDefault()
			);
		}

		// just in case first row is a zero vector
		rowsDiff[0] = true;

		return rowsDiff;
	}

	void HashTable::projectMatrix(const float* dataset, const int N, ThrustFloatV& dProjectedMatrix) {
		ThrustFloatV dDataset(dataset, dataset + N * d);
		ThrustFloatV dProjectionsMatrix(projectionsMatrix, projectionsMatrix + d * k);
		ThrustFloatV dOffsetVector(offsetVector, offsetVector + k);

		float * dProjectedMatrixPTR = thrust::raw_pointer_cast(dProjectedMatrix.data());

		multiplyMatrix(
			thrust::raw_pointer_cast(dDataset.data()),
			thrust::raw_pointer_cast(dProjectionsMatrix.data()),
			dProjectedMatrixPTR,
			N, d, k
		);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((k + dimBlock.x - 1)/dimBlock.x, (N + dimBlock.y - 1)/dimBlock.y);
		addVectorFromMatrix <<< dimGrid, dimBlock >>> (
			dProjectedMatrixPTR,
			thrust::raw_pointer_cast(dOffsetVector.data()),
			N, k
		);
		divideMatrixByScalar <<< dimGrid, dimBlock >>> (dProjectedMatrixPTR, w, N, k);
		floorMatrix <<< dimGrid, dimBlock >>> (dProjectedMatrixPTR, N, k);
	}
}

#endif // !__cuANN_HashTable__
