#include "QueryBinCalculator.h"
#include "utils.h"

namespace cuANN {
	ThrustIntV QueryBinCalculator::getBinsForProjectedQueries(
		const ThrustFloatV& dProjectedQueries, int Q, int k,
		int binsNumber, const float* binCodes
	) {
		ThrustFloatV dBinCodes(binCodes, binCodes + binsNumber * k);

		auto concatenatedMatrix = concatenateBinCodesAndProjectedQueries(dProjectedQueries, Q, k, binsNumber, dBinCodes);
		auto permutationIdxs = sortMatrixRows(concatenatedMatrix, binsNumber+Q, k);

		return calcBinIdxs(permutationIdxs, dProjectedQueries, dBinCodes, Q, k, binsNumber);
	}

	ThrustIntV QueryBinCalculator::calcBinIdxs(
			const ThrustUnsignedV& permutationIdxsVec,
			const ThrustFloatV& dProjectedQueries,
			const ThrustFloatV& dBinCodes,
			int Q, int k, int binsNumber
	) {
		auto binIdxsCandidates = calcIdxsCandidates(permutationIdxsVec, binsNumber, Q);

		dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
		dim3 dimGrid((Q + dimBlock.x - 1)/dimBlock.x);
		getActualBinIdxs<<<dimGrid, dimBlock>>>(
			thrust::raw_pointer_cast(binIdxsCandidates.data()),
			thrust::raw_pointer_cast(dProjectedQueries.data()),
			thrust::raw_pointer_cast(dBinCodes.data()),
			Q, k, binsNumber
		);

		return binIdxsCandidates;
	}

	ThrustIntV QueryBinCalculator::calcIdxsCandidates(const ThrustUnsignedV& permutationIdxsVec, int binsNumber, int Q) {
		thrust::host_vector<unsigned> sortedPermutationIdxsForQueries(permutationIdxsVec);
		thrust::host_vector<int> binIdxsForQueries(Q, -1);

		// start at 1 cause we need the element at i-1
		for (int i = 1; i < binsNumber + Q; ++i) {

			if(isAQueryIdx(i, sortedPermutationIdxsForQueries, binsNumber)) {
				auto idx = getQueryIdx(i, sortedPermutationIdxsForQueries, binsNumber);
				auto binIdx = sortedPermutationIdxsForQueries[i-1];

				if (idxIsNotAboutQuery(binIdx, binsNumber)) {
					auto prevIdx = getQueryIdx(i - 1, sortedPermutationIdxsForQueries, binsNumber);
					binIdx = binIdxsForQueries[prevIdx];
				}
				binIdxsForQueries[idx] = binIdx;
			}
		}

		return ThrustIntV(binIdxsForQueries);
	}

	ThrustUnsignedV QueryBinCalculator::sortMatrixRows(const ThrustFloatV& matrix, int rows, int cols) {
		ThrustUnsignedV dSortedPermutationIdxsForQueries(rows);
		radixSortMatrix(matrix, rows, cols, dSortedPermutationIdxsForQueries);

		return dSortedPermutationIdxsForQueries;
	}

	ThrustFloatV QueryBinCalculator::concatenateBinCodesAndProjectedQueries(
		const ThrustFloatV& dProjectedQueries, int Q, int k,
		int binsNumber, const ThrustFloatV& dBinCodes
	) {
		ThrustFloatV concatenated((binsNumber + Q)*k);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((k + dimBlock.x - 1)/dimBlock.x, (binsNumber + Q + dimBlock.y - 1)/dimBlock.y);

		concatenateMatricesBelow<<<dimGrid, dimBlock>>>(
			thrust::raw_pointer_cast(dBinCodes.data()), thrust::raw_pointer_cast(dProjectedQueries.data()),
			binsNumber, Q, k,
			thrust::raw_pointer_cast(concatenated.data())
		);

		return concatenated;
	}

	bool QueryBinCalculator::isAQueryIdx(const int idx, const thrust::host_vector<unsigned>& concatenatedPermutationIdxs, const int binsNumber) {
		return concatenatedPermutationIdxs[idx] >= binsNumber;
	}

	int QueryBinCalculator::getQueryIdx(int idx, const thrust::host_vector<unsigned>& concatenatedPermutationIdxs, const int binsNumber) {
		return concatenatedPermutationIdxs[idx] - binsNumber;
	}

	bool QueryBinCalculator::idxIsNotAboutQuery(const int binIdx, const int binsNumber) {
		return binIdx >= binsNumber;
	}
} /* namespace cuANN */
