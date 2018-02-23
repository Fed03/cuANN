#include "QueryBinCalculator.h"
#include "utils.h"

namespace cuANN {
	ThrustIntV QueryBinCalculator::getBinsForQueryHashes(
		const ThrustSizetV& queryHashes, int Q,
		int binsNumber, const size_t* binCodes
	) {
		ThrustSizetV dBinCodes(binCodes, binCodes + binsNumber);

		auto concatenatedHashes = concatenateBinCodesAndQueryHashes(queryHashes, Q, binsNumber, dBinCodes);
		auto permutationIdxs = sortHashes(concatenatedHashes, binsNumber+Q);

		return calcBinIdxs(permutationIdxs, queryHashes, dBinCodes, Q, binsNumber);
	}

	ThrustIntV QueryBinCalculator::calcBinIdxs(
			const ThrustUnsignedV& permutationIdxsVec,
			const ThrustSizetV& queryHashes,
			const ThrustSizetV& dBinCodes,
			int Q, int binsNumber
	) {
		auto binIdxsCandidates = calcIdxsCandidates(permutationIdxsVec, binsNumber, Q);

		dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
		dim3 dimGrid((Q + dimBlock.x - 1)/dimBlock.x);
		getActualBinIdxs<<<dimGrid, dimBlock>>>(
			thrust::raw_pointer_cast(binIdxsCandidates.data()),
			thrust::raw_pointer_cast(queryHashes.data()),
			thrust::raw_pointer_cast(dBinCodes.data()),
			Q, binsNumber
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

	ThrustUnsignedV QueryBinCalculator::sortHashes(ThrustSizetV& hashes, unsigned size) {
		ThrustUnsignedV dSortedPermutationIdxsForQueries(size);
		thrust::sequence(dSortedPermutationIdxsForQueries.begin(), dSortedPermutationIdxsForQueries.end());

		thrust::stable_sort_by_key(hashes.begin(), hashes.end(), dSortedPermutationIdxsForQueries.begin());

		return dSortedPermutationIdxsForQueries;
	}

	ThrustSizetV QueryBinCalculator::concatenateBinCodesAndQueryHashes(
		const ThrustSizetV& queryHashes, int Q,
		int binsNumber, const ThrustSizetV& dBinCodes
	) {
		ThrustSizetV concatenated(binsNumber + Q);

		thrust::copy_n(dBinCodes.begin(), binsNumber, concatenated.begin());
		thrust::copy_n(queryHashes.begin(), Q, concatenated.begin() + binsNumber);

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
