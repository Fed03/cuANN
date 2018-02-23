#ifndef __cuANN_QUERYBINCALCULATOR_H_
#define __cuANN_QUERYBINCALCULATOR_H_

#include "commons.h"

namespace cuANN {

class QueryBinCalculator {
public:
	static ThrustIntV getBinsForQueryHashes(
		const ThrustSizetV& dProjectedQueries, int Q,
		int binsNumber, const size_t* binCodes
	);
private:
	QueryBinCalculator(){}

	static ThrustSizetV concatenateBinCodesAndQueryHashes(
		const ThrustSizetV& dProjectedQueries, int Q,
		int binsNumber, const ThrustSizetV& dBinCodes
	);

	static ThrustUnsignedV sortHashes(ThrustSizetV& hashes, unsigned size);

	static ThrustIntV calcBinIdxs(
			const ThrustUnsignedV& permutationIdxsVec,
			const ThrustSizetV& queryHahes,
			const ThrustSizetV& dBinCodes,
			int Q, int binsNumber
	);

	static ThrustIntV calcIdxsCandidates(const ThrustUnsignedV& permutationIdxsVec, int binsNumber, int Q);

	static bool isAQueryIdx(const int idx, const thrust::host_vector<unsigned>& concatenatedPermutationIdxs, const int binsNumber);

	static int getQueryIdx(int idx, const thrust::host_vector<unsigned>& concatenatedPermutationIdxs, const int binsNumber);

	static bool idxIsNotAboutQuery(const int binIdx, const int binsNumber);
};

} /* namespace cuANN */

#endif /* __cuANN_QUERYBINCALCULATOR_H_ */
