#ifndef __cuANN_QUERYBINCALCULATOR_H_
#define __cuANN_QUERYBINCALCULATOR_H_

#include "commons.h"

namespace cuANN {

class QueryBinCalculator {
public:
	static ThrustIntV getBinsForProjectedQueries(
		const ThrustFloatV& dProjectedQueries, int Q, int k,
		int binsNumber, const float* binCodes
	);
private:
	QueryBinCalculator(){}

	static ThrustFloatV concatenateBinCodesAndProjectedQueries(
		const ThrustFloatV& dProjectedQueries, int Q, int k,
		int binsNumber, const ThrustFloatV& dBinCodes
	);

	static ThrustUnsignedV sortMatrixRows(const ThrustFloatV& matrix, int rows, int cols);

	static ThrustIntV calcBinIdxs(
			const ThrustUnsignedV& permutationIdxsVec,
			const ThrustFloatV& dProjectedQueries,
			const ThrustFloatV& dBinCodes,
			int Q, int k, int binsNumber
	);

	static ThrustIntV calcIdxsCandidates(const ThrustUnsignedV& permutationIdxsVec, int binsNumber, int Q);

	static bool isAQueryIdx(const int idx, const thrust::host_vector<unsigned>& concatenatedPermutationIdxs, const int binsNumber);

	static int getQueryIdx(int idx, const thrust::host_vector<unsigned>& concatenatedPermutationIdxs, const int binsNumber);

	static bool idxIsNotAboutQuery(const int binIdx, const int binsNumber);
};

} /* namespace cuANN */

#endif /* __cuANN_QUERYBINCALCULATOR_H_ */
