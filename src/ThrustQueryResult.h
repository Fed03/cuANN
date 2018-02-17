#ifndef __cuANN_THRUSTQUERYRESULT_H__
#define __cuANN_THRUSTQUERYRESULT_H__

#include "commons.h"

namespace cuANN {

	struct ThrustQueryResult {
		unsigned Q;
		unsigned resultSetSize;

		ThrustHUnsignedV resultStartingIdxs;
		ThrustHUnsignedV resultSizes;
		ThrustHUnsignedV resultSet;

		ThrustQueryResult(
			const ThrustHUnsignedV& resultStartingIdxs,
			const ThrustHUnsignedV& resultSizes,
			const ThrustHUnsignedV& resultSet,
			unsigned Q, unsigned resultSetSize
		);
	};

	inline ThrustQueryResult::ThrustQueryResult(
		const ThrustHUnsignedV& resultStartingIdxs,
		const ThrustHUnsignedV& resultSizes,
		const ThrustHUnsignedV& resultSet,
		unsigned Q, unsigned resultSetSize
	) {
		this->Q = Q;
		this->resultSetSize = resultSetSize;
		this->resultSizes = resultSizes;
		this->resultSet = resultSet;
		this->resultStartingIdxs = resultStartingIdxs;
	}

}  // namespace cuANN


#endif /* __cuANN_THRUSTQUERYRESULT_H__ */
