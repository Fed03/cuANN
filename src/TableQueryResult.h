#ifndef TABLEQUERYRESULT_H_
#define TABLEQUERYRESULT_H_

#include "commons.h"

namespace cuANN {

	struct TableQueryResult {
		unsigned Q;
		unsigned resultSetSize;

		ThrustHUnsignedV resultStartingIdxs;
		ThrustHUnsignedV resultSizes;
		ThrustHUnsignedV resultSet;

		TableQueryResult(
			const ThrustHUnsignedV& resultStartingIdxs,
			const ThrustHUnsignedV& resultSizes,
			const ThrustHUnsignedV& resultSet,
			unsigned Q, unsigned resultSetSize
		);
	};

	inline TableQueryResult::TableQueryResult(
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


#endif /* TABLEQUERYRESULT_H_ */
