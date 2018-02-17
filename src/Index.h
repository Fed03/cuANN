#ifndef __cuANN_INDEX_H_
#define __cuANN_INDEX_H_

#include "HashTable.h"
#include "Dataset.h"
#include <vector>
#include "ThrustQueryResult.h"
#include "QueryResult.h"

namespace cuANN {
	class Index
	{
	public:
		Index(int k, int L, Dataset * data, float w);

		~Index();

		bool refresh(int k, int L, Dataset * data, float w);

		bool buildIndex();

		std::vector<QueryResult> query(Dataset* queries, unsigned numberOfNeighbors);

	private:
		Dataset * dataset;
		int k;
		int L;
		float w;
		int d;
		int N;

		std::vector<HashTable*> tables;

		void allocateProjectionMemory();

		void generateRandomProjections();

		void freeProjectionMemory();

		void sortDistancesAndTheirIdxs(ThrustFloatV& dDistances, ThrustUnsignedV& dCandidatesIdxs, const ThrustQueryResult* mergedResult);

		ThrustFloatV calculateDistances(const float* queries, unsigned Q, const ThrustUnsignedV& dCandidatesIdxs, const ThrustQueryResult* result);

		ThrustQueryResult* mergeQueryResults(thrust::host_vector<ThrustQueryResult*>& results, unsigned Q);

		unsigned getMaxCandidatesNumber(thrust::host_vector<ThrustQueryResult*>& results);
	};
}

#endif /* __cuANN_INDEX_H_ */
