#ifndef __cuANN_LSH__
#define __cuANN_LSH__

#include "LSH.h"
#include "utils.h"

namespace cuANN {
	LSH::LSH(int k, int L, float w, Dataset* data) {
		this->dataset = data;
		index = new Index(k, L, this->dataset, w);
	}

	LSH::~LSH(){
		delete dataset;
	}

	void LSH::buildIndex() {
		this->index->buildIndex();
	}

	std::vector<QueryResult> LSH::queryIndex(Dataset* queries, int numberOfNeighbors) {
		return index->query(queries, numberOfNeighbors);
	}
}

#endif // !__cuANN_LSH__
