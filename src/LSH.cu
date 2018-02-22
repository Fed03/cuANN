#ifndef __cuANN_LSH__
#define __cuANN_LSH__

#include "LSH.h"
#include "utils.h"

namespace cuANN {
	LSH::LSH(int k, int L, float w, Dataset* data) {
		this->dataset = prepareDataset(data);
		index = new Index(k, L, this->dataset, w);
	}

	LSH::~LSH(){
		delete dataset;
	}

	void LSH::buildIndex() {
		this->index->buildIndex();
	}

	std::vector<QueryResult> LSH::queryIndex(Dataset* queries, int numberOfNeighbors) {
		auto colMajorQueries = prepareDataset(queries);
		return index->query(colMajorQueries, numberOfNeighbors);
	}

	Dataset* LSH::prepareDataset(Dataset* dataset) {
		int rows = dataset->N;
		int cols = dataset->d;

		thrust::device_vector<float> srcDataset(dataset->dataset, dataset->dataset + rows * cols);
		thrust::device_vector<float> destDataset(rows * cols);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);
		convertMatrixFromRawMajorToColumnMajor<<<dimGrid, dimBlock >>>(
			thrust::raw_pointer_cast(srcDataset.data()),
			thrust::raw_pointer_cast(destDataset.data()),
			rows, cols
		);

		float* colMajorDataset = (float *)malloc(rows * cols * sizeof(float));
		thrust::copy(destDataset.begin(), destDataset.end(), colMajorDataset);

		return new Dataset(colMajorDataset, rows, cols, rows);
	}
}

#endif // !__cuANN_LSH__
