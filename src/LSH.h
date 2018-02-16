#ifndef __cuANN_LSH_H_
#define __cuANN_LSH_H_

#include "Dataset.h"
#include "Index.h"

namespace cuANN {
	class LSH
	{
	public:
		LSH(int k, int L, float w, Dataset* data);
		LSH(const cuANN::LSH &) = delete;
		~LSH();

		void buildIndex();

		void query(Dataset* queries, unsigned numberOfNeighbors);

	private:
		Dataset * dataset;
		Index* index;

		Dataset* prepareDataset(Dataset* dataset);
	};
}

#endif /* __cuANN_LSH_H_ */
