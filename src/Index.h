#ifndef __cuANN_INDEX_H_
#define __cuANN_INDEX_H_

#include "HashTable.h"
#include "Dataset.h"

namespace cuANN {
	class Index
	{
	public:
		Index(int k, int L, Dataset * data, float w);

		~Index();

		bool refresh(int k, int L, Dataset * data, float w);

		bool buildIndex();

		void query(Dataset* queries, unsigned numberOfNeighbors);

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
	};
}

#endif /* __cuANN_INDEX_H_ */
