#ifndef __cuANN_Dataset__
#define __cuANN_Dataset__

#include <cstdlib>

namespace cuANN {
	struct Dataset
	{
		Dataset(float* dataset, int N, int d, int ld);

		~Dataset();

		float * dataset;
		int N;
		int d;
		int ld;
	};

	inline Dataset::Dataset(float* dataset, int N, int d, int ld) {
		this->dataset = dataset;
		this->N = N;
		this->d = d;
		this->ld = ld;
	}

	inline Dataset::~Dataset() {
		free(dataset);
	}
}

#endif
