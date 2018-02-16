#ifndef __cuANN_Index__
#define __cuANN_Index__

#include "commons.h"
#include "Index.h"
#include "TableQueryResult.h"

namespace cuANN {
	Index::Index(int k, int L, Dataset * data, float w) {
		this->k = 0;
		this->L = 0;
		this->dataset = 0;
		this->w = 0.0;
		this->d = 0;
		this->N = 0;

		this->tables = std::vector<HashTable*>();
		refresh(k, L, data, w);
	};

	Index::~Index() {
		freeProjectionMemory();
		tables.clear();
	}

	bool Index::refresh(int k, int L, Dataset * data, float w) {
		this->k = k;
		this->L = L;
		this->dataset = data;
		this->w = w;

		this->d = data->d;
		this->N = data->N;
		this->tables.resize(this->L);

		try
		{
			allocateProjectionMemory();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what();
			return false;
		}

		generateRandomProjections();

		return true;
	}

	bool Index::buildIndex() {
		for(const auto &table : tables) {
			table->hashDataset(dataset->dataset, dataset->N);
		}
		return true;
	}

	void Index::query(Dataset* queries, unsigned numberOfNeighbors) {
		unsigned Q = queries->N;
		std::vector<TableQueryResult*> results;
		results.resize(L);

		unsigned totalCandidatesNumber = 0;
		for (const auto& table : tables) {
			TableQueryResult* result = table->query(queries->dataset, Q);
			totalCandidatesNumber += result->resultSetSize;
			results.push_back(result);
		}

		ThrustHUnsignedV candidatesStartingIdxs(Q, 0);
		ThrustHUnsignedV candidatesSizes(Q, 0);
		ThrustHUnsignedV candidateIdxs(totalCandidatesNumber);

		unsigned queryOffset = 0;
		for (int query = 0; query < Q; ++query) {
			candidatesStartingIdxs[query] = queryOffset;
			for(const auto& tableResult: results) {
				thrust::copy_n(
					tableResult->resultSet.begin() + tableResult->resultStartingIdxs[query],
					tableResult->resultSizes[query],
					candidateIdxs.begin() + candidatesStartingIdxs[query] + candidatesSizes[query]
				);
				candidatesSizes[query] += tableResult->resultSizes[query];
			}

			auto candidatesForQueryBegin = candidateIdxs.begin() + candidatesStartingIdxs[query];
			auto candidatesForQueryEnd = candidatesForQueryBegin + candidatesSizes[query];
			thrust::sort(candidatesForQueryBegin, candidatesForQueryEnd);

			candidatesForQueryEnd = thrust::unique(candidatesForQueryBegin, candidatesForQueryEnd);
			candidatesSizes[query] = thrust::distance(candidatesForQueryBegin, candidatesForQueryEnd);
			queryOffset += candidatesSizes[query];
		}
	}

	void Index::allocateProjectionMemory() {
		for (int i = 0; i < L; i++)
		{
			auto table = new HashTable(k, d, w);
			table->allocateProjectionMemory();
			tables.push_back(std::move(table));
		}
	}

	void Index::generateRandomProjections() {
		curandGenerator_t uniform;
		curandGenerator_t normal;

		/*CURAND_CALL(curandCreateGenerator(&uniform, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandCreateGenerator(&normal, CURAND_RNG_PSEUDO_DEFAULT));*/
		curandCreateGenerator(&uniform, CURAND_RNG_PSEUDO_DEFAULT);
		curandCreateGenerator(&normal, CURAND_RNG_PSEUDO_DEFAULT);

		curandSetPseudoRandomGeneratorSeed(uniform, (unsigned long long) time(0));
		curandSetPseudoRandomGeneratorSeed(normal, (unsigned long long) time(0));

		for (const auto &table : tables) {
			table->generateProjection(&normal, &uniform);
		}

		curandDestroyGenerator(normal);
		curandDestroyGenerator(uniform);
	}

	void Index::freeProjectionMemory() {
		for (const auto &table : tables) {
			table->freeMemory();
		}
	}
}

#endif // !__cuANN_Index__
