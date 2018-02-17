#ifndef __cuANN_Index__
#define __cuANN_Index__

#include "commons.h"
#include "utils.h"
#include "Index.h"

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
		std::cout << "refreshing data" << std::endl;
		try
		{
			allocateProjectionMemory();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what();
			return false;
		}
		std::cout << "generating proj" << std::endl;
		generateRandomProjections();

		return true;
	}

	bool Index::buildIndex() {
		for(const auto &table : tables) {
			table->hashDataset(dataset->dataset, dataset->N);
		}
		return true;
	}

	std::vector<QueryResult> Index::query(Dataset* queries, unsigned numberOfNeighbors) {
		unsigned Q = queries->N;
		thrust::host_vector<ThrustQueryResult*> results;
		results.resize(L);

		for (const auto& table : tables) {
			results.push_back(table->query(queries->dataset, Q));
		}

		auto mergedResult = mergeQueryResults(results, Q);

		ThrustUnsignedV dCandidatesIdxs(mergedResult->resultSet);
		auto dDistances = calculateDistances(queries->dataset, Q, dCandidatesIdxs, mergedResult);
		sortDistancesAndTheirIdxs(dDistances, dCandidatesIdxs, mergedResult);

		std::vector<QueryResult> finalResult;
		for (unsigned query = 0; query < Q; ++query) {
			std::vector<unsigned> resultIdxsForQuery;
			thrust::copy_n(
				dCandidatesIdxs.begin() + mergedResult->resultStartingIdxs[query],
				std::min(numberOfNeighbors, mergedResult->resultSizes[query]),
				resultIdxsForQuery.begin()
			);
			finalResult.emplace_back(query, std::move(resultIdxsForQuery));
		}

		return finalResult;
	}

	void Index::sortDistancesAndTheirIdxs(ThrustFloatV& dDistances, ThrustUnsignedV& dCandidatesIdxs, const ThrustQueryResult* mergedResult) {
		ThrustUnsignedV dCandidatesStartingIdxs(mergedResult->resultStartingIdxs);
		ThrustUnsignedV dCandidatesSizes(mergedResult->resultSizes);

		unsigned Q = mergedResult->Q;
		for (int query = 0; query < Q; ++query) {
			thrust::sort_by_key(
				dDistances.begin() + dCandidatesStartingIdxs[query],
				dDistances.begin() + dCandidatesStartingIdxs[query] + dCandidatesSizes[query],
				dCandidatesIdxs.begin() + dCandidatesStartingIdxs[query],
				thrust::greater<float>()
			);
		}
	}

	ThrustFloatV Index::calculateDistances(const float* queries, unsigned Q, const ThrustUnsignedV& dCandidatesIdxs, const ThrustQueryResult* result) {
		unsigned distancesNumber = result->resultSetSize;
		ThrustFloatV dDistances(distancesNumber);

		ThrustUnsignedV dQueriesIdxsToCandidates(distancesNumber);
		for (int query = 0; query < Q; ++query) {
			thrust::fill_n(
				dQueriesIdxsToCandidates.begin() + result->resultStartingIdxs[query],
				result->resultSizes[query],
				query
			);
		}

		ThrustFloatV dQueries(queries, queries + Q * d);
		ThrustFloatV dDataset(dataset->dataset, dataset->dataset + N * d);

		dim3 dimBlock(BLOCK_SIZE_STRIDE_X, BLOCK_SIZE_STRIDE_Y);
		dim3 dimGrid((distancesNumber + dimBlock.x - 1)/ dimBlock.x);

		calcSquaredDistances<<<dimGrid, dimBlock>>>(
			thrust::raw_pointer_cast(dDataset.data()), N,
			thrust::raw_pointer_cast(dQueries.data()), Q,
			d,
			thrust::raw_pointer_cast(dCandidatesIdxs.data()),
			thrust::raw_pointer_cast(dQueriesIdxsToCandidates.data()),
			distancesNumber,
			thrust::raw_pointer_cast(dDistances.data())
		);

		return dDistances;
	}

	ThrustQueryResult* Index::mergeQueryResults(thrust::host_vector<ThrustQueryResult*>& results, unsigned Q) {
		unsigned maxCandidatesNumber = getMaxCandidatesNumber(results);

		ThrustHUnsignedV candidatesStartingIdxs(Q, 0);
		ThrustHUnsignedV candidatesSizes(Q, 0);
		ThrustHUnsignedV candidateIdxs(maxCandidatesNumber);

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

		auto totalCandidatesNumber = queryOffset;

		return new ThrustQueryResult(candidatesStartingIdxs, candidatesSizes, candidateIdxs, Q, totalCandidatesNumber);
	}

	unsigned Index::getMaxCandidatesNumber(thrust::host_vector<ThrustQueryResult*>& results) {
		thrust::host_vector<unsigned> resultsSizes(L);
		thrust::transform(results.begin(), results.end(), resultsSizes.begin(), [](ThrustQueryResult* q) {
			return q->resultSetSize;
		});
		return thrust::reduce(resultsSizes.begin(), resultsSizes.end());
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
