#ifndef __cuANN_QUERYRESULT_H__
#define __cuANN_QUERYRESULT_H__

namespace cuANN {
	struct QueryResult {
		unsigned queryIdx;
		std::vector<unsigned> resultIdx;
		unsigned resultSize;

		QueryResult(unsigned idx, std::vector<unsigned>&& result, unsigned numbersOfNeighbors) :
			queryIdx(idx), resultIdx(std::move(result)), resultSize(numbersOfNeighbors)
		{}
	};
}

#endif /* __cuANN_QUERYRESULT_H__ */
