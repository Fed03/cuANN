#ifndef __cuANN_QUERYRESULT_H__
#define __cuANN_QUERYRESULT_H__

namespace cuANN {
	struct QueryResult {
		unsigned queryIdx;
		std::vector<unsigned> resultIdx;

		QueryResult(unsigned idx, std::vector<unsigned>&& result) : queryIdx(idx), resultIdx(std::move(result)){

		}
	};
}

#endif /* __cuANN_QUERYRESULT_H__ */
