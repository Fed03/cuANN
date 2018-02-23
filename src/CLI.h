#ifndef __cuANN_CLI_H__
#define __cuANN_CLI_H__

#include <vector>
#include "QueryResult.h"
#include "argagg.hpp"
#include "Dataset.h"

using namespace std;

namespace cuANN {
	class CLI
	{
	public:
		CLI(int argc, char** argv);
		~CLI();

		int startLSH();

	private:
		argagg::parser argparser;
		int argcount;
		char** argvalue;
		vector<vector<int>> groundtruthIdxs;

		argagg::parser getParser();
		bool checkArgs(argagg::parser_results *args);
		Dataset * getDataset(std::string filePath);
		Dataset * getDataset(std::string filePath, int howMany);
		vector<vector<int>> loadGroundTruthIdxs(std::string filePath, int howMany);

		void printResults(const std::vector<QueryResult>& results);
	};
}

#endif // !__cuANN_CLI__
