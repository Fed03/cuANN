#ifndef __cuANN_CLI_H__
#define __cuANN_CLI_H__

#include "argagg.hpp"
#include "Dataset.h"

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

		argagg::parser getParser();
		bool checkArgs(argagg::parser_results *args);
		Dataset * getDataset(std::string filePath);
	};
}

#endif // !__cuANN_CLI__
