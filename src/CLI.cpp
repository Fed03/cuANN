#ifndef __cuANN_CLI__
#define __cuANN_CLI__

#include <iostream>
#include "CLI.h"
#include "LSH.h"
#include "FvecsReader.h"

namespace cuANN {
	CLI::CLI(int argc, char** argv) : argcount(argc), argvalue(argv), argparser(getParser())
	{
	}

	CLI::~CLI()
	{
	}

	int CLI::startLSH() {
		argagg::parser_results args;
		try {
			args = argparser.parse(argcount, argvalue);
		}
		catch (const std::exception& e) {
			std::cerr << argparser << std::endl
				<< "Encountered exception while parsing arguments: " << e.what()
				<< std::endl;
			return EXIT_FAILURE;
		}

		if (!checkArgs(&args)) {
			std::cerr << "Check that all the args are provided" << std::endl;
			return EXIT_FAILURE;
		}

		try
		{
			Dataset * dataset = getDataset(args["dataset"]);
			Dataset * queries = getDataset(args["queries"], args["numberOfQueries"]);

			LSH lsh(args["hashFunc"], args["tables"], args["binWidth"], dataset);
			lsh.buildIndex();
			auto result = lsh.query(queries, args["neighbors"]);
			std::cout << "arrivato";
		}
		catch (const std::exception& e )
		{
			std::cerr << e.what();
			return EXIT_FAILURE;
		}


		return 0;
	}

	argagg::parser CLI::getParser()
	{
		argagg::parser argparser{{
			{ "dataset", { "--dataset" }, "The dataset file in .fvecs format", 1 },
			{ "queries", { "--queries" }, "The queries file in .fvecs format", 1 },
			{ "binWidth", { "-w" }, "", 1},
			{ "numberOfQueries", { "-q" }, "How many query vectors to load", 1 },
			{ "neighbors", { "-n" }, "How many neighbors to return per query", 1 },
			{ "tables", { "-L" }, "The number of hash tables.", 1 },
			{ "hashFunc", { "-k" }, "The number of hash functions used to project the dataset.", 1 }
		}};
		return argparser;
	}

	bool CLI::checkArgs(argagg::parser_results* args)
	{
		std::string requiredArgs[] = { "dataset", "queries", "numberOfQueries", "neighbors", "tables", "hashFunc", "binWidth" };
		for (const auto &argName : requiredArgs) {
			if (!(*args)[argName]) return false;
		}

		return true;
	}
	Dataset * CLI::getDataset(std::string filePath)
	{
		auto f = new FvecsReader(filePath);
		return f->readAllVectors();
	}

	Dataset * CLI::getDataset(std::string filePath, int howMany)
	{
		auto f = new FvecsReader(filePath);
		return f->readVectors(howMany);
	}
}

#endif // !__cuANN_CLI__
