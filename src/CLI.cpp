#ifndef __cuANN_CLI__
#define __cuANN_CLI__

#include <iostream>
#include <iomanip>
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
			std::string datasetFilePath = args["dataset"];
			std::string queriesFilePath = args["queries"];
			int numberOfQueries = args["numberOfQueries"];
			int numberOfHashFuncs = args["hashFunc"];
			int numberOfProjTables = args["tables"];
			int numberOfNeighbors = args["neighbors"];
			float binWidth = args["binWidth"];

			Dataset * dataset = getDataset(datasetFilePath);
			Dataset * queries = getDataset(queriesFilePath, numberOfQueries);

			LSH lsh(numberOfHashFuncs, numberOfProjTables, binWidth, dataset);
			lsh.buildIndex();
			auto results = lsh.queryIndex(queries, numberOfNeighbors);
			printResults(results);
		}
		catch (const std::exception& e )
		{
			std::cerr << e.what();
			return EXIT_FAILURE;
		}


		return 0;
	}

	void CLI::printResults(const std::vector<QueryResult>& results) {
		for(const auto& result : results) {
			std::cout << "Query idx: " << result.queryIdx << std::endl;
			std::cout << "Result idx" << std::endl;
			for(const auto& id : result.resultIdx) {
				std::cout << std::right << std::setw(10) << id << std::endl;
			}
			std::cout << "==============" << std::endl;
		}
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
