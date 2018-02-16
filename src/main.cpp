#ifndef __cuANN_main__
#define __cuANN_main__

#include "CLI.h"

int main(int argc, char** argv) {
	cuANN::CLI cli(argc, argv);

	auto result = cli.startLSH();
	return result;
}

#endif // !__cuANN_main__
