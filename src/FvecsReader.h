#ifndef __FvecsReader__
#define __FvecsReader__


#include <string>
#include <fstream>
#include "Dataset.h"

using namespace std;

class FvecsReader
{
public:
	FvecsReader(string fileName);
	
	~FvecsReader();

	cuANN::Dataset* readAllVectors();
	
	cuANN::Dataset* readVectors(int howMany);

private:
	ifstream fvecsFile;
	int vectorDimension;
	static constexpr int STEP_SIZE = 4;

	bool readNextVector(float *realVector);

	long long getFileSize();
};

FvecsReader::FvecsReader(string fileName) {
	fvecsFile = ifstream(fileName, ios::in | ios::binary);
	if (fvecsFile.fail())
	{
		throw runtime_error("The file " + fileName + " cannot be opened");
	}
	fvecsFile.read(reinterpret_cast<char*>(&vectorDimension), STEP_SIZE);

	fvecsFile.seekg(0);
}

FvecsReader::~FvecsReader() {
	fvecsFile.close();
}

cuANN::Dataset* FvecsReader::readAllVectors() {
	long long fileSize = getFileSize();
	int howMany = fileSize / ((vectorDimension + 1) * STEP_SIZE);
	return readVectors(howMany);
}

cuANN::Dataset* FvecsReader::readVectors(int howMany) {
	float * dataset;
	dataset = (float *)malloc(howMany * vectorDimension * sizeof(float));
	for (int i = 0; i < howMany; i++)
	{
		if (!readNextVector(dataset + (i *vectorDimension))) {
			throw runtime_error("Couldn't read the required number of vectors");
		}
	}

	// stored in row-major order. ld=vectorDimension
	return new cuANN::Dataset(dataset, howMany, vectorDimension, vectorDimension);
}

bool FvecsReader::readNextVector(float *realVector) {
	fvecsFile.seekg(STEP_SIZE, ios::cur);
	fvecsFile.read(reinterpret_cast<char*>(realVector), vectorDimension*STEP_SIZE);
	return (bool)fvecsFile;
}

long long FvecsReader::getFileSize() {
	streampos currentPosition = fvecsFile.tellg();
	streampos begin, end;

	fvecsFile.seekg(0, ios::beg);
	begin = fvecsFile.tellg();

	fvecsFile.seekg(0, ios::end);
	end = fvecsFile.tellg();

	fvecsFile.seekg(currentPosition, ios::beg);

	return end - begin;
}

#endif
