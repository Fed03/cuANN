#ifndef __IvecsReader__
#define __IvecsReader__


#include <string>
#include <fstream>
#include <vector>

using namespace std;

class IvecsReader
{
public:
	IvecsReader(string fileName);
	
	~IvecsReader();

	vector<vector<int>> readAllGroundTruthIdxs();
	
	vector<vector<int>> readGroundTruthIdxs(int howMany);

private:
	ifstream ivecsFile;
	int vectorDimension;
	static constexpr int STEP_SIZE = 4;

	bool nextGroundTruthIdx(int* idxs);

	long long getFileSize();
};

IvecsReader::IvecsReader(string fileName) {
	ivecsFile = ifstream(fileName, ios::in | ios::binary);
	if (ivecsFile.fail())
	{
		throw runtime_error("The file " + fileName + " cannot be opened");
	}
	ivecsFile.read(reinterpret_cast<char*>(&vectorDimension), STEP_SIZE);

	ivecsFile.seekg(0);
}

IvecsReader::~IvecsReader() {
	ivecsFile.close();
}

vector<vector<int>> IvecsReader::readAllGroundTruthIdxs() {
	long long fileSize = getFileSize();
	int howMany = fileSize / ((vectorDimension + 1) * STEP_SIZE);
	return readGroundTruthIdxs(howMany);
}

vector<vector<int>> IvecsReader::readGroundTruthIdxs(int howMany) {
	vector<vector<int>> groundTruthIdxs;
	int* nextIdxs = (int*) malloc(vectorDimension* sizeof(int));
	for (int i = 0; i < howMany; i++)
	{
		if (!nextGroundTruthIdx(nextIdxs)) {
			throw runtime_error("Couldn't read the required number of vectors");
		}
		groundTruthIdxs.emplace_back(nextIdxs, nextIdxs + vectorDimension);
	}

	free(nextIdxs);

	return groundTruthIdxs;
}

bool IvecsReader::nextGroundTruthIdx(int* idxs) {
	ivecsFile.seekg(STEP_SIZE, ios::cur);
	ivecsFile.read(reinterpret_cast<char*>(idxs), vectorDimension*STEP_SIZE);
	return (bool)ivecsFile;
}

long long IvecsReader::getFileSize() {
	streampos currentPosition = ivecsFile.tellg();
	streampos begin, end;

	ivecsFile.seekg(0, ios::beg);
	begin = ivecsFile.tellg();

	ivecsFile.seekg(0, ios::end);
	end = ivecsFile.tellg();

	ivecsFile.seekg(currentPosition, ios::beg);

	return end - begin;
}

#endif
