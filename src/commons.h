#ifndef COMMONS_H_
#define COMMONS_H_

#include <thrust/device_vector.h>

#define BLOCK_SIZE 16
#define BLOCK_SIZE_STRIDE_Y 8
#define BLOCK_SIZE_STRIDE_X 32

typedef thrust::device_vector<float> ThrustFloatV;
typedef thrust::device_vector<unsigned> ThrustUnsignedV;
typedef thrust::device_vector<int> ThrustIntV;
typedef thrust::device_vector<bool> ThrustBoolV;

typedef thrust::host_vector<int> ThrustHIntV;
typedef thrust::host_vector<unsigned> ThrustHUnsignedV;


#endif /* COMMONS_H_ */
