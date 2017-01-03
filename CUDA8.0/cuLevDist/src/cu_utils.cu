#include "cu_utils.h"

__host__ __device__
long cu::pow2(const long val) {
	long result = 1;
	for (result = 1; result < val; result <<= 1);
	return result;
}
