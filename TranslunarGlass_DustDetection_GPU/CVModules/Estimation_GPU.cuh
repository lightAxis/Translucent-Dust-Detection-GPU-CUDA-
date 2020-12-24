#pragma once

#include "cuda_runtime.h"


#ifndef ESTIMATION_GPU
#define ESTIMATION_GPU

class Estimation_CUDA
{
public:
	void DoMeanShift(bool* bImage, const int& rows, const int& cols,
		float* centers, const int& centerNums,
		const float& radius, const float& deadDist);

	void MemoryPreAllocate(const int& rows, const int& cols, const int& centerNums, const int& iterLimit);
private:
	bool* mK_bImage;
	int* mK_rows_Const;
	int* mK_cols_Const;

	float* mK_centers;
	int* mK_centerNums_Const;

	// x_start , x_end , y_start , y_end
	// [i*4 + 0], max i is centerNums
	int* mK_temp_range;
	// max i is centerNums
	float* mK_temp_n;
	// dot_x_sum , dot_y_sum
	// [i*2 + 0], max i is centerNums
	int* mK_temp_dots;
	float* mK_temp_dist;

	float* mK_radius_Const;
	float* mK_deadDist_Const;

	int* mK_IterLimit;

	void MemoryFree();
public:
	Estimation_CUDA(const int& GPUNumber = 0);
	~Estimation_CUDA();
};

#endif
