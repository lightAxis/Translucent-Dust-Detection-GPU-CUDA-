
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <iostream>
#include <chrono>

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

__global__ void meanShiftKernal(bool* bImage, int* rows, int* cols,
	float* centers, int* centerNums,
	int* temp_range, int* dots, float* n, float* dist,
	float* radius, float* deadDist, int* iterLimit);


Estimation_CUDA::Estimation_CUDA(const int& GPUNumber)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}

Estimation_CUDA::~Estimation_CUDA()
{
	MemoryFree();

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

void Estimation_CUDA::MemoryPreAllocate(const int& rows, const int& cols, const int& centerNums, const int& iterLimit)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	//메모리 할당
	cudaMalloc(&mK_bImage, rows * cols * sizeof(bool));
	cudaMalloc(&mK_rows_Const, sizeof(int));
	cudaMalloc(&mK_cols_Const, sizeof(int));

	cudaMalloc(&mK_centers, centerNums * 5 * sizeof(float));
	cudaMalloc(&mK_centerNums_Const, sizeof(float));

	cudaMalloc(&mK_temp_range, centerNums * 4 * sizeof(int));
	cudaMalloc(&mK_temp_n, centerNums * sizeof(float));
	cudaMalloc(&mK_temp_dots, centerNums * 2 * sizeof(int));
	cudaMalloc(&mK_temp_dist, centerNums * sizeof(float));

	cudaMalloc(&mK_IterLimit, sizeof(int));
	cudaMemcpy(mK_IterLimit, &iterLimit, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&mK_radius_Const, sizeof(float));
	cudaMalloc(&mK_deadDist_Const, sizeof(float));

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time difference(mem allocate) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

}

void Estimation_CUDA::MemoryFree()
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//GPU 메모리 해제
	cudaFree(mK_bImage);
	cudaFree(mK_rows_Const);
	cudaFree(mK_cols_Const);

	cudaFree(mK_centers);
	cudaFree(mK_centerNums_Const);

	cudaFree(mK_temp_range);
	cudaFree(mK_temp_dots);
	cudaFree(mK_temp_n);
	cudaFree(mK_temp_dist);

	cudaFree(mK_IterLimit);

	cudaFree(mK_radius_Const);
	cudaFree(mK_deadDist_Const);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time difference(mem free) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

void Estimation_CUDA::DoMeanShift(bool* bImage, const int& rows, const int& cols,
	float* centers, const int& centerNums,
	const float& radius, const float& deadDist)
{

	//메모리 초기화
	cudaMemcpy(mK_bImage, bImage, rows * cols * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(mK_rows_Const, &rows, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(mK_cols_Const, &cols, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(mK_centers, centers, centerNums * 5 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(mK_centerNums_Const, &centerNums, sizeof(float), cudaMemcpyHostToDevice);

	//cudaMemcpy(K_temp_range, temp_range,centerNums * 4 * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(K_temp_n, temp_n, centerNums * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(K_temp_dots,temp_dots, centerNums * 2 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(mK_radius_Const, &radius, sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(mK_deadDist_Const, &deadDist, sizeof(float),cudaMemcpyHostToDevice);


	//연산 그리드 및 쓰레드 개수 설정
	const int ThreadNum = 64;
	const int BlockNum = (centerNums / 64) +1;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//GPU 커널 실행
	meanShiftKernal <<<BlockNum, ThreadNum>>>(mK_bImage, mK_rows_Const, mK_cols_Const,
		mK_centers, mK_centerNums_Const, 
		mK_temp_range, mK_temp_dots, mK_temp_n, mK_temp_dist,
		mK_radius_Const, mK_deadDist_Const, mK_IterLimit);


	

	//계산 완료된 메모리 호스트로 복사
	cudaMemcpy(centers, mK_centers, centerNums * 5 * sizeof(float), cudaMemcpyDeviceToHost);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time difference(gpu kernel) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

}

__global__ void meanShiftKernal(bool* bImage, int* rows, int* cols,
	float* centers, int* centerNums, 
	int* temp_range, int* dots, float* n, float* dist,
	float* radius, float* deadDist, int* iterLimit)
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int iter = 0;
	temp_range[i * 4 + 0] = 0;
	temp_range[i * 4 + 1] = 0;
	temp_range[i * 4 + 2] = 0;
	temp_range[i * 4 + 3] = 0;

	dist[i] = 0.0f;

	dots[i * 2 + 0] = 0;
	dots[i * 2 + 1] = 0;

	n[i] = 0.0f;



	if (i < *centerNums)
	{
		while (true)
		{
			//initialize temp_ranges;
			temp_range[i * 4 + 0] = (int)(centers[i * 5 + 2] - *radius);
			temp_range[i * 4 + 1] = (int)(centers[i * 5 + 2] + *radius);
			temp_range[i * 4 + 2] = (int)(centers[i * 5 + 3] - *radius);
			temp_range[i * 4 + 3] = (int)(centers[i * 5 + 3] + *radius);


			if (temp_range[i * 4 + 0] < 0) temp_range[i * 4 + 0] = 0;
			if (temp_range[i * 4 + 1] > *cols) temp_range[i * 4 + 1] = *cols;
			if (temp_range[i * 4 + 2] < 0) temp_range[i * 4 + 2] = 0;
			if (temp_range[i * 4 + 3] > *rows) temp_range[i * 4 + 3] = *rows;
		
			dots[i * 2 + 0] = 0;
			dots[i * 2 + 1] = 0;
			n[i] = 0.0f;
			for (int r = temp_range[i * 4 + 2]; r < temp_range[i * 4 + 3]; r++)
			{
				for (int c = temp_range[i * 4 + 0]; c < temp_range[i * 4 + 1]; c++)
				{

					if (bImage[r * (*cols) + c] == true)
					{
						dist[i] = hypotf(centers[i * 5 + 2] - c, centers[i * 5 + 3] - r);
						if (dist[i] < *radius)
						{
							dots[i * 2 + 0] += c;
							dots[i * 2 + 1] += r;
							n[i]++;
						}
					}
					
				}
			}
			
			if (n[i] > 0)
			{
				centers[i * 5 + 0] = dots[i * 2 + 0] / n[i];
				centers[i * 5 + 1] = dots[i * 2 + 1] / n[i];
			}
			

			dist[i] = hypotf(centers[i * 5 + 0] - centers[i * 5 + 2], centers[i * 5 + 1] - centers[i * 5 + 3]);
			if (dist[i] < *deadDist)
			{
				centers[i * 5 + 4] = n[i] / ((*radius) * (*radius) * (3.14159f));
				break;
			}
			else
			{
				centers[i * 5 + 2] = centers[i * 5 + 0];
				centers[i * 5 + 3] = centers[i * 5 + 1];
			}
			
			iter++;
			if (iter > * iterLimit)
			{
				centers[i * 5 + 4] = n[i] / ((*radius) * (*radius) * (3.14159f));
				break;
			}
			

		}
	}
	
	
}
