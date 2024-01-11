# Translucent-Dust-Detection-GPU-CUDA-
선행연구 : [Facade-Contaminant-Detection][https://github.com/lightAxis/Facade-Contaminant-Detection]

Using GPU library CUDA to accelerate processing time

## Contribution
Previous mean-shift process has many centroids together.
CPU processing cannot deal with it parallely.
Trying to use GPU to allocate each core a single centroid, process parallely.

Previous mean-shift process's centroids has no lifetime. Only if when the centroid is dead, it stops
This trait boders GPU processing time, as other centroids have to wait untill all centroids finish calculation.
So I made lifetime trait in mean-shift process. centroid dies when reached to maximum lifetime(maxIter) count.

## Result

![그림1](https://user-images.githubusercontent.com/62084431/103011839-f2b35680-457d-11eb-9e39-0d9bff6c4094.png)

Estimated Density
|maxLiftTime|Pic 1|Pic 2|Pic 3|
|:-:|:-:|:-:|:-:|
|Full|0.301|0.463|0.739|
|20  |0.301|0.463|0.738|
|15  |0.301|0.463|0.738|
|10  |0.300|0.463|0.732|
|8   |0.299|0.462|0.728|
|5   |0.295|0.459|0.719|


CPU Calculation Times(ms)
|maxLiftTime|Pic 1|Pic 2|Pic 3|
|:-:|:-:|:-:|:-:|
|Full|366|476|900|
|20  |337|476|747|
|15  |289|392|649|
|10  |245|327|508|
|8   |207|307|419|
|5   |145|196|283|

GPU Calculation Times(ms)
|maxLiftTime|Pic 1|Pic 2|Pic 3|
|:-:|:-:|:-:|:-:|
|Full|462|558|677|
|20  |282|290|237|
|15  |228|224|237|
|10  |163|163|167|
|8   |138|138|139|
|5   |97 |96 |98 |

![그림1](https://user-images.githubusercontent.com/62084431/103119057-044b4a00-46b5-11eb-96be-6d9b4bd88e7c.png)

![그림2](https://user-images.githubusercontent.com/62084431/103119058-057c7700-46b5-11eb-85b1-996f8a9a4ea2.png)


## Conclusion

Select maxLifeTime value of 10:
Reduces processing time when use CPU: 900ms -> 508 ms
Reduces processing time when use GPU: 900ms(CPU) -> 167(GPU)ms

Get 5.4 times Faster processing time than previous study


You can change maxLiftTime to trade-off processing time <-> estimation accuracy
In case of maxLifeTime=5, reduces 900ms -> 98ms. But estimation accuracy error rises to 2%

## Environment

OpenCV 4.01 / C++ / CUDA 10.1 / Visual Studio 2019

CPU - Intel i7-8750H @ 2.20GHz

GPU - GTX 1060
