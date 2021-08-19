# jetbot combine road following and collision avoidance
If we use the original nvidia model, GPU on jetbot cannot work successfully since it is unable to handle 2 model simultaneously.

By torch2trt library, I transform the original pytorch model to tensorrt model to raise performance, so that we can combine road following, endline parking, and collision avoidance.

After jetbot work well, I run a server on my laptop with Ubuntu 20.04 system, GeForce GTX 1060, NVIDIA driver 460, CUDA 10.1, cudnn 8.x, python 3.8.10, torch 1.5.0+cu101, torchvision 0.6.0+cu101, torch2trt 0.3.0, tensorrt 7.2.3.4, python flask 2.0.1.
