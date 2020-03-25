#ifndef DOF_GPU_H
#define DOF_GPU_H

/* dof_gpu.h.
 *
 * This file contains the interface to the CUDA functions ,
 * for rendering depth of field, based on Gaussian blurring
 * using separable convolution, with depth-dependent kernel size.
 * Separable convolution is based on convolution CUDA Sample with kernel-size adaptation
 */

#include "cuda.h"
#include "cuda_runtime.h"
#include <math.h> 
#include <algorithm>
#include <stdint.h>
#include <sl/Camera.hpp>


// Convert Depth map and Image to a point cloud
bool triangulateImageandZ(sl::Mat point_cloud, sl::Mat image, sl::Mat depth, sl::CameraParameters cam_params);

#endif //DOF_GPU_H
