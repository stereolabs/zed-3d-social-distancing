#include "cuUtils.h"

#define BLOCKDIM_X 32
#define BLOCKDIM_Y 8


int iDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__ float cvtuchar4tofloat32b(uchar4 colorIn) {
    uint8_t r = colorIn.z;
    uint8_t g = colorIn.y;
    uint8_t b = colorIn.x;
    uint8_t a = 255;
    uint32_t color_u;
    color_u = (((((uint32_t) a << 24) | (uint32_t) b << 16) | (uint32_t) g << 8) | (uint32_t) r);
    return *(float*) (&color_u);
}

__global__ void ZImage2XYZRGBA(float4* bufferXYZ, float *depth, uchar4* Image, float focale_x, float focale_y, float cx, float cy,
    unsigned int bufferStep, unsigned int depthStep, unsigned int imageStep, unsigned int width, unsigned int height, float depthMax) {
    // les coordonnees locaux de [0 W] et [0 H]
    unsigned int x_local = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_local = blockIdx.y * blockDim.y + threadIdx.y;

    if(x_local >= width || y_local >= height) return;

    float depth_ = (depth[depthStep * y_local + x_local] < depthMax) ? depth[depthStep * y_local + x_local] : depthMax;
    float4 xyza;
    xyza.x = (x_local - cx) * depth_ /focale_x;
    xyza.y = (y_local - cy) * depth_ * -1.f /focale_y; // -1 for OpenGL
    xyza.z = depth_ * -1.f; // -1 for OpenGL
    xyza.w = cvtuchar4tofloat32b(Image[y_local*imageStep + x_local]);
    bufferXYZ[bufferStep * y_local + x_local] = xyza;
}

bool triangulateImageandZ(sl::Mat point_cloud, sl::Mat image, sl::Mat depth, sl::CameraParameters cam_params)
{
    if (point_cloud.getWidth()!= image.getWidth()
            || point_cloud.getWidth()!= depth.getWidth()
            || point_cloud.getHeight()!= image.getHeight()
            || point_cloud.getHeight() != depth.getHeight())
        return false;

    unsigned int width = image.getWidth();
    unsigned int height = image.getHeight();

    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(width, BLOCKDIM_X), iDivUp(height, BLOCKDIM_Y));

    int PC_Step = point_cloud.getStep(sl::MEM::GPU);
    int depthStep = depth.getStep(sl::MEM::GPU);
    int imStep = image.getStep(sl::MEM::GPU);

    float fx = cam_params.fx;
    float fy = cam_params.fy;
    float cx = cam_params.cx;
    float cy = cam_params.cy;

    float4* pc_buffer = (float4*)point_cloud.getPtr<sl::float4>(sl::MEM::GPU);
    float* depth_buffer = (float*)depth.getPtr<sl::float1>(sl::MEM::GPU);
    uchar4* image_buffer = (uchar4*)image.getPtr<sl::uchar4>(sl::MEM::GPU);


   ZImage2XYZRGBA<< <grid, threads, 0, 0 >> > ((::float4 *)pc_buffer, depth_buffer, (::uchar4*)image_buffer, fx, fy, cx, cy,PC_Step, depthStep, imStep, width, height,40 );

   return true;
}

