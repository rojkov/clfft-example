#include <stdio.h>
#include <stdlib.h>

#include <clFFT.h>

#include "pgm.h"

#define MAX_SOURCE_SIZE (0x100000)
#define WORKSIZE 16

int setWorkSize(size_t *gws, size_t *lws, cl_int x, cl_int y) {
        switch (y) {
        case 1:
                gws[0] = x;
                gws[1] = 1;
                lws[0] = WORKSIZE;
                lws[1] = WORKSIZE;
                break;
        default:
                gws[0] = x;
                gws[1] = y;
                lws[0] = WORKSIZE;
                lws[1] = WORKSIZE;
                break;
        }

        return 0;
}

int main(void) {
        cl_int err;
        cl_platform_id platform = 0;
        cl_device_id device = 0;
        cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};
        cl_context ctx = 0;
        cl_kernel hpfl = 0;
        cl_program program = 0;
        cl_command_queue queue = 0;
        cl_mem bufX;
        float *X;
        int ret = 0;

        pgm_t ipgm;
        pgm_t opgm;

        FILE *fp;
        const char fileName[] = "./hpf.cl";
        size_t source_size;
        char *source_str;

        /* Load kernel source code */
        fp = fopen(fileName, "r");
        if (!fp) {
                fprintf(stderr, "Failed to load kernel.\n");
                exit(1);
        }
        source_str = (char *)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

        /* Read image */
        ret = readPGM(&ipgm, "lena.pgm");
        if (ret < 0) {
                fprintf(stderr, "Wrong input image format. Exiting...\n");
                exit(1);
        }

        const size_t N0 = ipgm.width, N1 = ipgm.width;
        char platform_name[128];
        char device_name[128];

        /* FFT library realted declarations */
        clfftPlanHandle planHandle;
        clfftDim dim = CLFFT_2D;
        size_t clLengths[2] = {N0, N1};

        /* Setup OpenCL environment. */
        err = clGetPlatformIDs(1, &platform, NULL);

        size_t ret_param_size = 0;
        err =
            clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
                              platform_name, &ret_param_size);
        printf("Platform found: %s\n", platform_name);

        err =
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

        err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                              device_name, &ret_param_size);
        printf("Device found on the above platform: %s\n", device_name);

        props[1] = (cl_context_properties)platform;
        ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
        queue = clCreateCommandQueue(ctx, device, 0, &err);

        /* Create kernel program from source */
        program = clCreateProgramWithSource(ctx, 1, (const char **)&source_str,
                                            (const size_t *)&source_size, &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create program. Error: %d\n", ret);
                exit(1);
        }

        /* Build kernel program */
        ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (ret != 0) {
                fprintf(stderr, "Can't build program. Error: %d\n", ret);
                exit(1);
        }

        /* Create OpenCL Kernel */
        hpfl = clCreateKernel(program, "highPassFilter", &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create kernel. Error: %d\n", ret);
                exit(1);
        }

        /* Setup clFFT. */
        clfftSetupData fftSetup;
        err = clfftInitSetupData(&fftSetup);
        err = clfftSetup(&fftSetup);

        /* Allocate host & initialize data. */
        /* Only allocation shown for simplicity. */
        size_t buffer_size = N0 * N1 * 2 * sizeof(*X);
        X = (float *)malloc(buffer_size);

        /* print input array just using the
         * indices to fill the array with data */
        printf("\nPerforming fft on an two dimensional array of size N0 x N1 : "
               "%lu x "
               "%lu\n",
               (unsigned long)N0, (unsigned long)N1);
        size_t i, j;

        for (i = 0; i < N0; ++i) {
                for (j = 0; j < N1; ++j) {
                        float x = ipgm.buf[j + i * N1];
                        float y = 0.0f;
                        size_t idx = 2 * (j + i * N1);
                        X[idx] = x;
                        X[idx + 1] = y;
                }
        }

        /* Prepare OpenCL memory objects and place data inside them. */
        bufX = clCreateBuffer(ctx, CL_MEM_READ_WRITE, buffer_size, NULL, &err);

        err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, buffer_size, X, 0,
                                   NULL, NULL);

        /* Create a default plan for a complex FFT. */
        err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);

        /* Set plan parameters. */
        err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
        err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED,
                             CLFFT_COMPLEX_INTERLEAVED);
        err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

        /* Bake the plan. */
        err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);

        /* Execute the plan. */
        err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0,
                                    NULL, NULL, &bufX, NULL, NULL);

        /* Apply high-pass filter */
        cl_int n = N0;
        cl_int radius = n / 8;
        size_t gws[2];
        size_t lws[2];
        ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&bufX);
        ret = clSetKernelArg(hpfl, 1, sizeof(cl_int), (void *)&n);
        ret = clSetKernelArg(hpfl, 2, sizeof(cl_int), (void *)&radius);
        setWorkSize(gws, lws, n, n);
        ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL,
                                     NULL);
        if (ret != 0) {
                fprintf(stderr, "Can't enqueue kernel. Error: %d\n", ret);
                exit(1);
        }

        /* Execute the plan for inverse FFT. */
        err = clfftEnqueueTransform(planHandle, CLFFT_BACKWARD, 1, &queue, 0,
                                    NULL, NULL, &bufX, NULL, NULL);

        /* Wait for calculations to be finished. */
        err = clFinish(queue);

        /* Fetch results of calculations. */
        err = clEnqueueReadBuffer(queue, bufX, CL_TRUE, 0, buffer_size, X, 0,
                                  NULL, NULL);

        float *ampd;
        ampd = (float *)malloc(N0 * N1 * sizeof(float));
        for (i = 0; i < N0; ++i) {
                for (j = 0; j < N1; ++j) {
                        size_t idx = 2 * (j + i * N1);
                        ampd[j + i * N1] =
                            sqrt(X[idx] * X[idx] + X[idx + 1] * X[idx + 1]);
                }
        }
        opgm.width = N0;
        opgm.height = N1;
        normalizeF2PGM(&opgm, ampd);
        free(ampd);
        writePGM(&opgm, "output.pgm");
	destroyPGM(&ipgm);
	destroyPGM(&opgm);

        /* Release OpenCL memory objects. */
        clReleaseMemObject(bufX);

        free(X);

        /* Release the plan. */
        err = clfftDestroyPlan(&planHandle);

        /* Release clFFT library. */
        clfftTeardown();

        /* Release OpenCL working objects. */
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);

        return ret;
}
