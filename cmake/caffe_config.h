/* Sources directory */
#define SOURCE_FOLDER "/home/xyzeng/run_rcnn/caffe-mpi_parallel_v3"

/* Binaries directory */
#define BINARY_FOLDER "/home/xyzeng/run_rcnn/caffe-mpi_parallel_v3/cmake"

/* NVIDA Cuda */
#define HAVE_CUDA

/* NVIDA cuDNN */
#define HAVE_CUDNN
#define USE_CUDNN

/* NVIDA cuDNN */
/* #undef CPU_ONLY */

/* Test device */
#define CUDA_TEST_DEVICE -1

/* Temporary (TODO: remove) */
#if 1
  #define CMAKE_SOURCE_DIR SOURCE_FOLDER "/src/"
  #define EXAMPLES_SOURCE_DIR BINARY_FOLDER "/examples/"
  #define CMAKE_EXT ".gen.cmake"
#else
  #define CMAKE_SOURCE_DIR "src/"
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define CMAKE_EXT ""
#endif

/* Matlab */
/* #undef HAVE_MATLAB */
