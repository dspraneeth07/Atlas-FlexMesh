#pragma once
// =============================================================================
// CORE/gpu_kernels_cuda.hpp
// CUDA device kernel stubs and lightweight GPU utilities used as an
// optional acceleration layer. The header provides:
//  - simple memory management helpers with CPU fallback
//  - illustrative CUDA kernels for dense and sparse primitives
//  - a thin cuBLAS/cuSPARSE wrapper for batched and SpMV operations
//
// This file is optional: it requires the NVIDIA CUDA toolkit and an
// appropriate GPU compute capability to enable device-backed paths.
// When `ENABLE_CUDA` is not defined, the header provides safe CPU
// fallbacks with identical APIs to simplify integration.
// =============================================================================

#ifdef ENABLE_CUDA
    #include <cuda_runtime.h>
    #include <cublas_  .h>
    #include <cusparse_  .h>
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

#include <cstdio>
#include <vector>

namespace atlas::core {

// ===========================================================================
// GPU MEMORY MANAGEMENT
// ===========================================================================

class GPUMemoryManager {
#ifdef ENABLE_CUDA
public:
    static double* allocate_gpu(size_t n_elements) noexcept {
        double* d_ptr = nullptr;
        cudaMalloc(&d_ptr, n_elements * sizeof(double));
        if (!d_ptr) {
            std::fprintf(stderr, "ERROR: GPU allocation failed (%zu elements)\n", n_elements);
        }
        return d_ptr;
    }
    
    static void deallocate_gpu(double* d_ptr) noexcept {
        if (d_ptr) cudaFree(d_ptr);
    }
    
    static void copy_to_gpu(
        double* d_dest,
        const double* h_src,
        size_t n_elements) noexcept
    {
        cudaMemcpy(d_dest, h_src, n_elements * sizeof(double), 
                  cudaMemcpyHostToDevice);
    }
    
    static void copy_to_host(
        double* h_dest,
        const double* d_src,
        size_t n_elements) noexcept
    {
        cudaMemcpy(h_dest, d_src, n_elements * sizeof(double), 
                  cudaMemcpyDeviceToHost);
    }
    
    static void synchronize() noexcept {
        cudaDeviceSynchronize();
    }

#else
public:
    static double* allocate_gpu(size_t n_elements) noexcept {
        // CPU fallback: allocate host memory
        return new double[n_elements];
    }
    
    static void deallocate_gpu(double* ptr) noexcept {
        delete[] ptr;
    }
    
    static void copy_to_gpu(
        double* dest,
        const double* src,
        size_t n_elements) noexcept
    {
        std::copy(src, src + n_elements, dest);
    }
    
    static void copy_to_host(
        double* dest,
        const double* src,
        size_t n_elements) noexcept
    {
        std::copy(src, src + n_elements, dest);
    }
    
    static void synchronize() noexcept {}

#endif
};

// ===========================================================================
// CUDA KERNELS (Elementary operations)
// ===========================================================================

#ifdef ENABLE_CUDA

__global__ void kernel_matrix_multiply(
    const double* A,    // n×m matrix
    const double* B,    // m×p matrix
    double* C,          // n×p matrix
    int n, int m, int p)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n && j < p) {
        double sum = 0.0;
        for (int k=0; k<m; ++k) {
            sum += A[i*m + k] * B[k*p + j];
        }
        C[i*p + j] = sum;
    }
}

__global__ void kernel_matrix_vector(
    const double* A,    // n×n matrix
    const double* x,    // n vector
    double* y,          // n vector
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double sum = 0.0;
        for (int j=0; j<n; ++j) {
            sum += A[i*n + j] * x[j];
        }
        y[i] = sum;
    }
}

__global__ void kernel_axpy(
    double alpha,
    const double* x,
    double* y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += alpha * x[i];
    }
}

__global__ void kernel_dot_product(
    const double* x,
    const double* y,
    double* result,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double sdata[256];
    
    sdata[threadIdx.x] = 0.0;
    if (i < n) {
        sdata[threadIdx.x] = x[i] * y[i];
    }
    __syncthreads();
    
    // Reduction within block
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(result, sdata[0]);
    }
}

#endif

// ===========================================================================
// GPU LINEAR ALGEBRA WRAPPER
// ===========================================================================

class GPULinearAlgebra {
#ifdef ENABLE_CUDA
private:
    cublasHandle_t cublas_handle = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;
    
public:
    GPULinearAlgebra() {
        cublasCreate(&cublas_handle);
        cusparseCreate(&cusparse_handle);
    }
    
    ~GPULinearAlgebra() {
        if (cublas_handle) cublasDestroy(cublas_handle);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
    }
    
    // Batched matrix multiply: C = A·B (multiple instances)
    void batch_matrix_multiply(
        int n_batch,
        int m,
        const double* d_A,  // Batch of m×m matrices
        const double* d_B,
        double* d_C) noexcept
    {
        double alpha = 1.0, beta = 0.0;
        
        // Use cuBLAS batched gemm if available
        std::vector<const double*> d_A_ptrs(n_batch);
        std::vector<const double*> d_B_ptrs(n_batch);
        std::vector<double*> d_C_ptrs(n_batch);
        
        for (int i=0; i<n_batch; ++i) {
            d_A_ptrs[i] = d_A + i*m*m;
            d_B_ptrs[i] = d_B + i*m*m;
            d_C_ptrs[i] = d_C + i*m*m;
        }
        
        // Would call cublasDgemmBatched here
        // (Requires CUDA 8.0+)
    }
    
    // Sparse matrix-vector product on GPU
    void sparse_matrix_vector(
        int n,
        int nnz,
        const double* d_values,
        const int* d_col_indices,
        const int* d_row_offsets,
        const double* d_x,
        double* d_y) noexcept
    {
        // Use cuSPARSE
        cusparseSpMatDescr_t mat_A;
        cusparseDnVecDescr_t vec_x, vec_y;
        
        cusparseCreateCsr(&mat_A, n, n, nnz,
                         (void*)d_row_offsets,
                         (void*)d_col_indices,
                         (void*)d_values,
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        
        cusparseCreateDnVec(&vec_x, n, (void*)d_x, CUDA_R_64F);
        cusparseCreateDnVec(&vec_y, n, (void*)d_y, CUDA_R_64F);
        
        double alpha = 1.0, beta = 0.0;
        
        size_t buffer_size = 0;
        cusparseSpMV_bufferSize(cusparse_handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, mat_A, vec_x, &beta, vec_y,
                               CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                               &buffer_size);
        
        void* d_buffer = nullptr;
        cudaMalloc(&d_buffer, buffer_size);
        
        cusparseSpMV(cusparse_handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, mat_A, vec_x, &beta, vec_y,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    d_buffer);
        
        cudaFree(d_buffer);
        cusparseDestroySpMat(mat_A);
        cusparseDestroyDnVec(vec_x);
        cusparseDestroyDnVec(vec_y);
    }

#else
public:
    void batch_matrix_multiply(
        int, int,
        const double*, const double*, double*) noexcept {}
    
    void sparse_matrix_vector(
        int, int,
        const double*, const int*, const int*,
        const double*, double*) noexcept {}

#endif
};

// ===========================================================================
// BATCH ELEMENT ASSEMBLY ON GPU (Advanced)
// ===========================================================================

#ifdef ENABLE_CUDA

__global__ void kernel_assemble_element_residuals(
    const double* u_nodal,      // Global displacement
    const int* elem_nodes,      // Connectivity
    double* elem_residuals,     // Output: element residuals
    int n_elements)
{
    int elem_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    if (elem_id >= n_elements) return;
    
    // Each warp processes one element
    // Thread 0-3: load element nodes
    // Thread 4-11: compute local stiffness
    // Thread 12-15: assemble to global
    
    int node_ids[4];
    if (thread_id < 4) {
        node_ids[thread_id] = elem_nodes[elem_id*4 + thread_id];
    }
    __syncwarp();
    
    // Load nodal displacements
    double u[12];  // 4 nodes × 3 DOFs
    if (thread_id < 12) {
        int node = node_ids[thread_id / 3];
        int dof = thread_id % 3;
        u[thread_id] = u_nodal[node*3 + dof];
    }
    __syncwarp();
    
    // Compute element residual (simplified)
    if (thread_id < 12) {
        elem_residuals[elem_id*12 + thread_id] = 0.0;  // Placeholder
    }
}

#endif

// ===========================================================================
// GPU SOLVER INTERFACE
// ===========================================================================

class GPUSolver {
public:
    bool is_gpu_available() const noexcept {
#ifdef ENABLE_CUDA
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        return device_count > 0;
#else
        return false;
#endif
    }
    
    void print_gpu_info() const noexcept {
#ifdef ENABLE_CUDA
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        
        if (device_count > 0) {
            std::printf("✓ GPU available: %d device(s)\n", device_count);
            
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::printf("  Device 0: %s\n", prop.name);
            std::printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            std::printf("  Global memory: %.1f GB\n", 
                prop.totalGlobalMem / 1e9);
        } else {
            std::printf("⊘ No GPU detected\n");
        }
#else
        std::printf("⊘ CUDA not enabled (recompile with -DENABLE_CUDA)\n");
#endif
    }
};

} // namespace atlas::core
