#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cooperative_groups.h>

// --- Utility Macros ---
#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)
#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define LIMIT2(a, ra, b, rb) (((a)<(ra)) && ((b)<(rb)))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CFLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// --- Kernel Configuration ---
constexpr int WARPSIZE = 32;

// Block Tiling dimensions
constexpr int BM = 128;
constexpr int BK = 16;
constexpr int BN = 128;

// Warp Tiling dimensions
constexpr int WM = 64;
constexpr int WN = 64;

// Sub-Warp Tiling dimensions (for register blocking)
constexpr int WSUBM = 64;
constexpr int WSUBN = 16;

// Thread Tiling dimensions
constexpr int TM = 8;
constexpr int TN = 4;

// --- Static Asserts for Configuration Validation ---
static_assert(BM % WM == 0, "Block M must be a multiple of Warp M.");
static_assert(BN % WN == 0, "Block N must be a multiple of Warp N.");
static_assert(WM % WSUBM == 0, "Warp M must be a multiple of Sub-Warp M.");
static_assert(WN % WSUBN == 0, "Warp N must be a multiple of Sub-Warp N.");
static_assert(WSUBM % TM == 0, "Sub-Warp M must be a multiple of Thread M.");
static_assert(WSUBN % TN == 0, "Sub-Warp N must be a multiple of Thread N.");
static_assert(TN % 4 == 0, "Thread N must be a multiple of 4 for float4 vectorization.");
static_assert((WSUBM / TM) * (WSUBN / TN) == WARPSIZE, "Thread mapping within a warp is incorrect.");


/**
 * @brief Loads tiles from global memory into shared memory for matrices A and B.
 * Matrix A is transposed on-the-fly into shared memory to facilitate coalesced memory access
 * during the computation phase. Vectorized float4 loads are used for efficiency.
 */
template <const int BM, const int BK, const int BN, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN, const int kExtraCol>
__device__ void load_global_to_shared(const float *A, const float *B, const int M, const int K, const int N, float *As, float *Bs) {
    // Ensure vectorization is possible
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);
    
    constexpr int thread_num = (BM/WM)*(BN/WN)*WARPSIZE;
    static_assert(BK*BM % (thread_num*4) == 0, "A tile size not divisible by thread load capacity.");
    static_assert(BK*BN % (thread_num*4) == 0, "B tile size not divisible by thread load capacity.");

    // --- Load Tile A (and transpose) ---
    constexpr int ldg_a_niter = BK*BM / (thread_num*4);
    constexpr int a_tile_stride = BM / ldg_a_niter;
    static_assert(BM % ldg_a_niter == 0, "A tile stride calculation error.");

    const int a_tile_row = threadIdx.x / (BK / 4);
    const int a_tile_col = (threadIdx.x % (BK / 4)) * 4;
    
    #pragma unroll
    for (int i = 0; i < ldg_a_niter; ++i) {
        int row_offset = i * a_tile_stride;
        float4 ldg_a_reg = CFLOAT4(A[OFFSET(a_tile_row + row_offset, a_tile_col, K)]);
        As[OFFSET(a_tile_col    , a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.x;
        As[OFFSET(a_tile_col + 1, a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.y;
        As[OFFSET(a_tile_col + 2, a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.z;
        As[OFFSET(a_tile_col + 3, a_tile_row + row_offset, BM + kExtraCol)] = ldg_a_reg.w;
    }

    // --- Load Tile B ---
    constexpr int ldg_b_niter = BK*BN / (thread_num*4);
    constexpr int b_tile_stride = BK / ldg_b_niter;
    static_assert(BK % ldg_b_niter == 0, "B tile stride calculation error.");

    const int b_tile_row = threadIdx.x / (BN / 4);
    const int b_tile_col = (threadIdx.x % (BN / 4)) * 4;

    #pragma unroll
    for (int i = 0; i < ldg_b_niter; ++i) {
        int row_offset = i * b_tile_stride;
        FLOAT4(Bs[OFFSET(b_tile_row + row_offset, b_tile_col, BN)]) = CFLOAT4(B[OFFSET(b_tile_row + row_offset, b_tile_col, N)]);
    }
}

/**
 * @brief Performs matrix multiplication on tiles in shared memory.
 * Each warp computes a WMxWN portion of the output block. Results are accumulated in registers.
 */
template <const int BM, const int BK, const int BN, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN, const int kExtraCol>
__device__ void compute_mma_from_shared(float *As, float *Bs, float* Areg, float* Breg, float* Creg) {
    constexpr int WMITER = WM / WSUBM;
    constexpr int WNITER = WN / WSUBN;

    const int warp_idx = threadIdx.x / WARPSIZE;
    const int wy = warp_idx / (BN / WN);
    const int wx = warp_idx % (BN / WN);

    const int thread_idx = threadIdx.x % WARPSIZE;
    const int ty = thread_idx / (WSUBN / TN);
    const int tx = thread_idx % (WSUBN / TN);

    // Loop over the K-dimension of the tile
    #pragma unroll
    for (uint k = 0; k < BK; ++k) {
        // Load sub-tiles from shared memory into registers
        #pragma unroll
        for (uint wsy = 0; wsy < WMITER; ++wsy) {
            #pragma unroll
            for (uint j = 0; j < TM; ++j) {
                Areg[wsy*TM + j] =
                    As[OFFSET(k, wy*WM + wsy*WSUBM + ty*TM + j, BM + kExtraCol)];
            }
        }
        #pragma unroll
        for (uint wsx = 0; wsx < WNITER; ++wsx) {
            #pragma unroll
            for (uint i = 0; i < TN; ++i) {
                Breg[wsx*TN + i] =
                    Bs[OFFSET(k, wx*WN + wsx*WSUBN + tx*TN + i, BN)];
            }
        }

        // Perform matrix multiplication using register data
        #pragma unroll
        for (uint wsy = 0; wsy < WMITER; ++wsy) {
            #pragma unroll
            for (uint wsx = 0; wsx < WNITER; ++wsx) {
                #pragma unroll
                for (int m = 0; m < TM; m++) {
                    #pragma unroll
                    for (int n = 0; n < TN; n++) {
                        Creg[OFFSET(wsy*TM + m, wsx*TN + n, WNITER*TN)] += Areg[wsy*TM + m] * Breg[wsx*TN + n];
                    }
                }
            }
        }
    }
}

/**
 * @brief Stores the computed tile from registers back to global memory.
 * Uses vectorized float4 stores for efficiency.
 */
template <const int BM, const int BK, const int BN, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void store_result_to_global(float *C, const int N, float *Creg) {
    constexpr int WMITER = WM / WSUBM;
    constexpr int WNITER = WN / WSUBN;

    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint wy = warp_idx / (BN / WN);
    const uint wx = warp_idx % (BN / WN);

    const uint thread_idx = threadIdx.x % WARPSIZE;
    const uint ty = thread_idx / (WSUBN / TN);
    const uint tx = thread_idx % (WSUBN / TN);

    // Pointer to the top-left of the warp's output tile
    C = &C[OFFSET(wy*WM, wx*WN, N)];

    #pragma unroll
    for (uint wsy = 0; wsy < WMITER; ++wsy) {
        #pragma unroll
        for (uint wsx = 0; wsx < WNITER; ++wsx) {
            float* Cws = &C[OFFSET(wsy*WSUBM, wsx*WSUBN, N)];
            #pragma unroll
            for (uint j = 0; j < TM; j += 1) {
                // Since TN is a multiple of 4, this loop is safe
                #pragma unroll
                for (uint i = 0; i < TN; i += 4) {
                    FLOAT4(Cws[OFFSET(ty*TM + j, tx*TN + i, N)])
                        = CFLOAT4(Creg[OFFSET(wsy*TM + j, wsx*TN + i, WNITER*TN)]);
                }
            }
        }
    }   
}

/**
 * @brief Main logic for a thread block, orchestrating the GEMM computation.
 * Uses double buffering to hide global memory latency by overlapping computation with data fetching.
 */
template<const int BM, const int BK, const int BN, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void gemm_block_tile(const float* A, const float* B, float* C, const int M, const int K, const int N) {
    // Padding to avoid shared memory bank conflicts when transposing A
    constexpr int kExtraCol = 4;
    constexpr int kAsSize = BK*(BM+kExtraCol);
    constexpr int kBsSize = BK*BN;

    // Double buffer in shared memory
    __shared__ float As[2][kAsSize];
    __shared__ float Bs[2][kBsSize];

    // Register arrays for computation
    constexpr uint WMITER = WM / WSUBM;
    constexpr uint WNITER = WN / WSUBN;
    float Areg[WMITER*TM];
    float Breg[WNITER*TN];
    float Creg[WMITER*TM * WNITER*TN] = {0.0f};

    // --- Double Buffering Main Loop ---
    int buffer_idx = 0;
    
    // 1. Load the first tile (k=0) into buffer 0
    load_global_to_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
        A, B, M, K, N, As[buffer_idx], Bs[buffer_idx]
    );
    __syncthreads(); // Ensure first load is complete before starting compute

    // 2. Loop over K dimension, computing on the current tile while fetching the next
    for (int k = BK; k < K; k += BK) {
        // Advance pointers to the next tile in global memory
        A += BK;
        B += BK * N;

        // Start loading the next tile (k+1) into the alternate buffer (1-buffer_idx)
        load_global_to_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
            A, B, M, K, N, As[1 - buffer_idx], Bs[1 - buffer_idx]
        );

        // Compute using the current tile (k) from buffer (buffer_idx)
        compute_mma_from_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
            As[buffer_idx], Bs[buffer_idx], Areg, Breg, Creg
        );
        
        // **PERFORMANCE FIX**: A single sync is sufficient here. It waits for both the async
        // load and the current computation to finish. This allows the GPU to overlap the
        // memory transfer and the arithmetic, hiding latency. The original code had two
        // syncs, which serialized these operations and negated the benefit of double buffering.
        __syncthreads();
        
        // Swap buffers for the next iteration
        buffer_idx = 1 - buffer_idx;
    }

    // 3. Compute the final tile which was pre-fetched in the last loop iteration
    compute_mma_from_shared<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN, kExtraCol>(
        As[buffer_idx], Bs[buffer_idx], Areg, Breg, Creg
    );

    // 4. Store final results from registers to global memory
    store_result_to_global<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN>(C, N, Creg);
}

/**
 * @brief Kernel entry point for the high-performance path (matrix sizes are divisible by tile sizes).
 */
template<const int BM, const int BK, const int BN, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN>
__global__ void matrix_multiplication_kernel(
    const float* A, const float* B, float* C, const int M, const int K, const int N) {
    // Calculate global block indices
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;

    // Set up pointers to the top-left of the matrices for this block
    A = &A[OFFSET(block_y*BM, 0, K)];
    B = &B[OFFSET(0, block_x*BN, N)];
    C = &C[OFFSET(block_y*BM, block_x*BN, N)];

    gemm_block_tile<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN>(A, B, C, M, K, N);
}

/**
 * @brief Host-callable wrapper to launch the high-performance kernel.
 */
void solve_uncheck(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock((BN/WN)*(BM/WM)*WARPSIZE);
    dim3 blocksPerGrid(N/BN, M/BM);
    
    matrix_multiplication_kernel<BM, BK, BN, WM, WN, WSUBM, WSUBN, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, K, N);
}


// ===================================================================================
// --- Native Fallback Implementation for Arbitrary Matrix Sizes ---
// ===================================================================================

template<const int BM, const int BK, const int BN, const int TM, const int TN>
__device__ void matrix_multiplication_inner_block_native(
    const float* A, const float* B, float* C, const int M, const int K, const int N,
    const int real_BM, const int real_BN
) {
    const int tx = (threadIdx.x % (BN / TN)) * TN;
    const int ty = (threadIdx.x / (BN / TN)) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float C_reg[TM][TN] = {0.0f};

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    for (int k_base = 0; k_base < K; k_base += BK) {
        const int real_BK = min(BK, K - k_base);
        __syncthreads(); // Ensure previous compute is done before overwriting shared memory

        // **PERFORMANCE FIX**: Reworked memory loading for coalesced access.
        // The original code had each thread making strided memory accesses, which is
        // very inefficient. This version has threads collaborate to load contiguous
        // chunks of memory, which is the canonical way to maximize bandwidth.
        
        // Load A tile into shared memory
        for(int i = tid; i < real_BM * real_BK; i += num_threads) {
            int row = i / real_BK;
            int col = i % real_BK;
            As[OFFSET(row, col, BK)] = A[OFFSET(row, col, K)];
        }
        // Load B tile into shared memory
        for(int i = tid; i < real_BK * real_BN; i += num_threads) {
            int row = i / real_BN;
            int col = i % real_BN;
            Bs[OFFSET(row, col, BN)] = B[OFFSET(row, col, N)];
        }
        __syncthreads(); // Wait for all loads to complete

        // Advance global pointers for the next iteration
        A += BK;
        B += BK * N;

        // Compute using shared memory tiles
        #pragma unroll
        for (int i = 0; i < real_BK; i++) {
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                #pragma unroll
                for (int l = 0; l < TN; l++) {
                    C_reg[j][l] += As[OFFSET(ty+j, i, BK)] * Bs[OFFSET(i, tx+l, BN)];
                }
            }
        }
    }

    // Write results from registers to global memory with boundary checks
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++) {
            if (LIMIT2(ty+j, real_BM, tx+l, real_BN)) {
                C[OFFSET(ty+j, tx+l, N)] = C_reg[j][l];
            }
        }
    }
}

template<const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void matrix_multiplication_kernel_native(
    const float* A, const float* B, float* C, const int M, const int K, const int N) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Calculate effective tile dimensions for boundary blocks
    const int real_BM = min(BM, M - by * BM);
    const int real_BN = min(BN, N - bx * BN);

    // Set up pointers for this block
    A = &A[OFFSET(by*BM, 0, K)];
    B = &B[OFFSET(0, bx*BN, N)];
    C = &C[OFFSET(by*BM, bx*BN, N)];

    matrix_multiplication_inner_block_native<BM, BK, BN, TM, TN>(A, B, C, M, K, N, real_BM, real_BN);
}

void solve_native(const float* A, const float* B, float* C, int M, int K, int N) {
    // Configuration for the native kernel
    const int BM_NATIVE = 128;
    const int BN_NATIVE = 128;
    const int BK_NATIVE = 8;
    const int TM_NATIVE = 8;
    const int TN_NATIVE = 8;
    static_assert(BM_NATIVE % TM_NATIVE == 0);
    static_assert(BN_NATIVE % TN_NATIVE == 0);
    static_assert((BN_NATIVE / TN_NATIVE) * (BM_NATIVE / TM_NATIVE) <= 1024, "Too many threads per block");

    dim3 threadsPerBlock((BN_NATIVE/TN_NATIVE)*(BM_NATIVE/TM_NATIVE));
    dim3 blocksPerGrid(CEIL_DIV(N, BN_NATIVE), CEIL_DIV(M, BM_NATIVE));
    
    matrix_multiplication_kernel_native<BM_NATIVE, BK_NATIVE, BN_NATIVE, TM_NATIVE, TN_NATIVE><<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, K, N);
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {    
    // Dispatch to the highly optimized kernel if dimensions are perfect multiples
    if (M % BM == 0 && N % BN == 0 && K % BK == 0) {
        solve_uncheck(A, B, C, M, K, N);
    } else {
        // Otherwise, use the fallback kernel that handles boundary conditions
        solve_native(A, B, C, M, K, N);
    }

    // Ensure the kernel is finished before returning control to the host
    cudaDeviceSynchronize();
}

