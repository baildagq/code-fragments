#include <cuda_runtime.h>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template<
    const int BLOCK_SIZE_M, //height of block of C that each  block calculate
    const int BLOCK_SIZE_N, //now: reduction dimension tile size (middle dimension tile)
    const int BLOCK_SIZE_K, //width of block of C that each block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X // width of block of C that each thread calculate
    >
__global__ void gemm_prefetch(float* A, float* B, float* C, int M, int N, int K) {
    
    // 现在语义改为：
    // A: M x N
    // B: N x K
    // C: M x K
    // 其中 N 是中间消去维度

    //线程参数
    //thread id
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    //block id
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    //the thread numbers in block of X,Y
    const unsigned int THREAD_X_PER_BLOCK = BLOCK_SIZE_K / THREAD_SIZE_X;
    const unsigned int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const unsigned int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;
    //thread id in cur Block
    unsigned int tid = ty * THREAD_X_PER_BLOCK + tx;

    //定义shared memory
    // As: [2][BLOCK_SIZE_N][BLOCK_SIZE_M]
    // Bs: [2][BLOCK_SIZE_N][BLOCK_SIZE_K]
    __shared__ float As[2][BLOCK_SIZE_N][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_N][BLOCK_SIZE_K];

    //register for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    //registers load global memory
    const unsigned int ldg_num_a = (BLOCK_SIZE_M * BLOCK_SIZE_N) / (4 * THREAD_NUM_PER_BLOCK);
    const unsigned int ldg_num_b = (BLOCK_SIZE_K * BLOCK_SIZE_N) / (4 * THREAD_NUM_PER_BLOCK);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    //读取A，B矩阵所需的索引(每个线程不同)
    //thread number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;

    //row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW; //该线程负责搬运的A矩阵行号
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
    const int A_TILE_COL = (tid % A_TILE_THREAD_PER_ROW) * 4; //该线程负责搬运的A矩阵列号
    const int B_TILE_COL = (tid % B_TILE_THREAD_PER_ROW) * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // A block row offset: by * BLOCK_SIZE_M
    // B block col offset: bx * BLOCK_SIZE_K
    A = &A[(BLOCK_SIZE_M * by) * N];
    B = &B[BLOCK_SIZE_K * bx];

    //第一个大迭代装载
    // load A from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = (i / A_TILE_ROW_STRIDE) * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i,   // row in A tile
            A_TILE_COL,             // col in A tile (along reduction dim N)
            N                       // A row stride
        )]);

        // As转置
        As[0][A_TILE_COL][A_TILE_ROW_START + i]     = ldg_a_reg[ldg_index];
        As[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
        As[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
        As[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
    }

    // load B from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
            B_TILE_ROW_START + i,   // row in B tile (along reduction dim N)
            B_TILE_COL,             // col in B tile (along output dim K)
            K                       // B row stride
        )]);
    }
    __syncthreads();

    // load A from shared memory to register
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }

    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    //开始大迭代
    int write_stage_idx = 1; //当前要写入的reg和sm块，第一个大迭代所需的数据已经预先装在stage 0里了
    int tile_idx = 0;

    do {
        tile_idx += BLOCK_SIZE_N;

        // load next tile from global mem(SM预取-从global memory到load register)
        if (tile_idx < N) {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
                int ldg_index = (i / A_TILE_ROW_STRIDE) * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i,   // row
                    tile_idx + A_TILE_COL,  // col along reduction dim N
                    N
                )]);
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_ROW_STRIDE) {
                int ldg_index = (i / B_TILE_ROW_STRIDE) * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row along reduction dim N
                    B_TILE_COL,                      // col along output dim K
                    K
                )]);
            }
        }

        //开始小迭代
        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE_N - 1; ++j) {
            // load next tile from shared mem to register 
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j + 1) % 2][thread_y]) =
                    FETCH_FLOAT4(As[load_stage_idx][j + 1][THREAD_SIZE_Y * ty + thread_y]);
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j + 1) % 2][thread_x]) =
                    FETCH_FLOAT4(Bs[load_stage_idx][j + 1][THREAD_SIZE_X * tx + thread_x]);
            }

            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] +=
                        frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
                }
            }
        }

        if (tile_idx < N) {
            // load A from load register to shared memory( SM预取 - 从load register到shared memory )
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
                int ldg_index = (i / A_TILE_ROW_STRIDE) * 4;
                // As转置
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]     = ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
                As[write_stage_idx][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
                As[write_stage_idx][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
            }

            // load B from load register to shared memory
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_ROW_STRIDE) {
                int ldg_index = (i / B_TILE_ROW_STRIDE) * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) =
                    FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }

            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        //最后完成下一轮小迭代第一个寄存器预取
        // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[0][thread_y]) =
                FETCH_FLOAT4(As[load_stage_idx ^ 1][0][THREAD_SIZE_Y * ty + thread_y]);
        }

        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[0][thread_x]) =
                FETCH_FLOAT4(Bs[load_stage_idx ^ 1][0][THREAD_SIZE_X * tx + thread_x]);
        }

        //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
        //在刚才的小迭代里：最后一个 fragment（索引 BLOCK_SIZE_N - 1）已经预取好了，但还没算
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }

    } while (tile_idx < N);

    // store back to C(vectorized float4)
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_K * bx + tx * THREAD_SIZE_X + thread_x,
                K
            )]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

__global__ void gemm_fallback_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 对应 C 的列，范围 [0, K)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 对应 C 的行，范围 [0, M)

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

extern "C" void solve(float* A, float* B, float* C, int M, int N, int K) {
    // 现在语义：
    // A: M x N
    // B: N x K
    // C: M x K

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 8;    // reduction tile size
    const int BLOCK_SIZE_K = 128;  // output width tile size
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;

    // 只有满足这些条件时，才走优化 kernel
    bool use_optimized =
        (M >= BLOCK_SIZE_M) &&
        (N >= BLOCK_SIZE_N) &&
        (K >= BLOCK_SIZE_K) &&
        (M % BLOCK_SIZE_M == 0) &&
        (N % BLOCK_SIZE_N == 0) &&
        (K % BLOCK_SIZE_K == 0);

    if (use_optimized) {
        dim3 dimBlock(BLOCK_SIZE_K / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(K / BLOCK_SIZE_K, M / BLOCK_SIZE_M);

        gemm_prefetch<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                      THREAD_SIZE_Y, THREAD_SIZE_X>
            <<<dimGrid, dimBlock>>>(A, B, C, M, N, K);
    } else {
        dim3 block(16, 16);
        dim3 grid((K + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);

        gemm_fallback_kernel<<<grid, block>>>(A, B, C, M, N, K);
    }

    cudaDeviceSynchronize();
}
