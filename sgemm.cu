__global__ void sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    if (row < M && col < N) {
        for (int k = 0; k < K; k ++) {
            sum += A[row * K + k] * B[k * N + col];
        }
    }
    C[row * N + col] = sum;
}


// A size: M * K
// B size: K * N
// 朴素方式的计算强度：
//      通过矩阵纬度考虑：2 (M * K * N) / (4 * M * N (2 * K)) = 1/4
//      通过元素纬度考虑：2K / (4 * 2 * K) = 1/4
// 如果能将所有元素都load到shared_memory中，则仅需读写 4 *(M * K + K * N + M * N) 次global_memory，但显然放不下
// 那对于C矩阵的一个小块儿 m * n, 则仅需读取A的 m * K + B的K * n, 此时shared_memory仍然放不下
// 则对于C矩阵的一个小块儿 m * h, 分 K/k 次读取A的 m * k + B的 k * n, 此时shared_memory应能放下 A的小块儿+B的小块儿 + C结果的小块儿
// 当前方式的计算强度：
//      通过矩阵纬度考虑（因为有数据复用，所以没法按元素纬度考虑）-> 不是考虑最终结果，而是单次过程计算与访寸比值
//      (2 * BM * BK * BN) / (4 * (BM * BK + BK * BN)) = BM * BN / (2 * (BM + BN)) = 1 / (2 * (1/BN + 1/BM))
//      BM + BN 固定，则由均值不等式可知，应该在 BM = BN 时值最大
//  step-1: 根据算术强度要求，选出合适的BM/BN的值（后面如果硬件不满足，再往低调整） 选值128 -> 1 / (2 * (1/64)) = 32，根据硬件的flops/io，可得满足计算强度
//  step-2: 根据shared_memory大小，选出合适的BK的值
//  step-3: ？？满足 2blocks/SM, block_size值选为256 -> blockDim.x; 指的是256个线程，这样同一个SM里能同时跑两个block


// 通过模版元编程，可以在编译阶段完成复杂的计算和逻辑判断，生成高度特化的内核代码
template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
    // tiling process, one block one tile
    __shared__ float tile_A[BM][BK];
    __shared__ float tile_B[BK][BN];

    // grid 纬度使用二维矩阵 -> 方便使每个block映射到一个tile上？A的tile还是C的tile？ 应该是C的tile
    // blockIdx
    // (x=0, y=0), (x=1, y=0), (x=2, y=0) ....
    // (x=0, y=1), (x=1, y=1), (x=2, y=1) ....
    // (x=0, y=2), (x=1, y=2), (x=2, y=2) ....
    //
    // block 使用一维矩阵 -> 因为block内部加载 tile_A, tile_B，计算以及写入 tile_C 的矩阵大小不一致，所以直接使用二维的矩阵不方便，处理每个部分时，使用一维模拟二维

    int global_row_0 = blockIdx.y * BM; // for A
    int global_col_0 = blockIdx.x * BN; // for B
    int tid = threadIdx.x;
    

    // 加载A线程布局
    constexpr int A_BLOCK_X = BK; // = 8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / BK; // = 32
    int a_thread_x = tid % A_BLOCK_X; // col
    int a_thread_y = tid / A_BLOCK_X; // row

    // 加载B线程布局
    constexpr int B_BLOCK_X = 32; // 选择原因是wrap_size
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_Y;
    int b_thread_x = tid % B_BLOCK_X; // col
    int b_thread_y = tid / B_BLOCK_Y; // row

    // 计算过程线程布局
    constexpr int C_BLOCK_X = 16; // 选择原因?
    constexpr int C_BLOCK_Y = BLOCK_SIZE / 16;
    int c_thread_x = tid % C_BLOCK_X; // col
    int c_thread_y = tid % C_BLOCK_Y; // row
    int Tx = BM / C_BLOCK_Y; // row
    int Ty = BN / C_BLOCK_X; // col
    int tile_C_each_thread_res[Tx][Ty]; // 寄存器


    // 沿K纬度分块
    for (int k = 0; k < K; k += BK) {
        // load A
#pragma unroll
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {
            int A_row = global_row_0 + i;
            int A_col = k + a_thread_x;
            tile_A[i][a_thread_x] = (A_row < M && A_col < K) ? A[A_row * K + A_col] : 0.0f;
        }

        // load B
#pragma unroll
        for (int i = b_thread_y; i < BK; i += B_BLOCK_Y) {
#pragma unroll
            for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
                int B_row = k + i;
                int B_col = global_col_0 + j;
                tile_B[i][j] = (B_row < K && B_col < N) ? B[B_row * N + B_col]: 0.0f;
            }
        }

        // 外积计算 As x Bs
        // 最终的C的size是BM * BN, 线程数量是 16 * 16，则每个线程需要计算保存的是 [BM / 16, BN / 16] 的值
        // 相当于全班同学分工从全局内存中搬运了tile矩阵进来，写在了黑板上
        // 每个同学再负责最终的矩阵的一个子集, 注意：子集并非C矩阵的连续部分
        for (int p = 0; p < BK; p ++) {
            for (int i = 0; i < Tx; i ++) {
                int row = c_thread_y + i * C_BLOCK_Y;
                for (int j = 0; j < Ty; j ++) {
                    int col = c_thread_x + j * C_BLOCK_X;
                    tile_C_each_thread_res[i][j] += A[row][p] * B[p][col];
                }
            }
        }
        __syncthreads();
    }

    // 写入最终结果
    for (int i = 0; i < Tx; i ++) {
        int row = global_row_0 + c_thread_y + i * C_BLOCK_Y;
        for (int j = 0; j < Ty; j ++) {
            int col = global_col_0 + c_thread_x + i * C_BLOCK_X;
            if (row < M && col < N) {
                C[row * K + col] = tile_C_each_thread_res[i][j];
            }
        }
    }
}



/*
 * BM = BN = 128
 * BK = 8
 */
template<int BM, int BN, int BK, int BLOCK_SIZE>
__global__ sgemm_v2(float* A, float* B, float* C, int M, int K, int N) {
    // use shared_memory to implement tiling gemm
    __shared__ float tile_A[BM][BK];
    __shared__ float tile_B[BK][BN];

    // ---- x ----> row dimenssion
    // ---- y ----> col dimenssion

    // one-block-one-tile of C
    int global_row_0 = blockIdx.x * BM;
    int global_col_0 = blockIdx.y * BN;

    // re-arange thread dimeession, simulate 2D thread
    int tid = threadIdx.x;
    // tile_A shape [BM, BK]
    int A_BLOCK_Y = BK;
    int A_BLOCK_X = BLOCK_SIZE / A_BLOCK_Y;
    int a_thread_x = tid / A_BLOCK_Y;
    int a_thread_y = tid % A_BLOCK_Y;

    for (int k = 0; k < K; k += BK) {
        // load A
#pragma unroll
        for (int i = a_thread_x; i < BM; i += A_BLOCK_X) {
            int global_row_A = global_row_0 + i;
            int global_col_A = 
        }




        // load B
        // calculate
    }
}
