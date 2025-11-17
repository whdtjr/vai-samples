#include "neuralNodes.h"
#include "npzLoader.h"
#include <stdexcept>
#include <string>
#include <unordered_map>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


static const char* src_relu = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main() 
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    out0[o] = max(in0[o], 0.0f);
})";

static const char* src_copy = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main() 
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    out0[o] = in0[o];
})";

static const char* src_setZero = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main() 
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    out0[o] = 0.0;
})";

static const char* src_maxpool = R"(
#version 450
#define FLT_MIN -3.402823466e+38
#define DISCARD_TAIL
layout(local_size_x = 64, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int H;      // input height
    int W;      // input width
    int C;
    int P;      // pooling size
};

void main()
{
    int h_ = int(gl_GlobalInvocationID.x);  // output row
    int w_ = int(gl_GlobalInvocationID.y);  // output col
    int c = int(gl_GlobalInvocationID.z);   // channel
#ifdef DISCARD_TAIL
    int H_ = H / P;  
    int W_ = W / P;  
#else
    int H_ = (H + P - 1) / P;
    int W_ = (W + P - 1) / P;
#endif
    if (h_ >= H_ || w_ >= W_ || c >= C)
        return;

    int h0 = h_ * P;  
    int w0 = w_ * P;     
    float maxVal = FLT_MIN;
    for (int dh=0; dh < P; ++dh) 
    {
        int h = h0 + dh;  
        for (int dw=0; dw < P; ++dw) 
        {
            int w = w0 + dw;

        #ifndef DISCARD_TAIL
            if (h < H && w < W) 
        #endif
            {
                maxVal = max(maxVal, in0[(h * W + w) * C + c]);
            }
        }
    }
    out0[(h_ * W_ + w_) * C + c] = maxVal;
})";

static const char* src_im2col = R"(
#version 450
layout(local_size_x = 64, local_size_y = 16) in;
layout(set = 0, binding = 0) writeonly buffer OutBuffer { float im2colOut[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int H;          // input height
    int W;          // input width
    int C;          // channels
    int K;          // kernel size
    int S;          // stride
    int H_out;      // output height
    int W_out;      // output width
};

void main()
{
    int i = int(gl_GlobalInvocationID.x);   // output position index
    int j = int(gl_GlobalInvocationID.y);   // kernel element index
    int KK = K * K;
    int CKK = C * KK;
    if (i >= H_out * W_out || j >= CKK)
        return;

    int h_out = i / W_out;      // output row
    int w_out = i % W_out;      // output col
    int c = j / KK;             // input channel
    int K_2 = K / 2;
    int k = j % KK;

    // Calculate input position with stride
    int h = h_out * S + (k / K) - K_2;
    int w = w_out * S + (k % K) - K_2;

    float value = 0.0;
    if (0 <= h && h < H && 0 <= w && w < W)
        value = in0[((h * W) + w) * C + c];

    im2colOut[i * CKK + j] = value;
})";

static const char* src_gemm_naive = R"(
#version 450
layout(local_size_x = 32, local_size_y = 32) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight { float B[]; };
layout(set = 0, binding = 3) buffer Bias { float b[]; };

// C(MxN) = A(MxK)*B(KxN) + b(1xN)
layout(push_constant) uniform PushConstants {
    int M;  // # of batchs
    int K;  // # of inputs
    int N;  // # of outputs
};

void main() 
{
    int n = int(gl_GlobalInvocationID.x); 
    int m = int(gl_GlobalInvocationID.y); 

    if (m >= M || n >= N) 
        return;

    float sum = b[n];
    for (int k = 0; k < K; ++k)
        sum += A[m * K + k] * B[k * N + n];

    C[m * N + n] = sum;
}
)";

//static const char* gemm_srcCode_vec = R"(
//#version 450
//layout(local_size_x = 32, local_size_y = 32) in;
//layout(set = 0, binding = 0) buffer OutBuffer { vec4 C[]; };
//layout(set = 0, binding = 1) buffer InBuffer { vec4 A[]; };
//layout(set = 0, binding = 2) buffer Weight { vec4 B[]; };
//layout(set = 0, binding = 3) buffer Bias { float b[]; };
//
//// C(MxN) = A(MxK)*B(KxN) + b(1xN)
//layout(push_constant) uniform PushConstants {
//    int M;  // # of batchs
//    int K;  // # of inputs
//    int N;  // # of outputs
//};
//
//void main() 
//{
//    int n = int(gl_GlobalInvocationID.x); 
//    int m = int(gl_GlobalInvocationID.y); 
//
//    if (m >= M || n >= N) 
//        return;
//
//    float sum = b[n];
//    for (int k = 0; k < K; ++k)
//        sum += A[m * K + k] * B[k * N + n];
//
//    C[m * N + n] = sum;
//})";

static const char* src_gemm_kSplit = R"(
#version 450
#define P 16
#extension GL_EXT_shader_atomic_float : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight   { float B[]; };
layout(set = 0, binding = 3) buffer Bias     { float b[]; };

layout(push_constant) uniform PushConstants {
    int M;  // batch
    int K;  // input size
    int N;  // output size
};

void main()
{
    int n = int(gl_GlobalInvocationID.x); // output column
    int m = int(gl_GlobalInvocationID.y); // output row
    int Pid = int(gl_GlobalInvocationID.z); // split index along K

    if (n >= N) return; 

    int k0 = Pid * P;
    float sum = (Pid==0) ? b[n] : 0.0;
    for (int p = 0; p < P; ++p) 
    {
        int k = k0 + p;
        if (k >= K) 
            break;
        sum += A[m * K + k] * B[k * N + n];
    }

    atomicAdd(C[m * N + n], sum);
}
)";

static const char* src_gemm_kSplit2 = R"(
#version 450
#define P 16
#extension GL_EXT_shader_atomic_float : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight   { float B[]; };
layout(set = 0, binding = 3) buffer Bias     { float b[]; };

layout(push_constant) uniform PushConstants {
    int M;  // batch
    int K;  // input size
    int N;  // output size
};

shared float sA;

void main()
{
    int n = int(gl_GlobalInvocationID.x); // output column
    int m = int(gl_GlobalInvocationID.y); // output row
    int Pid = int(gl_GlobalInvocationID.z); // split index along K

    if (n >= N) return; 

    int k0 = Pid * P;
    float sum = (Pid==0) ? b[n] : 0.0;
    for (int p = 0; p < P; ++p) 
    {
        int k = k0 + p;
        if (k >= K) 
            break;

        if (gl_LocalInvocationIndex.x == 0) 
            sA = A[m * K + k];
        barrier();

        sum += sA * B[k * N + n];
        barrier();
    }

    atomicAdd(C[m * N + n], sum);
}
)";

static const char* src_gemm_shared = R"(
#version 450
layout(local_size_x = 32, local_size_y = 32) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight { float B[]; };
layout(set = 0, binding = 3) buffer Bias { float b[]; };

// C(MxN) = A(MxK)*B(KxN) + b(1xN)  
layout(push_constant) uniform PushConstants {
    int M;  // # of batchs
    int K;  // # of inputs
    int N;  // # of outputs
};

shared float As[32 * 32];
shared float Bs[32 * 32];

void main() 
{
    int n = int(gl_GlobalInvocationID.x); 
    int m = int(gl_GlobalInvocationID.y); 
    int _n = int(gl_LocalInvocationID.x); 
    int _m = int(gl_LocalInvocationID.y); 
    bool validThread = (m < M && n < N);

    float acc = 0.0;
    int sharedIdx = _m * 32 + _n;
    for (int k0 = 0; k0 < K; k0 += 32) 
    {
        int n_ = k0 + _n;
        int m_ = k0 + _m;
        As[sharedIdx] = (m < M && n_ < K) ? A[m * K + n_] : 0.0; // A[m, n_]
        Bs[sharedIdx] = (m_ < K && n < N) ? B[m_ * N + n] : 0.0; // B[m_, n]
        barrier();

        for (int k = 0; k < 32; ++k) 
            acc += As[_m * 32 + k] * Bs[k * 32 + _n];
        barrier();
    }

    if (validThread)
        C[m * N + n] = acc + b[n];
})";

static const char* src_gemm_multiOut1d = R"(
#version 450
#define BN 64
#define BM 64
#define BK 16
#define TM 4
#define TRANSPOSE_A 0
// assert BN == BM == TM*BK
// Each workgroup computes a BMxBN results while it consists of (BM/TM)xBN threads.

layout(local_size_x = BN*BM/TM) in;     // (BM/TM) x BN
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight { float B[]; };
layout(set = 0, binding = 3) buffer Bias { float b[]; };

// C(MxN) = A(MxK)*B(KxN) + b(1xN)  
layout(push_constant) uniform PushConstants {
    int M;  // # of batchs
    int K;  // # of inputs
    int N;  // # of outputs
};

shared float As[BM * BK]; // BMxBK
shared float Bs[BK * BN]; // BKxBN

void main() 
{
    int tileCol = int(gl_WorkGroupID.x); 
    int tileRow = int(gl_WorkGroupID.y); 
    int t = int(gl_LocalInvocationID.x);

    int bk = t % BK, d_bk = t / BK;
    int bn = t % BN, d_bn = t / BN; 
    /*
    * The matrix row index m is computed in two different contexts:
    *   m = tileRow * BM + d_bk;            // Used when loading A to ensure coalesced global memory access.
    *   m = tileRow * BM + d_bn * TM + tm;  // Used when accessing C multiple times within a single thread.
    */
    int m = tileRow * BM + d_bk;            // d_bk : 0 ~ (BM*BN)/(TM*BK)-1 = BN-1 => assert BN == BM (Since 0 <= m - tileRow * BM < BM)
    int n = tileCol * BN + bn;

    float result[TM];   for (int tm = 0; tm < TM; ++tm) result[tm] = 0.0;
#if TRANSPOSE_A
    float regM[TM];
    int t_t = bk * BM + d_bk; // transposed index of t
#endif

    for (int k_ = 0; k_ < K; k_ += BK)
    {
        int k = k_ + bk;
    #if TRANSPOSE_A
        As[t_t] = (m < M) && (k < K) ? A[m * K + k] : 0.0;
    #else
        As[t] = (m < M) && (k < K) ? A[m * K + k] : 0.0;   
    #endif
        
        k = k_ + d_bn;                      // d_bn : 0 ~ (BM*BN)/(TM*BN)-1 = BM/TM-1 => assert BM/TM == BK (Since 0 <= m_ - k_ < BK)
        Bs[t] = (k < K) && (n < N) ? B[k * N + n] : 0.0;  
        barrier();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            float regN = Bs[dotIdx * BN + bn];

        #if TRANSPOSE_A
            for (int tm = 0; tm < TM; ++tm)
                regM[tm] = As[dotIdx * BM + (d_bn * TM + tm)];
            for (int tm = 0; tm < TM; ++tm)
                result[tm] += regM[tm] * regN;
        #else
            for (int tm = 0; tm < TM; ++tm)
                result[tm] += As[(d_bn * TM + tm) * BK + dotIdx] * regN;
        #endif
        }
        barrier();
    }

    if (n >= N)     // Return only after barrier(); skipping it may cause deadlock
        return;

    float bias = b[n];
    int m0 = tileRow * BM + d_bn * TM;

    for (int tm = 0; tm < TM; ++tm)
    {
        m = m0 + tm;
        if (m >= M)
            break;
        C[m * N + n] = result[tm] + bias;
    }
})";

static const char* src_batchnorm = R"(
#version 450
layout(local_size_x = 64, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) buffer Gamma { float gamma[]; };
layout(set = 0, binding = 3) buffer Beta { float beta[]; };
layout(set = 0, binding = 4) buffer Mean { float mean[]; };
layout(set = 0, binding = 5) buffer Var { float var[]; };
layout(push_constant) uniform PushConstants {
    int H;
    int W;
    int C;
};

void main()
{
    int h = int(gl_GlobalInvocationID.x);
    int w = int(gl_GlobalInvocationID.y);
    int c = int(gl_GlobalInvocationID.z);

    if (h >= H || w >= W || c >= C)
        return;

    int idx = (h * W + w) * C + c;
    float eps = 1e-5;
    float normalized = (in0[idx] - mean[c]) / sqrt(var[c] + eps);
    out0[idx] = gamma[c] * normalized + beta[c];
})";

static const char* src_globalAvgPool = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int H;
    int W;
    int C;
};

void main()
{
    int c = int(gl_GlobalInvocationID.x);

    if (c >= C)
        return;

    float sum = 0.0;
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            sum += in0[(h * W + w) * C + c];

    out0[c] = sum / float(H * W);
})";

static const char* src_add = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer1 { float in0[]; };
layout(set = 0, binding = 2) buffer InBuffer2 { float in1[]; };
layout(push_constant) uniform PushConstants {
    int N;
};

void main()
{
    int i = int(gl_GlobalInvocationID.x);

    if (i >= N)
        return;

    out0[i] = in0[i] + in1[i];
})";

static const char* src_gemm_multiOut2d = R"(
#version 450
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
/*
* assert: T/BK, T/BN is natural number
* assert: BM/(T/BK) = (BK*TM*TN)/BN is natural number
* assert: BK/(T/BN) = (BK*TM*TN)/BM is natural number
*/

layout(local_size_x = (BM*BN)/(TM*TN)) in; 
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight   { float B[]; };
layout(set = 0, binding = 3) buffer Bias     { float b[]; };

// C(MxN) = A(MxK)*B(KxN) + b(1xN)  
layout(push_constant) uniform PushConstants {
    int M;  // # of rows (batches)
    int K;  // # of inputs
    int N;  // # of outputs
};

shared float As[BM * BK]; // BMxBK
shared float Bs[BK * BN]; // BKxBN

void main() 
{
    int tileCol = int(gl_WorkGroupID.x);
    int tileRow = int(gl_WorkGroupID.y);
    int t = int(gl_LocalInvocationID.x);
    int T = int(gl_WorkGroupSize.x);        // 64 or 256 (== (BM*BN)/(TM*TN))

    // A/B 로드용 로컬 인덱스
    int innerRowA = t / BK;
    int innerColA = t % BK;
    int rowStrideA = T / BK;    // 8 or 32

    int innerRowB = t / BN;
    int innerColB = t % BN;
    int rowStrideB = T / BN;    // 1(64/64) or 2(256/128)

    // 이 스레드가 계산할 결과 블록 (TMxTN)
    int t_n = t % (BN / TN);
    int t_m = t / (BN / TN);

    // 결과와 레지스터 캐시
    float threadResults[TM * TN];
    for (int i = 0; i < TM*TN; ++i) threadResults[i] = 0.0;
    float regM[TM];
    float regN[TN];

    int m0 = tileRow * BM + innerRowA;
    int n0 = tileCol * BN + innerColB;
    int t_t = innerColA * BM + innerRowA; // transposed index of t

    for (int k0 = 0; k0 < K; k0 += BK) 
    {
        int k = k0 + innerColA;
        for (int bm = 0; bm < BM; bm += rowStrideA)  // 8(64/8) or 4(128/32)
        {
            int m = m0 + bm;
            /*
            * bm * BK + t 
            * == bm * BK + (innerRowA * BK + innerColA)
            * == (bm + innerRowA) * BK + innerColA
            * transpose => 
            * innerColA * BM + (bm + innerRowA)
            */
            As[bm + t_t] = (m < M && k < K) ? A[m * K + k] : 0.0;

        }
        
        int n = n0;
        for (int bk = 0; bk < BK; bk += rowStrideB) // 8(8/1) or 4(8/2)
        {
            k = k0 + innerRowB + bk; 
            Bs[bk * BN + t] = (k < K && n < N) ? B[k * N + n] : 0.0;
        }
        barrier();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) 
        {
            for (int tm = 0; tm < TM; ++tm)
                regM[tm] = As[dotIdx * BM + (t_m * TM + tm)];

            for (int tn = 0; tn < TN; ++tn)
                regN[tn] = Bs[dotIdx * BN + (t_n * TN + tn)];

            for (int tm = 0; tm < TM; ++tm)
                for (int tn = 0; tn < TN; ++tn)
                    threadResults[tm*TN + tn] += regM[tm] * regN[tn];
        }
        barrier();
    }

    m0 = tileRow * BM + t_m * TM;
    n0 = tileCol * BN + t_n * TN;

    for (int tn = 0; tn < TN; ++tn) 
    {
        int n = n0 + tn;
        if (n >= N)
            break;

        float bias = b[n];
        for (int tm = 0; tm < TM; ++tm) 
        {
            int m = m0 + tm;
            if (m >= M)
                break; 
            C[m * N + n] = threadResults[tm*TN + tn] + bias;
        }
    }
})";

Device netGlobalDevice = VulkanApp::get().device();

static DescriptorPool gDestSetPool = netGlobalDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 200}, 
    .maxSets = 100
});


static ComputePipeline requestPipeline(const char* src)
{
    static std::unordered_map<const char*, ComputePipeline> pipelineCache;

    auto [it, inserted] = pipelineCache.try_emplace(src);
    if (inserted)
        it->second = netGlobalDevice.createComputePipeline({src});
    return it->second;
}

static std::map<const char*, uint32_t> gGemmTileSize =
{
    {src_gemm_naive, 32},
    {src_gemm_shared, 32},
    {src_gemm_multiOut1d, 64},
    {src_gemm_multiOut2d, 128}
};

void loadShaders()
{
    requestPipeline(src_relu);
    // requestPipeline(src_copy);
    requestPipeline(src_setZero);
    requestPipeline(src_maxpool);
    requestPipeline(src_im2col);
    requestPipeline(src_gemm_naive);
    requestPipeline(src_gemm_kSplit);
    // requestPipeline(src_gemm_kSplit2);
    requestPipeline(src_gemm_shared);
    requestPipeline(src_gemm_multiOut1d);
    requestPipeline(src_gemm_multiOut2d);
    requestPipeline(src_batchnorm);
    requestPipeline(src_globalAvgPool);
    requestPipeline(src_add);
}

static std::runtime_error missingKey(const std::string& key)
{
    return std::runtime_error("Missing key '" + key + "' in NPZ weights");
}

void loadConvolutionWeights(ConvolutionNode& node, const NpzLoader& npz, const std::string& baseName)
{
    const std::string weightKey = baseName + ".weight";
    if (!npz.hasKey(weightKey))
        throw missingKey(weightKey);

    const NpyArray& w = npz[weightKey];
    if (w.shape.size() != 4)
        throw std::runtime_error("Unexpected weight shape for " + weightKey);

    const uint32_t outChannels = static_cast<uint32_t>(w.shape[0]);
    uint32_t rows = static_cast<uint32_t>(w.shape[1] * w.shape[2] * w.shape[3]);

    node["weight"] = Tensor(rows, outChannels).set(w.data);

    const std::string biasKey = baseName + ".bias";
    if (npz.hasKey(biasKey))
    {
        const NpyArray& b = npz[biasKey];
        node["bias"] = Tensor(static_cast<uint32_t>(b.shape[0])).set(b.data);
    }
    else
    {
        node["bias"] = Tensor(outChannels).set(std::vector<float>(outChannels, 0.0f));
    }
}

void loadBatchNormalizationWeights(BatchNormalizationNode& node, const NpzLoader& npz, const std::string& baseName)
{
    auto requireArray = [&](const std::string& key) -> const NpyArray& {
        if (!npz.hasKey(key))
            throw missingKey(key);
        return npz[key];
    };

    const auto& gamma = requireArray(baseName + ".weight");
    const auto& beta = requireArray(baseName + ".bias");
    const auto& mean = requireArray(baseName + ".running_mean");
    const auto& var = requireArray(baseName + ".running_var");

    const uint32_t channels = static_cast<uint32_t>(gamma.shape[0]);
    node["gamma"] = Tensor(channels).set(gamma.data);
    node["beta"] = Tensor(channels).set(beta.data);
    node["mean"] = Tensor(channels).set(mean.data);
    node["var"] = Tensor(channels).set(var.data);
}

/////////////////////////////////////////////////////////////////////////////////////////
// ConvolutionNode
/////////////////////////////////////////////////////////////////////////////////////////
ConvolutionNode::ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth, uint32_t stride)
:  C(inChannels), F(outChannels), K(kernelWidth), S(stride)
{
    _ASSERT(K % 2 == 1);
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("im2colOut", NodeSlot::internal);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    im2col = requestPipeline(src_im2col);
    im2colDescSet = im2col.descSetLayout(0).newDescSet(gDestSetPool);

    const char* gemmSrc = src_gemm_shared;

    gemm = requestPipeline(gemmSrc);
    gemmTileSize = gGemmTileSize.at(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);
}

void ConvolutionNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(-1, -1, C));
    _ASSERT((*this)["weight"].isShapeOf(C*K*K, F));
    _ASSERT((*this)["bias"].isShapeOf(F));

    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1];
    uint32_t H_out = (H + S - 1) / S;
    uint32_t W_out = (W + S - 1) / S;

    (*this)["im2colOut"] = Tensor(H_out, W_out, C*K*K);
    (*this)["out0"] = Tensor(H_out, W_out, F);
}

void ConvolutionNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    const auto& outShape = (*this)["out0"].shape();
    uint32_t H = inShape[0], W = inShape[1];
    uint32_t H_out = outShape[0], W_out = outShape[1];

    im2colDescSet.write({
        (*this)["im2colOut"].buffer(),
        (*this)["in0"].buffer(),
    });

    gemmDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["im2colOut"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t im2colConstants[] = {H, W, C, K, S, H_out, W_out};

    uint32_t M = H_out * W_out;
    uint32_t K_ = C * K * K;
    uint32_t N = F;
    uint32_t gemmConstants[] = {M, K_, N};

    cmdBuff
        .bindPipeline(im2col)
        .bindDescSets({im2colDescSet})
        .setPushConstants(0, sizeof(im2colConstants), im2colConstants)
        .dispatch(H_out * W_out, C * K * K)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["im2colOut"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        )

        .bindPipeline(gemm)
        .bindDescSets({gemmDescSet})
        .setPushConstants(0, sizeof(gemmConstants), gemmConstants)
        .dispatch0(CEIL_DIV(N, gemmTileSize), CEIL_DIV(M, gemmTileSize))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


/////////////////////////////////////////////////////////////////////////////////////////
// ReluNode
/////////////////////////////////////////////////////////////////////////////////////////
ReluNode::ReluNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    relu = requestPipeline(src_relu);
    reluDescSet = relu.descSetLayout(0).newDescSet(gDestSetPool);
}

void ReluNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void ReluNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    int I = 1;
    for (int dim : inShape) I *= dim;
    
    reluDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    int reluConstants[] = {I};

    cmdBuff
        .bindPipeline(relu)
        .setPushConstants(0, sizeof(reluConstants), reluConstants)
        .bindDescSets({reluDescSet})
        .dispatch(I)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  


/////////////////////////////////////////////////////////////////////////////////////////
// MaxPoolingNode
/////////////////////////////////////////////////////////////////////////////////////////
MaxPoolingNode::MaxPoolingNode(uint32_t poolSize)
: P(poolSize)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    maxpool = requestPipeline(src_maxpool);
    maxpoolDescSet = maxpool.descSetLayout(0).newDescSet(gDestSetPool);
}

void MaxPoolingNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    uint32_t H = inShape[0], W = inShape[1], C = inShape[2];

    if (discardTail)
        (*this)["out0"] = Tensor(H / P, W / P, C);
    else    
        (*this)["out0"] = Tensor((H + P - 1) / P, (W + P - 1) / P, C);
}

void MaxPoolingNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1], C = inShape[2];
    uint32_t H_ = discardTail ? H / P : (H + P - 1) / P;
    uint32_t W_ = discardTail ? W / P : (W + P - 1) / P;

    maxpoolDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    uint32_t maxpoolConstants[] = {H, W, C, P};

    cmdBuff
        .bindPipeline(maxpool)
        .bindDescSets({maxpoolDescSet})
        .setPushConstants(0, sizeof(maxpoolConstants), maxpoolConstants)
        .dispatch(H_, W_, C)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  



/////////////////////////////////////////////////////////////////////////////////////////
// FlattenNode
/////////////////////////////////////////////////////////////////////////////////////////
FlattenNode::FlattenNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
}

void FlattenNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    Tensor& outTensor = (*this)["out0"] = (*this)["in0"];
    outTensor.reshape(outTensor.numElements());  
}

void FlattenNode::run(CommandBuffer cmdBuff) 
{
}  


/////////////////////////////////////////////////////////////////////////////////////////
// FullyConnectedNode
/////////////////////////////////////////////////////////////////////////////////////////
FullyConnectedNode::FullyConnectedNode(uint32_t inDim, uint32_t outDim)
: I(inDim), O(outDim) 
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    /*const char* gemmSrc = src_gemm_naive;
    gemmTileSize = gGemmTileSize.at(gemmSrc);*/

    const char* gemmSrc = src_gemm_kSplit;

    gemm = requestPipeline(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);

	setZero = requestPipeline(src_setZero);
    setZeroDescSet = setZero.descSetLayout(0).newDescSet(gDestSetPool);
}

void FullyConnectedNode::prepare() 
{
    _ASSERT((*this)["in0"].isShapeOf(I));
    _ASSERT((*this)["weight"].isShapeOf(I, O));
    _ASSERT((*this)["bias"].isShapeOf(O));
    (*this)["out0"] = Tensor(O); 
}

void FullyConnectedNode::run(CommandBuffer cmdBuff)
{
    uint32_t M = 1;
    uint32_t K = (*this)["in0"].shape()[0];
    uint32_t N = (*this)["out0"].shape()[0];

    setZeroDescSet.write({
        (*this)["out0"].buffer(),
    });
    gemmDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

	uint32_t setZeroConstants[] = { N };
    uint32_t gemmConstants[] = {M, K, N};

    cmdBuff
		.bindPipeline(setZero)
		.bindDescSets({setZeroDescSet})
		.setPushConstants(0, sizeof(setZeroConstants), setZeroConstants)
		.dispatch(N)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
        )

        .bindPipeline(gemm)
        .bindDescSets({gemmDescSet})
        .setPushConstants(0, sizeof(gemmConstants), gemmConstants)
        .dispatch0(CEIL_DIV(N, 32), M, CEIL_DIV(K, 16))
        //.dispatch0(CEIL_DIV(N, gemmTileSize), CEIL_DIV(M, gemmTileSize))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );

}


/////////////////////////////////////////////////////////////////////////////////////////
// BatchNormalizationNode
/////////////////////////////////////////////////////////////////////////////////////////
BatchNormalizationNode::BatchNormalizationNode(uint32_t numChannels)
: C(numChannels)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("gamma", NodeSlot::input);
    addSlot("beta", NodeSlot::input);
    addSlot("mean", NodeSlot::input);
    addSlot("var", NodeSlot::input);

    batchnorm = requestPipeline(src_batchnorm);
    batchnormDescSet = batchnorm.descSetLayout(0).newDescSet(gDestSetPool);
}

void BatchNormalizationNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(-1, -1, C));
    _ASSERT((*this)["gamma"].isShapeOf(C));
    _ASSERT((*this)["beta"].isShapeOf(C));
    _ASSERT((*this)["mean"].isShapeOf(C));
    _ASSERT((*this)["var"].isShapeOf(C));

    const auto& inShape = (*this)["in0"].shape();
    (*this)["out0"] = Tensor(inShape);
}

void BatchNormalizationNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1], C = inShape[2];

    batchnormDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["gamma"].buffer(),
        (*this)["beta"].buffer(),
        (*this)["mean"].buffer(),
        (*this)["var"].buffer(),
    });

    uint32_t constants[] = {H, W, C};

    cmdBuff
        .bindPipeline(batchnorm)
        .bindDescSets({batchnormDescSet})
        .setPushConstants(0, sizeof(constants), constants)
        .dispatch(H, W, C)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


/////////////////////////////////////////////////////////////////////////////////////////
// GlobalAvgPoolNode
/////////////////////////////////////////////////////////////////////////////////////////
GlobalAvgPoolNode::GlobalAvgPoolNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    globalAvgPool = requestPipeline(src_globalAvgPool);
    globalAvgPoolDescSet = globalAvgPool.descSetLayout(0).newDescSet(gDestSetPool);
}

void GlobalAvgPoolNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    uint32_t C = inShape[2];

    (*this)["out0"] = Tensor(C);
}

void GlobalAvgPoolNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1], C = inShape[2];

    globalAvgPoolDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    uint32_t constants[] = {H, W, C};

    cmdBuff
        .bindPipeline(globalAvgPool)
        .bindDescSets({globalAvgPoolDescSet})
        .setPushConstants(0, sizeof(constants), constants)
        .dispatch(C)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


/////////////////////////////////////////////////////////////////////////////////////////
// AddNode
/////////////////////////////////////////////////////////////////////////////////////////
AddNode::AddNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("in1", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    add = requestPipeline(src_add);
    addDescSet = add.descSetLayout(0).newDescSet(gDestSetPool);
}

void AddNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    _ASSERT((*this)["in1"].validShape());
    _ASSERT((*this)["in0"].shape() == (*this)["in1"].shape());

    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void AddNode::run(CommandBuffer cmdBuff)
{
    uint32_t N = (*this)["in0"].numElements();

    addDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["in1"].buffer(),
    });

    uint32_t constants[] = {N};

    cmdBuff
        .bindPipeline(add)
        .bindDescSets({addDescSet})
        .setPushConstants(0, sizeof(constants), constants)
        .dispatch(N)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


/////////////////////////////////////////////////////////////////////////////////////////
// BasicBlock
/////////////////////////////////////////////////////////////////////////////////////////
BasicBlock::BasicBlock(uint32_t inChannels, uint32_t outChannels, uint32_t stride)
: conv1(inChannels, outChannels, 3, stride),
  bn1(outChannels),
  relu1(),
  conv2(outChannels, outChannels, 3, 1),
  bn2(outChannels),
  add(),
  relu2(),
  hasDownsample(stride != 1 || inChannels != outChannels),
  downsampleConv(nullptr),
  downsampleBN(nullptr)
{
    if (hasDownsample)
    {
        downsampleConv = new ConvolutionNode(inChannels, outChannels, 1, stride);
        downsampleBN = new BatchNormalizationNode(outChannels);
    }

    buildGraph();
}

BasicBlock::~BasicBlock()
{
    if (downsampleConv) delete downsampleConv;
    if (downsampleBN) delete downsampleBN;
}

void BasicBlock::buildGraph()
{
    // Main path: in0 -> conv1 -> bn1 -> relu1 -> conv2 -> bn2
    conv1.slot("out0") - bn1.slot("in0");
    bn1.slot("out0") - relu1.slot("in0");
    relu1.slot("out0") - conv2.slot("in0");
    conv2.slot("out0") - bn2.slot("in0");

    // Skip connection
    if (hasDownsample)
    {
        // Downsample path: in0 -> downsampleConv -> downsampleBN
        downsampleConv->slot("out0") - downsampleBN->slot("in0");

        // Add: bn2.out0 + downsampleBN.out0 -> add.out0
        bn2.slot("out0") - add.slot("in0");
        downsampleBN->slot("out0") - add.slot("in1");
    }
    else
    {
        // Add: bn2.out0 + input -> add.out0
        bn2.slot("out0") - add.slot("in0");
        // Input will be connected to add.in1
    }

    // Final relu
    add.slot("out0") - relu2.slot("in0");

    // Define group slots
    if (hasDownsample)
    {
        defineSlot("in0", conv1.slot("in0"));
        defineSlot("in0_skip", downsampleConv->slot("in0"));
    }
    else
    {
        defineSlot("in0", conv1.slot("in0"));
        defineSlot("in0_skip", add.slot("in1"));
    }
    defineSlot("out0", relu2.slot("out0"));
}

void BasicBlock::loadWeights(const NpzLoader& npz, const std::string& prefix)
{
    loadConvolutionWeights(conv1, npz, prefix + ".conv1");
    loadBatchNormalizationWeights(bn1, npz, prefix + ".bn1");
    loadConvolutionWeights(conv2, npz, prefix + ".conv2");
    loadBatchNormalizationWeights(bn2, npz, prefix + ".bn2");

    if (hasDownsample)
    {
        loadConvolutionWeights(*downsampleConv, npz, prefix + ".downsample.0");
        loadBatchNormalizationWeights(*downsampleBN, npz, prefix + ".downsample.1");
    }
}
