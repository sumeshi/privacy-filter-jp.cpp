// pf-gemm-bench — isolates ggml's CPU matmul throughput (GFLOP/s) by dtype and
// shape, to explain where CPU time goes vs a BLAS-backed framework (numpy/torch
// = MKL/oneDNN). result = mul_mat(a[K,M], b[K,N]) -> [M,N]; FLOPs = 2*M*N*K.
//
// Finding it was written to demonstrate: ggml's float matmul is ~40 GFLOP/s flat
// at every size (no cache-blocked SGEMM -- the kernels are per-row vec_dot, tuned
// for quantized weights where bandwidth dominates), while its q8_0 path is
// 5-7x faster. MKL's blocked SGEMM is 12-29x the ggml-f32 rate (run mkl side
// separately, e.g. torch.matmul). See docs/cpu-perf.md.
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double gflops(ggml_backend_t be, ggml_type ta, int M, int K, int N, int iters) {
    ggml_init_params p = { (size_t) 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, ta, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_gallocr * ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);

    std::vector<float> rb(K * N);
    for (auto & x : rb) x = (float) (rand() % 1000) / 1000.0f - 0.5f;  // values irrelevant for timing
    ggml_backend_tensor_set(b, rb.data(), 0, rb.size() * sizeof(float));
    std::vector<float> ra(K * M);
    for (auto & x : ra) x = (float) (rand() % 1000) / 1000.0f - 0.5f;
    if (ta == GGML_TYPE_F32) {
        ggml_backend_tensor_set(a, ra.data(), 0, ra.size() * sizeof(float));
    } else {
        std::vector<char> buf(ggml_nbytes(a));
        ggml_quantize_chunk(ta, ra.data(), buf.data(), 0, M, K, nullptr);
        ggml_backend_tensor_set(a, buf.data(), 0, buf.size());
    }

    ggml_backend_graph_compute(be, gf);  // warm
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) ggml_backend_graph_compute(be, gf);
    auto t1 = std::chrono::high_resolution_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count() / iters;
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return 2.0 * M * N * K / s / 1e9;
}

int main(int argc, char ** argv) {
    const int nth = argc > 1 ? std::atoi(argv[1]) : 12;
    ggml_backend_load_all();  // pick the best CPU variant in a GGML_BACKEND_DL build
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    auto set_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(be)), "ggml_backend_set_n_threads");
    if (set_fn) set_fn(be, nth);
    struct { const char * name; int M, K, N; } shp[] = {
        { "expert gate_up N=16 ", 1280, 640, 16 },
        { "expert gate_up N=64 ", 1280, 640, 64 },
        { "expert gate_up N=512", 1280, 640, 512 },
        { "expert down    N=16 ", 640, 640, 16 },
        { "large 4096^2   N=512", 4096, 4096, 512 },
    };
    std::printf("ggml CPU mul_mat GFLOP/s, %d threads\n", nth);
    std::printf("%-22s %10s %10s %10s\n", "shape (M,K,N)", "f32", "f16", "q8_0");
    for (auto & s : shp) {
        const int it = (double) s.M * s.N > 1e6 ? 20 : 200;
        std::printf("%-22s %10.1f %10.1f %10.1f\n", s.name,
                    gflops(be, GGML_TYPE_F32, s.M, s.K, s.N, it),
                    gflops(be, GGML_TYPE_F16, s.M, s.K, s.N, it),
                    gflops(be, GGML_TYPE_Q8_0, s.M, s.K, s.N, it));
    }
    ggml_backend_free(be);
    return 0;
}
