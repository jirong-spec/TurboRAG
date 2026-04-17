#include "tq_attention_ref.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

static inline float h2f(half x) {
    return __half2float(x);
}

std::vector<float> attention_ref_single_head(
    const std::vector<half>& q,
    const std::vector<half>& k_all,
    const std::vector<half>& v_all,
    int num_tokens,
    int head_dim) {

    if ((int)q.size() != head_dim) {
        throw std::runtime_error("q size mismatch");
    }
    if ((int)k_all.size() != num_tokens * head_dim) {
        throw std::runtime_error("k_all size mismatch");
    }
    if ((int)v_all.size() != num_tokens * head_dim) {
        throw std::runtime_error("v_all size mismatch");
    }

    std::vector<float> scores(num_tokens, 0.f);
    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);

    for (int t = 0; t < num_tokens; ++t) {
        float acc = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            acc += h2f(q[d]) * h2f(k_all[t * head_dim + d]);
        }
        scores[t] = acc * inv_sqrt_d;
    }

    float mx = *std::max_element(scores.begin(), scores.end());
    float denom = 0.f;
    for (int t = 0; t < num_tokens; ++t) {
        scores[t] = std::exp(scores[t] - mx);
        denom += scores[t];
    }
    for (int t = 0; t < num_tokens; ++t) {
        scores[t] /= denom;
    }

    std::vector<float> out(head_dim, 0.f);
    for (int t = 0; t < num_tokens; ++t) {
        float p = scores[t];
        for (int d = 0; d < head_dim; ++d) {
            out[d] += p * h2f(v_all[t * head_dim + d]);
        }
    }
    return out;
}

double mse_vec(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("mse_vec size mismatch");
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = (double)a[i] - (double)b[i];
        acc += x * x;
    }
    return acc / (double)a.size();
}

double max_abs_vec(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("max_abs_vec size mismatch");
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs((double)a[i] - (double)b[i]));
    }
    return m;
}
