#include <cmath>
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
#include <limits>

static inline double normal_pdf(double x) {
    static const double inv_sqrt_2pi = 0.39894228040143267794;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

static inline double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

static double gaussian_truncated_mean(double a, double b) {
    double Fa = normal_cdf(a);
    double Fb = normal_cdf(b);
    double Pa = normal_pdf(a);
    double Pb = normal_pdf(b);
    double Z = Fb - Fa;
    if (Z <= 1e-15) return 0.5 * (a + b);
    return (Pa - Pb) / Z;
}

static double gaussian_quantile(double p) {
    p = std::clamp(p, 1e-12, 1.0 - 1e-12);
    double lo = -10.0, hi = 10.0;
    for (int it = 0; it < 120; ++it) {
        double mid = 0.5 * (lo + hi);
        if (normal_cdf(mid) < p) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

struct LloydMaxResult {
    std::array<double, 8> levels;
    std::array<double, 7> thresholds;
};

LloydMaxResult design_gaussian_8level(int max_iter = 50, double tol = 1e-12) {
    constexpr int M = 8;
    std::array<double, M> c{};
    std::array<double, M - 1> t{};

    for (int i = 1; i < M; ++i) {
        t[i - 1] = gaussian_quantile(double(i) / M);
    }

    const double xmin = -6.0;
    const double xmax =  6.0;

    for (int it = 0; it < max_iter; ++it) {
        std::array<double, M> c_new{};
        std::array<double, M - 1> t_new{};

        for (int i = 0; i < M; ++i) {
            double a = (i == 0) ? xmin : t[i - 1];
            double b = (i == M - 1) ? xmax : t[i];
            c_new[i] = gaussian_truncated_mean(a, b);
        }

        for (int i = 0; i < M - 1; ++i) {
            t_new[i] = 0.5 * (c_new[i] + c_new[i + 1]);
        }

        double diff = 0.0;
        for (int i = 0; i < M; ++i) diff = std::max(diff, std::abs(c_new[i] - c[i]));
        for (int i = 0; i < M - 1; ++i) diff = std::max(diff, std::abs(t_new[i] - t[i]));

        c = c_new;
        t = t_new;

        if (diff < tol) break;
    }

    LloydMaxResult res;
    for (int i = 0; i < M; ++i) res.levels[i] = c[i];
    for (int i = 0; i < M - 1; ++i) res.thresholds[i] = t[i];
    return res;
}

std::array<double, 8> build_gaussian_codebook();

int main() {
    auto res = design_gaussian_8level();

    std::cout << "levels:\n";
    for (double v : res.levels) std::cout << v << " ";
    std::cout << "\nthresholds:\n";
    for (double v : res.thresholds) std::cout << v << " ";
    std::cout << "\n";
    return 0;
}
