// The C API from C++17. A std::unique_ptr owns the handle, and the
// context passes to eindir only once eindir_objective_new has returned a
// handle: on failure it never calls free_fn. eindir::version() is the only
// helper that eindir-core.hpp adds.
#include <eindir-core.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

// f(x) = sum_i w_i x_i^2
struct Weighted {
    std::vector<double> w;
};

DLManagedTensorVersioned borrowed(double *data, int64_t *shape) {
    DLManagedTensorVersioned t{};
    t.version.major = DLPACK_MAJOR_VERSION;
    t.version.minor = DLPACK_MINOR_VERSION;
    t.dl_tensor.data = data;
    t.dl_tensor.device.device_type = kDLCPU;
    t.dl_tensor.ndim = 1;
    t.dl_tensor.dtype.code = kDLFloat;
    t.dl_tensor.dtype.bits = 64;
    t.dl_tensor.dtype.lanes = 1;
    t.dl_tensor.shape = shape;
    return t;
}

eindir_status_t weighted_eval(void *user_data, const DLManagedTensorVersioned *x,
                              double *value_out) {
    const auto *f = static_cast<const Weighted *>(user_data);
    const DLTensor &t = x->dl_tensor;
    const auto n = static_cast<int64_t>(f->w.size());
    if (t.data == nullptr || t.ndim != 1 || t.shape[0] != n || t.dtype.code != kDLFloat ||
        t.dtype.bits != 64 || t.dtype.lanes != 1 || t.device.device_type != kDLCPU ||
        (t.strides != nullptr && t.strides[0] != 1)) {
        return EINDIR_INVALID_PARAMETER;
    }
    const auto *xs = reinterpret_cast<const double *>(static_cast<const char *>(t.data) +
                                                      t.byte_offset);
    double sum = 0.0;
    for (std::size_t i = 0; i < f->w.size(); ++i) {
        sum += f->w[i] * xs[i] * xs[i];
    }
    *value_out = sum;
    return EINDIR_SUCCESS;
}

void delete_weighted(void *user_data) { delete static_cast<Weighted *>(user_data); }

using Objective = std::unique_ptr<eindir_objective_t, decltype(&eindir_objective_free)>;

Objective make_weighted(std::vector<double> w, std::vector<double> low,
                        std::vector<double> high) {
    int64_t low_shape[1] = {static_cast<int64_t>(low.size())};
    int64_t high_shape[1] = {static_cast<int64_t>(high.size())};
    DLManagedTensorVersioned low_t = borrowed(low.data(), low_shape);
    DLManagedTensorVersioned high_t = borrowed(high.data(), high_shape);
    auto context = std::make_unique<Weighted>(Weighted{std::move(w)});
    const std::size_t dim = context->w.size();
    Objective obj(eindir_objective_new(dim, &low_t, &high_t, weighted_eval, nullptr,
                                       context.get(), delete_weighted),
                  eindir_objective_free);
    if (obj) {
        context.release();
    }
    return obj;
}

}  // namespace

int main() {
    if (std::strcmp(eindir::version(), eindir_core_version()) != 0) {
        std::fprintf(stderr, "FAIL: eindir::version()\n");
        return 1;
    }

    Objective obj = make_weighted({2.0, 3.0}, {-1.0, -1.0}, {1.0, 1.0});
    if (!obj) {
        std::fprintf(stderr, "FAIL: %s\n", eindir_last_error());
        return 1;
    }
    std::vector<double> x{0.5, -1.0};
    int64_t shape[1] = {2};
    DLManagedTensorVersioned x_t = borrowed(x.data(), shape);
    double f = 0.0;
    if (eindir_objective_eval(obj.get(), &x_t, &f) != EINDIR_SUCCESS || f != 3.5) {
        std::fprintf(stderr, "FAIL: eindir_objective_eval gave %g\n", f);
        return 1;
    }
    std::printf("eindir-core %s from C++: f(0.5, -1) = %g\n", eindir::version(), f);

    // Bounds of the wrong length: no handle, and the context is freed here.
    Objective refused = make_weighted({2.0, 3.0}, {-1.0}, {1.0});
    if (refused) {
        std::fprintf(stderr, "FAIL: bounds of length 1 for dim 2 were accepted\n");
        return 1;
    }
    std::printf("refused: %s\n", eindir_last_error());
    return 0;
}
