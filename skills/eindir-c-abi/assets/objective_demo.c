/* A C objective through the eindir C ABI: DLPack tensors for bounds and
 * points, value and gradient callbacks, error reporting, an embedded
 * handle with a semantic descriptor, and cleanup. Exits non-zero on the
 * first check that fails. */
#include <dlpack/dlpack.h>
#include <eindir-core.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 3

#define CHECK(cond, what)                                                   \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (eindir_last_error: %s)\n", (what),   \
                    eindir_last_error());                                   \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* A borrowed, contiguous, one-dimensional float64 CPU tensor over data. */
static struct DLManagedTensorVersioned vector(double *data, int64_t *shape) {
    struct DLManagedTensorVersioned t;
    memset(&t, 0, sizeof t);
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

/* eindir hands x to the callbacks unchecked, so they check it. */
static int is_f64_vector(const struct DLManagedTensorVersioned *t, int64_t n) {
    const DLTensor *x = &t->dl_tensor;
    return x->data != NULL && x->ndim == 1 && x->shape[0] == n &&
           x->dtype.code == kDLFloat && x->dtype.bits == 64 &&
           x->dtype.lanes == 1 && x->device.device_type == kDLCPU &&
           (x->strides == NULL || x->strides[0] == 1);
}

static double *elements(const struct DLManagedTensorVersioned *t) {
    return (double *)((char *)t->dl_tensor.data + t->dl_tensor.byte_offset);
}

/* f(x) = sum_i (x_i - centre_i)^2, counting evaluations. */
struct sphere {
    double centre[DIM];
    unsigned evals;
    int *freed;
};

static enum eindir_status_t sphere_eval(void *user_data,
                                        const struct DLManagedTensorVersioned *x,
                                        double *value_out) {
    struct sphere *s = user_data;
    const double *xs;
    double sum = 0.0;
    int i;
    if (!is_f64_vector(x, DIM)) {
        return EINDIR_INVALID_PARAMETER;
    }
    xs = elements(x);
    for (i = 0; i < DIM; ++i) {
        sum += (xs[i] - s->centre[i]) * (xs[i] - s->centre[i]);
    }
    s->evals++;
    *value_out = sum;
    return EINDIR_SUCCESS;
}

static enum eindir_status_t sphere_grad(void *user_data,
                                        const struct DLManagedTensorVersioned *x,
                                        struct DLManagedTensorVersioned *grad_out) {
    struct sphere *s = user_data;
    const double *xs;
    double *g;
    int i;
    if (!is_f64_vector(x, DIM) || !is_f64_vector(grad_out, DIM)) {
        return EINDIR_INVALID_PARAMETER;
    }
    xs = elements(x);
    g = elements(grad_out);
    for (i = 0; i < DIM; ++i) {
        g[i] = 2.0 * (xs[i] - s->centre[i]);
    }
    return EINDIR_SUCCESS;
}

static void free_sphere(void *user_data) {
    struct sphere *s = user_data;
    *s->freed = 1;
    free(s);
}

/* An objective embedded as the first member of a producer's own struct. */
struct producer {
    eindir_objective_t base;
    double low[DIM];
    double high[DIM];
    struct sphere sphere;
};

static const eindir_objective_descriptor_t produced = {
    .schema_id = "eindir.objective-descriptor.v1",
    .producer_id = "example.sphere",
    .length_unit = "angstrom",
    .energy_unit = "eV",
    .energy_sign = 1,
    .gradient_sign = 1,
    .operations =
        EINDIR_OBJECTIVE_OPERATION_ENERGY | EINDIR_OBJECTIVE_OPERATION_FORCES,
    .tensor_device_type = EINDIR_TENSOR_DEVICE_CPU,
    .tensor_dtype_code = EINDIR_TENSOR_DTYPE_FLOAT,
    .tensor_dtype_bits = 64,
    .tensor_dtype_lanes = 1,
    .tensor_layout = EINDIR_TENSOR_LAYOUT_CONTIGUOUS,
    .callback_lifetime = EINDIR_CALLBACK_LIFETIME_BORROWED_SYNC,
};

int main(void) {
    const eindir_abi_stamp_t stamp = eindir_core_abi_stamp();
    double low[DIM] = {-5.0, -5.0, -5.0};
    double high[DIM] = {5.0, 5.0, 5.0};
    double x[DIM] = {0.0, 0.0, 0.0};
    double g[DIM] = {0.0, 0.0, 0.0};
    double back[DIM] = {0.0, 0.0, 0.0};
    double pair[2] = {0.0, 0.0};
    int64_t shape[1] = {DIM};
    int64_t pair_shape[1] = {2};
    struct DLManagedTensorVersioned low_t = vector(low, shape);
    struct DLManagedTensorVersioned high_t = vector(high, shape);
    struct DLManagedTensorVersioned x_t = vector(x, shape);
    struct DLManagedTensorVersioned g_t = vector(g, shape);
    struct DLManagedTensorVersioned back_t = vector(back, shape);
    struct DLManagedTensorVersioned pair_t = vector(pair, pair_shape);
    struct sphere plain_ctx = {{1.0, 2.0, 3.0}, 0, NULL};
    struct sphere *ctx;
    eindir_objective_t *obj;
    eindir_objective_t *plain;
    eindir_objective_descriptor_t required;
    struct producer p;
    char message[256];
    double f = 0.0;
    int freed = 0;
    int i;

    CHECK(strcmp(eindir_core_abi_family(), "eindir.objective") == 0, "ABI family");
    CHECK(eindir_core_abi_compatible(&stamp), "ABI stamp");
    CHECK(stamp.dlpack_major == DLPACK_MAJOR_VERSION, "DLPack major version");
    CHECK(stamp.features & EINDIR_ABI_FEATURE_GRADIENT, "gradient feature bit");
    printf("eindir-core %s: ABI %u.%u, objective layout %u, DLPack %u.%u\n",
           eindir_core_version(), stamp.abi_major, stamp.abi_minor,
           stamp.objective_layout, stamp.dlpack_major, stamp.dlpack_minor);

    ctx = malloc(sizeof *ctx);
    CHECK(ctx != NULL, "malloc");
    *ctx = plain_ctx;
    ctx->freed = &freed;
    obj = eindir_objective_new(DIM, &low_t, &high_t, sphere_eval, sphere_grad,
                               ctx, free_sphere);
    CHECK(obj != NULL, "eindir_objective_new");
    CHECK(eindir_objective_dim(obj) == DIM, "eindir_objective_dim");
    CHECK(eindir_objective_has_grad(obj) == 1, "eindir_objective_has_grad");
    CHECK(eindir_objective_descriptor(obj) == NULL,
          "handles from eindir_objective_new carry no descriptor");
    low[0] = -1.0;
    CHECK(eindir_objective_bounds_low(obj, &back_t) == EINDIR_SUCCESS &&
              back[0] == -5.0,
          "eindir_objective_new copied the bounds");

    CHECK(eindir_objective_eval(obj, &x_t, &f) == EINDIR_SUCCESS && f == 14.0,
          "eindir_objective_eval");
    CHECK(eindir_objective_grad(obj, &x_t, &g_t) == EINDIR_SUCCESS &&
              g[0] == -2.0 && g[1] == -4.0 && g[2] == -6.0,
          "eindir_objective_grad");
    printf("f(0) = %g, grad f(0) = [%g, %g, %g], %u evaluation(s)\n", f, g[0],
           g[1], g[2], ctx->evals);

    CHECK(eindir_objective_new(DIM, &pair_t, &high_t, sphere_eval, NULL, NULL,
                               NULL) == NULL,
          "bounds of the wrong length are refused");
    printf("wrong bounds length: %s\n", eindir_last_error());
    CHECK(eindir_objective_eval(NULL, &x_t, &f) == EINDIR_INVALID_PARAMETER,
          "NULL objective");
    printf("NULL objective: %s\n", eindir_last_error());
    snprintf(message, sizeof message, "%s", eindir_last_error());
    CHECK(eindir_objective_eval(obj, &pair_t, &f) == EINDIR_INVALID_PARAMETER,
          "the callback refuses a 2-vector");
    CHECK(strcmp(eindir_last_error(), message) == 0,
          "a failing callback leaves the last error in place");

    plain = eindir_objective_new(DIM, &low_t, &high_t, sphere_eval, NULL,
                                 &plain_ctx, NULL);
    CHECK(plain != NULL && eindir_objective_has_grad(plain) == 0,
          "an objective without a gradient callback");
    CHECK(eindir_objective_grad(plain, &x_t, &g_t) == EINDIR_INVALID_PARAMETER,
          "grad without a gradient callback");
    printf("gradient without a callback: %s\n", eindir_last_error());
    snprintf(message, sizeof message, "%s", eindir_last_error());
    CHECK(eindir_objective_eval(plain, &x_t, &f) == EINDIR_SUCCESS,
          "eval after an error");
    CHECK(strcmp(eindir_last_error(), message) == 0,
          "a success leaves the last error in place");
    eindir_objective_free(plain);

    for (i = 0; i < DIM; ++i) {
        p.low[i] = -5.0;
        p.high[i] = 5.0;
    }
    p.sphere = plain_ctx;
    p.base.dim = DIM;
    p.base.low = p.low;
    p.base.high = p.high;
    p.base.eval_fn = sphere_eval;
    p.base.grad_fn = sphere_grad;
    p.base.user_data = &p.sphere;
    p.base.free_fn = NULL;
    p.base.descriptor = &produced;
    CHECK(eindir_objective_eval(&p.base, &x_t, &f) == EINDIR_SUCCESS && f == 14.0,
          "eval through an embedded handle");
    CHECK(eindir_objective_descriptor(&p.base) == &produced,
          "an embedded handle returns its descriptor");

    memset(&required, 0, sizeof required);
    required.schema_id = produced.schema_id;
    required.operations = EINDIR_OBJECTIVE_OPERATION_ENERGY;
    required.tensor_dtype_bits = 64;
    CHECK(eindir_objective_descriptor_compatible(&produced, &required) == 1,
          "zero and empty fields are wildcards");
    required.energy_unit = "hartree";
    CHECK(eindir_objective_descriptor_compatible(&produced, &required) == 0,
          "a unit mismatch is refused");
    printf("energy unit mismatch: %s\n", eindir_last_error());
    required.energy_unit = "";
    required.schema_id = "";
    CHECK(eindir_objective_descriptor_compatible(&produced, &required) == 0,
          "schema_id is never a wildcard");

    eindir_objective_free(obj);
    CHECK(freed == 1, "eindir_objective_free called free_fn");
    eindir_objective_free(NULL);
    printf("all checks passed\n");
    return 0;
}
