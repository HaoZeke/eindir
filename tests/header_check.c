#include <dlpack/dlpack.h>
#include <eindir-core.h>

#if DLPACK_MAJOR_VERSION != 1
#error "eindir-core.h must be checked against DLPack 1.x headers"
#endif

static enum eindir_status_t eval_cb(void *user_data,
                                    const struct DLManagedTensorVersioned *x,
                                    double *value_out) {
    (void)user_data;
    (void)x;
    *value_out = 0.0;
    return EINDIR_SUCCESS;
}

static enum eindir_status_t grad_cb(void *user_data,
                                    const struct DLManagedTensorVersioned *x,
                                    struct DLManagedTensorVersioned *grad_out) {
    (void)user_data;
    (void)x;
    (void)grad_out;
    return EINDIR_SUCCESS;
}

int main(void) {
    EindirEvalFn eval_fn = eval_cb;
    EindirGradFn grad_fn = grad_cb;
    enum eindir_status_t (*eval_api)(const struct eindir_objective_t *,
                                     const struct DLManagedTensorVersioned *,
                                     double *) = eindir_objective_eval;
    enum eindir_status_t (*grad_api)(const struct eindir_objective_t *,
                                     const struct DLManagedTensorVersioned *,
                                     struct DLManagedTensorVersioned *) =
        eindir_objective_grad;
    struct eindir_objective_t *(*new_api)(
        uintptr_t, const struct DLManagedTensorVersioned *,
        const struct DLManagedTensorVersioned *, EindirEvalFn, EindirGradFn,
        void *, EindirFreeFn) = eindir_objective_new;
    (void)eval_fn;
    (void)grad_fn;
    (void)eval_api;
    (void)grad_api;
    (void)new_api;
    return 0;
}
