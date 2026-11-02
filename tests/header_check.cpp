#include <dlpack/dlpack.h>
#include <eindir-core.hpp>

#include <type_traits>

#if DLPACK_MAJOR_VERSION != 1
#error "eindir-core.h must be checked against DLPack 1.x headers"
#endif

static_assert(std::is_same<decltype(&eindir_objective_eval),
                           eindir_status_t (*)(const eindir_objective_t *,
                                               const DLManagedTensorVersioned *,
                                               double *)>::value,
              "eindir_objective_eval must take the DLPack tensor type");
static_assert(std::is_same<decltype(&eindir_objective_grad),
                           eindir_status_t (*)(const eindir_objective_t *,
                                               const DLManagedTensorVersioned *,
                                               DLManagedTensorVersioned *)>::value,
              "eindir_objective_grad must take the DLPack tensor type");
static_assert(std::is_same<EindirEvalFn,
                           eindir_status_t (*)(void *,
                                               const DLManagedTensorVersioned *,
                                               double *)>::value,
              "EindirEvalFn must take the DLPack tensor type");

int main() {
    const char *(*version_api)() noexcept = eindir::version;
    const char *(*core_version_api)() = ::eindir_core_version;
    (void)version_api;
    (void)core_version_api;
    return 0;
}
