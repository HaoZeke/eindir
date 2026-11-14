---
name: eindir-c-abi
description: Build and install the eindir C library, link C or C++ programs to it, and define objectives in C with DLPack 1.0 tensors that Rust drivers can call. Use when compiling against eindir-core.h or eindir-core.hpp, writing eval or gradient callbacks, embedding eindir_objective_t in another library's struct, checking ABI compatibility or semantic descriptors, or when a program fails to start because libeindir_core.so.0.7 is missing.
license: See LICENSE in the eindir repository
compatibility: Rust 1.88 or newer with cargo-c 0.10.17 or newer, DLPack 1.0 headers, a C99 compiler and, for the C++ example, a C++17 compiler. The script needs Python 3.11 or newer. Without DLPACK_ROOT set, it downloads the DLPack archive, so it needs network access. Tested on Linux only.
---

# The C library

The C library exports fifteen functions: version and ABI queries, the objective handle (`eindir_objective_new`, `_free`, `_dim`, `_has_grad`, `_bounds_low`, `_bounds_high`, `_eval`, `_grad`), semantic descriptors and `eindir_last_error`.
Points, gradients and bounds cross the boundary as DLPack 1.0 `DLManagedTensorVersioned` tensors that the caller owns and lends for the length of one call.
The C and C++ blocks are excerpts of `assets/objective_demo.c` and `assets/raii_demo.cpp`, which `scripts/build_and_run.py` compiles and runs.

## Build and install

```bash
cargo install cargo-c        # once
cargo cinstall --release --prefix="$PREFIX" --libdir=lib
```

This installs `include/eindir-core.h`, `include/eindir-core.hpp`, `lib/libeindir_core.a`, `lib/libeindir_core.so.0.7.0` with the links `libeindir_core.so.0.7` and `libeindir_core.so`, and `lib/pkgconfig/eindir-core.pc`.
cargo-c turns on the `capi` feature by itself.

- A plain `cargo build --release` builds `libeindir_core.so` with no `eindir_` symbols at all.
  Add `--features capi`.
- `cargo cbuild --release` leaves the header and libraries under `target/<host triple>/release/`, but that `libeindir_core.so` has the soname `libeindir_core.so.0.7` and nothing of that name sits next to it.
  A program linked there fails to start with `error while loading shared libraries: libeindir_core.so.0.7: cannot open shared object file`.
  Install to a prefix, as above, or link the static archive.
- `eindir-core.h` includes `<dlpack/dlpack.h>`, so DLPack 1.0 must be on the include path.
  Ubuntu's `libdlpack-dev` is 0.6, which has no `DLManagedTensorVersioned`.
  The `ffi-smoke` job in `.github/workflows/ci.yml` pins the v1.0 archive and its sha256.
- To generate the header without cargo-c: `cbindgen --config cbindgen.toml --crate eindir-core --output eindir-core.h`.
  Under rustup with a default toolchain older than 1.88, cbindgen fails in `cargo metadata` with "feature `edition2024` is required".
  Set `RUSTUP_TOOLCHAIN=stable` for that command.
- With `-Dwith_examples=true`, `meson.build` calls `subdir('examples')`, but `examples/` has one Rust example and no `meson.build`.

## Compile and link

```bash
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
cc -std=c99 -Wall -Wextra -Wpedantic -Werror -I"$DLPACK_ROOT/include" \
  $(pkg-config --cflags eindir-core) objective_demo.c \
  $(pkg-config --libs eindir-core) -Wl,-rpath,"$PREFIX/lib" -o objective_demo
```

For a static link, name `"$PREFIX/lib/libeindir_core.a"` in place of `-leindir_core` and add the system libraries that `pkg-config --static --libs eindir-core` lists after it.
C++ uses the same flags with `c++ -std=c++17`.

## Tensors

A borrowed, contiguous float64 vector on the CPU.
Zeroing the struct leaves `strides` NULL (contiguous), `byte_offset` 0 and no deleter.
The `shape` array must outlive the call.

```c
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
```

## Callbacks

`eindir_objective_eval` and `eindir_objective_grad` check only that the handle, the point and the output are not NULL, then hand the point to your callback as it is.
The callback has to check the shape, dtype, device and strides, and honour `byte_offset`:

```c
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
```

- The status the callback returns is what `eval` or `grad` returns.
- A callback cannot set the error message.
  After a callback fails, `eindir_last_error()` still returns whatever an earlier eindir call left there.
  Keep your own error state in `user_data`.
- `eindir_last_error()` is per thread, and a successful call does not clear it.
  Read it only right after a call that returned an error status or NULL.
- Nothing checks `eval_fn`, which must not be NULL.
  `grad_fn` may be NULL: then `eindir_objective_has_grad` returns 0 and `eindir_objective_grad` returns `EINDIR_INVALID_PARAMETER` with "no gradient callback".
- Rust drivers see a C objective through `EindirObjectiveWrapper` in `src/ffi.rs`.
  It turns a failed value callback into NaN, returns zeros when there is no gradient callback, and ignores the status of the gradient callback.

## Ownership

```c
    ctx = malloc(sizeof *ctx);
    CHECK(ctx != NULL, "malloc");
    *ctx = plain_ctx;
    ctx->freed = &freed;
    obj = eindir_objective_new(DIM, &low_t, &high_t, sphere_eval, sphere_grad,
                               ctx, free_sphere);
    CHECK(obj != NULL, "eindir_objective_new");
```

- `eindir_objective_new` copies the bounds, so their tensors can go once it returns.
  It checks that they are 1-D, `dim` long and not NULL, then reads `dim` doubles from `data`, ignoring dtype, strides and `byte_offset`.
  Pass contiguous float64 with a zero offset.
- When it returns NULL it has not called `free_fn`, so `user_data` is still yours.
- `eindir_objective_free` calls `free_fn(user_data)` when both are set, frees the bounds and the handle, and does nothing for NULL.
- A library can instead embed the handle as the first member of its own struct and pass `&p.base` wherever an `eindir_objective_t *` is wanted.
  The embedder owns `low`, `high` and everything else.
  Never call `eindir_objective_free` on an embedded handle.

```c
struct producer {
    eindir_objective_t base;
    double low[DIM];
    double high[DIM];
    struct sphere sphere;
};
```

```c
    p.base.dim = DIM;
    p.base.low = p.low;
    p.base.high = p.high;
    p.base.eval_fn = sphere_eval;
    p.base.grad_fn = sphere_grad;
    p.base.user_data = &p.sphere;
    p.base.free_fn = NULL;
    p.base.descriptor = &produced;
```

## ABI version and descriptors

`eindir_core_abi_family()` returns `"eindir.objective"`, and `eindir_core_abi_stamp()` returns an `eindir_abi_stamp_t` with the ABI version, the objective layout, the size and alignment of `eindir_objective_t`, the DLPack version and the feature bits.
`eindir_core_abi_compatible(&stamp)` compares a consumer's `eindir_abi_stamp_t` with the library's.
The ABI major, objective layout, size, alignment and DLPack major must be equal.
The consumer's ABI and DLPack minors must be no newer than the library's, and it must set no feature bit the library lacks.
The library advertises `EINDIR_ABI_FEATURE_BATCH`, but the C API has no batched call.
Evaluate one point per call.

Besides the ABI version, a descriptor states units, sign conventions, operations, and the device, dtype and layout of the tensors.
`eindir_objective_new` leaves the `descriptor` field NULL and takes no argument for it.
Producers that embed the handle set the field themselves, as above.
A consumer fills in the fields it needs and compares the two descriptors.

```c
    memset(&required, 0, sizeof required);
    required.schema_id = produced.schema_id;
    required.operations = EINDIR_OBJECTIVE_OPERATION_ENERGY;
    required.tensor_dtype_bits = 64;
    CHECK(eindir_objective_descriptor_compatible(&produced, &required) == 1,
          "zero and empty fields are wildcards");
    required.energy_unit = "hartree";
    CHECK(eindir_objective_descriptor_compatible(&produced, &required) == 0,
          "a unit mismatch is refused");
```

Empty or NULL strings and zero numbers in `required` match anything, and `operations` must be a subset.
`schema_id` is never a wildcard.
An empty one is refused, whatever the header comment says.
A mismatch sets the message "eindir objective semantic descriptor mismatch" without naming the field.

## C++

`eindir-core.hpp` includes the C header and adds `eindir::version()`, nothing more.
Own handles with `std::unique_ptr`, and hand the context over only once `eindir_objective_new` has succeeded:

```cpp
using Objective = std::unique_ptr<eindir_objective_t, decltype(&eindir_objective_free)>;
```

```cpp
    auto context = std::make_unique<Weighted>(Weighted{std::move(w)});
    const std::size_t dim = context->w.size();
    Objective obj(eindir_objective_new(dim, &low_t, &high_t, weighted_eval, nullptr,
                                       context.get(), delete_weighted),
                  eindir_objective_free);
    if (obj) {
        context.release();
    }
    return obj;
```

## Run the examples

```bash
RUSTUP_TOOLCHAIN=stable python scripts/build_and_run.py
```

The script installs the library under `target/skill-c-abi/prefix` of the eindir checkout, fetches DLPack 1.0 unless `DLPACK_ROOT` is set, then compiles both examples with `$CC` and `$CXX` and runs them, under valgrind too when it is installed.
`objective_demo` prints the library version, the ABI version, the objective layout and the DLPack version, then each error message it provokes, and ends with `all checks passed`.
`raii_demo` prints `f(0.5, -1) = 3.5` and the message for bounds of the wrong length.
