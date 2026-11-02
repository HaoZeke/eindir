The generated `eindir-core.h` is valid C again: cbindgen no longer wraps it
in a C++ namespace, and tensor arguments use the DLPack
`struct DLManagedTensorVersioned` tag so C and C++ consumers can include it.
