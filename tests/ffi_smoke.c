#include <eindir-core.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const eindir_abi_stamp_t stamp = eindir_core_abi_stamp();
    if (strcmp(eindir_core_abi_family(), "eindir.objective") != 0 ||
        stamp.abi_major != 1 || stamp.objective_layout != 2 ||
        stamp.objective_size == 0 || stamp.objective_align == 0 ||
        stamp.dlpack_major != 1 ||
        eindir_core_abi_compatible(&stamp) != 1) {
        return 1;
    }
    printf("%s abi=%u.%u layout=%u dlpack=%u.%u features=%llu\n",
           eindir_core_version(), stamp.abi_major, stamp.abi_minor,
           stamp.objective_layout, stamp.dlpack_major, stamp.dlpack_minor,
           (unsigned long long)stamp.features);
    return 0;
}
