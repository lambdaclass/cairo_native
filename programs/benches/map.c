#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>


typedef struct map_return_values
{
    uint64_t remaining_gas;
    struct {
        uint8_t discriminant;
        union {
            uint64_t ok[4];
            struct {
                void *ptr;
                uint32_t len;
                uint32_t cap;
            } err;
        };
    } result;
} map_return_values_t;


int main()
{
    map_return_values_t return_values;
    void (*ptr)(map_return_values_t *, void *, uint64_t);
    void *handle;

    handle = dlopen(NULL, RTLD_LAZY);

    *(void **) (&ptr) = dlsym(handle, "_mlir_ciface_map::map::main");
    ptr(&return_values, NULL, 0);

    return 0;
}
