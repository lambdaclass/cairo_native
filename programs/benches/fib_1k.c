#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>


typedef struct fib_return_values
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
} fib_return_values_t;


int main()
{
    fib_return_values_t return_values;
    void (*ptr)(fib_return_values_t *, void *, uint64_t);
    void *handle;

    handle = dlopen(NULL, RTLD_LAZY);

    *(void **) (&ptr) = dlsym(handle, "_mlir_ciface_fib_1k::fib_1k::main");
    ptr(&return_values, NULL, 0);

    return 0;
}
