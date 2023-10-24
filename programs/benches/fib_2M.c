#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>


typedef struct fib_return_values
{
    unsigned __int128 remaining_gas;
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


static void run_bench(fib_return_values_t *, void *, uint64_t)
    __attribute__((weakref("_mlir_ciface_fib_2M::fib_2M::main")));


int main()
{
    fib_return_values_t return_values;

    run_bench(&return_values, NULL, 0);

    // discriminant 0 == Result::Ok
    // return value 0 == ok in hyperfine
    return return_values.result.discriminant;
}
