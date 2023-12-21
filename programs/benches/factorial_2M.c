#include <stddef.h>
#include <stdint.h>


typedef struct factorial_return_values
{
    unsigned __int128 remaining_gas;
    struct {
        uint8_t discriminant;
        union {
            uint64_t ok[4];
            struct {
                void* ptr;
                uint32_t len;
                uint32_t cap;
            } err;
        };
    } result;
} factorial_return_values_t;


static void run_bench(factorial_return_values_t*, void*, uint64_t)
__attribute__((weakref("_mlir_ciface_factorial_2M::factorial_2M::main(f1)")));


int main()
{
    factorial_return_values_t return_values;

    run_bench(&return_values, NULL, 0);

    // discriminant 0 == Result::Ok
    // return value 0 == ok in hyperfine
    return return_values.result.discriminant;
}
