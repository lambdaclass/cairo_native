#include <assert.h>
#include <stdint.h>


typedef struct fib_return_values
{
    uint64_t range_check_counter;
    uint64_t remaining_gas;
    struct {
        uint8_t discriminant;
        struct {
            void *ptr;
            uint32_t start;
            uint32_t end;
            uint32_t cap;
        } err;
    } result;
} fib_return_values_t;

static void run_bench(fib_return_values_t *, uint64_t)
    __attribute__((weakref("_mlir_ciface_fib_2M::fib_2M::main(f1)")));

extern uint64_t* cairo_native__set_costs_builtin(uint64_t*);

int main()
{
    uint64_t BuiltinCosts[7] = {1, 4050, 583, 4085, 491, 230, 604};

    cairo_native__set_costs_builtin(&BuiltinCosts[0]);

    fib_return_values_t return_values;

    run_bench(&return_values, 0xFFFFFFFFFFFFFFFF);
    assert((return_values.result.discriminant & 0x1) == 0);

    return 0;
}
