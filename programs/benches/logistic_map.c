#include <assert.h>
#include <stdint.h>


typedef struct map_return_values
{
    uint64_t range_check_counter;
    unsigned __int128 remaining_gas;
    struct {
        uint8_t discriminant;
        struct {
            void *ptr;
            uint32_t start;
            uint32_t end;
            uint32_t cap;
        } err;
    } result;
} map_return_values_t;

static void run_bench(map_return_values_t *, uint64_t)
    __attribute__((weakref("_mlir_ciface_logistic_map::logistic_map::main(f2)")));

extern uint64_t* cairo_native__set_costs_builtin(uint64_t*);

int main()
{
    uint64_t BuiltinCosts[7] = {1, 4050, 583, 4085, 491, 230, 604};

    cairo_native__set_costs_builtin(&BuiltinCosts[0]);

    map_return_values_t return_values;

    run_bench(&return_values, 0);
    assert((return_values.result.discriminant & 0x1) == 0);

    return 0;
}
