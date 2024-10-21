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
            uint32_t len;
            uint32_t cap;
        } err;
    } result;
} map_return_values_t;


static void run_bench(map_return_values_t *, uint64_t)
    __attribute__((weakref("_mlir_ciface_logistic_map::logistic_map::main(f2)")));


int main()
{
    map_return_values_t return_values;

    run_bench(&return_values, 0);
    assert(return_values.result.discriminant == 0);

    return 0;
}
