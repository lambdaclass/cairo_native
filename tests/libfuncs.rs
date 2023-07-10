mod common;
use crate::common::{
    casm_variant_to_sierra, felt, get_result_success, run_native_program, run_vm_program,
};
use common::load_cairo;
use pretty_assertions::assert_eq;
use serde_json::json;

#[test]
fn enum_init() {
    let (source, program, runner) = load_cairo! {
        enum MySmallEnum {
            A: felt252,
        }

        enum MyEnum {
            A: felt252,
            B: u8,
            C: u16,
            D: u32,
            E: u64,
        }

        fn run_test() -> (MySmallEnum, MyEnum, MyEnum, MyEnum, MyEnum, MyEnum) {
            (
                MySmallEnum::A(-1),
                MyEnum::A(5678),
                MyEnum::B(90),
                MyEnum::C(9012),
                MyEnum::D(34567890),
                MyEnum::E(1234567890123456),
            )
        }
    };

    let result_vm = run_vm_program(
        &(source.clone(), program.clone(), runner),
        "run_test",
        &[],
        None,
    )
    .unwrap();

    let vm_results = get_result_success(result_vm.value);

    let result = run_native_program(&(source, program), "run_test", json!([]));
    assert_eq!(
        result,
        json!([[
            [vm_results[0].parse::<i64>().unwrap(), felt(&vm_results[1])],
            [
                casm_variant_to_sierra(vm_results[2].parse::<i64>().unwrap(), 5),
                felt(&vm_results[3])
            ],
            [
                casm_variant_to_sierra(vm_results[4].parse::<i64>().unwrap(), 5),
                vm_results[5].parse::<i64>().unwrap()
            ],
            [
                casm_variant_to_sierra(vm_results[6].parse::<i64>().unwrap(), 5),
                vm_results[7].parse::<i64>().unwrap()
            ],
            [
                casm_variant_to_sierra(vm_results[8].parse::<i64>().unwrap(), 5),
                vm_results[9].parse::<i64>().unwrap()
            ],
            [
                casm_variant_to_sierra(vm_results[10].parse::<i64>().unwrap(), 5),
                vm_results[11].parse::<i64>().unwrap()
            ],
        ]])
    );
}
