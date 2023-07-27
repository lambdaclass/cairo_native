mod common;
use crate::common::{
    casm_variant_to_sierra, felt, get_run_result, run_native_program, run_vm_program,
};
use common::load_cairo;
use pretty_assertions::assert_eq;
use serde_json::json;

#[test]
fn enum_init() {
    let enum_init = load_cairo! {
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

    let result_vm = run_vm_program(&enum_init, "run_test", &[], None).unwrap();

    let vm_results = get_run_result(&result_vm.value);

    let result = run_native_program(&enum_init, "run_test", json!([]));
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

#[test]
fn enum_match() {
    let enum_match = load_cairo! {
        enum MyEnum {
            A: felt252,
            B: u8,
            C: u16,
            D: u32,
            E: u64,
        }

        fn match_a() -> felt252 {
            let x = MyEnum::A(5);
            match x {
                MyEnum::A(x) => x,
                MyEnum::B(_) => 0,
                MyEnum::C(_) => 1,
                MyEnum::D(_) => 2,
                MyEnum::E(_) => 3,
            }
        }

        fn match_b() -> u8 {
            let x = MyEnum::B(5_u8);
            match x {
                MyEnum::A(_) => 0_u8,
                MyEnum::B(x) => x,
                MyEnum::C(_) => 1_u8,
                MyEnum::D(_) => 2_u8,
                MyEnum::E(_) => 3_u8,
            }
        }
    };

    let result_vm = run_vm_program(&enum_match, "match_a", &[], None).unwrap();

    let vm_results = get_run_result(&result_vm.value);

    let result = run_native_program(&enum_match, "match_a", json!([]));
    assert_eq!(result, json!([felt(&vm_results[0])]));

    let result_vm = run_vm_program(&enum_match, "match_b", &[], None).unwrap();

    let vm_results = get_run_result(&result_vm.value);

    let result = run_native_program(&enum_match, "match_b", json!([]));
    assert_eq!(result, json!([vm_results[0].parse::<i64>().unwrap()]));
}
