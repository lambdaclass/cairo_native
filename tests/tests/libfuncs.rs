//use crate::common::{compare_outputs, load_cairo, run_native_program, run_vm_program};
use crate::common::{compare_outputs, load_cairo, run_native_program, run_vm_program};
//use cairo_native::starknet::DummySyscallHandler;
use cairo_native::starknet::DummySyscallHandler;
//

//#[test]
#[test]
//fn enum_init() {
fn enum_init() {
//    let program = load_cairo! {
    let program = load_cairo! {
//        enum MySmallEnum {
        enum MySmallEnum {
//            A: felt252,
            A: felt252,
//        }
        }
//

//        enum MyEnum {
        enum MyEnum {
//            A: felt252,
            A: felt252,
//            B: u8,
            B: u8,
//            C: u16,
            C: u16,
//            D: u32,
            D: u32,
//            E: u64,
            E: u64,
//        }
        }
//

//        fn run_test() -> (MySmallEnum, MyEnum, MyEnum, MyEnum, MyEnum, MyEnum) {
        fn run_test() -> (MySmallEnum, MyEnum, MyEnum, MyEnum, MyEnum, MyEnum) {
//            (
            (
//                MySmallEnum::A(-1),
                MySmallEnum::A(-1),
//                MyEnum::A(5678),
                MyEnum::A(5678),
//                MyEnum::B(90),
                MyEnum::B(90),
//                MyEnum::C(9012),
                MyEnum::C(9012),
//                MyEnum::D(34567890),
                MyEnum::D(34567890),
//                MyEnum::E(1234567890123456),
                MyEnum::E(1234567890123456),
//            )
            )
//        }
        }
//    };
    };
//

//    let result_vm = run_vm_program(&program, "run_test", &[], None).unwrap();
    let result_vm = run_vm_program(&program, "run_test", &[], None).unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        &program,
        &program,
//        "run_test",
        "run_test",
//        &[],
        &[],
//        None,
        None,
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &program.1,
        &program.1,
//        &program.2.find_function("run_test").unwrap().id,
        &program.2.find_function("run_test").unwrap().id,
//        &result_vm,
        &result_vm,
//        &result_native,
        &result_native,
//    )
    )
//    .unwrap();
    .unwrap();
//}
}
//

//#[test]
#[test]
//fn enum_match() {
fn enum_match() {
//    let program = load_cairo! {
    let program = load_cairo! {
//        enum MyEnum {
        enum MyEnum {
//            A: felt252,
            A: felt252,
//            B: u8,
            B: u8,
//            C: u16,
            C: u16,
//            D: u32,
            D: u32,
//            E: u64,
            E: u64,
//        }
        }
//

//        fn match_a() -> felt252 {
        fn match_a() -> felt252 {
//            let x = MyEnum::A(5);
            let x = MyEnum::A(5);
//            match x {
            match x {
//                MyEnum::A(x) => x,
                MyEnum::A(x) => x,
//                MyEnum::B(_) => 0,
                MyEnum::B(_) => 0,
//                MyEnum::C(_) => 1,
                MyEnum::C(_) => 1,
//                MyEnum::D(_) => 2,
                MyEnum::D(_) => 2,
//                MyEnum::E(_) => 3,
                MyEnum::E(_) => 3,
//            }
            }
//        }
        }
//

//        fn match_b() -> u8 {
        fn match_b() -> u8 {
//            let x = MyEnum::B(5_u8);
            let x = MyEnum::B(5_u8);
//            match x {
            match x {
//                MyEnum::A(_) => 0_u8,
                MyEnum::A(_) => 0_u8,
//                MyEnum::B(x) => x,
                MyEnum::B(x) => x,
//                MyEnum::C(_) => 1_u8,
                MyEnum::C(_) => 1_u8,
//                MyEnum::D(_) => 2_u8,
                MyEnum::D(_) => 2_u8,
//                MyEnum::E(_) => 3_u8,
                MyEnum::E(_) => 3_u8,
//            }
            }
//        }
        }
//    };
    };
//

//    let result_vm = run_vm_program(&program, "match_a", &[], None).unwrap();
    let result_vm = run_vm_program(&program, "match_a", &[], None).unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        &program,
        &program,
//        "match_a",
        "match_a",
//        &[],
        &[],
//        None,
        None,
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &program.1,
        &program.1,
//        &program.2.find_function("match_a").unwrap().id,
        &program.2.find_function("match_a").unwrap().id,
//        &result_vm,
        &result_vm,
//        &result_native,
        &result_native,
//    )
    )
//    .unwrap();
    .unwrap();
//

//    let result_vm = run_vm_program(&program, "match_b", &[], None).unwrap();
    let result_vm = run_vm_program(&program, "match_b", &[], None).unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        &program,
        &program,
//        "match_b",
        "match_b",
//        &[],
        &[],
//        None,
        None,
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &program.1,
        &program.1,
//        &program.2.find_function("match_b").unwrap().id,
        &program.2.find_function("match_b").unwrap().id,
//        &result_vm,
        &result_vm,
//        &result_native,
        &result_native,
//    )
    )
//    .unwrap();
    .unwrap();
//}
}
