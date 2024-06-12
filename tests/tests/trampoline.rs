//use crate::common::load_cairo;
use crate::common::load_cairo;
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use cairo_native::{
use cairo_native::{
//    context::NativeContext,
    context::NativeContext,
//    execution_result::{BuiltinStats, ExecutionResult},
    execution_result::{BuiltinStats, ExecutionResult},
//    executor::JitNativeExecutor,
    executor::JitNativeExecutor,
//    utils::find_function_id,
    utils::find_function_id,
//    values::JitValue,
    values::JitValue,
//    OptLevel,
    OptLevel,
//};
};
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//

//fn run_program(program: &Program, entry_point: &str, args: &[JitValue]) -> ExecutionResult {
fn run_program(program: &Program, entry_point: &str, args: &[JitValue]) -> ExecutionResult {
//    let entry_point_id = find_function_id(program, entry_point);
    let entry_point_id = find_function_id(program, entry_point);
//

//    let context = NativeContext::new();
    let context = NativeContext::new();
//    let module = context.compile(program, None).unwrap();
    let module = context.compile(program, None).unwrap();
//    // FIXME: There are some bugs with non-zero LLVM optimization levels.
    // FIXME: There are some bugs with non-zero LLVM optimization levels.
//    let executor = JitNativeExecutor::from_native_module(module, OptLevel::None);
    let executor = JitNativeExecutor::from_native_module(module, OptLevel::None);
//

//    executor.invoke_dynamic(entry_point_id, args, None).unwrap()
    executor.invoke_dynamic(entry_point_id, args, None).unwrap()
//}
}
//

//#[test]
#[test]
//fn invoke0() {
fn invoke0() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main() {}
        fn main() {}
//    };
    };
//

//    assert_eq!(
    assert_eq!(
//        run_program(&program, &format!("{0}::{0}::main", module_name), &[]),
        run_program(&program, &format!("{0}::{0}::main", module_name), &[]),
//        ExecutionResult {
        ExecutionResult {
//            remaining_gas: None,
            remaining_gas: None,
//            return_value: JitValue::Struct {
            return_value: JitValue::Struct {
//                fields: Vec::new(),
                fields: Vec::new(),
//                debug_name: None,
                debug_name: None,
//            },
            },
//            builtin_stats: BuiltinStats::default(),
            builtin_stats: BuiltinStats::default(),
//        },
        },
//    );
    );
//}
}
//

//#[test]
#[test]
//fn invoke1_felt252() {
fn invoke1_felt252() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main(x: felt252) -> felt252 {
        fn main(x: felt252) -> felt252 {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: Felt| {
    let r = |x: Felt| {
//        let x = JitValue::Felt252(x);
        let x = JitValue::Felt252(x);
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r(0.into());
    r(0.into());
//    r(1.into());
    r(1.into());
//    r(10.into());
    r(10.into());
//    r(Felt::MAX);
    r(Felt::MAX);
//}
}
//

//#[test]
#[test]
//fn invoke1_u8() {
fn invoke1_u8() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main(x: u8) -> u8 {
        fn main(x: u8) -> u8 {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: u8| {
    let r = |x: u8| {
//        let x = JitValue::Uint8(x);
        let x = JitValue::Uint8(x);
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r(0);
    r(0);
//    r(1);
    r(1);
//    r(10);
    r(10);
//    r(u8::MAX);
    r(u8::MAX);
//}
}
//

//#[test]
#[test]
//fn invoke1_u16() {
fn invoke1_u16() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main(x: u16) -> u16 {
        fn main(x: u16) -> u16 {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: u16| {
    let r = |x: u16| {
//        let x = JitValue::Uint16(x);
        let x = JitValue::Uint16(x);
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r(0);
    r(0);
//    r(1);
    r(1);
//    r(10);
    r(10);
//    r(u16::MAX);
    r(u16::MAX);
//}
}
//

//#[test]
#[test]
//fn invoke1_u32() {
fn invoke1_u32() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main(x: u32) -> u32 {
        fn main(x: u32) -> u32 {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: u32| {
    let r = |x: u32| {
//        let x = JitValue::Uint32(x);
        let x = JitValue::Uint32(x);
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r(0);
    r(0);
//    r(1);
    r(1);
//    r(10);
    r(10);
//    r(u32::MAX);
    r(u32::MAX);
//}
}
//

//#[test]
#[test]
//fn invoke1_u64() {
fn invoke1_u64() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main(x: u64) -> u64 {
        fn main(x: u64) -> u64 {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: u64| {
    let r = |x: u64| {
//        let x = JitValue::Uint64(x);
        let x = JitValue::Uint64(x);
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r(0);
    r(0);
//    r(1);
    r(1);
//    r(10);
    r(10);
//    r(u64::MAX);
    r(u64::MAX);
//}
}
//

//#[test]
#[test]
//fn invoke1_u128() {
fn invoke1_u128() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main(x: u128) -> u128 {
        fn main(x: u128) -> u128 {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: u128| {
    let r = |x: u128| {
//        let x = JitValue::Uint128(x);
        let x = JitValue::Uint128(x);
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r(0);
    r(0);
//    r(1);
    r(1);
//    r(10);
    r(10);
//    r(u128::MAX);
    r(u128::MAX);
//}
}
//

//#[test]
#[test]
//fn invoke1_tuple1_felt252() {
fn invoke1_tuple1_felt252() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main(x: (felt252,)) -> (felt252,) {
        fn main(x: (felt252,)) -> (felt252,) {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: (Felt,)| {
    let r = |x: (Felt,)| {
//        let x = JitValue::Struct {
        let x = JitValue::Struct {
//            fields: vec![JitValue::Felt252(x.0)],
            fields: vec![JitValue::Felt252(x.0)],
//            debug_name: None,
            debug_name: None,
//        };
        };
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r((0.into(),));
    r((0.into(),));
//    r((1.into(),));
    r((1.into(),));
//    r((10.into(),));
    r((10.into(),));
//    r((Felt::MAX,));
    r((Felt::MAX,));
//}
}
//

//#[test]
#[test]
//fn invoke1_tuple1_u64() {
fn invoke1_tuple1_u64() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main(x: (u64,)) -> (u64,) {
        fn main(x: (u64,)) -> (u64,) {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: (u64,)| {
    let r = |x: (u64,)| {
//        let x = JitValue::Struct {
        let x = JitValue::Struct {
//            fields: vec![JitValue::Uint64(x.0)],
            fields: vec![JitValue::Uint64(x.0)],
//            debug_name: None,
            debug_name: None,
//        };
        };
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r((0,));
    r((0,));
//    r((1,));
    r((1,));
//    r((10,));
    r((10,));
//    r((u64::MAX,));
    r((u64::MAX,));
//}
}
//

//#[test]
#[test]
//fn invoke1_tuple5_u8_u16_u32_u64_u128() {
fn invoke1_tuple5_u8_u16_u32_u64_u128() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main(x: (u8, u16, u32, u64, u128)) -> (u8, u16, u32, u64, u128) {
        fn main(x: (u8, u16, u32, u64, u128)) -> (u8, u16, u32, u64, u128) {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: (u8, u16, u32, u64, u128)| {
    let r = |x: (u8, u16, u32, u64, u128)| {
//        let x = JitValue::Struct {
        let x = JitValue::Struct {
//            fields: vec![
            fields: vec![
//                JitValue::Uint8(x.0),
                JitValue::Uint8(x.0),
//                JitValue::Uint16(x.1),
                JitValue::Uint16(x.1),
//                JitValue::Uint32(x.2),
                JitValue::Uint32(x.2),
//                JitValue::Uint64(x.3),
                JitValue::Uint64(x.3),
//                JitValue::Uint128(x.4),
                JitValue::Uint128(x.4),
//            ],
            ],
//            debug_name: None,
            debug_name: None,
//        };
        };
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r((0, 0, 0, 0, 0));
    r((0, 0, 0, 0, 0));
//    r((1, 1, 1, 1, 1));
    r((1, 1, 1, 1, 1));
//    r((10, 10, 10, 10, 10));
    r((10, 10, 10, 10, 10));
//    r((u8::MAX, u16::MAX, u32::MAX, u64::MAX, u128::MAX));
    r((u8::MAX, u16::MAX, u32::MAX, u64::MAX, u128::MAX));
//}
}
//

//#[test]
#[test]
//fn invoke1_array_felt252() {
fn invoke1_array_felt252() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        fn main(x: Array<felt252>) -> Array<felt252> {
        fn main(x: Array<felt252>) -> Array<felt252> {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: Vec<Felt>| {
    let r = |x: Vec<Felt>| {
//        let x = JitValue::Array(x.into_iter().map(JitValue::Felt252).collect());
        let x = JitValue::Array(x.into_iter().map(JitValue::Felt252).collect());
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r(vec![]);
    r(vec![]);
//    r(vec![0.into()]);
    r(vec![0.into()]);
//    r(vec![0.into(), 1.into()]);
    r(vec![0.into(), 1.into()]);
//    r(vec![0.into(), 1.into(), 10.into()]);
    r(vec![0.into(), 1.into(), 10.into()]);
//    r(vec![0.into(), 1.into(), 10.into(), Felt::MAX]);
    r(vec![0.into(), 1.into(), 10.into(), Felt::MAX]);
//}
}
//

//#[test]
#[test]
//fn invoke1_enum1_unit() {
fn invoke1_enum1_unit() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        enum MyEnum {
        enum MyEnum {
//            A: ()
            A: ()
//        }
        }
//

//        fn main(x: MyEnum) -> MyEnum {
        fn main(x: MyEnum) -> MyEnum {
//            x
            x
//        }
        }
//    };
    };
//

//    let x = JitValue::Enum {
    let x = JitValue::Enum {
//        tag: 0,
        tag: 0,
//        value: Box::new(JitValue::Struct {
        value: Box::new(JitValue::Struct {
//            fields: Vec::new(),
            fields: Vec::new(),
//            debug_name: None,
            debug_name: None,
//        }),
        }),
//        debug_name: Some("MyEnum".into()),
        debug_name: Some("MyEnum".into()),
//    };
    };
//    assert_eq!(
    assert_eq!(
//        run_program(
        run_program(
//            &program,
            &program,
//            &format!("{0}::{0}::main", module_name),
            &format!("{0}::{0}::main", module_name),
//            &[x.clone()]
            &[x.clone()]
//        ),
        ),
//        ExecutionResult {
        ExecutionResult {
//            remaining_gas: None,
            remaining_gas: None,
//            return_value: x,
            return_value: x,
//            builtin_stats: BuiltinStats::default(),
            builtin_stats: BuiltinStats::default(),
//        },
        },
//    );
    );
//}
}
//

//#[test]
#[test]
//fn invoke1_enum1_u64() {
fn invoke1_enum1_u64() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        enum MyEnum {
        enum MyEnum {
//            A: u64
            A: u64
//        }
        }
//

//        fn main(x: MyEnum) -> MyEnum {
        fn main(x: MyEnum) -> MyEnum {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: u64| {
    let r = |x: u64| {
//        let x = JitValue::Enum {
        let x = JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Uint64(x)),
            value: Box::new(JitValue::Uint64(x)),
//            debug_name: Some("MyEnum".into()),
            debug_name: Some("MyEnum".into()),
//        };
        };
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r(0);
    r(0);
//    r(1);
    r(1);
//    r(10);
    r(10);
//    r(u64::MAX);
    r(u64::MAX);
//}
}
//

//#[test]
#[test]
//fn invoke1_enum1_felt252() {
fn invoke1_enum1_felt252() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        enum MyEnum {
        enum MyEnum {
//            A: felt252
            A: felt252
//        }
        }
//

//        fn main(x: MyEnum) -> MyEnum {
        fn main(x: MyEnum) -> MyEnum {
//            x
            x
//        }
        }
//    };
    };
//

//    let r = |x: Felt| {
    let r = |x: Felt| {
//        let x = JitValue::Enum {
        let x = JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Felt252(x)),
            value: Box::new(JitValue::Felt252(x)),
//            debug_name: Some("MyEnum".into()),
            debug_name: Some("MyEnum".into()),
//        };
        };
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r(0.into());
    r(0.into());
//    r(1.into());
    r(1.into());
//    r(10.into());
    r(10.into());
//    r(Felt::MAX);
    r(Felt::MAX);
//}
}
//

//#[test]
#[test]
//fn invoke1_enum2_u8_u16() {
fn invoke1_enum2_u8_u16() {
//    let (module_name, program, _) = load_cairo! {
    let (module_name, program, _) = load_cairo! {
//        enum MyEnum {
        enum MyEnum {
//            A: u8,
            A: u8,
//            B: u16,
            B: u16,
//        }
        }
//

//        fn main(x: MyEnum) -> MyEnum {
        fn main(x: MyEnum) -> MyEnum {
//            x
            x
//        }
        }
//    };
    };
//

//    enum MyEnum {
    enum MyEnum {
//        A(u8),
        A(u8),
//        B(u16),
        B(u16),
//    }
    }
//

//    let r = |x: MyEnum| {
    let r = |x: MyEnum| {
//        let x = match x {
        let x = match x {
//            MyEnum::A(x) => JitValue::Enum {
            MyEnum::A(x) => JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Uint8(x)),
                value: Box::new(JitValue::Uint8(x)),
//                debug_name: Some("MyEnum".into()),
                debug_name: Some("MyEnum".into()),
//            },
            },
//            MyEnum::B(x) => JitValue::Enum {
            MyEnum::B(x) => JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: Box::new(JitValue::Uint16(x)),
                value: Box::new(JitValue::Uint16(x)),
//                debug_name: Some("MyEnum".into()),
                debug_name: Some("MyEnum".into()),
//            },
            },
//        };
        };
//        assert_eq!(
        assert_eq!(
//            run_program(
            run_program(
//                &program,
                &program,
//                &format!("{0}::{0}::main", module_name),
                &format!("{0}::{0}::main", module_name),
//                &[x.clone()]
                &[x.clone()]
//            ),
            ),
//            ExecutionResult {
            ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: x,
                return_value: x,
//                builtin_stats: BuiltinStats::default(),
                builtin_stats: BuiltinStats::default(),
//            },
            },
//        );
        );
//    };
    };
//

//    r(MyEnum::A(0));
    r(MyEnum::A(0));
//    r(MyEnum::A(1));
    r(MyEnum::A(1));
//    r(MyEnum::A(10));
    r(MyEnum::A(10));
//    r(MyEnum::A(u8::MAX));
    r(MyEnum::A(u8::MAX));
//    r(MyEnum::B(0));
    r(MyEnum::B(0));
//    r(MyEnum::B(1));
    r(MyEnum::B(1));
//    r(MyEnum::B(10));
    r(MyEnum::B(10));
//    r(MyEnum::B(u16::MAX));
    r(MyEnum::B(u16::MAX));
//}
}
