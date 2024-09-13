use crate::common::load_cairo;
use cairo_lang_sierra::program::Program;
use cairo_native::{
    context::NativeContext,
    execution_result::{BuiltinStats, ExecutionResult},
    executor::JitNativeExecutor,
    utils::find_function_id,
    OptLevel, Value,
};
use starknet_types_core::felt::Felt;

fn run_program(program: &Program, entry_point: &str, args: &[Value]) -> ExecutionResult {
    let entry_point_id = find_function_id(program, entry_point);

    let context = NativeContext::new();
    let module = context.compile(program).unwrap();
    // FIXME: There are some bugs with non-zero LLVM optimization levels.
    let executor = JitNativeExecutor::from_native_module(module, OptLevel::None);

    executor.invoke_dynamic(entry_point_id, args, None).unwrap()
}

#[test]
fn invoke0() {
    let (module_name, program, _) = load_cairo! {
        fn main() {}
    };

    assert_eq!(
        run_program(&program, &format!("{0}::{0}::main", module_name), &[]),
        ExecutionResult {
            remaining_gas: None,
            return_value: Value::Struct {
                fields: Vec::new(),
                debug_name: None,
            },
            builtin_stats: BuiltinStats::default(),
        },
    );
}

#[test]
fn invoke1_felt252() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: felt252) -> felt252 {
            x
        }
    };

    let r = |x: Felt| {
        let x = Value::Felt252(x);
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r(0.into());
    r(1.into());
    r(10.into());
    r(Felt::MAX);
}

#[test]
fn invoke1_u8() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: u8) -> u8 {
            x
        }
    };

    let r = |x: u8| {
        let x = Value::Uint8(x);
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r(0);
    r(1);
    r(10);
    r(u8::MAX);
}

#[test]
fn invoke1_u16() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: u16) -> u16 {
            x
        }
    };

    let r = |x: u16| {
        let x = Value::Uint16(x);
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r(0);
    r(1);
    r(10);
    r(u16::MAX);
}

#[test]
fn invoke1_u32() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: u32) -> u32 {
            x
        }
    };

    let r = |x: u32| {
        let x = Value::Uint32(x);
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r(0);
    r(1);
    r(10);
    r(u32::MAX);
}

#[test]
fn invoke1_u64() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: u64) -> u64 {
            x
        }
    };

    let r = |x: u64| {
        let x = Value::Uint64(x);
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r(0);
    r(1);
    r(10);
    r(u64::MAX);
}

#[test]
fn invoke1_u128() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: u128) -> u128 {
            x
        }
    };

    let r = |x: u128| {
        let x = Value::Uint128(x);
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r(0);
    r(1);
    r(10);
    r(u128::MAX);
}

#[test]
fn invoke1_tuple1_felt252() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: (felt252,)) -> (felt252,) {
            x
        }
    };

    let r = |x: (Felt,)| {
        let x = Value::Struct {
            fields: vec![Value::Felt252(x.0)],
            debug_name: None,
        };
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r((0.into(),));
    r((1.into(),));
    r((10.into(),));
    r((Felt::MAX,));
}

#[test]
fn invoke1_tuple1_u64() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: (u64,)) -> (u64,) {
            x
        }
    };

    let r = |x: (u64,)| {
        let x = Value::Struct {
            fields: vec![Value::Uint64(x.0)],
            debug_name: None,
        };
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r((0,));
    r((1,));
    r((10,));
    r((u64::MAX,));
}

#[test]
fn invoke1_tuple5_u8_u16_u32_u64_u128() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: (u8, u16, u32, u64, u128)) -> (u8, u16, u32, u64, u128) {
            x
        }
    };

    let r = |x: (u8, u16, u32, u64, u128)| {
        let x = Value::Struct {
            fields: vec![
                Value::Uint8(x.0),
                Value::Uint16(x.1),
                Value::Uint32(x.2),
                Value::Uint64(x.3),
                Value::Uint128(x.4),
            ],
            debug_name: None,
        };
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r((0, 0, 0, 0, 0));
    r((1, 1, 1, 1, 1));
    r((10, 10, 10, 10, 10));
    r((u8::MAX, u16::MAX, u32::MAX, u64::MAX, u128::MAX));
}

#[test]
fn invoke1_array_felt252() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: Array<felt252>) -> Array<felt252> {
            x
        }
    };

    let r = |x: Vec<Felt>| {
        let x = Value::Array(x.into_iter().map(Value::Felt252).collect());
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r(vec![]);
    r(vec![0.into()]);
    r(vec![0.into(), 1.into()]);
    r(vec![0.into(), 1.into(), 10.into()]);
    r(vec![0.into(), 1.into(), 10.into(), Felt::MAX]);
}

#[test]
fn invoke1_enum1_unit() {
    let (module_name, program, _) = load_cairo! {
        enum MyEnum {
            A: ()
        }

        fn main(x: MyEnum) -> MyEnum {
            x
        }
    };

    let x = Value::Enum {
        tag: 0,
        value: Box::new(Value::Struct {
            fields: Vec::new(),
            debug_name: None,
        }),
        debug_name: Some("MyEnum".into()),
    };
    assert_eq!(
        run_program(
            &program,
            &format!("{0}::{0}::main", module_name),
            &[x.clone()]
        ),
        ExecutionResult {
            remaining_gas: None,
            return_value: x,
            builtin_stats: BuiltinStats::default(),
        },
    );
}

#[test]
fn invoke1_enum1_u64() {
    let (module_name, program, _) = load_cairo! {
        enum MyEnum {
            A: u64
        }

        fn main(x: MyEnum) -> MyEnum {
            x
        }
    };

    let r = |x: u64| {
        let x = Value::Enum {
            tag: 0,
            value: Box::new(Value::Uint64(x)),
            debug_name: Some("MyEnum".into()),
        };
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r(0);
    r(1);
    r(10);
    r(u64::MAX);
}

#[test]
fn invoke1_enum1_felt252() {
    let (module_name, program, _) = load_cairo! {
        enum MyEnum {
            A: felt252
        }

        fn main(x: MyEnum) -> MyEnum {
            x
        }
    };

    let r = |x: Felt| {
        let x = Value::Enum {
            tag: 0,
            value: Box::new(Value::Felt252(x)),
            debug_name: Some("MyEnum".into()),
        };
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r(0.into());
    r(1.into());
    r(10.into());
    r(Felt::MAX);
}

#[test]
fn invoke1_enum2_u8_u16() {
    let (module_name, program, _) = load_cairo! {
        enum MyEnum {
            A: u8,
            B: u16,
        }

        fn main(x: MyEnum) -> MyEnum {
            x
        }
    };

    enum MyEnum {
        A(u8),
        B(u16),
    }

    let r = |x: MyEnum| {
        let x = match x {
            MyEnum::A(x) => Value::Enum {
                tag: 0,
                value: Box::new(Value::Uint8(x)),
                debug_name: Some("MyEnum".into()),
            },
            MyEnum::B(x) => Value::Enum {
                tag: 1,
                value: Box::new(Value::Uint16(x)),
                debug_name: Some("MyEnum".into()),
            },
        };
        assert_eq!(
            run_program(
                &program,
                &format!("{0}::{0}::main", module_name),
                &[x.clone()]
            ),
            ExecutionResult {
                remaining_gas: None,
                return_value: x,
                builtin_stats: BuiltinStats::default(),
            },
        );
    };

    r(MyEnum::A(0));
    r(MyEnum::A(1));
    r(MyEnum::A(10));
    r(MyEnum::A(u8::MAX));
    r(MyEnum::B(0));
    r(MyEnum::B(1));
    r(MyEnum::B(10));
    r(MyEnum::B(u16::MAX));
}

#[test]
fn invoke1_box_felt252() {
    let (module_name, program, _) = load_cairo! {
        fn main(x: Box<felt252>) -> felt252 {
            x.unbox()
        }
    };

    assert_eq!(
        run_program(
            &program,
            &format!("{0}::{0}::main", module_name),
            &[Value::Felt252(42.into())],
        ),
        ExecutionResult {
            remaining_gas: None,
            return_value: Value::Felt252(42.into()),
            builtin_stats: BuiltinStats::default(),
        }
    );
}

#[test]
fn invoke1_nullable_felt252() {
    let (module_name, program, _) = load_cairo! {
        use core::nullable::{match_nullable, FromNullableResult};

        fn main(x: Nullable<felt252>) -> Option<felt252> {
            match match_nullable(x) {
                FromNullableResult::Null(()) => Option::None(()),
                FromNullableResult::NotNull(x) => Option::Some(x.unbox()),
            }
        }
    };

    assert_eq!(
        run_program(
            &program,
            &format!("{0}::{0}::main", module_name),
            &[Value::Felt252(42.into())],
        ),
        ExecutionResult {
            remaining_gas: None,
            return_value: Value::Enum {
                tag: 0,
                value: Box::new(Value::Felt252(42.into())),
                debug_name: None
            },
            builtin_stats: BuiltinStats::default(),
        }
    );
    assert_eq!(
        run_program(
            &program,
            &format!("{0}::{0}::main", module_name),
            &[Value::Null],
        ),
        ExecutionResult {
            remaining_gas: None,
            return_value: Value::Enum {
                tag: 1,
                value: Box::new(Value::Struct {
                    fields: Vec::new(),
                    debug_name: None
                }),
                debug_name: None
            },
            builtin_stats: BuiltinStats::default(),
        }
    );
}

#[test]
fn test_deserialize_param_bug() {
    let (module_name, program, _) = load_cairo! {
        fn main(
            b0: u64,            // Pedersen
            b1: u64,            // RangeCheck
            b2: u64,            // Bitwise
            b3: u128,           // GasBuiltin
            b4: u64,            // System
            arg0: Span<felt252> // Arguments
        ) -> (u64, u64, u64, u128, u64, Span<felt252>) {
            (b0, b1, b2, b3, b4, arg0)
        }
    };

    let args = vec![
        Value::Uint64(0),
        Value::Uint64(0),
        Value::Uint64(0),
        Value::Uint128(0),
        Value::Uint64(0),
        Value::Struct {
            fields: vec![Value::Array(vec![
                Value::Felt252(1.into()),
                Value::Felt252(2.into()),
            ])],
            debug_name: None,
        },
    ];
    assert_eq!(
        run_program(&program, &format!("{0}::{0}::main", module_name), &args),
        ExecutionResult {
            remaining_gas: None,
            return_value: Value::Struct {
                fields: args,
                debug_name: None
            },
            builtin_stats: BuiltinStats::default(),
        },
    );
}
