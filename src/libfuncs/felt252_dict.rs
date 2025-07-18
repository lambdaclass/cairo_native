//! # `Felt` dictionary libfuncs

use super::LibfuncHelper;
use crate::{
    error::{panic::ToNativeAssertError, Result},
    execution_result::SEGMENT_ARENA_BUILTIN_SIZE,
    metadata::{
        felt252_dict::Felt252DictOverrides, runtime_bindings::RuntimeBindingsMeta, MetadataStorage,
    },
    native_panic,
    types::TypeBuilder,
    utils::BlockExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        felt252_dict::Felt252DictConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{llvm, ods},
    ir::{r#type::IntegerType, Block, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252DictConcreteLibfunc,
) -> Result<()> {
    match selector {
        Felt252DictConcreteLibfunc::New(info) => {
            build_new(context, registry, entry, location, helper, metadata, info)
        }
        Felt252DictConcreteLibfunc::Squash(info) => {
            build_squash(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_new<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // We increase the segment arena builtin by 1 usage.
    // https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/felt252_dict.rs?plain=1#L45-L49
    let segment_arena = super::increment_builtin_counter_by(
        context,
        entry,
        location,
        entry.arg(0)?,
        SEGMENT_ARENA_BUILTIN_SIZE,
    )?;

    let value_type_id = match registry.get_type(&info.signature.branch_signatures[0].vars[1].ty)? {
        CoreTypeConcrete::Felt252Dict(info) => &info.ty,
        _ => native_panic!("entered unreachable code"),
    };

    let drop_fn = {
        let mut dict_overrides = metadata
            .remove::<Felt252DictOverrides>()
            .unwrap_or_default();

        let drop_fn = match dict_overrides.build_drop_fn(
            context,
            helper,
            registry,
            metadata,
            value_type_id,
        )? {
            Some(drop_fn_symbol) => Some(
                entry.append_op_result(
                    ods::llvm::mlir_addressof(
                        context,
                        llvm::r#type::pointer(context, 0),
                        drop_fn_symbol,
                        location,
                    )
                    .into(),
                )?,
            ),
            None => None,
        };

        metadata.insert(dict_overrides);

        drop_fn
    };

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .to_native_assert_error("runtime library should be available")?;
    let dict_ptr = runtime_bindings.dict_new(
        context,
        helper,
        entry,
        location,
        drop_fn,
        registry.get_type(value_type_id)?.layout(registry)?,
    )?;

    helper.br(entry, 0, &[segment_arena, dict_ptr], location)
}

pub fn build_squash<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = entry.arg(0)?;
    let gas = entry.arg(1)?;
    let segment_arena = entry.arg(2)?;
    let dict_ptr = entry.arg(3)?;

    // Increase the segment arena builtin by 1 usage.
    // https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/felt252_dict.rs?plain=1#L148-L151
    let segment_arena = super::increment_builtin_counter_by(
        context,
        entry,
        location,
        segment_arena,
        SEGMENT_ARENA_BUILTIN_SIZE,
    )?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .to_native_assert_error("runtime library should be available")?;

    let range_check_ptr =
        entry.alloca1(context, location, IntegerType::new(context, 64).into(), 0)?;
    entry.store(context, location, range_check_ptr, range_check)?;
    let gas_ptr = entry.alloca1(context, location, IntegerType::new(context, 64).into(), 0)?;
    entry.store(context, location, gas_ptr, gas)?;

    runtime_bindings.dict_squash(
        context,
        helper,
        entry,
        dict_ptr,
        range_check_ptr,
        gas_ptr,
        location,
    )?;

    let range_check = entry.load(
        context,
        location,
        range_check_ptr,
        IntegerType::new(context, 64).into(),
    )?;
    let gas_builtin = entry.load(
        context,
        location,
        gas_ptr,
        IntegerType::new(context, 64).into(),
    )?;

    helper.br(
        entry,
        0,
        &[range_check, gas_builtin, segment_arena, entry.arg(3)?],
        location,
    )
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{
            jit_dict, jit_enum, jit_struct, load_cairo, run_program, run_program_assert_output,
        },
        values::Value,
    };

    #[test]
    fn run_dict_new() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() {
                let mut _dict: Felt252Dict<u32> = Default::default();
            }
        );

        run_program_assert_output(&program, "run_test", &[], jit_struct!());
    }

    #[test]
    fn run_dict_insert() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;
            fn run_test() -> u32 {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict.get(2)
            }
        );

        run_program_assert_output(&program, "run_test", &[], 1u32.into());
    }

    #[test]
    fn run_dict_insert_ret_dict() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> Felt252Dict<u32> {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(1, 2_u32);
                dict.insert(2, 3_u32);
                dict.insert(3, 4_u32);
                dict.insert(4, 5_u32);
                dict.insert(5, 6_u32);
                dict
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            ),
        );
    }

    #[test]
    fn run_dict_deserialize() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test(mut dict: Felt252Dict<u32>) -> Felt252Dict<u32> {
                dict
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            )],
            jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            ),
        );
    }

    #[test]
    fn run_dict_deserialize2() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test(mut dict: Felt252Dict<u32>) -> (felt252, Felt252Dict<u32>) {
                (0, dict)
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            )],
            jit_struct!(
                Value::Felt252(0.into()),
                jit_dict!(
                    1 => 2u32,
                    2 => 3u32,
                    3 => 4u32,
                    4 => 5u32,
                    5 => 6u32,
                )
            ),
        );
    }

    #[test]
    fn run_dict_deserialize_struct() {
        let program = load_cairo! {
            use core::{dict::Felt252DictTrait, nullable::Nullable};

            fn run_test() -> Felt252Dict<Nullable<(u32, u64, u128)>> {
                let mut x: Felt252Dict<Nullable<(u32, u64, u128)>> = Default::default();
                x.insert(0, NullableTrait::new((1_u32, 2_u64, 3_u128)));
                x.insert(1, NullableTrait::new((2_u32, 3_u64, 4_u128)));
                x.insert(2, NullableTrait::new((3_u32, 4_u64, 5_u128)));
                x
            }
        };

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_dict!(
                0 => jit_struct!(1u32.into(), 2u64.into(), 3u128.into()),
                1 => jit_struct!(2u32.into(), 3u64.into(), 4u128.into()),
                2 => jit_struct!(3u32.into(), 4u64.into(), 5u128.into()),
            ),
        );
    }

    #[test]
    fn run_dict_deserialize_enum() {
        let program = load_cairo! {
            use core::{dict::Felt252DictTrait, nullable::Nullable};

            #[derive(Drop)]
            enum MyEnum {
                A: u32,
                B: u64,
                C: u128,
            }

            fn run_test() -> Felt252Dict<Nullable<MyEnum>> {
                let mut x: Felt252Dict<Nullable<MyEnum>> = Default::default();
                x.insert(0, NullableTrait::new(MyEnum::A(1)));
                x.insert(1, NullableTrait::new(MyEnum::B(2)));
                x.insert(2, NullableTrait::new(MyEnum::C(3)));
                x
            }
        };

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_dict!(
                0 => jit_enum!(0, 1u32.into()),
                1 => jit_enum!(1, 2u64.into()),
                2 => jit_enum!(2, 3u128.into()),
            ),
        );
    }

    #[test]
    fn run_dict_squash() {
        let program = load_cairo! {
            use core::dict::{Felt252Dict, Felt252DictEntryTrait, SquashedFelt252DictImpl};

            pub fn main() {
                // The squash libfunc has a fixed range check cost of 2.

                // If no big keys, 3 per unique key access.
                let mut dict: Felt252Dict<felt252> = Default::default();
                dict.insert(1, 1); // 3
                dict.insert(2, 2); // 3
                dict.insert(3, 3); // 3
                dict.insert(4, 4); // 3
                dict.insert(5, 4); // 3
                dict.insert(6, 4); // 3
                let _ = dict.squash(); // 2
                // SUBTOTAL: 20

                // A dictionary has big keys if there is at least one key greater than
                // the range check bound (2**128 - 1).

                // If has big keys, 2 for first unique key access,
                // and 6 each of the remaining unique key accesses.
                let mut dict: Felt252Dict<felt252> = Default::default();
                dict.insert(1, 1); // 2
                dict.insert(0xF00000000000000000000000000000002, 1); // 6
                dict.insert(3, 1); // 6
                dict.insert(0xF00000000000000000000000000000004, 1); // 6
                dict.insert(5, 1); // 6
                dict.insert(0xF00000000000000000000000000000006, 1); // 6
                dict.insert(7, 1); // 6
                let _ = dict.squash(); // 2
                // SUBTOTAL: 40


                // If no big keys, 3 per unique key access.
                // Each repeated key adds an extra range check usage.
                let mut dict: Felt252Dict<felt252> = Default::default();
                dict.insert(1, 1); // 3
                dict.insert(2, 1); // 3
                dict.insert(3, 1); // 3
                dict.insert(4, 1); // 3
                dict.insert(1, 1); // 1
                dict.insert(2, 1); // 1
                dict.insert(1, 1); // 1
                dict.insert(2, 1); // 1
                dict.insert(1, 1); // 1
                dict.insert(2, 1); // 1
                let _ = dict.squash(); // 2
                // SUBTOTAL: 20


                // If has big keys, 2 for first unique key access,
                // and 6 each of the remaining unique key accesses.
                // Each repeated key access adds an extra range check usage.
                let mut dict: Felt252Dict<felt252> = Default::default();
                dict.insert(1, 1); // 2
                dict.insert(0xF00000000000000000000000000000002, 1); // 6
                dict.insert(1, 1); // 1
                dict.insert(0xF00000000000000000000000000000002, 1); // 1
                dict.insert(1, 1); // 1
                dict.insert(0xF00000000000000000000000000000002, 1); // 1
                dict.insert(1, 1); // 1
                dict.insert(0xF00000000000000000000000000000002, 1); // 1
                dict.insert(1, 1); // 1
                dict.insert(0xF00000000000000000000000000000002, 1); // 1
                dict.insert(1, 1); // 1
                dict.insert(0xF00000000000000000000000000000002, 1); // 1
                let _ = dict.squash(); // 2
                // SUBTOTAL: 20

                // TOTAL: 100
            }
        };

        let result = run_program(&program, "main", &[]);
        assert_eq!(result.builtin_stats.range_check, 100);
    }
}
