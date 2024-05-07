//! # `Felt` dictionary libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        felt252_dict::Felt252DictConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith,
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
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let segment_arena =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let op = runtime_bindings.dict_alloc_new(context, helper, entry, location)?;
    let dict_ptr = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[segment_arena, dict_ptr], location));
    Ok(())
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
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
    let gas_builtin = entry.argument(1)?.into();
    let segment_arena =
        super::increment_builtin_counter(context, entry, location, entry.argument(2)?.into())?;
    let dict_ptr = entry.argument(3)?.into();

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let gas_refund = runtime_bindings
        .dict_gas_refund(context, helper, entry, dict_ptr, location)?
        .result(0)?
        .into();
    let gas_refund = entry
        .append_operation(arith::extui(
            gas_refund,
            IntegerType::new(context, 128).into(),
            location,
        ))
        .result(0)?
        .into();

    let new_gas_builtin = entry
        .append_operation(arith::addi(gas_builtin, gas_refund, location))
        .result(0)?
        .into();

    entry.append_operation(helper.br(
        0,
        &[
            range_check,
            new_gas_builtin,
            segment_arena,
            entry.argument(3)?.into(),
        ],
        location,
    ));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_dict, jit_struct, load_cairo, run_program_assert_output};

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
}
