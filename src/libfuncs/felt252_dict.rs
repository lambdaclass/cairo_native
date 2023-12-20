//! # `Felt` dictionary libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        felt252_dict::Felt252DictConcreteLibfunc, lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252DictConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        Felt252DictConcreteLibfunc::New(info) => {
            build_new(context, registry, entry, location, helper, metadata, info)
        }
        Felt252DictConcreteLibfunc::Squash(info) => {
            build_squash(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_new<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let _dict_ptr_type = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[0].ty,
    )?;

    let segment_arena = entry.argument(0)?.into();

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let op = runtime_bindings.dict_alloc_new(context, helper, entry, location)?;

    let dict_ptr = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[segment_arena, dict_ptr], location));

    Ok(())
}

pub fn build_squash<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    entry.append_operation(helper.br(
        0,
        &[
            entry.argument(0)?.into(),
            entry.argument(1)?.into(),
            entry.argument(2)?.into(),
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
                let mut dict: Felt252Dict<u32> = Default::default();
            }
        );

        run_program_assert_output(&program, "run_test", &[], &[jit_struct!()]);
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

        run_program_assert_output(&program, "run_test", &[], &[1u32.into()]);
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
            &[jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            )],
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
            &[jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            )],
        );
    }
}
