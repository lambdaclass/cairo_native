//! # `Felt` dictionary entry libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{
        realloc_bindings::ReallocBindingsMeta, runtime_bindings::RuntimeBindingsMeta,
        MetadataStorage,
    },
    types::TypeBuilder,
    utils::{get_integer_layout, BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        felt252_dict::Felt252DictEntryConcreteLibfunc,
        lib_func::SignatureAndTypeConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{cf, llvm},
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        Identifier, Location, Value, ValueLike,
    },
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
    selector: &Felt252DictEntryConcreteLibfunc,
) -> Result<()> {
    match selector {
        Felt252DictEntryConcreteLibfunc::Get(info) => {
            build_get(context, registry, entry, location, helper, metadata, info)
        }
        Felt252DictEntryConcreteLibfunc::Finalize(info) => {
            build_finalize(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_get<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let (key_ty, key_layout) = registry.build_type_with_layout(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[1].ty,
    )?;

    let entry_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let (value_ty, value_layout) = registry.build_type_with_layout(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;

    let dict_ptr = entry.argument(0)?.into();
    let key_value = entry.argument(1)?.into();

    let key_ptr = helper
        .init_block()
        .alloca1(context, location, key_ty, key_layout.align())?;

    entry.store(context, location, key_ptr, key_value)?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let op = runtime_bindings.dict_get(context, helper, entry, dict_ptr, key_ptr, location)?;
    let result_ptr: Value = op.result(0)?.into();

    let null_ptr = entry.append_op_result(
        OperationBuilder::new("llvm.mlir.zero", location)
            .add_results(&[result_ptr.r#type()])
            .build()?,
    )?;

    // need llvm instead of arith to compare pointers
    let is_null_ptr = entry.append_op_result(
        OperationBuilder::new("llvm.icmp", location)
            .add_operands(&[result_ptr, null_ptr])
            .add_attributes(&[(
                Identifier::new(context, "predicate"),
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            )])
            .add_results(&[IntegerType::new(context, 1).into()])
            .build()?,
    )?;

    let block_is_null = helper.append_block(Block::new(&[]));
    let block_is_found = helper.append_block(Block::new(&[]));
    let block_final = helper.append_block(Block::new(&[
        (llvm::r#type::pointer(context, 0), location),
        (value_ty, location),
    ]));

    entry.append_operation(cf::cond_br(
        context,
        is_null_ptr,
        block_is_null,
        block_is_found,
        &[],
        &[],
        location,
    ));

    // null block
    {
        let alloc_size = block_is_null.const_int(context, location, value_layout.size(), 64)?;

        let value_ptr = block_is_null.append_op_result(ReallocBindingsMeta::realloc(
            context, result_ptr, alloc_size, location,
        ))?;

        let default_value = registry
            .get_type(&info.branch_signatures()[0].vars[1].ty)?
            .build_default(
                context,
                registry,
                block_is_null,
                location,
                helper,
                metadata,
                &info.branch_signatures()[0].vars[1].ty,
            )?;

        block_is_null.append_operation(cf::br(block_final, &[value_ptr, default_value], location));
    }

    // found block
    {
        let loaded_val_ptr = block_is_found.load(context, location, result_ptr, value_ty)?;
        block_is_found.append_operation(cf::br(
            block_final,
            &[result_ptr, loaded_val_ptr],
            location,
        ));
    }

    // construct the struct

    let entry_value = block_final.append_op_result(llvm::undef(entry_ty, location))?;

    let value_ptr = block_final.argument(0)?.into();
    let value = block_final.argument(1)?.into();

    let entry_value = block_final.insert_value(context, location, entry_value, key_value, 0)?;

    let entry_value = block_final.insert_value(context, location, entry_value, value_ptr, 1)?;

    let entry_value = block_final.insert_value(context, location, entry_value, dict_ptr, 2)?;

    block_final.append_operation(helper.br(0, &[entry_value, value], location));

    Ok(())
}

pub fn build_finalize<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let key_ty = IntegerType::new(context, 252).into();
    let key_layout = get_integer_layout(252);

    let entry_value = entry.argument(0)?.into();
    let new_value = entry.argument(1)?.into();

    let key_value = entry.extract_value(context, location, entry_value, key_ty, 0)?;

    let value_ptr = entry.extract_value(
        context,
        location,
        entry_value,
        llvm::r#type::pointer(context, 0),
        1,
    )?;

    let dict_ptr = entry.extract_value(
        context,
        location,
        entry_value,
        llvm::r#type::pointer(context, 0),
        2,
    )?;

    entry.store(context, location, value_ptr, new_value)?;

    let key_ptr = helper
        .init_block()
        .alloca1(context, location, key_ty, key_layout.align())?;

    entry.store(context, location, key_ptr, key_value)?;

    // call insert

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    runtime_bindings.dict_insert(
        context, helper, entry, dict_ptr, key_ptr, value_ptr, location,
    )?;

    entry.append_operation(helper.br(0, &[dict_ptr], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_dict, load_cairo, run_program_assert_output};

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
    fn run_dict_insert_big() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u64 {
                let mut dict: Felt252Dict<u64> = Default::default();
                dict.insert(200000000, 4_u64);
                dict.get(200000000)
            }
        );

        run_program_assert_output(&program, "run_test", &[], 4u64.into());
    }

    #[test]
    fn run_dict_insert_ret_dict() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> Felt252Dict<u32> {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_dict!(
                2 => 1u32
            ),
        );
    }

    #[test]
    fn run_dict_insert_multiple() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u32 {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict.insert(3, 1_u32);
                dict.insert(4, 1_u32);
                dict.insert(5, 1_u32);
                dict.insert(6, 1_u32);
                dict.insert(7, 1_u32);
                dict.insert(8, 1_u32);
                dict.insert(9, 1_u32);
                dict.insert(10, 1_u32);
                dict.insert(11, 1_u32);
                dict.insert(12, 1_u32);
                dict.insert(13, 1_u32);
                dict.insert(14, 1_u32);
                dict.insert(15, 1_u32);
                dict.insert(16, 1_u32);
                dict.insert(17, 1_u32);
                dict.insert(18, 1345432_u32);
                dict.get(18)
            }
        );

        run_program_assert_output(&program, "run_test", &[], 1345432_u32.into());
    }
}
