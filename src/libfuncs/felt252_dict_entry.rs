//! # `Felt` dictionary entry libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{
        drop_overrides::DropOverridesMeta, realloc_bindings::ReallocBindingsMeta,
        runtime_bindings::RuntimeBindingsMeta, MetadataStorage,
    },
    types::TypeBuilder,
    utils::{BlockExt, ProgramRegistryExt},
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
    dialect::{cf, llvm, ods},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location},
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
    let value_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;

    let dict_ptr = entry.argument(0)?.into();
    let entry_key = entry.argument(1)?.into();

    let entry_key_ptr =
        helper
            .init_block()
            .alloca1(context, location, key_ty, key_layout.align())?;
    entry.store(context, location, entry_key_ptr, entry_key)?;

    // Double pointer. Avoid allocating an element on a dict getter.
    let entry_value_ptr_ptr = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .unwrap()
        .dict_get(context, helper, entry, dict_ptr, entry_key_ptr, location)?;
    let entry_value_ptr = entry.load(
        context,
        location,
        entry_value_ptr_ptr,
        llvm::r#type::pointer(context, 0),
    )?;

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
    let is_vacant = entry.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            entry_value_ptr,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

    let block_occupied = helper.append_block(Block::new(&[]));
    let block_vacant = helper.append_block(Block::new(&[]));
    let block_final = helper.append_block(Block::new(&[(value_ty, location)]));
    entry.append_operation(cf::cond_br(
        context,
        is_vacant,
        block_vacant,
        block_occupied,
        &[],
        &[],
        location,
    ));

    {
        let value = block_occupied.load(context, location, entry_value_ptr, value_ty)?;
        block_occupied.append_operation(cf::br(block_final, &[value], location));
    }

    {
        let value = registry
            .get_type(&info.branch_signatures()[0].vars[1].ty)?
            .build_default(
                context,
                registry,
                block_vacant,
                location,
                helper,
                metadata,
                &info.branch_signatures()[0].vars[1].ty,
            )?;
        block_vacant.append_operation(cf::br(block_final, &[value], location));
    }

    let entry = block_final.append_op_result(llvm::undef(entry_ty, location))?;
    let entry =
        block_final.insert_values(context, location, entry, &[dict_ptr, entry_value_ptr_ptr])?;

    block_final.append_operation(helper.br(0, &[entry, block_final.argument(0)?.into()], location));
    Ok(())
}

pub fn build_finalize<'ctx, 'this>(
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

    let (value_ty, value_layout) = registry.build_type_with_layout(
        context,
        helper,
        registry,
        metadata,
        &info.signature.param_signatures[1].ty,
    )?;

    let dict_entry = entry.argument(0)?.into();
    let entry_value = entry.argument(1)?.into();

    let dict_ptr = entry.extract_value(
        context,
        location,
        dict_entry,
        llvm::r#type::pointer(context, 0),
        0,
    )?;
    let value_ptr_ptr = entry.extract_value(
        context,
        location,
        dict_entry,
        llvm::r#type::pointer(context, 0),
        1,
    )?;

    let value_ptr = entry.load(
        context,
        location,
        value_ptr_ptr,
        llvm::r#type::pointer(context, 0),
    )?;

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
    let is_vacant = entry.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            value_ptr,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

    let block_occupied = helper.append_block(Block::new(&[]));
    let block_vacant = helper.append_block(Block::new(&[]));
    let block_final =
        helper.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));
    entry.append_operation(cf::cond_br(
        context,
        is_vacant,
        block_vacant,
        block_occupied,
        &[],
        &[],
        location,
    ));

    {
        match metadata.get::<DropOverridesMeta>() {
            Some(drop_overrides_meta)
                if drop_overrides_meta.is_overriden(&info.signature.param_signatures[1].ty) =>
            {
                let value = block_occupied.load(context, location, value_ptr, value_ty)?;
                drop_overrides_meta.invoke_override(
                    context,
                    block_occupied,
                    location,
                    &info.signature.param_signatures[1].ty,
                    value,
                )?;
            }
            _ => {}
        }

        block_occupied.append_operation(cf::br(block_final, &[value_ptr], location));
    }

    {
        let value_len = block_vacant.const_int(context, location, value_layout.size(), 64)?;
        let value_ptr = block_vacant.append_op_result(ReallocBindingsMeta::realloc(
            context, null_ptr, value_len, location,
        ))?;

        block_vacant.store(context, location, value_ptr_ptr, value_ptr)?;
        block_vacant.append_operation(cf::br(block_final, &[value_ptr], location));
    }

    block_final.store(
        context,
        location,
        block_final.argument(0)?.into(),
        entry_value,
    )?;
    block_final.append_operation(helper.br(0, &[dict_ptr], location));

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
