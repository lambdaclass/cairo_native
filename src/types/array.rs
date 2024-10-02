//! # Array type
//!
//! An array type is a dynamically allocated list of items.
//!
//! ## Layout
//!
//! Being dynamically allocated, we just need to keep the pointer to the data, its length and
//! its capacity:
//!
//! | Index | Type           | Description              |
//! | ----- | -------------- | ------------------------ |
//! |   0   | `!llvm.ptr<T>` | Pointer to the data[^1]. |
//! |   1   | `i32`          | Array start offset[^2].  |
//! |   1   | `i32`          | Array end offset[^2].    |
//! |   2   | `i32`          | Allocated capacity[^2].  |
//!
//! [^1]: When capacity is zero, this field is not guaranteed to be valid.
//! [^2]: Those numbers are number of items, **not bytes**.

use super::{TypeBuilder, WithSelf};
use crate::{
    error::Result,
    libfuncs::LibfuncHelper,
    metadata::{
        dup_overrides::DupOverrideMeta, realloc_bindings::ReallocBindingsMeta, MetadataStorage,
    },
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::dialect::{arith::CmpiPredicate, func, scf};
use melior::ir::Region;
use melior::{
    dialect::{arith, cf, llvm, ods},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Type, Value},
    Context,
};
use std::cell::Cell;

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    DupOverrideMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `snapshot_take`.

            Ok(Some(build_dup(context, module, registry, metadata, &info)?))
        },
    )?;

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    Ok(llvm::r#type::r#struct(
        context,
        &[ptr_ty, len_ty, len_ty, len_ty],
        false,
    ))
}

fn build_dup<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: &WithSelf<InfoAndTypeConcreteType>,
) -> Result<Region<'ctx>> {
    let location = Location::unknown(context);
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, module));
    }

    let value_ty = registry.build_type(context, module, registry, metadata, info.self_ty())?;
    let elem_ty = registry.get_type(&info.ty)?;
    let elem_stride = elem_ty.layout(registry)?.pad_to_align().size();
    let elem_ty = elem_ty.build(context, module, registry, metadata, &info.ty)?;

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    let src_value = entry.argument(0)?.into();
    let value_ptr = entry.extract_value(
        context,
        location,
        src_value,
        llvm::r#type::pointer(context, 0),
        0,
    )?;
    let value_start = entry.extract_value(
        context,
        location,
        src_value,
        IntegerType::new(context, 32).into(),
        1,
    )?;
    let value_end = entry.extract_value(
        context,
        location,
        src_value,
        IntegerType::new(context, 32).into(),
        2,
    )?;

    let value_len = entry.append_op_result(arith::subi(value_end, value_start, location))?;

    let k0 = entry.const_int(context, location, 0, 32)?;
    let value_is_empty = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        value_len,
        k0,
        location,
    ))?;

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;

    let block_realloc = region.append_block(Block::new(&[]));
    let block_finish =
        region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));
    entry.append_operation(cf::cond_br(
        context,
        value_is_empty,
        &block_finish,
        &block_realloc,
        &[null_ptr],
        &[],
        location,
    ));

    {
        let elem_stride = block_realloc.const_int(context, location, elem_stride, 64)?;

        let dst_value_len = {
            let value_len = block_realloc.append_op_result(arith::extui(
                value_len,
                IntegerType::new(context, 64).into(),
                location,
            ))?;

            block_realloc.append_op_result(arith::muli(value_len, elem_stride, location))?
        };
        let dst_value_ptr = {
            block_realloc.append_op_result(ReallocBindingsMeta::realloc(
                context,
                null_ptr,
                dst_value_len,
                location,
            ))?
        };

        let src_value_ptr = {
            let value_offset = block_realloc.append_op_result(arith::extui(
                value_start,
                IntegerType::new(context, 64).into(),
                location,
            ))?;

            let src_value_offset =
                block_realloc.append_op_result(arith::muli(value_offset, elem_stride, location))?;
            block_realloc.append_op_result(llvm::get_element_ptr_dynamic(
                context,
                value_ptr,
                &[src_value_offset],
                IntegerType::new(context, 8).into(),
                llvm::r#type::pointer(context, 0),
                location,
            ))?
        };

        match metadata.get::<DupOverrideMeta>() {
            Some(dup_override_meta) if dup_override_meta.is_overriden(&info.ty) => {
                let k0 = block_realloc.const_int(context, location, 0, 64)?;
                block_realloc.append_operation(scf::r#for(
                    k0,
                    dst_value_len,
                    elem_stride,
                    {
                        let region = Region::new();
                        let block = region.append_block(Block::new(&[(
                            IntegerType::new(context, 64).into(),
                            location,
                        )]));

                        let idx = block.argument(0)?.into();

                        let src_value_ptr =
                            block.append_op_result(llvm::get_element_ptr_dynamic(
                                context,
                                src_value_ptr,
                                &[idx],
                                IntegerType::new(context, 8).into(),
                                llvm::r#type::pointer(context, 0),
                                location,
                            ))?;
                        let dst_value_ptr =
                            block.append_op_result(llvm::get_element_ptr_dynamic(
                                context,
                                dst_value_ptr,
                                &[idx],
                                IntegerType::new(context, 8).into(),
                                llvm::r#type::pointer(context, 0),
                                location,
                            ))?;

                        let value = block.load(context, location, src_value_ptr, elem_ty)?;
                        let values = dup_override_meta
                            .invoke_override(context, &block, location, &info.ty, value)?;
                        block.store(context, location, src_value_ptr, values.0)?;
                        block.store(context, location, dst_value_ptr, values.1)?;

                        block.append_operation(scf::r#yield(&[], location));
                        region
                    },
                    location,
                ));
            }
            _ => {
                block_realloc.append_operation(
                    ods::llvm::intr_memcpy(
                        context,
                        dst_value_ptr,
                        src_value_ptr,
                        dst_value_len,
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                        location,
                    )
                    .into(),
                );
            }
        }

        block_realloc.append_operation(cf::br(&block_finish, &[dst_value_ptr], location));
    }

    {
        let dst_value = block_finish.append_op_result(llvm::undef(value_ty, location))?;
        let dst_value = block_finish.insert_values(
            context,
            location,
            dst_value,
            &[block_finish.argument(0)?.into(), k0, value_len, value_len],
        )?;

        block_finish.append_operation(func::r#return(&[src_value, dst_value], location));
    }

    Ok(region)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_drop<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
    value: Value<'ctx, 'this>,
) -> Result<()> {
    let payload_type = registry.get_type(&info.ty)?;
    let payload_ty = registry.build_type(context, helper, registry, metadata, &info.ty)?;

    let array_ty = registry.build_type(context, helper, registry, metadata, info.self_ty())?;
    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let ptr = entry.extract_value(context, location, value, ptr_ty, 0)?;

    let start = entry.extract_value(context, location, value, len_ty, 1)?;
    let end = entry.extract_value(context, location, value, len_ty, 2)?;
    let step = entry.const_int_from_type(context, location, 1, len_ty)?;

    entry.append_operation(scf::r#for(
        start,
        end,
        step,
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[(len_ty, location)]));

            let index = block.argument(0)?.into();
            let payload_ptr = block.append_op_result(llvm::get_element_ptr_dynamic(
                context,
                ptr,
                &[index],
                payload_ty,
                ptr_ty,
                location,
            ))?;

            let payload_value = block.load(context, location, payload_ptr, payload_ty)?;

            let helper = LibfuncHelper {
                module: helper.module,
                init_block: helper.init_block,
                region: &region,
                blocks_arena: helper.blocks_arena,
                last_block: Cell::new(&block),
                branches: Vec::new(),
                results: Vec::new(),
            };
            payload_type.build_drop(
                context,
                registry,
                &block,
                location,
                &helper,
                metadata,
                &info.ty,
                payload_value,
            )?;

            block.append_operation(scf::r#yield(&[], location));

            region
        },
        location,
    ));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{load_cairo, run_program},
        values::Value,
    };
    use pretty_assertions_sorted::assert_eq;

    #[test]
    fn test_array_snapshot_deep_clone() {
        let program = load_cairo! {
            fn run_test() -> @Array<Array<felt252>> {
                let mut inputs: Array<Array<felt252>> = ArrayTrait::new();
                inputs.append(array![1, 2, 3]);
                inputs.append(array![4, 5, 6]);

                @inputs
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            Value::Array(vec![
                Value::Array(vec![
                    Value::Felt252(1.into()),
                    Value::Felt252(2.into()),
                    Value::Felt252(3.into()),
                ]),
                Value::Array(vec![
                    Value::Felt252(4.into()),
                    Value::Felt252(5.into()),
                    Value::Felt252(6.into()),
                ]),
            ]),
        );
    }
}
