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
//! The pointer to the allocation (which is **not the data**) contains:
//!   1. Reference counter.
//!   2. Padding.
//!   3. Array data. Its address is the pointer to the data stored in the type.
//!
//! [^1]: When capacity is zero, this field is not guaranteed to be valid.
//! [^2]: Those numbers are number of items, **not bytes**.

use super::{TypeBuilder, WithSelf};
use crate::{
    error::Result,
    metadata::{
        drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta,
        realloc_bindings::ReallocBindingsMeta, MetadataStorage,
    },
    utils::{get_integer_layout, BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::ir::Region;
use melior::{
    dialect::{arith, llvm},
    ir::{r#type::IntegerType, Block, Location, Module, Type},
    Context,
};
use melior::{
    dialect::{arith::CmpiPredicate, func, scf},
    ir::attribute::DenseI32ArrayAttribute,
};

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
    DupOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `build_dup`.

            Ok(Some(build_dup(context, module, registry, metadata, &info)?))
        },
    )?;
    DropOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `build_drop`.

            Ok(Some(build_drop(
                context, module, registry, metadata, &info,
            )?))
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
    let value_ty = registry.build_type(context, module, registry, metadata, info.self_ty())?;

    let elem_layout = registry.get_type(&info.ty)?.layout(registry)?;
    let refcount_offset = get_integer_layout(32)
        .align_to(elem_layout.align())
        .unwrap()
        .pad_to_align()
        .size();

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    let array_cap = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        IntegerType::new(context, 32).into(),
        3,
    )?;
    let k0 = entry.const_int(context, location, 0, 32)?;
    let is_empty = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        array_cap,
        k0,
        location,
    ))?;

    entry.append_operation(scf::r#if(
        is_empty,
        &[],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            metadata
                .get_mut::<crate::metadata::debug_utils::DebugUtils>()
                .unwrap()
                .debug_print(
                    context,
                    module,
                    &block,
                    "[MEM] Cloning empty array.",
                    location,
                )?;

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let array_ptr = block.extract_value(
                context,
                location,
                entry.argument(0)?.into(),
                llvm::r#type::pointer(context, 0),
                0,
            )?;

            let refcount_ptr = block.append_op_result(llvm::get_element_ptr(
                context,
                array_ptr,
                DenseI32ArrayAttribute::new(context, &[-(refcount_offset as i32)]),
                IntegerType::new(context, 8).into(),
                llvm::r#type::pointer(context, 0),
                location,
            ))?;
            let ref_count = block.load(
                context,
                location,
                refcount_ptr,
                IntegerType::new(context, 32).into(),
            )?;

            metadata
                .get_mut::<crate::metadata::debug_utils::DebugUtils>()
                .unwrap()
                .debug_print(
                    context,
                    module,
                    &block,
                    "[MEM] Cloning non-empty array (ref_count += 1). Original ref_count:",
                    location,
                )?;
            metadata
                .get_mut::<crate::metadata::debug_utils::DebugUtils>()
                .unwrap()
                .print_i32(context, module, &block, ref_count, location)?;

            let k1 = block.const_int(context, location, 1, 32)?;
            let ref_count = block.append_op_result(arith::addi(ref_count, k1, location))?;
            block.store(context, location, refcount_ptr, ref_count)?;

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        location,
    ));

    entry.append_operation(func::r#return(
        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
        location,
    ));
    Ok(region)

    // let location = Location::unknown(context);
    // if metadata.get::<ReallocBindingsMeta>().is_none() {
    //     metadata.insert(ReallocBindingsMeta::new(context, module));
    // }

    // let value_ty = registry.build_type(context, module, registry, metadata, info.self_ty())?;
    // let elem_ty = registry.get_type(&info.ty)?;
    // let elem_stride = elem_ty.layout(registry)?.pad_to_align().size();
    // let elem_ty = elem_ty.build(context, module, registry, metadata, &info.ty)?;

    // let region = Region::new();
    // let entry = region.append_block(Block::new(&[(value_ty, location)]));

    // let src_value = entry.argument(0)?.into();
    // let value_ptr = entry.extract_value(
    //     context,
    //     location,
    //     src_value,
    //     llvm::r#type::pointer(context, 0),
    //     0,
    // )?;
    // let value_start = entry.extract_value(
    //     context,
    //     location,
    //     src_value,
    //     IntegerType::new(context, 32).into(),
    //     1,
    // )?;
    // let value_end = entry.extract_value(
    //     context,
    //     location,
    //     src_value,
    //     IntegerType::new(context, 32).into(),
    //     2,
    // )?;

    // let value_len = entry.append_op_result(arith::subi(value_end, value_start, location))?;

    // let k0 = entry.const_int(context, location, 0, 32)?;
    // let value_is_empty = entry.append_op_result(arith::cmpi(
    //     context,
    //     CmpiPredicate::Eq,
    //     value_len,
    //     k0,
    //     location,
    // ))?;

    // let null_ptr =
    //     entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;

    // let block_realloc = region.append_block(Block::new(&[]));
    // let block_finish =
    //     region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));
    // entry.append_operation(cf::cond_br(
    //     context,
    //     value_is_empty,
    //     &block_finish,
    //     &block_realloc,
    //     &[null_ptr],
    //     &[],
    //     location,
    // ));

    // {
    //     let elem_stride = block_realloc.const_int(context, location, elem_stride, 64)?;

    //     let dst_value_len = {
    //         let value_len = block_realloc.append_op_result(arith::extui(
    //             value_len,
    //             IntegerType::new(context, 64).into(),
    //             location,
    //         ))?;

    //         block_realloc.append_op_result(arith::muli(value_len, elem_stride, location))?
    //     };
    //     let dst_value_ptr = {
    //         block_realloc.append_op_result(ReallocBindingsMeta::realloc(
    //             context,
    //             null_ptr,
    //             dst_value_len,
    //             location,
    //         ))?
    //     };

    //     let src_value_ptr = {
    //         let value_offset = block_realloc.append_op_result(arith::extui(
    //             value_start,
    //             IntegerType::new(context, 64).into(),
    //             location,
    //         ))?;

    //         let src_value_offset =
    //             block_realloc.append_op_result(arith::muli(value_offset, elem_stride, location))?;
    //         block_realloc.append_op_result(llvm::get_element_ptr_dynamic(
    //             context,
    //             value_ptr,
    //             &[src_value_offset],
    //             IntegerType::new(context, 8).into(),
    //             llvm::r#type::pointer(context, 0),
    //             location,
    //         ))?
    //     };

    //     match metadata.get::<DupOverridesMeta>() {
    //         Some(dup_override_meta) if dup_override_meta.is_overriden(&info.ty) => {
    //             let k0 = block_realloc.const_int(context, location, 0, 64)?;
    //             block_realloc.append_operation(scf::r#for(
    //                 k0,
    //                 dst_value_len,
    //                 elem_stride,
    //                 {
    //                     let region = Region::new();
    //                     let block = region.append_block(Block::new(&[(
    //                         IntegerType::new(context, 64).into(),
    //                         location,
    //                     )]));

    //                     let idx = block.argument(0)?.into();

    //                     let src_value_ptr =
    //                         block.append_op_result(llvm::get_element_ptr_dynamic(
    //                             context,
    //                             src_value_ptr,
    //                             &[idx],
    //                             IntegerType::new(context, 8).into(),
    //                             llvm::r#type::pointer(context, 0),
    //                             location,
    //                         ))?;
    //                     let dst_value_ptr =
    //                         block.append_op_result(llvm::get_element_ptr_dynamic(
    //                             context,
    //                             dst_value_ptr,
    //                             &[idx],
    //                             IntegerType::new(context, 8).into(),
    //                             llvm::r#type::pointer(context, 0),
    //                             location,
    //                         ))?;

    //                     let value = block.load(context, location, src_value_ptr, elem_ty)?;
    //                     let values = dup_override_meta
    //                         .invoke_override(context, &block, location, &info.ty, value)?;
    //                     block.store(context, location, src_value_ptr, values.0)?;
    //                     block.store(context, location, dst_value_ptr, values.1)?;

    //                     block.append_operation(scf::r#yield(&[], location));
    //                     region
    //                 },
    //                 location,
    //             ));
    //         }
    //         _ => {
    //             block_realloc.append_operation(
    //                 ods::llvm::intr_memcpy(
    //                     context,
    //                     dst_value_ptr,
    //                     src_value_ptr,
    //                     dst_value_len,
    //                     IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
    //                     location,
    //                 )
    //                 .into(),
    //             );
    //         }
    //     }

    //     block_realloc.append_operation(cf::br(&block_finish, &[dst_value_ptr], location));
    // }

    // {
    //     let dst_value = block_finish.append_op_result(llvm::undef(value_ty, location))?;
    //     let dst_value = block_finish.insert_values(
    //         context,
    //         location,
    //         dst_value,
    //         &[block_finish.argument(0)?.into(), k0, value_len, value_len],
    //     )?;

    //     block_finish.append_operation(func::r#return(&[src_value, dst_value], location));
    // }

    // Ok(region)
}

fn build_drop<'ctx>(
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
    let elem_layout = registry.get_type(&info.ty)?.layout(registry)?;
    let refcount_offset = get_integer_layout(32)
        .align_to(elem_layout.align())
        .unwrap()
        .pad_to_align()
        .size();

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    let array_ptr = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        llvm::r#type::pointer(context, 0),
        0,
    )?;

    let array_cap = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        IntegerType::new(context, 32).into(),
        3,
    )?;
    let k0 = entry.const_int(context, location, 0, 32)?;
    let is_empty = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        array_cap,
        k0,
        location,
    ))?;

    entry.append_operation(scf::r#if(
        is_empty,
        &[],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let refcount_ptr = block.append_op_result(llvm::get_element_ptr(
                context,
                array_ptr,
                DenseI32ArrayAttribute::new(context, &[-(refcount_offset as i32)]),
                IntegerType::new(context, 8).into(),
                llvm::r#type::pointer(context, 0),
                location,
            ))?;
            let ref_count = block.load(
                context,
                location,
                refcount_ptr,
                IntegerType::new(context, 32).into(),
            )?;

            let k1 = block.const_int(context, location, 1, 32)?;
            let is_shared = block.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Ne,
                ref_count,
                k1,
                location,
            ))?;

            block.append_operation(scf::r#if(
                is_shared,
                &[],
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    let ref_count = block.append_op_result(arith::subi(ref_count, k1, location))?;
                    block.store(context, location, refcount_ptr, ref_count)?;

                    metadata
                        .get_mut::<crate::metadata::debug_utils::DebugUtils>()
                        .unwrap()
                        .debug_print(
                            context,
                            module,
                            &block,
                            "[MEM] Dropping non-empty array (ref_count -= 1). Original ref_count:",
                            location,
                        )?;
                    metadata
                        .get_mut::<crate::metadata::debug_utils::DebugUtils>()
                        .unwrap()
                        .print_i32(context, module, &block, ref_count, location)?;

                    block.append_operation(scf::r#yield(&[], location));
                    region
                },
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    metadata
                        .get_mut::<crate::metadata::debug_utils::DebugUtils>()
                        .unwrap()
                        .debug_print(
                            context,
                            module,
                            &block,
                            "[MEM] Dropping non-empty array (ref_count -= 1). Freeing memory.",
                            location,
                        )?;

                    match metadata.get::<DropOverridesMeta>() {
                        Some(drop_overrides_meta) if drop_overrides_meta.is_overriden(&info.ty) => {
                            let value_start = block.extract_value(
                                context,
                                location,
                                entry.argument(0)?.into(),
                                IntegerType::new(context, 32).into(),
                                1,
                            )?;
                            let value_end = block.extract_value(
                                context,
                                location,
                                entry.argument(0)?.into(),
                                IntegerType::new(context, 32).into(),
                                2,
                            )?;

                            let value_start = block.append_op_result(arith::extui(
                                value_start,
                                IntegerType::new(context, 64).into(),
                                location,
                            ))?;
                            let value_end = block.append_op_result(arith::extui(
                                value_end,
                                IntegerType::new(context, 64).into(),
                                location,
                            ))?;

                            let elem_stride =
                                block.const_int(context, location, elem_stride, 64)?;
                            let offset_start = block.append_op_result(arith::muli(
                                value_start,
                                elem_stride,
                                location,
                            ))?;
                            let offset_end = block.append_op_result(arith::muli(
                                value_end,
                                elem_stride,
                                location,
                            ))?;

                            block.append_operation(scf::r#for(
                                offset_start,
                                offset_end,
                                elem_stride,
                                {
                                    let region = Region::new();
                                    let block = region.append_block(Block::new(&[(
                                        IntegerType::new(context, 64).into(),
                                        location,
                                    )]));

                                    let elem_offset = block.argument(0)?.into();
                                    let elem_ptr =
                                        block.append_op_result(llvm::get_element_ptr_dynamic(
                                            context,
                                            array_ptr,
                                            &[elem_offset],
                                            IntegerType::new(context, 8).into(),
                                            llvm::r#type::pointer(context, 0),
                                            location,
                                        ))?;
                                    let elem_val =
                                        block.load(context, location, elem_ptr, elem_ty)?;

                                    drop_overrides_meta.invoke_override(
                                        context, &block, location, &info.ty, elem_val,
                                    )?;

                                    block.append_operation(scf::r#yield(&[], location));
                                    region
                                },
                                location,
                            ));
                        }
                        _ => {}
                    }

                    block.append_operation(ReallocBindingsMeta::free(
                        context,
                        refcount_ptr,
                        location,
                    ));
                    block.append_operation(scf::r#yield(&[], location));
                    region
                },
                location,
            ));

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        location,
    ));

    entry.append_operation(func::r#return(&[], location));
    Ok(region)
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
