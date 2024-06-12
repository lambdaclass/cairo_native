////! # Enum-related libfuncs
//! # Enum-related libfuncs
////!
//!
////! Check out [the enum type](crate::types::enum) for more information on enum layouts.
//! Check out [the enum type](crate::types::enum) for more information on enum layouts.
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::{Error, Result},
    error::{Error, Result},
//    metadata::{enum_snapshot_variants::EnumSnapshotVariantsMeta, MetadataStorage},
    metadata::{enum_snapshot_variants::EnumSnapshotVariantsMeta, MetadataStorage},
//    types::TypeBuilder,
    types::TypeBuilder,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        enm::{EnumConcreteLibfunc, EnumFromBoundedIntConcreteLibfunc, EnumInitConcreteLibfunc},
        enm::{EnumConcreteLibfunc, EnumFromBoundedIntConcreteLibfunc, EnumInitConcreteLibfunc},
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//        ConcreteLibfunc,
        ConcreteLibfunc,
//    },
    },
//    ids::ConcreteTypeId,
    ids::ConcreteTypeId,
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{
    dialect::{
//        arith, cf,
        arith, cf,
//        llvm::{self, AllocaOptions, LoadStoreOptions},
        llvm::{self, AllocaOptions, LoadStoreOptions},
//        ods,
        ods,
//    },
    },
//    ir::{
    ir::{
//        attribute::{DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute},
        attribute::{DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute},
//        r#type::IntegerType,
        r#type::IntegerType,
//        Block, Location, Value,
        Block, Location, Value,
//    },
    },
//    Context,
    Context,
//};
};
//use std::num::TryFromIntError;
use std::num::TryFromIntError;
//

///// Select and call the correct libfunc builder function from the selector.
/// Select and call the correct libfunc builder function from the selector.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: &EnumConcreteLibfunc,
    selector: &EnumConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        EnumConcreteLibfunc::Init(info) => {
        EnumConcreteLibfunc::Init(info) => {
//            build_init(context, registry, entry, location, helper, metadata, info)
            build_init(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EnumConcreteLibfunc::Match(info) => {
        EnumConcreteLibfunc::Match(info) => {
//            build_match(context, registry, entry, location, helper, metadata, info)
            build_match(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EnumConcreteLibfunc::SnapshotMatch(info) => {
        EnumConcreteLibfunc::SnapshotMatch(info) => {
//            build_snapshot_match(context, registry, entry, location, helper, metadata, info)
            build_snapshot_match(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        EnumConcreteLibfunc::FromBoundedInt(info) => {
        EnumConcreteLibfunc::FromBoundedInt(info) => {
//            build_from_bounded_int(context, registry, entry, location, helper, metadata, info)
            build_from_bounded_int(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `enum_init` libfunc.
/// Generate MLIR operations for the `enum_init` libfunc.
//pub fn build_init<'ctx, 'this>(
pub fn build_init<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &EnumInitConcreteLibfunc,
    info: &EnumInitConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let val = build_enum_value(
    let val = build_enum_value(
//        context,
        context,
//        registry,
        registry,
//        entry,
        entry,
//        location,
        location,
//        helper,
        helper,
//        metadata,
        metadata,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//        &info.signature.param_signatures[0].ty,
        &info.signature.param_signatures[0].ty,
//        info.index,
        info.index,
//    )?;
    )?;
//    entry.append_operation(helper.br(0, &[val], location));
    entry.append_operation(helper.br(0, &[val], location));
//

//    Ok(())
    Ok(())
//}
}
//

//#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
//pub fn build_enum_value<'ctx, 'this>(
pub fn build_enum_value<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    payload_value: Value<'ctx, 'this>,
    payload_value: Value<'ctx, 'this>,
//    enum_type: &ConcreteTypeId,
    enum_type: &ConcreteTypeId,
//    variant_type: &ConcreteTypeId,
    variant_type: &ConcreteTypeId,
//    variant_index: usize,
    variant_index: usize,
//) -> Result<Value<'ctx, 'this>> {
) -> Result<Value<'ctx, 'this>> {
//    let type_info = registry.get_type(enum_type)?;
    let type_info = registry.get_type(enum_type)?;
//    let payload_type_info = registry.get_type(variant_type)?;
    let payload_type_info = registry.get_type(variant_type)?;
//

//    let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
    let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        type_info.variants().unwrap(),
        type_info.variants().unwrap(),
//    )?;
    )?;
//

//    Ok(match variant_tys.len() {
    Ok(match variant_tys.len() {
//        0 | 1 => payload_value,
        0 | 1 => payload_value,
//        _ => {
        _ => {
//            let enum_ty = llvm::r#type::r#struct(
            let enum_ty = llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    tag_ty,
                    tag_ty,
//                    if payload_type_info.is_zst(registry) {
                    if payload_type_info.is_zst(registry) {
//                        llvm::r#type::array(IntegerType::new(context, 8).into(), 0)
                        llvm::r#type::array(IntegerType::new(context, 8).into(), 0)
//                    } else {
                    } else {
//                        variant_tys[variant_index].0
                        variant_tys[variant_index].0
//                    },
                    },
//                ],
                ],
//                false,
                false,
//            );
            );
//

//            let tag_val = entry
            let tag_val = entry
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(
                    IntegerAttribute::new(
//                        tag_ty,
                        tag_ty,
//                        variant_index
                        variant_index
//                            .try_into()
                            .try_into()
//                            .expect("couldnt convert index to i64"),
                            .expect("couldnt convert index to i64"),
//                    )
                    )
//                    .into(),
                    .into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            let val = entry
            let val = entry
//                .append_operation(llvm::undef(enum_ty, location))
                .append_operation(llvm::undef(enum_ty, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let val = entry
            let val = entry
//                .append_operation(llvm::insert_value(
                .append_operation(llvm::insert_value(
//                    context,
                    context,
//                    val,
                    val,
//                    DenseI64ArrayAttribute::new(context, &[0]),
                    DenseI64ArrayAttribute::new(context, &[0]),
//                    tag_val,
                    tag_val,
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let mut val = if payload_type_info.is_zst(registry) {
            let mut val = if payload_type_info.is_zst(registry) {
//                val
                val
//            } else {
            } else {
//                entry
                entry
//                    .append_operation(llvm::insert_value(
                    .append_operation(llvm::insert_value(
//                        context,
                        context,
//                        val,
                        val,
//                        DenseI64ArrayAttribute::new(context, &[1]),
                        DenseI64ArrayAttribute::new(context, &[1]),
//                        payload_value,
                        payload_value,
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into()
                    .into()
//            };
            };
//

//            if type_info.is_memory_allocated(registry) {
            if type_info.is_memory_allocated(registry) {
//                let k1 = helper
                let k1 = helper
//                    .init_block()
                    .init_block()
//                    .append_operation(arith::constant(
                    .append_operation(arith::constant(
//                        context,
                        context,
//                        IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
                        IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//                let stack_ptr = helper
                let stack_ptr = helper
//                    .init_block()
                    .init_block()
//                    .append_operation(llvm::alloca(
                    .append_operation(llvm::alloca(
//                        context,
                        context,
//                        k1,
                        k1,
//                        llvm::r#type::pointer(context, 0),
                        llvm::r#type::pointer(context, 0),
//                        location,
                        location,
//                        AllocaOptions::new()
                        AllocaOptions::new()
//                            .align(Some(IntegerAttribute::new(
                            .align(Some(IntegerAttribute::new(
//                                IntegerType::new(context, 64).into(),
                                IntegerType::new(context, 64).into(),
//                                layout.align() as i64,
                                layout.align() as i64,
//                            )))
                            )))
//                            .elem_type(Some(TypeAttribute::new(
                            .elem_type(Some(TypeAttribute::new(
//                                type_info.build(context, helper, registry, metadata, enum_type)?,
                                type_info.build(context, helper, registry, metadata, enum_type)?,
//                            ))),
                            ))),
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                // Convert the enum from the concrete variant to the internal representation.
                // Convert the enum from the concrete variant to the internal representation.
//                entry.append_operation(llvm::store(
                entry.append_operation(llvm::store(
//                    context,
                    context,
//                    val,
                    val,
//                    stack_ptr,
                    stack_ptr,
//                    location,
                    location,
//                    LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                    LoadStoreOptions::new().align(Some(IntegerAttribute::new(
//                        IntegerType::new(context, 64).into(),
                        IntegerType::new(context, 64).into(),
//                        layout.align() as i64,
                        layout.align() as i64,
//                    ))),
                    ))),
//                ));
                ));
//                val = entry.load(
                val = entry.load(
//                    context,
                    context,
//                    location,
                    location,
//                    stack_ptr,
                    stack_ptr,
//                    type_info.build(context, helper, registry, metadata, enum_type)?,
                    type_info.build(context, helper, registry, metadata, enum_type)?,
//                    Some(layout.align()),
                    Some(layout.align()),
//                )?;
                )?;
//            };
            };
//

//            val
            val
//        }
        }
//    })
    })
//}
}
//

///// Generate MLIR operations for the `enum_init` libfunc.
/// Generate MLIR operations for the `enum_init` libfunc.
//pub fn build_from_bounded_int<'ctx, 'this>(
pub fn build_from_bounded_int<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &EnumFromBoundedIntConcreteLibfunc,
    info: &EnumFromBoundedIntConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let inp_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let inp_ty = registry.get_type(&info.param_signatures()[0].ty)?;
//    let varaint_selector_type: IntegerType = inp_ty
    let varaint_selector_type: IntegerType = inp_ty
//        .build(
        .build(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &info.param_signatures()[0].ty,
            &info.param_signatures()[0].ty,
//        )?
        )?
//        .try_into()?;
        .try_into()?;
//    let enum_type = registry.get_type(&info.branch_signatures()[0].vars[0].ty)?;
    let enum_type = registry.get_type(&info.branch_signatures()[0].vars[0].ty)?;
//    // we assume its never memory allocated since its always a enum with only a tag
    // we assume its never memory allocated since its always a enum with only a tag
//    assert!(!enum_type.is_memory_allocated(registry));
    assert!(!enum_type.is_memory_allocated(registry));
//

//    let enum_ty = enum_type.build(
    let enum_ty = enum_type.build(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//

//    let tag_bits = info.n_variants.next_power_of_two().trailing_zeros();
    let tag_bits = info.n_variants.next_power_of_two().trailing_zeros();
//    let tag_type = IntegerType::new(context, tag_bits);
    let tag_type = IntegerType::new(context, tag_bits);
//

//    let mut tag_value: Value = entry.argument(0)?.into();
    let mut tag_value: Value = entry.argument(0)?.into();
//

//    match tag_type.width().cmp(&varaint_selector_type.width()) {
    match tag_type.width().cmp(&varaint_selector_type.width()) {
//        std::cmp::Ordering::Less => {
        std::cmp::Ordering::Less => {
//            tag_value = entry.append_op_result(
            tag_value = entry.append_op_result(
//                ods::llvm::trunc(context, tag_type.into(), tag_value, location).into(),
                ods::llvm::trunc(context, tag_type.into(), tag_value, location).into(),
//            )?;
            )?;
//        }
        }
//        std::cmp::Ordering::Equal => {}
        std::cmp::Ordering::Equal => {}
//        std::cmp::Ordering::Greater => {
        std::cmp::Ordering::Greater => {
//            tag_value = entry.append_op_result(
            tag_value = entry.append_op_result(
//                ods::llvm::zext(context, tag_type.into(), tag_value, location).into(),
                ods::llvm::zext(context, tag_type.into(), tag_value, location).into(),
//            )?;
            )?;
//        }
        }
//    };
    };
//

//    let value = entry.append_op_result(llvm::undef(enum_ty, location))?;
    let value = entry.append_op_result(llvm::undef(enum_ty, location))?;
//    let value = entry.insert_value(context, location, value, tag_value, 0)?;
    let value = entry.insert_value(context, location, value, tag_value, 0)?;
//

//    entry.append_operation(helper.br(0, &[value], location));
    entry.append_operation(helper.br(0, &[value], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `enum_match` libfunc.
/// Generate MLIR operations for the `enum_match` libfunc.
//pub fn build_match<'ctx, 'this>(
pub fn build_match<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let type_info = registry.get_type(&info.param_signatures()[0].ty)?;
    let type_info = registry.get_type(&info.param_signatures()[0].ty)?;
//

//    let variant_ids = type_info.variants().unwrap();
    let variant_ids = type_info.variants().unwrap();
//    match variant_ids.len() {
    match variant_ids.len() {
//        0 | 1 => {
        0 | 1 => {
//            entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
            entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
//        }
        }
//        _ => {
        _ => {
//            let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
            let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
//                context,
                context,
//                helper,
                helper,
//                registry,
                registry,
//                metadata,
                metadata,
//                variant_ids,
                variant_ids,
//            )?;
            )?;
//

//            let (stack_ptr, tag_val) = if type_info.is_memory_allocated(registry) {
            let (stack_ptr, tag_val) = if type_info.is_memory_allocated(registry) {
//                let stack_ptr = helper.init_block().alloca1(
                let stack_ptr = helper.init_block().alloca1(
//                    context,
                    context,
//                    location,
                    location,
//                    type_info.build(
                    type_info.build(
//                        context,
                        context,
//                        helper,
                        helper,
//                        registry,
                        registry,
//                        metadata,
                        metadata,
//                        &info.param_signatures()[0].ty,
                        &info.param_signatures()[0].ty,
//                    )?,
                    )?,
//                    Some(layout.align()),
                    Some(layout.align()),
//                )?;
                )?;
//                entry.store(
                entry.store(
//                    context,
                    context,
//                    location,
                    location,
//                    stack_ptr,
                    stack_ptr,
//                    entry.argument(0)?.into(),
                    entry.argument(0)?.into(),
//                    Some(layout.align()),
                    Some(layout.align()),
//                );
                );
//                let tag_val =
                let tag_val =
//                    entry.load(context, location, stack_ptr, tag_ty, Some(layout.align()))?;
                    entry.load(context, location, stack_ptr, tag_ty, Some(layout.align()))?;
//

//                (Some(stack_ptr), tag_val)
                (Some(stack_ptr), tag_val)
//            } else {
            } else {
//                let tag_val = entry
                let tag_val = entry
//                    .append_operation(llvm::extract_value(
                    .append_operation(llvm::extract_value(
//                        context,
                        context,
//                        entry.argument(0)?.into(),
                        entry.argument(0)?.into(),
//                        DenseI64ArrayAttribute::new(context, &[0]),
                        DenseI64ArrayAttribute::new(context, &[0]),
//                        tag_ty,
                        tag_ty,
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                (None, tag_val)
                (None, tag_val)
//            };
            };
//

//            let default_block = helper.append_block(Block::new(&[]));
            let default_block = helper.append_block(Block::new(&[]));
//            let variant_blocks = variant_tys
            let variant_blocks = variant_tys
//                .iter()
                .iter()
//                .map(|_| helper.append_block(Block::new(&[])))
                .map(|_| helper.append_block(Block::new(&[])))
//                .collect::<Vec<_>>();
                .collect::<Vec<_>>();
//

//            let case_values = (0..variant_tys.len())
            let case_values = (0..variant_tys.len())
//                .map(i64::try_from)
                .map(i64::try_from)
//                .collect::<std::result::Result<Vec<_>, TryFromIntError>>()?;
                .collect::<std::result::Result<Vec<_>, TryFromIntError>>()?;
//

//            entry.append_operation(cf::switch(
            entry.append_operation(cf::switch(
//                context,
                context,
//                &case_values,
                &case_values,
//                tag_val,
                tag_val,
//                tag_ty,
                tag_ty,
//                (default_block, &[]),
                (default_block, &[]),
//                &variant_blocks
                &variant_blocks
//                    .iter()
                    .iter()
//                    .copied()
                    .copied()
//                    .map(|block| (block, [].as_slice()))
                    .map(|block| (block, [].as_slice()))
//                    .collect::<Vec<_>>(),
                    .collect::<Vec<_>>(),
//                location,
                location,
//            )?);
            )?);
//

//            // Default block.
            // Default block.
//            {
            {
//                let val = default_block
                let val = default_block
//                    .append_operation(arith::constant(
                    .append_operation(arith::constant(
//                        context,
                        context,
//                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0).into(),
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0).into(),
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                default_block.append_operation(cf::assert(
                default_block.append_operation(cf::assert(
//                    context,
                    context,
//                    val,
                    val,
//                    "Invalid enum tag.",
                    "Invalid enum tag.",
//                    location,
                    location,
//                ));
                ));
//                default_block.append_operation(llvm::unreachable(location));
                default_block.append_operation(llvm::unreachable(location));
//            }
            }
//

//            // Enum variants.
            // Enum variants.
//            for (i, (block, (payload_ty, _))) in
            for (i, (block, (payload_ty, _))) in
//                variant_blocks.into_iter().zip(variant_tys).enumerate()
                variant_blocks.into_iter().zip(variant_tys).enumerate()
//            {
            {
//                let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);
                let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);
//

//                let payload_val = match stack_ptr {
                let payload_val = match stack_ptr {
//                    Some(stack_ptr) => {
                    Some(stack_ptr) => {
//                        let val = block.load(
                        let val = block.load(
//                            context,
                            context,
//                            location,
                            location,
//                            stack_ptr,
                            stack_ptr,
//                            enum_ty,
                            enum_ty,
//                            Some(layout.align()),
                            Some(layout.align()),
//                        )?;
                        )?;
//                        block.extract_value(context, location, val, payload_ty, 1)?
                        block.extract_value(context, location, val, payload_ty, 1)?
//                    }
                    }
//                    None => {
                    None => {
//                        // If the enum is not memory-allocated it means that:
                        // If the enum is not memory-allocated it means that:
//                        //   - Either it's a C-style enum and all payloads have the same type.
                        //   - Either it's a C-style enum and all payloads have the same type.
//                        //   - Or the enum only has a single non-memory-allocated variant.
                        //   - Or the enum only has a single non-memory-allocated variant.
//                        if variant_ids.len() == 1 {
                        if variant_ids.len() == 1 {
//                            entry.argument(0)?.into()
                            entry.argument(0)?.into()
//                        } else {
                        } else {
//                            assert!(registry.get_type(&variant_ids[i])?.is_zst(registry));
                            assert!(registry.get_type(&variant_ids[i])?.is_zst(registry));
//                            block
                            block
//                                .append_operation(llvm::undef(payload_ty, location))
                                .append_operation(llvm::undef(payload_ty, location))
//                                .result(0)?
                                .result(0)?
//                                .into()
                                .into()
//                        }
                        }
//                    }
                    }
//                };
                };
//

//                block.append_operation(helper.br(i, &[payload_val], location));
                block.append_operation(helper.br(i, &[payload_val], location));
//            }
            }
//        }
        }
//    }
    }
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `enum_snapshot_match` libfunc.
/// Generate MLIR operations for the `enum_snapshot_match` libfunc.
//pub fn build_snapshot_match<'ctx, 'this>(
pub fn build_snapshot_match<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let type_info = registry.get_type(&info.param_signatures()[0].ty)?;
    let type_info = registry.get_type(&info.param_signatures()[0].ty)?;
//

//    // This libfunc's implementation is identical to `enum_match` aside from fetching the snapshotted enum's variants from the metadata:
    // This libfunc's implementation is identical to `enum_match` aside from fetching the snapshotted enum's variants from the metadata:
//    let variant_ids = metadata
    let variant_ids = metadata
//        .get::<EnumSnapshotVariantsMeta>()
        .get::<EnumSnapshotVariantsMeta>()
//        .ok_or(Error::MissingMetadata)?
        .ok_or(Error::MissingMetadata)?
//        .get_variants(&info.param_signatures()[0].ty)
        .get_variants(&info.param_signatures()[0].ty)
//        .expect("enum should always have variants")
        .expect("enum should always have variants")
//        .clone();
        .clone();
//    match variant_ids.len() {
    match variant_ids.len() {
//        0 | 1 => {
        0 | 1 => {
//            entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
            entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
//        }
        }
//        _ => {
        _ => {
//            let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
            let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
//                context,
                context,
//                helper,
                helper,
//                registry,
                registry,
//                metadata,
                metadata,
//                &variant_ids,
                &variant_ids,
//            )?;
            )?;
//

//            let (stack_ptr, tag_val) = if type_info.is_memory_allocated(registry) {
            let (stack_ptr, tag_val) = if type_info.is_memory_allocated(registry) {
//                let stack_ptr = helper.init_block().alloca1(
                let stack_ptr = helper.init_block().alloca1(
//                    context,
                    context,
//                    location,
                    location,
//                    type_info.build(
                    type_info.build(
//                        context,
                        context,
//                        helper,
                        helper,
//                        registry,
                        registry,
//                        metadata,
                        metadata,
//                        &info.param_signatures()[0].ty,
                        &info.param_signatures()[0].ty,
//                    )?,
                    )?,
//                    Some(layout.align()),
                    Some(layout.align()),
//                )?;
                )?;
//                entry.store(
                entry.store(
//                    context,
                    context,
//                    location,
                    location,
//                    stack_ptr,
                    stack_ptr,
//                    entry.argument(0)?.into(),
                    entry.argument(0)?.into(),
//                    Some(layout.align()),
                    Some(layout.align()),
//                );
                );
//                let tag_val =
                let tag_val =
//                    entry.load(context, location, stack_ptr, tag_ty, Some(layout.align()))?;
                    entry.load(context, location, stack_ptr, tag_ty, Some(layout.align()))?;
//

//                (Some(stack_ptr), tag_val)
                (Some(stack_ptr), tag_val)
//            } else {
            } else {
//                let tag_val = entry
                let tag_val = entry
//                    .append_operation(llvm::extract_value(
                    .append_operation(llvm::extract_value(
//                        context,
                        context,
//                        entry.argument(0)?.into(),
                        entry.argument(0)?.into(),
//                        DenseI64ArrayAttribute::new(context, &[0]),
                        DenseI64ArrayAttribute::new(context, &[0]),
//                        tag_ty,
                        tag_ty,
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                (None, tag_val)
                (None, tag_val)
//            };
            };
//

//            let default_block = helper.append_block(Block::new(&[]));
            let default_block = helper.append_block(Block::new(&[]));
//            let variant_blocks = variant_tys
            let variant_blocks = variant_tys
//                .iter()
                .iter()
//                .map(|_| helper.append_block(Block::new(&[])))
                .map(|_| helper.append_block(Block::new(&[])))
//                .collect::<Vec<_>>();
                .collect::<Vec<_>>();
//

//            let case_values = (0..variant_tys.len())
            let case_values = (0..variant_tys.len())
//                .map(i64::try_from)
                .map(i64::try_from)
//                .collect::<std::result::Result<Vec<_>, TryFromIntError>>()?;
                .collect::<std::result::Result<Vec<_>, TryFromIntError>>()?;
//

//            entry.append_operation(cf::switch(
            entry.append_operation(cf::switch(
//                context,
                context,
//                &case_values,
                &case_values,
//                tag_val,
                tag_val,
//                tag_ty,
                tag_ty,
//                (default_block, &[]),
                (default_block, &[]),
//                &variant_blocks
                &variant_blocks
//                    .iter()
                    .iter()
//                    .copied()
                    .copied()
//                    .map(|block| (block, [].as_slice()))
                    .map(|block| (block, [].as_slice()))
//                    .collect::<Vec<_>>(),
                    .collect::<Vec<_>>(),
//                location,
                location,
//            )?);
            )?);
//

//            // Default block.
            // Default block.
//            {
            {
//                let val = default_block
                let val = default_block
//                    .append_operation(arith::constant(
                    .append_operation(arith::constant(
//                        context,
                        context,
//                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0).into(),
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0).into(),
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                default_block.append_operation(cf::assert(
                default_block.append_operation(cf::assert(
//                    context,
                    context,
//                    val,
                    val,
//                    "Invalid enum tag.",
                    "Invalid enum tag.",
//                    location,
                    location,
//                ));
                ));
//                default_block.append_operation(llvm::unreachable(location));
                default_block.append_operation(llvm::unreachable(location));
//            }
            }
//

//            // Enum variants.
            // Enum variants.
//            for (i, (block, (payload_ty, _))) in
            for (i, (block, (payload_ty, _))) in
//                variant_blocks.into_iter().zip(variant_tys).enumerate()
                variant_blocks.into_iter().zip(variant_tys).enumerate()
//            {
            {
//                let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);
                let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);
//

//                let payload_val = match stack_ptr {
                let payload_val = match stack_ptr {
//                    Some(stack_ptr) => {
                    Some(stack_ptr) => {
//                        let val = block.load(
                        let val = block.load(
//                            context,
                            context,
//                            location,
                            location,
//                            stack_ptr,
                            stack_ptr,
//                            enum_ty,
                            enum_ty,
//                            Some(layout.align()),
                            Some(layout.align()),
//                        )?;
                        )?;
//                        block.extract_value(context, location, val, payload_ty, 1)?
                        block.extract_value(context, location, val, payload_ty, 1)?
//                    }
                    }
//                    None => {
                    None => {
//                        // If the enum is not memory-allocated it means that:
                        // If the enum is not memory-allocated it means that:
//                        //   - Either it's a C-style enum and all payloads have the same type.
                        //   - Either it's a C-style enum and all payloads have the same type.
//                        //   - Or the enum only has a single non-memory-allocated variant.
                        //   - Or the enum only has a single non-memory-allocated variant.
//                        if variant_ids.len() == 1 {
                        if variant_ids.len() == 1 {
//                            entry.argument(0)?.into()
                            entry.argument(0)?.into()
//                        } else {
                        } else {
//                            assert!(registry.get_type(&variant_ids[i])?.is_zst(registry));
                            assert!(registry.get_type(&variant_ids[i])?.is_zst(registry));
//                            block
                            block
//                                .append_operation(llvm::undef(payload_ty, location))
                                .append_operation(llvm::undef(payload_ty, location))
//                                .result(0)?
                                .result(0)?
//                                .into()
                                .into()
//                        }
                        }
//                    }
                    }
//                };
                };
//

//                block.append_operation(helper.br(i, &[payload_val], location));
                block.append_operation(helper.br(i, &[payload_val], location));
//            }
            }
//        }
        }
//    }
    }
//

//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output};
    use crate::utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output};
//    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::Program;
//    use lazy_static::lazy_static;
    use lazy_static::lazy_static;
//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//

//    lazy_static! {
    lazy_static! {
//        static ref ENUM_INIT: (String, Program) = load_cairo! {
        static ref ENUM_INIT: (String, Program) = load_cairo! {
//            enum MySmallEnum {
            enum MySmallEnum {
//                A: felt252,
                A: felt252,
//            }
            }
//

//            enum MyEnum {
            enum MyEnum {
//                A: felt252,
                A: felt252,
//                B: u8,
                B: u8,
//                C: u16,
                C: u16,
//                D: u32,
                D: u32,
//                E: u64,
                E: u64,
//            }
            }
//

//            fn run_test() -> (MySmallEnum, MyEnum, MyEnum, MyEnum, MyEnum, MyEnum) {
            fn run_test() -> (MySmallEnum, MyEnum, MyEnum, MyEnum, MyEnum, MyEnum) {
//                (
                (
//                    MySmallEnum::A(-1),
                    MySmallEnum::A(-1),
//                    MyEnum::A(5678),
                    MyEnum::A(5678),
//                    MyEnum::B(90),
                    MyEnum::B(90),
//                    MyEnum::C(9012),
                    MyEnum::C(9012),
//                    MyEnum::D(34567890),
                    MyEnum::D(34567890),
//                    MyEnum::E(1234567890123456),
                    MyEnum::E(1234567890123456),
//                )
                )
//            }
            }
//        };
        };
//        static ref ENUM_MATCH: (String, Program) = load_cairo! {
        static ref ENUM_MATCH: (String, Program) = load_cairo! {
//            enum MyEnum {
            enum MyEnum {
//                A: felt252,
                A: felt252,
//                B: u8,
                B: u8,
//                C: u16,
                C: u16,
//                D: u32,
                D: u32,
//                E: u64,
                E: u64,
//            }
            }
//

//            fn match_a() -> felt252 {
            fn match_a() -> felt252 {
//                let x = MyEnum::A(5);
                let x = MyEnum::A(5);
//                match x {
                match x {
//                    MyEnum::A(x) => x,
                    MyEnum::A(x) => x,
//                    MyEnum::B(_) => 0,
                    MyEnum::B(_) => 0,
//                    MyEnum::C(_) => 1,
                    MyEnum::C(_) => 1,
//                    MyEnum::D(_) => 2,
                    MyEnum::D(_) => 2,
//                    MyEnum::E(_) => 3,
                    MyEnum::E(_) => 3,
//                }
                }
//            }
            }
//

//            fn match_b() -> u8 {
            fn match_b() -> u8 {
//                let x = MyEnum::B(5_u8);
                let x = MyEnum::B(5_u8);
//                match x {
                match x {
//                    MyEnum::A(_) => 0_u8,
                    MyEnum::A(_) => 0_u8,
//                    MyEnum::B(x) => x,
                    MyEnum::B(x) => x,
//                    MyEnum::C(_) => 1_u8,
                    MyEnum::C(_) => 1_u8,
//                    MyEnum::D(_) => 2_u8,
                    MyEnum::D(_) => 2_u8,
//                    MyEnum::E(_) => 3_u8,
                    MyEnum::E(_) => 3_u8,
//                }
                }
//            }
            }
//        };
        };
//    }
    }
//

//    #[test]
    #[test]
//    fn enum_init() {
    fn enum_init() {
//        run_program_assert_output(
        run_program_assert_output(
//            &ENUM_INIT,
            &ENUM_INIT,
//            "run_test",
            "run_test",
//            &[],
            &[],
//            jit_struct!(
            jit_struct!(
//                jit_enum!(0, Felt::from(-1).into()),
                jit_enum!(0, Felt::from(-1).into()),
//                jit_enum!(0, Felt::from(5678).into()),
                jit_enum!(0, Felt::from(5678).into()),
//                jit_enum!(1, 90u8.into()),
                jit_enum!(1, 90u8.into()),
//                jit_enum!(2, 9012u16.into()),
                jit_enum!(2, 9012u16.into()),
//                jit_enum!(3, 34567890u32.into()),
                jit_enum!(3, 34567890u32.into()),
//                jit_enum!(4, 1234567890123456u64.into()),
                jit_enum!(4, 1234567890123456u64.into()),
//            ),
            ),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn enum_match() {
    fn enum_match() {
//        run_program_assert_output(&ENUM_MATCH, "match_a", &[], Felt::from(5).into());
        run_program_assert_output(&ENUM_MATCH, "match_a", &[], Felt::from(5).into());
//        run_program_assert_output(&ENUM_MATCH, "match_b", &[], 5u8.into());
        run_program_assert_output(&ENUM_MATCH, "match_b", &[], 5u8.into());
//    }
    }
//}
}
