//! # Enum-related libfuncs
//!
//! Check out [the enum type](crate::types::enum) for more information on enum layouts.

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt,
    error::{Error, Result},
    metadata::{enum_snapshot_variants::EnumSnapshotVariantsMeta, MetadataStorage},
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        enm::{EnumConcreteLibfunc, EnumInitConcreteLibfunc},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith, cf,
        llvm::{self, AllocaOptions, LoadStoreOptions},
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute},
        r#type::IntegerType,
        Block, Location,
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
    selector: &EnumConcreteLibfunc,
) -> Result<()> {
    match selector {
        EnumConcreteLibfunc::Init(info) => {
            build_init(context, registry, entry, location, helper, metadata, info)
        }
        EnumConcreteLibfunc::Match(info) => {
            build_match(context, registry, entry, location, helper, metadata, info)
        }
        EnumConcreteLibfunc::SnapshotMatch(info) => {
            build_snapshot_match(context, registry, entry, location, helper, metadata, info)
        }
        EnumConcreteLibfunc::FromBoundedInt(_) => todo!(),
    }
}

/// Generate MLIR operations for the `enum_init` libfunc.
pub fn build_init<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &EnumInitConcreteLibfunc,
) -> Result<()> {
    let type_info = registry.get_type(&info.branch_signatures()[0].vars[0].ty)?;
    let payload_type_info = registry.get_type(&info.signature.param_signatures[0].ty)?;

    let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
        context,
        helper,
        registry,
        metadata,
        type_info.variants().unwrap(),
    )?;

    match variant_tys.len() {
        0 | 1 => {
            entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
        }
        _ => {
            let enum_ty = llvm::r#type::r#struct(
                context,
                &[
                    tag_ty,
                    if payload_type_info.is_zst(registry) {
                        llvm::r#type::array(IntegerType::new(context, 8).into(), 0)
                    } else {
                        variant_tys[info.index].0
                    },
                ],
                false,
            );

            let tag_val = entry
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(tag_ty, info.index as i64).into(),
                    location,
                ))
                .result(0)?
                .into();

            let val = entry
                .append_operation(llvm::undef(enum_ty, location))
                .result(0)?
                .into();
            let val = entry
                .append_operation(llvm::insert_value(
                    context,
                    val,
                    DenseI64ArrayAttribute::new(context, &[0]),
                    tag_val,
                    location,
                ))
                .result(0)?
                .into();
            let mut val = if payload_type_info.is_zst(registry) {
                val
            } else {
                entry
                    .append_operation(llvm::insert_value(
                        context,
                        val,
                        DenseI64ArrayAttribute::new(context, &[1]),
                        entry.argument(0)?.into(),
                        location,
                    ))
                    .result(0)?
                    .into()
            };

            if type_info.is_memory_allocated(registry) {
                let k1 = helper
                    .init_block()
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let stack_ptr = helper
                    .init_block()
                    .append_operation(llvm::alloca(
                        context,
                        k1,
                        llvm::r#type::opaque_pointer(context),
                        location,
                        AllocaOptions::new()
                            .align(Some(IntegerAttribute::new(
                                IntegerType::new(context, 64).into(),
                                layout.align() as i64,
                            )))
                            .elem_type(Some(TypeAttribute::new(type_info.build(
                                context,
                                helper,
                                registry,
                                metadata,
                                &info.branch_signatures()[0].vars[0].ty,
                            )?))),
                    ))
                    .result(0)?
                    .into();

                // Convert the enum from the concrete variant to the internal representation.
                entry.append_operation(llvm::store(
                    context,
                    val,
                    stack_ptr,
                    location,
                    LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                        IntegerType::new(context, 64).into(),
                        layout.align() as i64,
                    ))),
                ));
                val = entry.load(
                    context,
                    location,
                    stack_ptr,
                    type_info.build(
                        context,
                        helper,
                        registry,
                        metadata,
                        &info.branch_signatures()[0].vars[0].ty,
                    )?,
                    Some(layout.align()),
                )?;
            };

            entry.append_operation(helper.br(0, &[val], location));
        }
    }

    Ok(())
}

/// Generate MLIR operations for the `enum_match` libfunc.
pub fn build_match<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let type_info = registry.get_type(&info.param_signatures()[0].ty)?;

    let variant_ids = type_info.variants().unwrap();
    match variant_ids.len() {
        0 | 1 => {
            entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
        }
        _ => {
            let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
                context,
                helper,
                registry,
                metadata,
                variant_ids,
            )?;

            let (stack_ptr, tag_val) = if type_info.is_memory_allocated(registry) {
                let stack_ptr = helper.init_block().alloca1(
                    context,
                    location,
                    type_info.build(
                        context,
                        helper,
                        registry,
                        metadata,
                        &info.param_signatures()[0].ty,
                    )?,
                    Some(layout.align()),
                )?;
                entry.store(
                    context,
                    location,
                    stack_ptr,
                    entry.argument(0)?.into(),
                    Some(layout.align()),
                );
                let tag_val =
                    entry.load(context, location, stack_ptr, tag_ty, Some(layout.align()))?;

                (Some(stack_ptr), tag_val)
            } else {
                let tag_val = entry
                    .append_operation(llvm::extract_value(
                        context,
                        entry.argument(0)?.into(),
                        DenseI64ArrayAttribute::new(context, &[0]),
                        tag_ty,
                        location,
                    ))
                    .result(0)?
                    .into();

                (None, tag_val)
            };

            let default_block = helper.append_block(Block::new(&[]));
            let variant_blocks = variant_tys
                .iter()
                .map(|_| helper.append_block(Block::new(&[])))
                .collect::<Vec<_>>();

            let case_values = (0..variant_tys.len())
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;

            entry.append_operation(cf::switch(
                context,
                &case_values,
                tag_val,
                tag_ty,
                (default_block, &[]),
                &variant_blocks
                    .iter()
                    .copied()
                    .map(|block| (block, [].as_slice()))
                    .collect::<Vec<_>>(),
                location,
            )?);

            // Default block.
            {
                let val = default_block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0).into(),
                        location,
                    ))
                    .result(0)?
                    .into();

                default_block.append_operation(cf::assert(
                    context,
                    val,
                    "Invalid enum tag.",
                    location,
                ));
                default_block.append_operation(llvm::unreachable(location));
            }

            // Enum variants.
            for (i, (block, (payload_ty, _))) in
                variant_blocks.into_iter().zip(variant_tys).enumerate()
            {
                let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);

                let payload_val = match stack_ptr {
                    Some(stack_ptr) => {
                        let val = block.load(
                            context,
                            location,
                            stack_ptr,
                            enum_ty,
                            Some(layout.align()),
                        )?;
                        block.extract_value(context, location, val, payload_ty, 1)?
                    }
                    None => {
                        // If the enum is not memory-allocated it means that:
                        //   - Either it's a C-style enum and all payloads have the same type.
                        //   - Or the enum only has a single non-memory-allocated variant.
                        if variant_ids.len() == 1 {
                            entry.argument(0)?.into()
                        } else {
                            assert!(registry.get_type(&variant_ids[i])?.is_zst(registry));
                            block
                                .append_operation(llvm::undef(payload_ty, location))
                                .result(0)?
                                .into()
                        }
                    }
                };

                block.append_operation(helper.br(i, &[payload_val], location));
            }
        }
    }

    Ok(())
}

/// Generate MLIR operations for the `enum_snapshot_match` libfunc.
pub fn build_snapshot_match<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let type_info = registry.get_type(&info.param_signatures()[0].ty)?;

    // This libfunc's implementation is identical to `enum_match` aside from fetching the snapshotted enum's variants from the metadata:
    let variant_ids = metadata
        .get::<EnumSnapshotVariantsMeta>()
        .ok_or(Error::MissingMetadata)?
        .get_variants(&info.param_signatures()[0].ty)
        .expect("enum should always have variants")
        .clone();
    match variant_ids.len() {
        0 | 1 => {
            entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
        }
        _ => {
            let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
                context,
                helper,
                registry,
                metadata,
                &variant_ids,
            )?;

            let (stack_ptr, tag_val) = if type_info.is_memory_allocated(registry) {
                let stack_ptr = helper.init_block().alloca1(
                    context,
                    location,
                    type_info.build(
                        context,
                        helper,
                        registry,
                        metadata,
                        &info.param_signatures()[0].ty,
                    )?,
                    Some(layout.align()),
                )?;
                entry.store(
                    context,
                    location,
                    stack_ptr,
                    entry.argument(0)?.into(),
                    Some(layout.align()),
                );
                let tag_val =
                    entry.load(context, location, stack_ptr, tag_ty, Some(layout.align()))?;

                (Some(stack_ptr), tag_val)
            } else {
                let tag_val = entry
                    .append_operation(llvm::extract_value(
                        context,
                        entry.argument(0)?.into(),
                        DenseI64ArrayAttribute::new(context, &[0]),
                        tag_ty,
                        location,
                    ))
                    .result(0)?
                    .into();

                (None, tag_val)
            };

            let default_block = helper.append_block(Block::new(&[]));
            let variant_blocks = variant_tys
                .iter()
                .map(|_| helper.append_block(Block::new(&[])))
                .collect::<Vec<_>>();

            let case_values = (0..variant_tys.len())
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;

            entry.append_operation(cf::switch(
                context,
                &case_values,
                tag_val,
                tag_ty,
                (default_block, &[]),
                &variant_blocks
                    .iter()
                    .copied()
                    .map(|block| (block, [].as_slice()))
                    .collect::<Vec<_>>(),
                location,
            )?);

            // Default block.
            {
                let val = default_block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0).into(),
                        location,
                    ))
                    .result(0)?
                    .into();

                default_block.append_operation(cf::assert(
                    context,
                    val,
                    "Invalid enum tag.",
                    location,
                ));
                default_block.append_operation(llvm::unreachable(location));
            }

            // Enum variants.
            for (i, (block, (payload_ty, _))) in
                variant_blocks.into_iter().zip(variant_tys).enumerate()
            {
                let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);

                let payload_val = match stack_ptr {
                    Some(stack_ptr) => {
                        let val = block.load(
                            context,
                            location,
                            stack_ptr,
                            enum_ty,
                            Some(layout.align()),
                        )?;
                        block.extract_value(context, location, val, payload_ty, 1)?
                    }
                    None => {
                        // If the enum is not memory-allocated it means that:
                        //   - Either it's a C-style enum and all payloads have the same type.
                        //   - Or the enum only has a single non-memory-allocated variant.
                        if variant_ids.len() == 1 {
                            entry.argument(0)?.into()
                        } else {
                            assert!(registry.get_type(&variant_ids[i])?.is_zst(registry));
                            block
                                .append_operation(llvm::undef(payload_ty, location))
                                .result(0)?
                                .into()
                        }
                    }
                };

                block.append_operation(helper.br(i, &[payload_val], location));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output};
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use starknet_types_core::felt::Felt;

    lazy_static! {
        static ref ENUM_INIT: (String, Program) = load_cairo! {
            enum MySmallEnum {
                A: felt252,
            }

            enum MyEnum {
                A: felt252,
                B: u8,
                C: u16,
                D: u32,
                E: u64,
            }

            fn run_test() -> (MySmallEnum, MyEnum, MyEnum, MyEnum, MyEnum, MyEnum) {
                (
                    MySmallEnum::A(-1),
                    MyEnum::A(5678),
                    MyEnum::B(90),
                    MyEnum::C(9012),
                    MyEnum::D(34567890),
                    MyEnum::E(1234567890123456),
                )
            }
        };
        static ref ENUM_MATCH: (String, Program) = load_cairo! {
            enum MyEnum {
                A: felt252,
                B: u8,
                C: u16,
                D: u32,
                E: u64,
            }

            fn match_a() -> felt252 {
                let x = MyEnum::A(5);
                match x {
                    MyEnum::A(x) => x,
                    MyEnum::B(_) => 0,
                    MyEnum::C(_) => 1,
                    MyEnum::D(_) => 2,
                    MyEnum::E(_) => 3,
                }
            }

            fn match_b() -> u8 {
                let x = MyEnum::B(5_u8);
                match x {
                    MyEnum::A(_) => 0_u8,
                    MyEnum::B(x) => x,
                    MyEnum::C(_) => 1_u8,
                    MyEnum::D(_) => 2_u8,
                    MyEnum::E(_) => 3_u8,
                }
            }
        };
    }

    #[test]
    fn enum_init() {
        run_program_assert_output(
            &ENUM_INIT,
            "run_test",
            &[],
            jit_struct!(
                jit_enum!(0, Felt::from(-1).into()),
                jit_enum!(0, Felt::from(5678).into()),
                jit_enum!(1, 90u8.into()),
                jit_enum!(2, 9012u16.into()),
                jit_enum!(3, 34567890u32.into()),
                jit_enum!(4, 1234567890123456u64.into()),
            ),
        );
    }

    #[test]
    fn enum_match() {
        run_program_assert_output(&ENUM_MATCH, "match_a", &[], Felt::from(5).into());
        run_program_assert_output(&ENUM_MATCH, "match_b", &[], 5u8.into());
    }
}
