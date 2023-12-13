//! # Enum-related libfuncs
//!
//! Check out [the enum type](crate::types::enum) for more information on enum layouts.

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{enum_snapshot_variants::EnumSnapshotVariantsMeta, MetadataStorage},
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        enm::{EnumConcreteLibfunc, EnumInitConcreteLibfunc},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
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
use std::num::TryFromIntError;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &EnumConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
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
    }
}

/// Generate MLIR operations for the `enum_init` libfunc.
pub fn build_init<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &EnumInitConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
        context,
        helper,
        registry,
        metadata,
        registry
            .get_type(&info.branch_signatures()[0].vars[0].ty)?
            .variants()
            .unwrap(),
    )?;

    let ptr_ty = llvm::r#type::opaque_pointer(context);
    let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, variant_tys[info.index].0], false);

    let k1 = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    // Allocating only the space necessary for the current variant. This shouldn't cause any
    // problems because the data won't be changed in place.
    let stack_ptr = helper
        .init_block()
        .append_operation(llvm::alloca(
            context,
            k1,
            ptr_ty,
            location,
            AllocaOptions::new()
                .align(Some(IntegerAttribute::new(
                    layout.align() as i64,
                    IntegerType::new(context, 64).into(),
                )))
                .elem_type(Some(TypeAttribute::new(enum_ty))),
        ))
        .result(0)?
        .into();

    let tag_val = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(info.index as i64, tag_ty).into(),
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
    let val = entry
        .append_operation(llvm::insert_value(
            context,
            val,
            DenseI64ArrayAttribute::new(context, &[1]),
            entry.argument(0)?.into(),
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(llvm::store(
        context,
        val,
        stack_ptr,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    entry.append_operation(helper.br(0, &[stack_ptr], location));
    Ok(())
}

/// Generate MLIR operations for the `enum_match` libfunc.
pub fn build_match<'ctx, 'this, TType, TLibfunc>(
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
    let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
        context,
        helper,
        registry,
        metadata,
        registry
            .get_type(&info.param_signatures()[0].ty)?
            .variants()
            .unwrap(),
    )?;

    let tag_val = entry
        .append_operation(llvm::load(
            context,
            entry.argument(0)?.into(),
            tag_ty,
            location,
            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                layout.align() as i64,
                IntegerType::new(context, 64).into(),
            ))),
        ))
        .result(0)?
        .into();

    let default_block = helper.append_block(Block::new(&[]));
    let variant_blocks = variant_tys
        .iter()
        .map(|_| helper.append_block(Block::new(&[])))
        .collect::<Vec<_>>();

    let case_values = (0..variant_tys.len())
        .map(i64::try_from)
        .collect::<std::result::Result<Vec<_>, TryFromIntError>>()?;

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
                IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
                location,
            ))
            .result(0)?
            .into();

        default_block.append_operation(cf::assert(context, val, "Invalid enum tag.", location));
        default_block.append_operation(llvm::unreachable(location));
    }

    // Enum variants.
    for (i, (block, (payload_ty, _))) in variant_blocks.into_iter().zip(variant_tys).enumerate() {
        let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);

        let val = block
            .append_operation(llvm::load(
                context,
                entry.argument(0)?.into(),
                enum_ty,
                location,
                LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                    layout.align() as i64,
                    IntegerType::new(context, 64).into(),
                ))),
            ))
            .result(0)?
            .into();

        let payload_val = block
            .append_operation(llvm::extract_value(
                context,
                val,
                DenseI64ArrayAttribute::new(context, &[1]),
                payload_ty,
                location,
            ))
            .result(0)?
            .into();

        block.append_operation(helper.br(i, &[payload_val], location));
    }

    Ok(())
}

/// Generate MLIR operations for the `enum_snapshot_match` libfunc.
pub fn build_snapshot_match<'ctx, 'this, TType, TLibfunc>(
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
    // This libfunc's implementation is identical to `enum_match` aside from fetching the snapshotted enum's variants from the metadata:
    let variants = metadata
        .get::<EnumSnapshotVariantsMeta>()
        .unwrap()
        .get_variants(&info.param_signatures()[0].ty)
        .unwrap()
        .clone();
    let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
        context, helper, registry, metadata, &variants,
    )?;

    let tag_val = entry
        .append_operation(llvm::load(
            context,
            entry.argument(0)?.into(),
            tag_ty,
            location,
            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                layout.align() as i64,
                IntegerType::new(context, 64).into(),
            ))),
        ))
        .result(0)?
        .into();

    let default_block = helper.append_block(Block::new(&[]));
    let variant_blocks = variant_tys
        .iter()
        .map(|_| helper.append_block(Block::new(&[])))
        .collect::<Vec<_>>();

    let case_values = (0..variant_tys.len())
        .map(i64::try_from)
        .collect::<std::result::Result<Vec<_>, TryFromIntError>>()?;

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
                IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
                location,
            ))
            .result(0)?
            .into();

        default_block.append_operation(cf::assert(context, val, "Invalid enum tag.", location));
        default_block.append_operation(llvm::unreachable(location));
    }

    // Enum variants.
    for (i, (block, (payload_ty, _))) in variant_blocks.into_iter().zip(variant_tys).enumerate() {
        let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);

        let val = block
            .append_operation(llvm::load(
                context,
                entry.argument(0)?.into(),
                enum_ty,
                location,
                LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                    layout.align() as i64,
                    IntegerType::new(context, 64).into(),
                ))),
            ))
            .result(0)?
            .into();

        let payload_val = block
            .append_operation(llvm::extract_value(
                context,
                val,
                DenseI64ArrayAttribute::new(context, &[1]),
                payload_ty,
                location,
            ))
            .result(0)?
            .into();

        block.append_operation(helper.br(i, &[payload_val], location));
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output};
    use cairo_felt::Felt252;
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;

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
            &[jit_struct!(
                jit_enum!(0, Felt252::new(-1).into()),
                jit_enum!(0, Felt252::new(5678).into()),
                jit_enum!(1, 90u8.into()),
                jit_enum!(2, 9012u16.into()),
                jit_enum!(3, 34567890u32.into()),
                jit_enum!(4, 1234567890123456u64.into()),
            )],
        );
    }

    #[test]
    fn enum_match() {
        run_program_assert_output(&ENUM_MATCH, "match_a", &[], &[Felt252::new(5).into()]);

        run_program_assert_output(&ENUM_MATCH, "match_b", &[], &[5u8.into()]);
    }
}
