//! # Enum-related libfuncs
//!
//! Check out [the enum type](crate::types::enum) for more information on enum layouts.

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
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
        llvm::{self, LoadStoreOptions},
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location,
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
        EnumConcreteLibfunc::SnapshotMatch(_) => todo!(),
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
    let (layout, (tag_ty, tag_layout), variant_tys) = crate::types::r#enum::get_type_for_variants(
        context,
        helper,
        registry,
        metadata,
        registry
            .get_type(&info.branch_signatures()[0].vars[0].ty)?
            .variants()
            .unwrap(),
    )?;

    let enum_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)?
        .build(context, helper, registry, metadata)?;

    let op0 = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(info.index.try_into()?, tag_ty).into(),
        location,
    ));

    let concrete_enum_ty = llvm::r#type::r#struct(
        context,
        &[
            tag_ty,
            llvm::r#type::array(
                IntegerType::new(context, 8).into(),
                tag_layout
                    .padding_needed_for(variant_tys[info.index].1.align())
                    .try_into()?,
            ),
            variant_tys[info.index].0,
        ],
        false,
    );

    let op1 = entry.append_operation(llvm::undef(concrete_enum_ty, location));
    let op2 = entry.append_operation(llvm::insert_value(
        context,
        op1.result(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[0]),
        op0.result(0)?.into(),
        location,
    ));
    let op3 = entry.append_operation(llvm::insert_value(
        context,
        op2.result(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[2]),
        entry.argument(0)?.into(),
        location,
    ));

    let op4 = helper.init_block().append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
        location,
    ));
    let op5 = helper.init_block().append_operation(
        OperationBuilder::new("llvm.alloca", location)
            .add_attributes(&[(
                Identifier::new(context, "alignment"),
                IntegerAttribute::new(
                    layout.align().try_into()?,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
            )])
            .add_operands(&[op4.result(0)?.into()])
            .add_results(&[llvm::r#type::pointer(enum_ty, 0)])
            .build(),
    );

    let op6 = entry.append_operation(
        OperationBuilder::new("llvm.bitcast", location)
            .add_operands(&[op5.result(0)?.into()])
            .add_results(&[llvm::r#type::pointer(concrete_enum_ty, 0)])
            .build(),
    );
    entry.append_operation(llvm::store(
        context,
        op3.result(0)?.into(),
        op6.result(0)?.into(),
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            layout.align().try_into()?,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    let op7 = entry.append_operation(llvm::load(
        context,
        op5.result(0)?.into(),
        enum_ty,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            layout.align().try_into()?,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    entry.append_operation(helper.br(0, &[op7.result(0)?.into()], location));

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
    let (layout, (tag_ty, tag_layout), variant_tys) = crate::types::r#enum::get_type_for_variants(
        context,
        helper,
        registry,
        metadata,
        registry
            .get_type(&info.param_signatures()[0].ty)?
            .variants()
            .unwrap(),
    )?;

    let enum_ty = registry
        .get_type(&info.param_signatures()[0].ty)?
        .build(context, helper, registry, metadata)?;

    let op0 = helper.init_block().append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
        location,
    ));
    let op1 = helper.init_block().append_operation(
        OperationBuilder::new("llvm.alloca", location)
            .add_attributes(&[(
                Identifier::new(context, "alignment"),
                IntegerAttribute::new(
                    layout.align().try_into()?,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
            )])
            .add_operands(&[op0.result(0)?.into()])
            .add_results(&[llvm::r#type::pointer(enum_ty, 0)])
            .build(),
    );
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        op1.result(0)?.into(),
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            layout.align().try_into()?,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    let default_block = helper.append_block(Block::new(&[]));
    let variant_blocks = variant_tys
        .iter()
        .map(|_| helper.append_block(Block::new(&[])))
        .collect::<Vec<_>>();

    let op2 = entry.append_operation(llvm::extract_value(
        context,
        entry.argument(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[0]),
        tag_ty,
        location,
    ));

    let case_values: Vec<i64> = (0..variant_tys.len())
        .map(|n| i64::try_from(n))
        .collect::<std::result::Result<Vec<_>, TryFromIntError>>()?;

    entry.append_operation(cf::switch(
        context,
        &case_values,
        op2.result(0)?.into(),
        tag_ty,
        (default_block, &[]),
        &variant_blocks
            .iter()
            .copied()
            .map(|block| (block, [].as_slice()))
            .collect::<Vec<_>>(),
        location,
    )?);

    {
        let op3 = default_block.append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
            location,
        ));

        default_block.append_operation(cf::assert(
            context,
            op3.result(0)?.into(),
            "Invalid enum tag.",
            location,
        ));
        default_block.append_operation(OperationBuilder::new("llvm.unreachable", location).build());
    }

    for (i, (block, (payload_ty, payload_layout))) in
        variant_blocks.into_iter().zip(variant_tys).enumerate()
    {
        let concrete_enum_ty = llvm::r#type::r#struct(
            context,
            &[
                tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    tag_layout
                        .padding_needed_for(payload_layout.align())
                        .try_into()?,
                ),
                payload_ty,
            ],
            false,
        );

        let op3 = block.append_operation(
            OperationBuilder::new("llvm.bitcast", location)
                .add_operands(&[op1.result(0)?.into()])
                .add_results(&[llvm::r#type::pointer(concrete_enum_ty, 0)])
                .build(),
        );
        let op4 = block.append_operation(llvm::load(
            context,
            op3.result(0)?.into(),
            concrete_enum_ty,
            location,
            LoadStoreOptions::default(),
        ));
        let op5 = block.append_operation(llvm::extract_value(
            context,
            op4.result(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[2]),
            payload_ty,
            location,
        ));

        block.append_operation(helper.br(i, &[op5.result(0)?.into()], location));
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{felt, load_cairo, run_program};
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use serde_json::json;

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
        let r = || run_program(&ENUM_INIT, "run_test", json!([]));

        assert_eq!(
            r(),
            json!([[
                [0, felt("-1")],
                [0, felt("5678")],
                [1, 90u8],
                [2, 9012u16],
                [3, 34567890u32],
                [4, 1234567890123456u64],
            ]])
        );
    }

    #[test]
    fn enum_match() {
        let result_a = run_program(&ENUM_MATCH, "match_a", json!([]));
        let result_b = run_program(&ENUM_MATCH, "match_b", json!([]));

        assert_eq!(result_a, json!([felt("5")]));
        assert_eq!(result_b, json!([5]));
    }
}
