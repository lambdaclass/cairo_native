//! # Poseidon hashing libfuncs
//!

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    types::TypeBuilder,
    utils::get_integer_layout,
};
use cairo_lang_sierra::{
    extensions::{
        lib_func::SignatureOnlyConcreteLibfunc, poseidon::PoseidonConcreteLibfunc, ConcreteLibfunc,
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith,
        llvm::{self, LoadStoreOptions},
        ods,
    },
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        Identifier, Location,
    },
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
    selector: &PoseidonConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        PoseidonConcreteLibfunc::HadesPermutation(info) => {
            build_poseidon(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_poseidon<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'ctx Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, '_>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let felt252_ty = registry
        .get_type(&info.param_signatures()[1].ty)?
        .build(context, helper, registry, metadata)?;

    let i256_ty = IntegerType::new(context, 256).into();
    let layout_i256 = get_integer_layout(256);

    let poseidon_builtin = entry.argument(0)?.into();
    let op0 = entry.argument(1)?.into();
    let op1 = entry.argument(2)?.into();
    let op2 = entry.argument(3)?.into();

    // We must extend to i256 because bswap must be an even number of bytes.

    let const_1 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let op0_ptr = entry
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        layout_i256.align().try_into()?,
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[const_1])
                .add_results(&[llvm::r#type::pointer(i256_ty, 0)])
                .build(),
        )
        .result(0)?
        .into();
    let op1_ptr = entry
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        layout_i256.align().try_into()?,
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[const_1])
                .add_results(&[llvm::r#type::pointer(i256_ty, 0)])
                .build(),
        )
        .result(0)?
        .into();
    let op2_ptr = entry
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        layout_i256.align().try_into()?,
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[const_1])
                .add_results(&[llvm::r#type::pointer(i256_ty, 0)])
                .build(),
        )
        .result(0)?
        .into();

    let op0_i256 = entry
        .append_operation(ods::arith::extui(i256_ty, op0, location).into())
        .result(0)?
        .into();
    let op1_i256 = entry
        .append_operation(ods::arith::extui(i256_ty, op1, location).into())
        .result(0)?
        .into();
    let op2_i256 = entry
        .append_operation(ods::arith::extui(i256_ty, op2, location).into())
        .result(0)?
        .into();

    let op0_be = entry
        .append_operation(llvm::intr_bswap(op0_i256, i256_ty, location))
        .result(0)?
        .into();
    let op1_be = entry
        .append_operation(llvm::intr_bswap(op1_i256, i256_ty, location))
        .result(0)?
        .into();
    let op2_be = entry
        .append_operation(llvm::intr_bswap(op2_i256, i256_ty, location))
        .result(0)?
        .into();

    entry.append_operation(llvm::store(
        context,
        op0_be,
        op0_ptr,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            layout_i256.align().try_into()?,
            IntegerType::new(context, 64).into(),
        ))),
    ));
    entry.append_operation(llvm::store(
        context,
        op1_be,
        op1_ptr,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            layout_i256.align().try_into()?,
            IntegerType::new(context, 64).into(),
        ))),
    ));
    entry.append_operation(llvm::store(
        context,
        op2_be,
        op2_ptr,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            layout_i256.align().try_into()?,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    runtime_bindings
        .libfunc_poseidon(context, helper, entry, op0_ptr, op1_ptr, op2_ptr, location)?;

    let op0_be = entry
        .append_operation(llvm::load(
            context,
            op0_ptr,
            i256_ty,
            location,
            LoadStoreOptions::default().align(Some(IntegerAttribute::new(
                layout_i256.align().try_into()?,
                IntegerType::new(context, 64).into(),
            ))),
        ))
        .result(0)?
        .into();
    let op1_be = entry
        .append_operation(llvm::load(
            context,
            op1_ptr,
            i256_ty,
            location,
            LoadStoreOptions::default().align(Some(IntegerAttribute::new(
                layout_i256.align().try_into()?,
                IntegerType::new(context, 64).into(),
            ))),
        ))
        .result(0)?
        .into();
    let op2_be = entry
        .append_operation(llvm::load(
            context,
            op2_ptr,
            i256_ty,
            location,
            LoadStoreOptions::default().align(Some(IntegerAttribute::new(
                layout_i256.align().try_into()?,
                IntegerType::new(context, 64).into(),
            ))),
        ))
        .result(0)?
        .into();

    let op0_i256 = entry
        .append_operation(llvm::intr_bswap(op0_be, i256_ty, location))
        .result(0)?
        .into();
    let op1_i256 = entry
        .append_operation(llvm::intr_bswap(op1_be, i256_ty, location))
        .result(0)?
        .into();
    let op2_i256 = entry
        .append_operation(llvm::intr_bswap(op2_be, i256_ty, location))
        .result(0)?
        .into();

    let op0 = entry
        .append_operation(arith::trunci(op0_i256, felt252_ty, location))
        .result(0)?
        .into();
    let op1 = entry
        .append_operation(arith::trunci(op1_i256, felt252_ty, location))
        .result(0)?
        .into();
    let op2 = entry
        .append_operation(arith::trunci(op2_i256, felt252_ty, location))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[poseidon_builtin, op0, op1, op2], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{felt, load_cairo, run_program};
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn run_poseidon() {
        let program = load_cairo!(
            use core::poseidon::hades_permutation;

            fn run_test(a: felt252, b: felt252, c: felt252) -> (felt252, felt252, felt252) {
                hades_permutation(a, b, c)
            }
        );

        let result = run_program(
            &program,
            "run_test",
            json!([(), felt("2"), felt("4"), felt("4")]),
        );
        assert_eq!(
            result,
            json!([
                null,
                [
                    [
                        1451956842u64,
                        4009088405u64,
                        2701936287u64,
                        2751428444u64,
                        149168921,
                        2640934343u64,
                        3983554576u64,
                        60350433,
                    ],
                    [
                        3195313629u64,
                        2105130109,
                        1451822522,
                        671253646,
                        2960065512u64,
                        1827556108,
                        776192228,
                        87841256,
                    ],
                    [
                        2919724085u64,
                        3626733544u64,
                        1688675005,
                        3047144720u64,
                        201114190,
                        1215855961,
                        2130126335,
                        88328284,
                    ],
                ]
            ])
        );
    }
}
