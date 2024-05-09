//! # Pedersen hashing libfuncs
//!

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt,
    error::Result,
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::{get_integer_layout, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        pedersen::PedersenConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith,
        llvm::{self, LoadStoreOptions},
        ods,
    },
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
    selector: &PedersenConcreteLibfunc,
) -> Result<()> {
    match selector {
        PedersenConcreteLibfunc::PedersenHash(info) => {
            build_pedersen(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_pedersen<'ctx>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'ctx Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, '_>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let pedersen_builtin =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[1].ty,
    )?;

    let i256_ty = IntegerType::new(context, 256).into();
    let layout_i256 = get_integer_layout(256);

    let lhs = entry.argument(1)?.into();
    let rhs = entry.argument(2)?.into();

    // We must extend to i256 because bswap must be an even number of bytes.

    let lhs_ptr =
        helper
            .init_block()
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
    let rhs_ptr =
        helper
            .init_block()
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
    let dst_ptr =
        helper
            .init_block()
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;

    let lhs_i256 = entry.append_op_result(arith::extui(lhs, i256_ty, location))?;
    let rhs_i256 = entry.append_op_result(arith::extui(rhs, i256_ty, location))?;

    let lhs_be =
        entry.append_op_result(ods::llvm::intr_bswap(context, lhs_i256, location).into())?;

    let rhs_be =
        entry.append_op_result(ods::llvm::intr_bswap(context, rhs_i256, location).into())?;

    entry.append_operation(llvm::store(
        context,
        lhs_be,
        lhs_ptr,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            IntegerType::new(context, 64).into(),
            layout_i256.align().try_into()?,
        ))),
    ));
    entry.append_operation(llvm::store(
        context,
        rhs_be,
        rhs_ptr,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            IntegerType::new(context, 64).into(),
            layout_i256.align().try_into()?,
        ))),
    ));

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    runtime_bindings
        .libfunc_pedersen(context, helper, entry, dst_ptr, lhs_ptr, rhs_ptr, location)?;

    let result_be = entry.append_op_result(llvm::load(
        context,
        dst_ptr,
        i256_ty,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            IntegerType::new(context, 64).into(),
            layout_i256.align().try_into()?,
        ))),
    ))?;

    let op = entry.append_op_result(ods::llvm::intr_bswap(context, result_be, location).into())?;

    let result = entry.append_op_result(arith::trunci(op, felt252_ty, location))?;

    entry.append_operation(helper.br(0, &[pedersen_builtin, result], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{load_cairo, run_program_assert_output};

    use starknet_types_core::felt::Felt;

    #[test]
    fn run_pedersen() {
        let program = load_cairo!(
            use core::pedersen::pedersen;

            fn run_test(a: felt252, b: felt252) -> felt252 {
                pedersen(a, b)
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[Felt::from(2).into(), Felt::from(4).into()],
            Felt::from_dec_str(
                "2178161520066714737684323463974044933282313051386084149915030950231093462467",
            )
            .unwrap()
            .into(),
        );
    }
}
