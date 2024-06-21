//! # Poseidon hashing libfuncs
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
        poseidon::PoseidonConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm, ods},
    ir::{r#type::IntegerType, Block, Location},
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
    selector: &PoseidonConcreteLibfunc,
) -> Result<()> {
    match selector {
        PoseidonConcreteLibfunc::HadesPermutation(info) => {
            build_hades_permutation(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_hades_permutation<'ctx>(
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

    let poseidon_builtin =
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

    let op0 = entry.argument(1)?.into();
    let op1 = entry.argument(2)?.into();
    let op2 = entry.argument(3)?.into();

    // We must extend to i256 because bswap must be an even number of bytes.

    let op0_ptr =
        helper
            .init_block()
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
    let op1_ptr =
        helper
            .init_block()
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;
    let op2_ptr =
        helper
            .init_block()
            .alloca1(context, location, i256_ty, Some(layout_i256.align()))?;

    let op0_i256 =
        entry.append_op_result(ods::arith::extui(context, i256_ty, op0, location).into())?;

    let op1_i256 =
        entry.append_op_result(ods::arith::extui(context, i256_ty, op1, location).into())?;
    let op2_i256 =
        entry.append_op_result(ods::arith::extui(context, i256_ty, op2, location).into())?;

    let op0_be = entry.append_op_result(llvm::intr_bswap(op0_i256, i256_ty, location))?;
    let op1_be = entry.append_op_result(llvm::intr_bswap(op1_i256, i256_ty, location))?;
    let op2_be = entry.append_op_result(llvm::intr_bswap(op2_i256, i256_ty, location))?;

    entry.store(
        context,
        location,
        op0_ptr,
        op0_be,
        Some(layout_i256.align()),
    )?;
    entry.store(
        context,
        location,
        op1_ptr,
        op1_be,
        Some(layout_i256.align()),
    )?;
    entry.store(
        context,
        location,
        op2_ptr,
        op2_be,
        Some(layout_i256.align()),
    )?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    runtime_bindings
        .libfunc_hades_permutation(context, helper, entry, op0_ptr, op1_ptr, op2_ptr, location)?;

    let op0_be = entry.load(
        context,
        location,
        op0_ptr,
        i256_ty,
        Some(layout_i256.align()),
    )?;
    let op1_be = entry.load(
        context,
        location,
        op1_ptr,
        i256_ty,
        Some(layout_i256.align()),
    )?;
    let op2_be = entry.load(
        context,
        location,
        op2_ptr,
        i256_ty,
        Some(layout_i256.align()),
    )?;

    let op0_i256 = entry.append_op_result(llvm::intr_bswap(op0_be, i256_ty, location))?;
    let op1_i256 = entry.append_op_result(llvm::intr_bswap(op1_be, i256_ty, location))?;
    let op2_i256 = entry.append_op_result(llvm::intr_bswap(op2_be, i256_ty, location))?;

    let op0 = entry.append_op_result(arith::trunci(op0_i256, felt252_ty, location))?;
    let op1 = entry.append_op_result(arith::trunci(op1_i256, felt252_ty, location))?;
    let op2 = entry.append_op_result(arith::trunci(op2_i256, felt252_ty, location))?;

    entry.append_operation(helper.br(0, &[poseidon_builtin, op0, op1, op2], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_struct, load_cairo, run_program_assert_output};

    use starknet_types_core::felt::Felt;

    #[test]
    fn run_hades_permutation() {
        let program = load_cairo!(
            use core::poseidon::hades_permutation;

            fn run_test(a: felt252, b: felt252, c: felt252) -> (felt252, felt252, felt252) {
                hades_permutation(a, b, c)
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[
                Felt::from(2).into(),
                Felt::from(4).into(),
                Felt::from(4).into(),
            ],
            jit_struct!(
                Felt::from_dec_str(
                    "1627044480024625333712068603977073585655327747658231320998869768849911913066"
                )
                .unwrap()
                .into(),
                Felt::from_dec_str(
                    "2368195581807763724810563135784547417602556730014791322540110420941926079965"
                )
                .unwrap()
                .into(),
                Felt::from_dec_str(
                    "2381325839211954898363395375151559373051496038592329842107874845056395867189"
                )
                .unwrap()
                .into(),
            ),
        );
    }
}
