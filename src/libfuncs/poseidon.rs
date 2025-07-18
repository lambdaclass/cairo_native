//! # Poseidon hashing libfuncs
//!

use super::LibfuncHelper;
use crate::{
    error::{panic::ToNativeAssertError, Result},
    execution_result::POSEIDON_BUILTIN_SIZE,
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::{get_integer_layout, BlockExt, ProgramRegistryExt},
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
    dialect::ods,
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
        .to_native_assert_error("runtime library should be available")?;

    // The sierra-to-casm compiler uses the poseidon builtin a total of 1 time.
    // https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/poseidon.rs?plain=1#L19
    let poseidon_builtin = super::increment_builtin_counter_by(
        context,
        entry,
        location,
        entry.arg(0)?,
        POSEIDON_BUILTIN_SIZE,
    )?;

    let felt252_ty =
        registry.build_type(context, helper, metadata, &info.param_signatures()[1].ty)?;

    let i256_ty = IntegerType::new(context, 256).into();
    let layout_i256 = get_integer_layout(256);

    let op0 = entry.arg(1)?;
    let op1 = entry.arg(2)?;
    let op2 = entry.arg(3)?;

    // We must extend to i256 because bswap must be an even number of bytes.

    let op0_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;
    let op1_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;
    let op2_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;

    let op0_i256 =
        entry.append_op_result(ods::arith::extui(context, i256_ty, op0, location).into())?;

    let op1_i256 =
        entry.append_op_result(ods::arith::extui(context, i256_ty, op1, location).into())?;
    let op2_i256 =
        entry.append_op_result(ods::arith::extui(context, i256_ty, op2, location).into())?;

    entry.store(context, location, op0_ptr, op0_i256)?;
    entry.store(context, location, op1_ptr, op1_i256)?;
    entry.store(context, location, op2_ptr, op2_i256)?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .to_native_assert_error("runtime library should be available")?;

    runtime_bindings
        .libfunc_hades_permutation(context, helper, entry, op0_ptr, op1_ptr, op2_ptr, location)?;

    let op0_i256 = entry.load(context, location, op0_ptr, i256_ty)?;
    let op1_i256 = entry.load(context, location, op1_ptr, i256_ty)?;
    let op2_i256 = entry.load(context, location, op2_ptr, i256_ty)?;

    let op0 = entry.trunci(op0_i256, felt252_ty, location)?;
    let op1 = entry.trunci(op1_i256, felt252_ty, location)?;
    let op2 = entry.trunci(op2_i256, felt252_ty, location)?;

    helper.br(entry, 0, &[poseidon_builtin, op0, op1, op2], location)
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
