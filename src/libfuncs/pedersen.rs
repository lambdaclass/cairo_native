//! # Pedersen hashing libfuncs
//!

use super::LibfuncHelper;
use crate::{
    error::{panic::ToNativeAssertError, Result},
    execution_result::PEDERSEN_BUILTIN_SIZE,
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::{get_integer_layout, BlockExt, ProgramRegistryExt},
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
        .to_native_assert_error("runtime library should be available")?;

    // The sierra-to-casm compiler uses the pedersen builtin a total of 1 time.
    // https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/pedersen.rs?plain=1#L23
    let pedersen_builtin = super::increment_builtin_counter_by(
        context,
        entry,
        location,
        entry.arg(0)?,
        PEDERSEN_BUILTIN_SIZE,
    )?;

    let felt252_ty =
        registry.build_type(context, helper, metadata, &info.param_signatures()[1].ty)?;

    let i256_ty = IntegerType::new(context, 256).into();
    let layout_i256 = get_integer_layout(256);

    let lhs = entry.arg(1)?;
    let rhs = entry.arg(2)?;

    // We must extend to i256 because bswap must be an even number of bytes.

    let lhs_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;
    let rhs_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;
    let dst_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;

    let lhs_i256 = entry.extui(lhs, i256_ty, location)?;
    let rhs_i256 = entry.extui(rhs, i256_ty, location)?;

    entry.store(context, location, lhs_ptr, lhs_i256)?;
    entry.store(context, location, rhs_ptr, rhs_i256)?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .to_native_assert_error("runtime library should be available")?;

    runtime_bindings
        .libfunc_pedersen(context, helper, entry, dst_ptr, lhs_ptr, rhs_ptr, location)?;

    let result = entry.load(context, location, dst_ptr, i256_ty)?;
    let result = entry.trunci(result, felt252_ty, location)?;

    helper.br(entry, 0, &[pedersen_builtin, result], location)
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
