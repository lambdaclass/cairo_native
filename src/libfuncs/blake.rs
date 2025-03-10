use cairo_lang_sierra::{
    extensions::{
        blake::BlakeConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, BlockLike, Location},
    Context,
};

use crate::{
    error::{panic::ToNativeAssertError, Result},
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::BlockExt,
};

use super::LibfuncHelper;

pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &BlakeConcreteLibfunc,
) -> Result<()> {
    match selector {
        BlakeConcreteLibfunc::Blake2sCompress(info) => build_blake_operation(
            context, registry, entry, location, helper, metadata, info, false,
        ),
        BlakeConcreteLibfunc::Blake2sFinalize(info) => build_blake_operation(
            context, registry, entry, location, helper, metadata, info, true,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_blake_operation<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
    finalize: bool,
) -> Result<()> {
    let state_ptr = entry.arg(0)?;
    let bytes_count = entry.arg(1)?;
    let message = entry.arg(2)?;
    let k_finalize = entry.const_int(context, location, finalize, 1)?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .to_native_assert_error("runtime library should be available")?;

    runtime_bindings.libfunc_blake_compress(
        context,
        helper,
        entry,
        state_ptr,
        message,
        bytes_count,
        k_finalize,
        location,
    )?;

    entry.append_operation(helper.br(0, &[state_ptr], location));

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        utils::test::{jit_struct, load_cairo, run_program},
        Value,
    };

    #[test]
    fn test_blake() {
        let program = load_cairo!(
            use core::blake::{blake2s_compress, blake2s_finalize};

            fn run_test() -> [u32; 8] nopanic {
                let state = BoxTrait::new([0_u32; 8]);
                let msg = BoxTrait::new([0_u32; 16]);
                let byte_count = 64_u32;

                let _res = blake2s_compress(state, byte_count, msg).unbox();

                blake2s_finalize(state, byte_count, msg).unbox()
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_struct!(
                Value::Uint32(128291589),
                Value::Uint32(1454945417),
                Value::Uint32(3191583614),
                Value::Uint32(1491889056),
                Value::Uint32(794023379),
                Value::Uint32(651000200),
                Value::Uint32(3725903680),
                Value::Uint32(1044330286),
            )
        );
    }
}
