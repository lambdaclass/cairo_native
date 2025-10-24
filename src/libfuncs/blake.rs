use cairo_lang_sierra::{
    extensions::{
        blake::BlakeConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    helpers::{ArithBlockExt, BuiltinBlockExt},
    ir::{Block, Location},
    Context,
};

use crate::{
    error::{panic::ToNativeAssertError, Result},
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
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

/// Performs a blake2s compression.
///
/// `bytes_count`: total amount of bytes hashed after hashing the message.
/// `finalize`: wether the libfunc call is a finalize or not.
/// ```cairo
/// pub extern fn blake2s_compress(
///     state: Blake2sState, byte_count: u32, msg: Blake2sInput,
/// ) -> Blake2sState nopanic;
/// ```
///
/// Similar to `blake2s_compress`, but it marks the end of the compression
/// ```cairo
/// pub extern fn blake2s_finalize(
///     state: Blake2sState, byte_count: u32, msg: Blake2sInput,
/// ) -> Blake2sState nopanic;
/// ```
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
    let k_finalize = entry.const_int(context, location, finalize as u8, 1)?;

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

    helper.br(entry, 0, &[state_ptr], location)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        utils::test::{jit_struct, load_cairo, run_program},
        Value,
    };

    // This test is taken from the Blake2s-256 implementeation RFC-7693, Appendix B.
    // https://www.rfc-editor.org/rfc/rfc7693#appendix-B.
    #[test]
    fn test_blake_3_bytes_compress() {
        let program = load_cairo!(
            use core::blake::{blake2s_compress, blake2s_finalize};

            fn run_test() -> [u32; 8] nopanic {
                let initial_state: Box<[u32; 8]> = BoxTrait::new([
                    0x6B08E647, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
                ]);
                // This number represents the bytes for "abc" string.
                let abc_bytes = 0x00636261;
                let msg: Box<[u32; 16]>  = BoxTrait::new([abc_bytes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

                blake2s_finalize(initial_state, 3, msg).unbox()
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_struct!(
                Value::Uint32(0x8C5E8C50),
                Value::Uint32(0xE2147C32),
                Value::Uint32(0xA32BA7E1),
                Value::Uint32(0x2F45EB4E),
                Value::Uint32(0x208B4537),
                Value::Uint32(0x293AD69E),
                Value::Uint32(0x4C9B994D),
                Value::Uint32(0x82596786),
            )
        );
    }
}
