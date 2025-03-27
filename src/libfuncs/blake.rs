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

// Used when initializing the v vector
pub const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const SIGMA: [[u32; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

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
