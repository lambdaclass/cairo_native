//! # Bitwise libfuncs

use super::LibfuncHelper;
use crate::{block_ext::BlockExt, error::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith,
    ir::{Block, Location},
    Context,
};

/// Generate MLIR operations for the `bitwise` libfunc.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let bitwise =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let lhs = entry.argument(1)?.into();
    let rhs = entry.argument(2)?.into();

    let logical_and = entry.append_op_result(arith::andi(lhs, rhs, location))?;
    let logical_xor = entry.append_op_result(arith::xori(lhs, rhs, location))?;
    let logical_or = entry.append_op_result(arith::ori(lhs, rhs, location))?;

    entry.append_operation(helper.br(
        0,
        &[bitwise, logical_and, logical_or, logical_xor],
        location,
    ));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_struct, load_cairo, run_program};
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;

    lazy_static! {
        static ref BITWISE: (String, Program) = load_cairo! {
            use core::integer::bitwise;

            fn run_test(lhs: u128, rhs: u128) -> (u128, u128, u128) {
                bitwise(lhs, rhs)
            }
        };
    }

    #[test]
    fn bitwise() {
        let r = |lhs, rhs| run_program(&BITWISE, "run_test", &[lhs, rhs]).return_value;

        assert_eq!(
            r(
                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
            ),
            jit_struct!(
                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into()
            )
        );
        assert_eq!(
            r(
                0x00000000_00000000_00000000_00000000u128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
            ),
            jit_struct!(
                0x00000000_00000000_00000000_00000000u128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into()
            )
        );
        assert_eq!(
            r(
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
            ),
            jit_struct!(
                0x00000000_00000000_00000000_00000000u128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into()
            )
        );
        assert_eq!(
            r(
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
            ),
            jit_struct!(
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
            )
        );
    }
}
