//! # Bitwise libfuncs
//!
//! TODO

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
    extensions::{lib_func::SignatureOnlyConcreteLibfunc, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith,
    ir::{Block, Location},
    Context,
};

/// Generate MLIR operations for the `bitwise` libfunc.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let lhs = entry.argument(1)?.into();
    let rhs = entry.argument(2)?.into();

    let logical_and = entry
        .append_operation(arith::andi(lhs, rhs, location))
        .result(0)?
        .into();
    let logical_xor = entry
        .append_operation(arith::xori(lhs, rhs, location))
        .result(0)?
        .into();
    let logical_or = entry
        .append_operation(arith::ori(lhs, rhs, location))
        .result(0)?
        .into();

    entry.append_operation(helper.br(
        0,
        &[
            entry.argument(0)?.into(),
            logical_and,
            logical_xor,
            logical_or,
        ],
        location,
    ));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{load_cairo, run_program};
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use serde_json::json;

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
        let r = |lhs, rhs| run_program(&BITWISE, "run_test", json!([(), lhs, rhs]));

        assert_eq!(
            r(
                0x00000000_00000000_00000000_00000000u128,
                0x00000000_00000000_00000000_00000000u128,
            ),
            json!([
                (),
                [
                    0x00000000_00000000_00000000_00000000u128,
                    0x00000000_00000000_00000000_00000000u128,
                    0x00000000_00000000_00000000_00000000u128,
                ]
            ])
        );
        assert_eq!(
            r(
                0x00000000_00000000_00000000_00000000u128,
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128,
            ),
            json!([
                (),
                [
                    0x00000000_00000000_00000000_00000000u128,
                    0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128,
                    0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128,
                ]
            ])
        );
        assert_eq!(
            r(
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128,
                0x00000000_00000000_00000000_00000000u128,
            ),
            json!([
                (),
                [
                    0x00000000_00000000_00000000_00000000u128,
                    0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128,
                    0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128,
                ]
            ])
        );
        assert_eq!(
            r(
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128,
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128,
            ),
            json!([
                (),
                [
                    0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128,
                    0x00000000_00000000_00000000_00000000u128,
                    0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128,
                ]
            ])
        );
    }
}
