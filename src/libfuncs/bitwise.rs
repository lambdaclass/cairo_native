////! # Bitwise libfuncs
//! # Bitwise libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{block_ext::BlockExt, error::Result, metadata::MetadataStorage};
use crate::{block_ext::BlockExt, error::Result, metadata::MetadataStorage};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::arith,
    dialect::arith,
//    ir::{Block, Location},
    ir::{Block, Location},
//    Context,
    Context,
//};
};
//

///// Generate MLIR operations for the `bitwise` libfunc.
/// Generate MLIR operations for the `bitwise` libfunc.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let bitwise =
    let bitwise =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let lhs = entry.argument(1)?.into();
    let lhs = entry.argument(1)?.into();
//    let rhs = entry.argument(2)?.into();
    let rhs = entry.argument(2)?.into();
//

//    let logical_and = entry.append_op_result(arith::andi(lhs, rhs, location))?;
    let logical_and = entry.append_op_result(arith::andi(lhs, rhs, location))?;
//    let logical_xor = entry.append_op_result(arith::xori(lhs, rhs, location))?;
    let logical_xor = entry.append_op_result(arith::xori(lhs, rhs, location))?;
//    let logical_or = entry.append_op_result(arith::ori(lhs, rhs, location))?;
    let logical_or = entry.append_op_result(arith::ori(lhs, rhs, location))?;
//

//    entry.append_operation(helper.br(
    entry.append_operation(helper.br(
//        0,
        0,
//        &[bitwise, logical_and, logical_xor, logical_or],
        &[bitwise, logical_and, logical_xor, logical_or],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::utils::test::{jit_struct, load_cairo, run_program};
    use crate::utils::test::{jit_struct, load_cairo, run_program};
//    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::Program;
//    use lazy_static::lazy_static;
    use lazy_static::lazy_static;
//

//    lazy_static! {
    lazy_static! {
//        static ref BITWISE: (String, Program) = load_cairo! {
        static ref BITWISE: (String, Program) = load_cairo! {
//            use core::integer::bitwise;
            use core::integer::bitwise;
//

//            fn run_test(lhs: u128, rhs: u128) -> (u128, u128, u128) {
            fn run_test(lhs: u128, rhs: u128) -> (u128, u128, u128) {
//                bitwise(lhs, rhs)
                bitwise(lhs, rhs)
//            }
            }
//        };
        };
//    }
    }
//

//    #[test]
    #[test]
//    fn bitwise() {
    fn bitwise() {
//        let r = |lhs, rhs| run_program(&BITWISE, "run_test", &[lhs, rhs]).return_value;
        let r = |lhs, rhs| run_program(&BITWISE, "run_test", &[lhs, rhs]).return_value;
//

//        assert_eq!(
        assert_eq!(
//            r(
            r(
//                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
//                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
//            ),
            ),
//            jit_struct!(
            jit_struct!(
//                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
//                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
//                0x00000000_00000000_00000000_00000000u128.into()
                0x00000000_00000000_00000000_00000000u128.into()
//            )
            )
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(
            r(
//                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
//                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
//            ),
            ),
//            jit_struct!(
            jit_struct!(
//                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
//                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
//                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into()
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into()
//            )
            )
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(
            r(
//                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
//                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
//            ),
            ),
//            jit_struct!(
            jit_struct!(
//                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
//                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
//                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into()
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into()
//            )
            )
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(
            r(
//                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
//                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
//            ),
            ),
//            jit_struct!(
            jit_struct!(
//                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into(),
//                0x00000000_00000000_00000000_00000000u128.into(),
                0x00000000_00000000_00000000_00000000u128.into(),
//                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into()
                0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128.into()
//            )
            )
//        );
        );
//    }
    }
//}
}
