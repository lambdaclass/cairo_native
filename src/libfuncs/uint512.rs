////! # `u512`-related libfuncs
//! # `u512`-related libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt, error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt,
    block_ext::BlockExt, error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        int::unsigned512::Uint512Concrete,
        int::unsigned512::Uint512Concrete,
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//        ConcreteLibfunc,
        ConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{arith, llvm},
    dialect::{arith, llvm},
//    ir::{r#type::IntegerType, Block, Location, Value},
    ir::{r#type::IntegerType, Block, Location, Value},
//    Context,
    Context,
//};
};
//

///// Select and call the correct libfunc builder function from the selector.
/// Select and call the correct libfunc builder function from the selector.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: &Uint512Concrete,
    selector: &Uint512Concrete,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        Uint512Concrete::DivModU256(info) => {
        Uint512Concrete::DivModU256(info) => {
//            build_divmod_u256(context, registry, entry, location, helper, metadata, info)
            build_divmod_u256(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `u512_safe_divmod_by_u256` libfunc.
/// Generate MLIR operations for the `u512_safe_divmod_by_u256` libfunc.
//pub fn build_divmod_u256<'ctx, 'this>(
pub fn build_divmod_u256<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let i128_ty = IntegerType::new(context, 128).into();
    let i128_ty = IntegerType::new(context, 128).into();
//    let i512_ty = IntegerType::new(context, 512).into();
    let i512_ty = IntegerType::new(context, 512).into();
//

//    let guarantee_type = registry.build_type(
    let guarantee_type = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.output_types()[0][3],
        &info.output_types()[0][3],
//    )?;
    )?;
//

//    let lhs_struct: Value = entry.argument(1)?.into();
    let lhs_struct: Value = entry.argument(1)?.into();
//    let rhs_struct: Value = entry.argument(2)?.into();
    let rhs_struct: Value = entry.argument(2)?.into();
//

//    let dividend = (
    let dividend = (
//        entry.extract_value(context, location, lhs_struct, i128_ty, 0)?,
        entry.extract_value(context, location, lhs_struct, i128_ty, 0)?,
//        entry.extract_value(context, location, lhs_struct, i128_ty, 1)?,
        entry.extract_value(context, location, lhs_struct, i128_ty, 1)?,
//        entry.extract_value(context, location, lhs_struct, i128_ty, 2)?,
        entry.extract_value(context, location, lhs_struct, i128_ty, 2)?,
//        entry.extract_value(context, location, lhs_struct, i128_ty, 3)?,
        entry.extract_value(context, location, lhs_struct, i128_ty, 3)?,
//    );
    );
//    let divisor = (
    let divisor = (
//        entry.extract_value(context, location, rhs_struct, i128_ty, 0)?,
        entry.extract_value(context, location, rhs_struct, i128_ty, 0)?,
//        entry.extract_value(context, location, rhs_struct, i128_ty, 1)?,
        entry.extract_value(context, location, rhs_struct, i128_ty, 1)?,
//    );
    );
//

//    let dividend = (
    let dividend = (
//        entry.append_op_result(arith::extui(dividend.0, i512_ty, location))?,
        entry.append_op_result(arith::extui(dividend.0, i512_ty, location))?,
//        entry.append_op_result(arith::extui(dividend.1, i512_ty, location))?,
        entry.append_op_result(arith::extui(dividend.1, i512_ty, location))?,
//        entry.append_op_result(arith::extui(dividend.2, i512_ty, location))?,
        entry.append_op_result(arith::extui(dividend.2, i512_ty, location))?,
//        entry.append_op_result(arith::extui(dividend.3, i512_ty, location))?,
        entry.append_op_result(arith::extui(dividend.3, i512_ty, location))?,
//    );
    );
//    let divisor = (
    let divisor = (
//        entry.append_op_result(arith::extui(divisor.0, i512_ty, location))?,
        entry.append_op_result(arith::extui(divisor.0, i512_ty, location))?,
//        entry.append_op_result(arith::extui(divisor.1, i512_ty, location))?,
        entry.append_op_result(arith::extui(divisor.1, i512_ty, location))?,
//    );
    );
//

//    let k128 = entry.const_int_from_type(context, location, 128, i512_ty)?;
    let k128 = entry.const_int_from_type(context, location, 128, i512_ty)?;
//    let k256 = entry.const_int_from_type(context, location, 256, i512_ty)?;
    let k256 = entry.const_int_from_type(context, location, 256, i512_ty)?;
//    let k384 = entry.const_int_from_type(context, location, 384, i512_ty)?;
    let k384 = entry.const_int_from_type(context, location, 384, i512_ty)?;
//

//    let dividend = (
    let dividend = (
//        dividend.0,
        dividend.0,
//        entry.append_op_result(arith::shli(dividend.1, k128, location))?,
        entry.append_op_result(arith::shli(dividend.1, k128, location))?,
//        entry.append_op_result(arith::shli(dividend.2, k256, location))?,
        entry.append_op_result(arith::shli(dividend.2, k256, location))?,
//        entry.append_op_result(arith::shli(dividend.3, k384, location))?,
        entry.append_op_result(arith::shli(dividend.3, k384, location))?,
//    );
    );
//    let divisor = (
    let divisor = (
//        divisor.0,
        divisor.0,
//        entry.append_op_result(arith::shli(divisor.1, k128, location))?,
        entry.append_op_result(arith::shli(divisor.1, k128, location))?,
//    );
    );
//

//    let dividend = {
    let dividend = {
//        let lhs_01 = entry.append_op_result(arith::ori(dividend.0, dividend.1, location))?;
        let lhs_01 = entry.append_op_result(arith::ori(dividend.0, dividend.1, location))?;
//        let lhs_23 = entry.append_op_result(arith::ori(dividend.2, dividend.3, location))?;
        let lhs_23 = entry.append_op_result(arith::ori(dividend.2, dividend.3, location))?;
//

//        entry.append_op_result(arith::ori(lhs_01, lhs_23, location))?
        entry.append_op_result(arith::ori(lhs_01, lhs_23, location))?
//    };
    };
//    let divisor = entry.append_op_result(arith::ori(divisor.0, divisor.1, location))?;
    let divisor = entry.append_op_result(arith::ori(divisor.0, divisor.1, location))?;
//

//    let result_div = entry.append_op_result(arith::divui(dividend, divisor, location))?;
    let result_div = entry.append_op_result(arith::divui(dividend, divisor, location))?;
//    let result_rem = entry.append_op_result(arith::remui(dividend, divisor, location))?;
    let result_rem = entry.append_op_result(arith::remui(dividend, divisor, location))?;
//

//    let result_div = (
    let result_div = (
//        entry.append_op_result(arith::trunci(result_div, i128_ty, location))?,
        entry.append_op_result(arith::trunci(result_div, i128_ty, location))?,
//        entry.append_op_result(arith::trunci(
        entry.append_op_result(arith::trunci(
//            entry.append_op_result(arith::shrui(result_div, k128, location))?,
            entry.append_op_result(arith::shrui(result_div, k128, location))?,
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))?,
        ))?,
//        entry.append_op_result(arith::trunci(
        entry.append_op_result(arith::trunci(
//            entry.append_op_result(arith::shrui(result_div, k256, location))?,
            entry.append_op_result(arith::shrui(result_div, k256, location))?,
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))?,
        ))?,
//        entry.append_op_result(arith::trunci(
        entry.append_op_result(arith::trunci(
//            entry.append_op_result(arith::shrui(result_div, k384, location))?,
            entry.append_op_result(arith::shrui(result_div, k384, location))?,
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))?,
        ))?,
//    );
    );
//

//    let result_rem = (
    let result_rem = (
//        entry.append_op_result(arith::trunci(result_rem, i128_ty, location))?,
        entry.append_op_result(arith::trunci(result_rem, i128_ty, location))?,
//        entry.append_op_result(arith::trunci(
        entry.append_op_result(arith::trunci(
//            entry.append_op_result(arith::shrui(result_rem, k128, location))?,
            entry.append_op_result(arith::shrui(result_rem, k128, location))?,
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))?,
        ))?,
//    );
    );
//

//    let result_div_val = entry.append_op_result(llvm::undef(
    let result_div_val = entry.append_op_result(llvm::undef(
//        llvm::r#type::r#struct(context, &[i128_ty, i128_ty, i128_ty, i128_ty], false),
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty, i128_ty, i128_ty], false),
//        location,
        location,
//    ))?;
    ))?;
//    let result_div_val = entry.insert_values(
    let result_div_val = entry.insert_values(
//        context,
        context,
//        location,
        location,
//        result_div_val,
        result_div_val,
//        &[result_div.0, result_div.1, result_div.2, result_div.3],
        &[result_div.0, result_div.1, result_div.2, result_div.3],
//    )?;
    )?;
//

//    let result_rem_val = entry.append_op_result(llvm::undef(
    let result_rem_val = entry.append_op_result(llvm::undef(
//        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
//        location,
        location,
//    ))?;
    ))?;
//    let result_rem_val = entry.insert_values(
    let result_rem_val = entry.insert_values(
//        context,
        context,
//        location,
        location,
//        result_rem_val,
        result_rem_val,
//        &[result_rem.0, result_rem.1],
        &[result_rem.0, result_rem.1],
//    )?;
    )?;
//

//    let guarantee = entry.append_op_result(llvm::undef(guarantee_type, location))?;
    let guarantee = entry.append_op_result(llvm::undef(guarantee_type, location))?;
//

//    entry.append_operation(helper.br(
    entry.append_operation(helper.br(
//        0,
        0,
//        &[
        &[
//            range_check,
            range_check,
//            result_div_val,
            result_div_val,
//            result_rem_val,
            result_rem_val,
//            guarantee,
            guarantee,
//            guarantee,
            guarantee,
//            guarantee,
            guarantee,
//            guarantee,
            guarantee,
//            guarantee,
            guarantee,
//        ],
        ],
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
//    use crate::{
    use crate::{
//        utils::test::{jit_struct, load_cairo, run_program_assert_output},
        utils::test::{jit_struct, load_cairo, run_program_assert_output},
//        values::JitValue,
        values::JitValue,
//    };
    };
//    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::Program;
//    use lazy_static::lazy_static;
    use lazy_static::lazy_static;
//    use num_bigint::BigUint;
    use num_bigint::BigUint;
//    use num_traits::One;
    use num_traits::One;
//

//    lazy_static! {
    lazy_static! {
//        static ref UINT512_DIVMOD_U256: (String, Program) = load_cairo! {
        static ref UINT512_DIVMOD_U256: (String, Program) = load_cairo! {
//            use core::integer::{u512, u512_safe_divmod_by_u256};
            use core::integer::{u512, u512_safe_divmod_by_u256};
//

//            fn run_test(lhs: u512, rhs: NonZero<u256>) -> (u512, u256) {
            fn run_test(lhs: u512, rhs: NonZero<u256>) -> (u512, u256) {
//                let (lhs, rhs, _, _, _, _, _) = u512_safe_divmod_by_u256(lhs, rhs);
                let (lhs, rhs, _, _, _, _, _) = u512_safe_divmod_by_u256(lhs, rhs);
//                (lhs, rhs)
                (lhs, rhs)
//            }
            }
//        };
        };
//    }
    }
//

//    #[test]
    #[test]
//    fn u512_safe_divmod_by_u256() {
    fn u512_safe_divmod_by_u256() {
//        fn u512(value: BigUint) -> JitValue {
        fn u512(value: BigUint) -> JitValue {
//            assert!(value.bits() <= 512);
            assert!(value.bits() <= 512);
//            jit_struct!(
            jit_struct!(
//                JitValue::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
                JitValue::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
//                JitValue::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
                JitValue::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
//                JitValue::Uint128(((&value >> 256u32) & &u128::MAX.into()).try_into().unwrap()),
                JitValue::Uint128(((&value >> 256u32) & &u128::MAX.into()).try_into().unwrap()),
//                JitValue::Uint128(((&value >> 384u32) & &u128::MAX.into()).try_into().unwrap()),
                JitValue::Uint128(((&value >> 384u32) & &u128::MAX.into()).try_into().unwrap()),
//            )
            )
//        }
        }
//

//        fn u256(value: BigUint) -> JitValue {
        fn u256(value: BigUint) -> JitValue {
//            assert!(value.bits() <= 256);
            assert!(value.bits() <= 256);
//            jit_struct!(
            jit_struct!(
//                JitValue::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
                JitValue::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
//                JitValue::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
                JitValue::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
//            )
            )
//        }
        }
//

//        #[track_caller]
        #[track_caller]
//        fn r2(lhs: BigUint, rhs: BigUint, output_u512: BigUint, output_u256: BigUint) {
        fn r2(lhs: BigUint, rhs: BigUint, output_u512: BigUint, output_u256: BigUint) {
//            let lhs = u512(lhs);
            let lhs = u512(lhs);
//            let rhs = u256(rhs);
            let rhs = u256(rhs);
//            let output_u512 = u512(output_u512);
            let output_u512 = u512(output_u512);
//            let output_u256 = u256(output_u256);
            let output_u256 = u256(output_u256);
//            run_program_assert_output(
            run_program_assert_output(
//                &UINT512_DIVMOD_U256,
                &UINT512_DIVMOD_U256,
//                "run_test",
                "run_test",
//                &[lhs, rhs],
                &[lhs, rhs],
//                jit_struct!(output_u512, output_u256),
                jit_struct!(output_u512, output_u256),
//            );
            );
//        }
        }
//

//        r2(0u32.into(), 1u32.into(), 0u32.into(), 0u32.into());
        r2(0u32.into(), 1u32.into(), 0u32.into(), 0u32.into());
//        r2(
        r2(
//            0u32.into(),
            0u32.into(),
//            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256u32) - 2u32,
//            0u32.into(),
            0u32.into(),
//            0u32.into(),
            0u32.into(),
//        );
        );
//        r2(
        r2(
//            0u32.into(),
            0u32.into(),
//            (BigUint::one() << 256u32) - 1u32,
            (BigUint::one() << 256u32) - 1u32,
//            0u32.into(),
            0u32.into(),
//            0u32.into(),
            0u32.into(),
//        );
        );
//

//        r2(1u32.into(), 1u32.into(), 1u32.into(), 0u32.into());
        r2(1u32.into(), 1u32.into(), 1u32.into(), 0u32.into());
//        r2(
        r2(
//            1u32.into(),
            1u32.into(),
//            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256u32) - 2u32,
//            0u32.into(),
            0u32.into(),
//            1u32.into(),
            1u32.into(),
//        );
        );
//        r2(
        r2(
//            1u32.into(),
            1u32.into(),
//            (BigUint::one() << 256u32) - 1u32,
            (BigUint::one() << 256u32) - 1u32,
//            0u32.into(),
            0u32.into(),
//            1u32.into(),
            1u32.into(),
//        );
        );
//

//        r2(
        r2(
//            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 512u32) - 2u32,
//            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256u32) - 2u32,
//            (BigUint::one() << 256) + 2u32,
            (BigUint::one() << 256) + 2u32,
//            2u32.into(),
            2u32.into(),
//        );
        );
//        r2(
        r2(
//            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 512u32) - 2u32,
//            1u32.into(),
            1u32.into(),
//            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 512u32) - 2u32,
//            0u32.into(),
            0u32.into(),
//        );
        );
//        r2(
        r2(
//            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 512u32) - 2u32,
//            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256u32) - 2u32,
//            (BigUint::one() << 256) + 2u32,
            (BigUint::one() << 256) + 2u32,
//            2u32.into(),
            2u32.into(),
//        );
        );
//        r2(
        r2(
//            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 512u32) - 2u32,
//            (BigUint::one() << 256u32) - 1u32,
            (BigUint::one() << 256u32) - 1u32,
//            BigUint::one() << 256u32,
            BigUint::one() << 256u32,
//            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256u32) - 2u32,
//        );
        );
//

//        r2(
        r2(
//            (BigUint::one() << 512u32) - 1u32,
            (BigUint::one() << 512u32) - 1u32,
//            1u32.into(),
            1u32.into(),
//            (BigUint::one() << 512u32) - 1u32,
            (BigUint::one() << 512u32) - 1u32,
//            0u32.into(),
            0u32.into(),
//        );
        );
//        r2(
        r2(
//            (BigUint::one() << 512u32) - 1u32,
            (BigUint::one() << 512u32) - 1u32,
//            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256u32) - 2u32,
//            (BigUint::one() << 256) + 2u32,
            (BigUint::one() << 256) + 2u32,
//            3u32.into(),
            3u32.into(),
//        );
        );
//        r2(
        r2(
//            (BigUint::one() << 512u32) - 1u32,
            (BigUint::one() << 512u32) - 1u32,
//            (BigUint::one() << 256u32) - 1u32,
            (BigUint::one() << 256u32) - 1u32,
//            (BigUint::one() << 256) + 1u32,
            (BigUint::one() << 256) + 1u32,
//            0u32.into(),
            0u32.into(),
//        );
        );
//    }
    }
//}
}
