//! # `u512`-related libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::unsigned512::Uint512Concrete,
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm},
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        r#type::IntegerType,
        Block, Location, Value,
    },
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
    selector: &Uint512Concrete,
) -> Result<()> {
    match selector {
        Uint512Concrete::DivModU256(info) => {
            build_divmod_u256(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `u512_safe_divmod_by_u256` libfunc.
pub fn build_divmod_u256<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let i128_ty = IntegerType::new(context, 128).into();
    let i512_ty = IntegerType::new(context, 512).into();

    let guarantee_type = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[0][3],
    )?;

    let lhs_struct: Value = entry.argument(1)?.into();
    let rhs_struct: Value = entry.argument(2)?.into();

    let lhs = (
        entry
            .append_operation(llvm::extract_value(
                context,
                lhs_struct,
                DenseI64ArrayAttribute::new(context, &[0]),
                i128_ty,
                location,
            ))
            .result(0)?
            .into(),
        entry
            .append_operation(llvm::extract_value(
                context,
                lhs_struct,
                DenseI64ArrayAttribute::new(context, &[1]),
                i128_ty,
                location,
            ))
            .result(0)?
            .into(),
        entry
            .append_operation(llvm::extract_value(
                context,
                lhs_struct,
                DenseI64ArrayAttribute::new(context, &[2]),
                i128_ty,
                location,
            ))
            .result(0)?
            .into(),
        entry
            .append_operation(llvm::extract_value(
                context,
                lhs_struct,
                DenseI64ArrayAttribute::new(context, &[3]),
                i128_ty,
                location,
            ))
            .result(0)?
            .into(),
    );
    let rhs = (
        entry
            .append_operation(llvm::extract_value(
                context,
                rhs_struct,
                DenseI64ArrayAttribute::new(context, &[0]),
                i128_ty,
                location,
            ))
            .result(0)?
            .into(),
        entry
            .append_operation(llvm::extract_value(
                context,
                rhs_struct,
                DenseI64ArrayAttribute::new(context, &[1]),
                i128_ty,
                location,
            ))
            .result(0)?
            .into(),
    );

    let lhs = (
        entry
            .append_operation(arith::extui(lhs.0, i512_ty, location))
            .result(0)?
            .into(),
        entry
            .append_operation(arith::extui(lhs.1, i512_ty, location))
            .result(0)?
            .into(),
        entry
            .append_operation(arith::extui(lhs.2, i512_ty, location))
            .result(0)?
            .into(),
        entry
            .append_operation(arith::extui(lhs.3, i512_ty, location))
            .result(0)?
            .into(),
    );
    let rhs = (
        entry
            .append_operation(arith::extui(rhs.0, i512_ty, location))
            .result(0)?
            .into(),
        entry
            .append_operation(arith::extui(rhs.1, i512_ty, location))
            .result(0)?
            .into(),
    );

    let k128 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(128, i512_ty).into(),
            location,
        ))
        .result(0)?
        .into();
    let k256 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(256, i512_ty).into(),
            location,
        ))
        .result(0)?
        .into();
    let k384 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(384, i512_ty).into(),
            location,
        ))
        .result(0)?
        .into();

    let lhs = (
        lhs.0,
        entry
            .append_operation(arith::shli(lhs.1, k128, location))
            .result(0)?
            .into(),
        entry
            .append_operation(arith::shli(lhs.2, k256, location))
            .result(0)?
            .into(),
        entry
            .append_operation(arith::shli(lhs.3, k384, location))
            .result(0)?
            .into(),
    );
    let rhs = (
        rhs.0,
        entry
            .append_operation(arith::shli(rhs.1, k128, location))
            .result(0)?
            .into(),
    );

    let lhs = {
        let lhs_01 = entry
            .append_operation(arith::ori(lhs.0, lhs.1, location))
            .result(0)?
            .into();
        let lhs_23 = entry
            .append_operation(arith::ori(lhs.2, lhs.3, location))
            .result(0)?
            .into();

        entry
            .append_operation(arith::ori(lhs_01, lhs_23, location))
            .result(0)?
            .into()
    };
    let rhs = entry
        .append_operation(arith::ori(rhs.0, rhs.1, location))
        .result(0)?
        .into();

    let result_div = entry
        .append_operation(arith::divui(lhs, rhs, location))
        .result(0)?
        .into();
    let result_rem = entry
        .append_operation(arith::remui(lhs, rhs, location))
        .result(0)?
        .into();

    let result_div = (
        entry
            .append_operation(arith::trunci(result_div, i128_ty, location))
            .result(0)?
            .into(),
        entry
            .append_operation(arith::trunci(
                entry
                    .append_operation(arith::shrui(result_div, k128, location))
                    .result(0)?
                    .into(),
                i128_ty,
                location,
            ))
            .result(0)?
            .into(),
        entry
            .append_operation(arith::trunci(
                entry
                    .append_operation(arith::shrui(result_div, k256, location))
                    .result(0)?
                    .into(),
                i128_ty,
                location,
            ))
            .result(0)?
            .into(),
        entry
            .append_operation(arith::trunci(
                entry
                    .append_operation(arith::shrui(result_div, k384, location))
                    .result(0)?
                    .into(),
                i128_ty,
                location,
            ))
            .result(0)?
            .into(),
    );

    let result_rem = (
        entry
            .append_operation(arith::trunci(result_rem, i128_ty, location))
            .result(0)?
            .into(),
        entry
            .append_operation(arith::trunci(
                entry
                    .append_operation(arith::shrui(result_rem, k128, location))
                    .result(0)?
                    .into(),
                i128_ty,
                location,
            ))
            .result(0)?
            .into(),
    );

    let result_div_val = entry
        .append_operation(llvm::undef(
            llvm::r#type::r#struct(context, &[i128_ty, i128_ty, i128_ty, i128_ty], false),
            location,
        ))
        .result(0)?
        .into();
    let result_div_val = entry
        .append_operation(llvm::insert_value(
            context,
            result_div_val,
            DenseI64ArrayAttribute::new(context, &[0]),
            result_div.0,
            location,
        ))
        .result(0)?
        .into();
    let result_div_val = entry
        .append_operation(llvm::insert_value(
            context,
            result_div_val,
            DenseI64ArrayAttribute::new(context, &[1]),
            result_div.1,
            location,
        ))
        .result(0)?
        .into();
    let result_div_val = entry
        .append_operation(llvm::insert_value(
            context,
            result_div_val,
            DenseI64ArrayAttribute::new(context, &[2]),
            result_div.2,
            location,
        ))
        .result(0)?
        .into();
    let result_div_val = entry
        .append_operation(llvm::insert_value(
            context,
            result_div_val,
            DenseI64ArrayAttribute::new(context, &[3]),
            result_div.3,
            location,
        ))
        .result(0)?
        .into();

    let result_rem_val = entry
        .append_operation(llvm::undef(
            llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
            location,
        ))
        .result(0)?
        .into();
    let result_rem_val = entry
        .append_operation(llvm::insert_value(
            context,
            result_rem_val,
            DenseI64ArrayAttribute::new(context, &[0]),
            result_rem.0,
            location,
        ))
        .result(0)?
        .into();
    let result_rem_val = entry
        .append_operation(llvm::insert_value(
            context,
            result_rem_val,
            DenseI64ArrayAttribute::new(context, &[1]),
            result_rem.1,
            location,
        ))
        .result(0)?
        .into();

    let op = entry.append_operation(llvm::undef(guarantee_type, location));
    let guarantee = op.result(0)?.into();

    entry.append_operation(helper.br(
        0,
        &[
            range_check,
            result_div_val,
            result_rem_val,
            guarantee,
            guarantee,
            guarantee,
            guarantee,
            guarantee,
        ],
        location,
    ));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_struct, load_cairo, run_program_assert_output},
        values::JitValue,
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use num_bigint::BigUint;
    use num_traits::One;

    lazy_static! {
        static ref UINT512_DIVMOD_U256: (String, Program) = load_cairo! {
            use core::integer::{u512, u512_safe_divmod_by_u256};

            fn run_test(lhs: u512, rhs: NonZero<u256>) -> (u512, u256) {
                let (lhs, rhs, _, _, _, _, _) = u512_safe_divmod_by_u256(lhs, rhs);
                (lhs, rhs)
            }
        };
    }

    #[test]
    fn u512_safe_divmod_by_u256() {
        fn u512(value: BigUint) -> JitValue {
            assert!(value.bits() <= 512);
            jit_struct!(
                JitValue::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
                JitValue::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
                JitValue::Uint128(((&value >> 256u32) & &u128::MAX.into()).try_into().unwrap()),
                JitValue::Uint128(((&value >> 384u32) & &u128::MAX.into()).try_into().unwrap()),
            )
        }

        fn u256(value: BigUint) -> JitValue {
            assert!(value.bits() <= 256);
            jit_struct!(
                JitValue::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
                JitValue::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
            )
        }

        #[track_caller]
        fn r2(lhs: BigUint, rhs: BigUint, output_u512: BigUint, output_u256: BigUint) {
            let lhs = u512(lhs);
            let rhs = u256(rhs);
            let output_u512 = u512(output_u512);
            let output_u256 = u256(output_u256);
            run_program_assert_output(
                &UINT512_DIVMOD_U256,
                "run_test",
                &[lhs, rhs],
                jit_struct!(output_u512, output_u256),
            );
        }

        r2(0u32.into(), 1u32.into(), 0u32.into(), 0u32.into());
        r2(
            0u32.into(),
            (BigUint::one() << 256u32) - 2u32,
            0u32.into(),
            0u32.into(),
        );
        r2(
            0u32.into(),
            (BigUint::one() << 256u32) - 1u32,
            0u32.into(),
            0u32.into(),
        );

        r2(1u32.into(), 1u32.into(), 1u32.into(), 0u32.into());
        r2(
            1u32.into(),
            (BigUint::one() << 256u32) - 2u32,
            0u32.into(),
            1u32.into(),
        );
        r2(
            1u32.into(),
            (BigUint::one() << 256u32) - 1u32,
            0u32.into(),
            1u32.into(),
        );

        r2(
            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256) + 2u32,
            2u32.into(),
        );
        r2(
            (BigUint::one() << 512u32) - 2u32,
            1u32.into(),
            (BigUint::one() << 512u32) - 2u32,
            0u32.into(),
        );
        r2(
            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256) + 2u32,
            2u32.into(),
        );
        r2(
            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 256u32) - 1u32,
            BigUint::one() << 256u32,
            (BigUint::one() << 256u32) - 2u32,
        );

        r2(
            (BigUint::one() << 512u32) - 1u32,
            1u32.into(),
            (BigUint::one() << 512u32) - 1u32,
            0u32.into(),
        );
        r2(
            (BigUint::one() << 512u32) - 1u32,
            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256) + 2u32,
            3u32.into(),
        );
        r2(
            (BigUint::one() << 512u32) - 1u32,
            (BigUint::one() << 256u32) - 1u32,
            (BigUint::one() << 256) + 1u32,
            0u32.into(),
        );
    }
}
