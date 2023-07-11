//! # StarkNet libfuncs
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
    extensions::{
        consts::SignatureAndConstConcreteLibfunc, lib_func::SignatureOnlyConcreteLibfunc,
        starknet::StarkNetConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith::{self, CmpiPredicate},
    ir::{operation::OperationBuilder, r#type::IntegerType, Attribute, Block, Location, ValueLike},
    Context,
};
use num_bigint::Sign;
use std::ops::Neg;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &StarkNetConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        StarkNetConcreteLibfunc::CallContract(_) => todo!(),
        StarkNetConcreteLibfunc::ClassHashConst(_) => todo!(),
        StarkNetConcreteLibfunc::ClassHashTryFromFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::ClassHashToFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::ContractAddressConst(_) => todo!(),
        StarkNetConcreteLibfunc::ContractAddressTryFromFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::ContractAddressToFelt252(_) => todo!(),
        StarkNetConcreteLibfunc::StorageRead(_) => todo!(),
        StarkNetConcreteLibfunc::StorageWrite(_) => todo!(),
        StarkNetConcreteLibfunc::StorageBaseAddressConst(info) => build_storage_base_address_const(
            context, registry, entry, location, helper, metadata, info,
        ),
        StarkNetConcreteLibfunc::StorageBaseAddressFromFelt252(info) => {
            build_storage_base_address_from_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::StorageAddressFromBase(info) => build_storage_address_from_base(
            context, registry, entry, location, helper, metadata, info,
        ),
        StarkNetConcreteLibfunc::StorageAddressFromBaseAndOffset(info) => {
            build_storage_address_from_base_and_offset(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::StorageAddressToFelt252(info) => build_storage_address_to_felt252(
            context, registry, entry, location, helper, metadata, info,
        ),
        StarkNetConcreteLibfunc::StorageAddressTryFromFelt252(info) => {
            build_storage_address_try_from_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::EmitEvent(_) => todo!(),
        StarkNetConcreteLibfunc::GetBlockHash(_) => todo!(),
        StarkNetConcreteLibfunc::GetExecutionInfo(_) => todo!(),
        StarkNetConcreteLibfunc::Deploy(_) => todo!(),
        StarkNetConcreteLibfunc::Keccak(_) => todo!(),
        StarkNetConcreteLibfunc::LibraryCall(_) => todo!(),
        StarkNetConcreteLibfunc::ReplaceClass(_) => todo!(),
        StarkNetConcreteLibfunc::SendMessageToL1(_) => todo!(),
        StarkNetConcreteLibfunc::Testing(_) => todo!(),
        StarkNetConcreteLibfunc::Secp256(_) => todo!(),
    }
}

pub fn build_storage_base_address_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndConstConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let value = match info.c.sign() {
        Sign::Minus => (&info.c).neg().to_biguint().unwrap(),
        _ => info.c.to_biguint().unwrap(),
    };

    let value = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(context, &format!("{value} : i252")).unwrap(),
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

pub fn build_storage_base_address_from_felt252<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
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
    let k_limit = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(
                context,
                "3618502788666131106986593281521497120414687020801267626233049500247285300992 : i252",
            )
            .unwrap(),
            location,
        ))
        .result(0)?
        .into();

    let limited_value = entry
        .append_operation(arith::subi(entry.argument(1)?.into(), k_limit, location))
        .result(0)?
        .into();

    let is_within_limit = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ult,
            entry.argument(1)?.into(),
            k_limit,
            location,
        ))
        .result(0)?
        .into();
    let value = entry
        .append_operation(
            OperationBuilder::new("arith.select", location)
                .add_operands(&[is_within_limit, entry.argument(1)?.into(), limited_value])
                .add_results(&[IntegerType::new(context, 252).into()])
                .build(),
        )
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into(), value], location));
    Ok(())
}

pub fn build_storage_address_from_base<'ctx, 'this, TType, TLibfunc>(
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
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
    Ok(())
}

pub fn build_storage_address_from_base_and_offset<'ctx, 'this, TType, TLibfunc>(
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
{
    let offset = entry
        .append_operation(arith::extui(
            entry.argument(1)?.into(),
            entry.argument(0)?.r#type(),
            location,
        ))
        .result(0)?
        .into();
    let addr = entry
        .append_operation(arith::addi(entry.argument(0)?.into(), offset, location))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[addr], location));
    Ok(())
}

pub fn build_storage_address_to_felt252<'ctx, 'this, TType, TLibfunc>(
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
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
    Ok(())
}

pub fn build_storage_address_try_from_felt252<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
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
    let value = entry.argument(1)?.into();

    let limit = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(
                context,
                "106710729501573572985208420194530329073740042555888586719489 : i252",
            )
            .unwrap(),
            location,
        ))
        .result(0)?
        .into();
    let is_in_range = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ult,
            value,
            limit,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        is_in_range,
        [0, 1],
        [
            &[entry.argument(0)?.into(), value],
            &[entry.argument(0)?.into()],
        ],
        location,
    ));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        types::felt252::PRIME,
        utils::test::{load_cairo, run_program},
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use num_bigint::{BigInt, Sign};
    use serde_json::json;
    use std::ops::Neg;

    lazy_static! {
        static ref STORAGE_BASE_ADDRESS_FROM_FELT252: (String, Program) = load_cairo! {
            use starknet::storage_access::{StorageBaseAddress, storage_base_address_from_felt252};

            fn run_program(value: felt252) -> StorageBaseAddress {
                storage_base_address_from_felt252(value)
            }
        };
        static ref STORAGE_ADDRESS_FROM_BASE: (String, Program) = load_cairo! {
            use starknet::storage_access::{StorageAddress, StorageBaseAddress, storage_address_from_base};

            fn run_program(value: StorageBaseAddress) -> StorageAddress {
                storage_address_from_base(value)
            }
        };
        static ref STORAGE_ADDRESS_FROM_BASE_AND_OFFSET: (String, Program) = load_cairo! {
            use starknet::storage_access::{StorageAddress, StorageBaseAddress, storage_address_from_base_and_offset};

            fn run_program(addr: StorageBaseAddress, offset: u8) -> StorageAddress {
                storage_address_from_base_and_offset(addr, offset)
            }
        };
        static ref STORAGE_ADDRESS_TO_FELT252: (String, Program) = load_cairo! {
            use starknet::storage_access::{StorageAddress, storage_address_to_felt252};

            fn run_program(value: StorageAddress) -> felt252 {
                storage_address_to_felt252(value)
            }
        };
        static ref STORAGE_ADDRESS_TRY_FROM_FELT252: (String, Program) = load_cairo! {
            use starknet::storage_access::{StorageAddress, storage_address_try_from_felt252};

            fn run_program(value: felt252) -> Option<StorageAddress> {
                storage_address_try_from_felt252(value)
            }
        };
    }

    // Parse numeric string into felt, wrapping negatives around the prime modulo.
    fn f(value: &str) -> [u32; 8] {
        let value = value.parse::<BigInt>().unwrap();
        let value = match value.sign() {
            Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
            _ => value.to_biguint().unwrap(),
        };

        let mut u32_digits = value.to_u32_digits();
        u32_digits.resize(8, 0);
        u32_digits.try_into().unwrap()
    }

    #[test]
    fn storage_base_address_from_felt252() {
        let r = |value| {
            run_program(
                &STORAGE_BASE_ADDRESS_FROM_FELT252,
                "run_program",
                json!([(), value]),
            )
        };

        assert_eq!(r(f("0")), json!([(), f("0")]));
        assert_eq!(r(f("1")), json!([(), f("1")]));
        assert_eq!(
            r(f("-1")),
            json!([
                (),
                f("106710729501573572985208420194530329073740042555888586719488")
            ])
        );
        assert_eq!(
            r(f(
                "3618502788666131106986593281521497120414687020801267626233049500247285300992"
            )),
            json!([(), f("0")])
        );
    }

    #[test]
    fn storage_address_from_base() {
        let r = |value| run_program(&STORAGE_ADDRESS_FROM_BASE, "run_program", json!([value]));

        assert_eq!(r(f("0")), json!([f("0")]));
        assert_eq!(r(f("1")), json!([f("1")]));
        assert_eq!(
            r(f(
                "106710729501573572985208420194530329073740042555888586719488"
            )),
            json!([f(
                "106710729501573572985208420194530329073740042555888586719488"
            )])
        );
    }

    #[test]
    fn storage_address_from_base_and_offset() {
        let r = |addr, offset| {
            run_program(
                &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
                "run_program",
                json!([addr, offset]),
            )
        };

        assert_eq!(r(f("0"), 0u8), json!([f("0")]));
        assert_eq!(r(f("1"), 0u8), json!([f("1")]));
        assert_eq!(
            r(
                f("106710729501573572985208420194530329073740042555888586719488"),
                0u8
            ),
            json!([f(
                "106710729501573572985208420194530329073740042555888586719488"
            )])
        );

        assert_eq!(r(f("0"), 1u8), json!([f("1")]));
        assert_eq!(r(f("1"), 1u8), json!([f("2")]));
        assert_eq!(
            r(
                f("106710729501573572985208420194530329073740042555888586719488"),
                1u8
            ),
            json!([f(
                "106710729501573572985208420194530329073740042555888586719489"
            )])
        );

        assert_eq!(r(f("0"), 255u8), json!([f("255")]));
        assert_eq!(r(f("1"), 255u8), json!([f("256")]));
        assert_eq!(
            r(
                f("106710729501573572985208420194530329073740042555888586719488"),
                255u8
            ),
            json!([f(
                "106710729501573572985208420194530329073740042555888586719743"
            )])
        );
    }

    #[test]
    fn storage_address_to_felt252() {
        let r = |value| run_program(&STORAGE_ADDRESS_TO_FELT252, "run_program", json!([value]));

        assert_eq!(r(f("0")), json!([f("0")]));
        assert_eq!(r(f("1")), json!([f("1")]));
        assert_eq!(
            r(f(
                "106710729501573572985208420194530329073740042555888586719488"
            ),),
            json!([f(
                "106710729501573572985208420194530329073740042555888586719488"
            )])
        );
    }

    #[test]
    fn storage_address_try_from_felt252() {
        let r = |value| {
            run_program(
                &STORAGE_ADDRESS_TRY_FROM_FELT252,
                "run_program",
                json!([(), value]),
            )
        };

        assert_eq!(r(f("0")), json!([(), [0, f("0")]]));
        assert_eq!(r(f("1")), json!([(), [0, f("1")]]));
        assert_eq!(
            r(f(
                "106710729501573572985208420194530329073740042555888586719488"
            ),),
            json!([
                (),
                [
                    0,
                    f("106710729501573572985208420194530329073740042555888586719488")
                ]
            ])
        );
        assert_eq!(r(f("-1"),), json!([(), [1, []]]));
    }
}
