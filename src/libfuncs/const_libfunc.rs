//! # Const libfuncs

use std::str::FromStr;

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{
        prime_modulo::PrimeModuloMeta, realloc_bindings::ReallocBindingsMeta, MetadataStorage,
    },
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        const_type::{
            ConstAsBoxConcreteLibfunc, ConstAsImmediateConcreteLibfunc, ConstConcreteLibfunc,
        },
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
    },
    program::GenericArg,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith,
        llvm::{self, r#type::opaque_pointer, LoadStoreOptions},
    },
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Attribute, Block, Location, Value},
    Context,
};
use num_bigint::{BigInt, ToBigInt};
use starknet_types_core::felt::Felt;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &ConstConcreteLibfunc,
) -> Result<()> {
    match selector {
        ConstConcreteLibfunc::AsBox(info) => {
            build_const_as_box(context, registry, entry, location, helper, metadata, info)
        }
        ConstConcreteLibfunc::AsImmediate(info) => {
            build_const_as_immediate(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `const_as_box` libfunc.
pub fn build_const_as_box<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConstAsBoxConcreteLibfunc,
) -> Result<()> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }
    let inner_type = registry.get_type(&info.const_type)?;
    let inner_layout = inner_type.layout(registry)?;

    // Create box
    let value_len = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(
                inner_layout.pad_to_align().size().try_into()?,
                IntegerType::new(context, 64).into(),
            )
            .into(),
            location,
        ))
        .result(0)?
        .into();

    let ptr = entry
        .append_operation(llvm::nullptr(opaque_pointer(context), location))
        .result(0)?
        .into();
    let ptr = entry
        .append_operation(ReallocBindingsMeta::realloc(
            context, ptr, value_len, location,
        ))
        .result(0)?
        .into();

    // Create constant
    let const_type = match &inner_type {
        CoreTypeConcrete::Const(inner) => inner,
        _ => unreachable!(),
    };

    // const_type.inner_data Should be one of the following:
    // - A single value, if the inner type is a simple numeric type (e.g., `felt252`, `u32`,
    //   etc.).
    // - A list of const types, if the inner type is a struct. The type of each const type must be
    //   the same as the corresponding struct member type.
    // - A selector (a single value) followed by a const type, if the inner type is an enum. The
    //   type of the const type must be the same as the corresponding enum variant type.

    let inner_type = registry.get_type(&const_type.inner_ty)?;
    let inner_ty =
        registry.build_type(context, helper, registry, metadata, &const_type.inner_ty)?;

    match inner_type {
        CoreTypeConcrete::Struct(struct_info) => {
            dbg!("struct!");
            dbg!(&const_type.inner_data);
            todo!()
        }
        CoreTypeConcrete::Enum(_enum_info) => {
            match &const_type.inner_data[..] {
                [GenericArg::Value(variant_index), GenericArg::Type(enum_ty)] => {
                    todo!()
                }
                _ => return Err(Error::ConstDataMismatch),
            }
            dbg!("enum!");
            dbg!(&const_type.inner_data);
            todo!()
        }
        CoreTypeConcrete::NonZero(_) => match &const_type.inner_data[..] {
            [GenericArg::Type(_inner)] => {
                todo!()
            }
            _ => return Err(Error::ConstDataMismatch),
        },
        inner_type => match &const_type.inner_data[..] {
            [GenericArg::Value(value)] => {
                let mlir_value: Value = match inner_type {
                    CoreTypeConcrete::Felt252(_) => {
                        let value = if value.sign() == num_bigint::Sign::Minus {
                            let prime = metadata
                                .get::<PrimeModuloMeta<Felt>>()
                                .ok_or(Error::MissingMetadata)?
                                .prime();

                            value + prime.to_bigint().expect("Prime to BigInt shouldn't fail")
                        } else {
                            value.clone()
                        };

                        entry
                            .append_operation(arith::constant(
                                context,
                                Attribute::parse(context, &format!("{} : {}", value, inner_ty))
                                    .unwrap(),
                                location,
                            ))
                            .result(0)?
                            .into()
                    }
                    // any other int type
                    _ => entry
                        .append_operation(arith::constant(
                            context,
                            Attribute::parse(context, &format!("{} : {}", value, inner_ty))
                                .unwrap(),
                            location,
                        ))
                        .result(0)?
                        .into(),
                };

                // Store constant in box
                entry.append_operation(llvm::store(
                    context,
                    mlir_value,
                    ptr,
                    location,
                    LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                        inner_layout.align() as i64,
                        IntegerType::new(context, 64).into(),
                    ))),
                ));
            }
            _ => return Err(Error::ConstDataMismatch),
        },
    }

    /*
    if let Some(variants) = registry.get_type(&const_type.inner_ty)?.variants() {
        use num_traits::cast::ToPrimitive;
        let tag = match &const_type.inner_data[0] {
            GenericArg::Value(val) => val.to_usize().unwrap_or_default(),
            // Enum tag should always be value
            _ => todo!(),
        };
        let mut value = const_type.inner_data[1].clone();
        let variant_ty = &variants[tag];
        let variant_type = registry.build_type(context, helper, registry, metadata, variant_ty)?;
        if let GenericArg::Type(ref t) = value {
            // In this case we have to fetch the number from the debug name
            // We get stuff like "Const<felt252, 1234>"
            let str_val = t
                .debug_name
                .as_ref()
                .unwrap()
                .split(' ')
                .last()
                .unwrap()
                .trim_end_matches('>');
            let val = if let Ok(val) = BigInt::from_str(str_val) {
                val
            } else {
                // Unit type enum
                entry.append_operation(helper.br(0, &[ptr], location));
                return Ok(());
            };
            value = GenericArg::Value(val);
        }

        if variant_ty
            .debug_name
            .as_ref()
            .is_some_and(|name| name == "felt252")
        {
            if let cairo_lang_sierra::program::GenericArg::Value(ref num) = value {
                if num.sign() == num_bigint::Sign::Minus {
                    let prime = metadata
                        .get::<PrimeModuloMeta<Felt>>()
                        .ok_or(Error::MissingMetadata)?
                        .prime();
                    let value_mod_prime =
                        num + prime.to_bigint().expect("Prime to BigInt shouldn't fail");
                    let generic_arg = GenericArg::Value(value_mod_prime);
                    value = generic_arg;
                }
            }
        }

        let value = entry
            .append_operation(arith::constant(
                context,
                Attribute::parse(context, &format!("{} : {}", value, variant_type)).unwrap(),
                location,
            ))
            .result(0)?
            .into();

        // Store constant in box
        entry.append_operation(llvm::store(
            context,
            value,
            ptr,
            location,
            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                inner_layout.align() as i64,
                IntegerType::new(context, 64).into(),
            ))),
        ));
    }
    */

    entry.append_operation(helper.br(0, &[ptr], location));
    Ok(())
}

/// Generate MLIR operations for the `const_as_immediate` libfunc.
pub fn build_const_as_immediate<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConstAsImmediateConcreteLibfunc,
) -> Result<()> {
    let const_ty = registry.get_type(&info.const_type)?;

    // Create constant
    let const_type = match &const_ty {
        CoreTypeConcrete::Const(inner) => inner,
        _ => unreachable!(),
    };

    // const_type.inner_data Should be one of the following:
    // - A single value, if the inner type is a simple numeric type (e.g., `felt252`, `u32`,
    //   etc.).
    // - A list of const types, if the inner type is a struct. The type of each const type must be
    //   the same as the corresponding struct member type.
    // - A selector (a single value) followed by a const type, if the inner type is an enum. The
    //   type of the const type must be the same as the corresponding enum variant type.

    let inner_type = registry.get_type(&const_type.inner_ty)?;
    let inner_ty =
        registry.build_type(context, helper, registry, metadata, &const_type.inner_ty)?;

    let result = match inner_type {
        CoreTypeConcrete::Struct(struct_info) => {
            dbg!("struct!");
            dbg!(&const_type.inner_data);
            todo!()
        }
        CoreTypeConcrete::Enum(_enum_info) => {
            match &const_type.inner_data[..] {
                [GenericArg::Value(variant_index), GenericArg::Type(enum_ty)] => {
                    todo!()
                }
                _ => return Err(Error::ConstDataMismatch),
            }
            dbg!("enum!");
            dbg!(&const_type.inner_data);
            todo!()
        }
        CoreTypeConcrete::NonZero(_) => match &const_type.inner_data[..] {
            [GenericArg::Type(_inner)] => {
                todo!()
            }
            _ => return Err(Error::ConstDataMismatch),
        },
        inner_type => match &const_type.inner_data[..] {
            [GenericArg::Value(value)] => {
                let mlir_value: Value = match inner_type {
                    CoreTypeConcrete::Felt252(_) => {
                        let value = if value.sign() == num_bigint::Sign::Minus {
                            let prime = metadata
                                .get::<PrimeModuloMeta<Felt>>()
                                .ok_or(Error::MissingMetadata)?
                                .prime();

                            value + prime.to_bigint().expect("Prime to BigInt shouldn't fail")
                        } else {
                            value.clone()
                        };

                        entry
                            .append_operation(arith::constant(
                                context,
                                Attribute::parse(context, &format!("{} : {}", value, inner_ty))
                                    .unwrap(),
                                location,
                            ))
                            .result(0)?
                            .into()
                    }
                    // any other int type
                    _ => entry
                        .append_operation(arith::constant(
                            context,
                            Attribute::parse(context, &format!("{} : {}", value, inner_ty))
                                .unwrap(),
                            location,
                        ))
                        .result(0)?
                        .into(),
                };

                mlir_value
            }
            _ => return Err(Error::ConstDataMismatch),
        },
    };

    entry.append_operation(helper.br(0, &[result], location));
    Ok(())
}

/*
./cairo2/bin/cairo-compile -r -s program.cairo > program.sierra
cargo r --bin  cairo-native-dump -- program.cairo


use core::box::BoxTrait;

enum MyEnum {
    A: u32,
    B: u16,
}

struct Hello {
    x: MyEnum,
}

fn run_test() -> Hello {
    let x = BoxTrait::new(Hello {
        x: MyEnum::A(2)
    });
    x.unbox()
}
*/
