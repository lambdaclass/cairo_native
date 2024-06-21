//! # Const libfuncs

use super::LibfuncHelper;
use crate::block_ext::BlockExt;
use crate::{
    error::{Error, Result},
    libfuncs::{r#enum::build_enum_value, r#struct::build_struct_value},
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
            ConstConcreteType,
        },
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
    },
    program::GenericArg,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith,
        llvm::{self, r#type::pointer},
    },
    ir::{Attribute, Block, Location, Value},
    Context,
};
use num_bigint::ToBigInt;
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

    let const_type_outer = registry.get_type(&info.const_type)?;

    // Create constant
    let const_type = match &const_type_outer {
        CoreTypeConcrete::Const(inner) => inner,
        _ => unreachable!(),
    };

    let value = build_const_type_value(
        context, registry, entry, location, helper, metadata, const_type,
    )?;

    let const_ty = registry.get_type(&const_type.inner_ty)?;
    let inner_layout = const_ty.layout(registry)?;

    // Create box
    let value_len = entry.const_int(context, location, inner_layout.pad_to_align().size(), 64)?;

    let ptr = entry.append_op_result(llvm::zero(pointer(context, 0), location))?;
    let ptr = entry.append_op_result(ReallocBindingsMeta::realloc(
        context, ptr, value_len, location,
    ))?;

    // Store constant in box
    entry.store(context, location, ptr, value, Some(inner_layout.align()))?;

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

    let const_type = match &const_ty {
        CoreTypeConcrete::Const(inner) => inner,
        _ => unreachable!(),
    };

    let value = build_const_type_value(
        context, registry, entry, location, helper, metadata, const_type,
    )?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

pub fn build_const_type_value<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConstConcreteType,
) -> Result<Value<'ctx, 'this>> {
    // const_type.inner_data Should be one of the following:
    // - A single value, if the inner type is a simple numeric type (e.g., `felt252`, `u32`,
    //   etc.).
    // - A list of const types, if the inner type is a struct. The type of each const type must be
    //   the same as the corresponding struct member type.
    // - A selector (a single value) followed by a const type, if the inner type is an enum. The
    //   type of the const type must be the same as the corresponding enum variant type.

    let inner_type = registry.get_type(&info.inner_ty)?;
    let inner_ty = registry.build_type(context, helper, registry, metadata, &info.inner_ty)?;

    match inner_type {
        CoreTypeConcrete::Struct(_) => {
            let mut fields = Vec::new();

            for field in &info.inner_data {
                match field {
                    GenericArg::Type(const_field_ty) => {
                        let field_type = registry.get_type(const_field_ty)?;

                        let const_field_type = match &field_type {
                            CoreTypeConcrete::Const(inner) => inner,
                            _ => unreachable!(),
                        };

                        let field_value = build_const_type_value(
                            context,
                            registry,
                            entry,
                            location,
                            helper,
                            metadata,
                            const_field_type,
                        )?;
                        fields.push(field_value);
                    }
                    _ => return Err(Error::ConstDataMismatch),
                }
            }

            build_struct_value(
                context,
                registry,
                entry,
                location,
                helper,
                metadata,
                &info.inner_ty,
                &fields,
            )
        }
        CoreTypeConcrete::Enum(_enum_info) => match &info.inner_data[..] {
            [GenericArg::Value(variant_index), GenericArg::Type(payload_ty)] => {
                let payload_type = registry.get_type(payload_ty)?;
                let const_payload_type = match payload_type {
                    CoreTypeConcrete::Const(inner) => inner,
                    _ => unreachable!(),
                };

                let payload_value = build_const_type_value(
                    context,
                    registry,
                    entry,
                    location,
                    helper,
                    metadata,
                    const_payload_type,
                )?;

                build_enum_value(
                    context,
                    registry,
                    entry,
                    location,
                    helper,
                    metadata,
                    payload_value,
                    &info.inner_ty,
                    payload_ty,
                    variant_index.try_into().unwrap(),
                )
            }
            _ => Err(Error::ConstDataMismatch),
        },
        CoreTypeConcrete::NonZero(_) => match &info.inner_data[..] {
            // Copied from the sierra to casm lowering
            // NonZero is the same type as the inner type in native.
            [GenericArg::Type(inner)] => {
                let inner_type = registry.get_type(inner)?;
                let const_inner_type = match inner_type {
                    CoreTypeConcrete::Const(inner) => inner,
                    _ => unreachable!(),
                };

                build_const_type_value(
                    context,
                    registry,
                    entry,
                    location,
                    helper,
                    metadata,
                    const_inner_type,
                )
            }
            _ => Err(Error::ConstDataMismatch),
        },
        inner_type => match &info.inner_data[..] {
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

                        entry.append_op_result(arith::constant(
                            context,
                            Attribute::parse(context, &format!("{} : {}", value, inner_ty))
                                .unwrap(),
                            location,
                        ))?
                    }
                    // any other int type
                    _ => entry.append_op_result(arith::constant(
                        context,
                        Attribute::parse(context, &format!("{} : {}", value, inner_ty)).unwrap(),
                        location,
                    ))?,
                };

                Ok(mlir_value)
            }
            _ => Err(Error::ConstDataMismatch),
        },
    }
}

#[cfg(test)]
pub mod test {
    use crate::{
        utils::test::{jit_struct, load_cairo, run_program},
        values::JitValue,
    };

    #[test]
    fn run_const_as_box() {
        let program = load_cairo!(
            use core::box::BoxTrait;

            struct Hello {
                x: i32,
            }

            fn run_test() -> Hello {
                let x = BoxTrait::new(Hello {
                    x: -2
                });
                x.unbox()
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;
        assert_eq!(result, jit_struct!(JitValue::Sint32(-2)));
    }
}
