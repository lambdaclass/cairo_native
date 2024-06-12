////! # Const libfuncs
//! # Const libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    error::{Error, Result},
    error::{Error, Result},
//    libfuncs::{r#enum::build_enum_value, r#struct::build_struct_value},
    libfuncs::{r#enum::build_enum_value, r#struct::build_struct_value},
//    metadata::{
    metadata::{
//        prime_modulo::PrimeModuloMeta, realloc_bindings::ReallocBindingsMeta, MetadataStorage,
        prime_modulo::PrimeModuloMeta, realloc_bindings::ReallocBindingsMeta, MetadataStorage,
//    },
    },
//    types::TypeBuilder,
    types::TypeBuilder,
//    utils::ProgramRegistryExt,
    utils::ProgramRegistryExt,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        const_type::{
        const_type::{
//            ConstAsBoxConcreteLibfunc, ConstAsImmediateConcreteLibfunc, ConstConcreteLibfunc,
            ConstAsBoxConcreteLibfunc, ConstAsImmediateConcreteLibfunc, ConstConcreteLibfunc,
//            ConstConcreteType,
            ConstConcreteType,
//        },
        },
//        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
//    },
    },
//    program::GenericArg,
    program::GenericArg,
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{
    dialect::{
//        arith,
        arith,
//        llvm::{self, r#type::pointer, LoadStoreOptions},
        llvm::{self, r#type::pointer, LoadStoreOptions},
//    },
    },
//    ir::{attribute::IntegerAttribute, r#type::IntegerType, Attribute, Block, Location, Value},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Attribute, Block, Location, Value},
//    Context,
    Context,
//};
};
//use num_bigint::ToBigInt;
use num_bigint::ToBigInt;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
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
//    selector: &ConstConcreteLibfunc,
    selector: &ConstConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        ConstConcreteLibfunc::AsBox(info) => {
        ConstConcreteLibfunc::AsBox(info) => {
//            build_const_as_box(context, registry, entry, location, helper, metadata, info)
            build_const_as_box(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ConstConcreteLibfunc::AsImmediate(info) => {
        ConstConcreteLibfunc::AsImmediate(info) => {
//            build_const_as_immediate(context, registry, entry, location, helper, metadata, info)
            build_const_as_immediate(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `const_as_box` libfunc.
/// Generate MLIR operations for the `const_as_box` libfunc.
//pub fn build_const_as_box<'ctx, 'this>(
pub fn build_const_as_box<'ctx, 'this>(
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
//    info: &ConstAsBoxConcreteLibfunc,
    info: &ConstAsBoxConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let const_type_outer = registry.get_type(&info.const_type)?;
    let const_type_outer = registry.get_type(&info.const_type)?;
//

//    // Create constant
    // Create constant
//    let const_type = match &const_type_outer {
    let const_type = match &const_type_outer {
//        CoreTypeConcrete::Const(inner) => inner,
        CoreTypeConcrete::Const(inner) => inner,
//        _ => unreachable!(),
        _ => unreachable!(),
//    };
    };
//

//    let value = build_const_type_value(
    let value = build_const_type_value(
//        context, registry, entry, location, helper, metadata, const_type,
        context, registry, entry, location, helper, metadata, const_type,
//    )?;
    )?;
//

//    let const_ty = registry.get_type(&const_type.inner_ty)?;
    let const_ty = registry.get_type(&const_type.inner_ty)?;
//    let inner_layout = const_ty.layout(registry)?;
    let inner_layout = const_ty.layout(registry)?;
//

//    // Create box
    // Create box
//    let value_len = entry
    let value_len = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(
            IntegerAttribute::new(
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                inner_layout.pad_to_align().size().try_into()?,
                inner_layout.pad_to_align().size().try_into()?,
//            )
            )
//            .into(),
            .into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let ptr = entry
    let ptr = entry
//        .append_operation(llvm::zero(pointer(context, 0), location))
        .append_operation(llvm::zero(pointer(context, 0), location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let ptr = entry
    let ptr = entry
//        .append_operation(ReallocBindingsMeta::realloc(
        .append_operation(ReallocBindingsMeta::realloc(
//            context, ptr, value_len, location,
            context, ptr, value_len, location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Store constant in box
    // Store constant in box
//

//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        value,
        value,
//        ptr,
        ptr,
//        location,
        location,
//        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
//            IntegerType::new(context, 64).into(),
            IntegerType::new(context, 64).into(),
//            inner_layout.align() as i64,
            inner_layout.align() as i64,
//        ))),
        ))),
//    ));
    ));
//

//    entry.append_operation(helper.br(0, &[ptr], location));
    entry.append_operation(helper.br(0, &[ptr], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `const_as_immediate` libfunc.
/// Generate MLIR operations for the `const_as_immediate` libfunc.
//pub fn build_const_as_immediate<'ctx, 'this>(
pub fn build_const_as_immediate<'ctx, 'this>(
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
//    info: &ConstAsImmediateConcreteLibfunc,
    info: &ConstAsImmediateConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let const_ty = registry.get_type(&info.const_type)?;
    let const_ty = registry.get_type(&info.const_type)?;
//

//    let const_type = match &const_ty {
    let const_type = match &const_ty {
//        CoreTypeConcrete::Const(inner) => inner,
        CoreTypeConcrete::Const(inner) => inner,
//        _ => unreachable!(),
        _ => unreachable!(),
//    };
    };
//

//    let value = build_const_type_value(
    let value = build_const_type_value(
//        context, registry, entry, location, helper, metadata, const_type,
        context, registry, entry, location, helper, metadata, const_type,
//    )?;
    )?;
//

//    entry.append_operation(helper.br(0, &[value], location));
    entry.append_operation(helper.br(0, &[value], location));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_const_type_value<'ctx, 'this>(
pub fn build_const_type_value<'ctx, 'this>(
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
//    info: &ConstConcreteType,
    info: &ConstConcreteType,
//) -> Result<Value<'ctx, 'this>> {
) -> Result<Value<'ctx, 'this>> {
//    // const_type.inner_data Should be one of the following:
    // const_type.inner_data Should be one of the following:
//    // - A single value, if the inner type is a simple numeric type (e.g., `felt252`, `u32`,
    // - A single value, if the inner type is a simple numeric type (e.g., `felt252`, `u32`,
//    //   etc.).
    //   etc.).
//    // - A list of const types, if the inner type is a struct. The type of each const type must be
    // - A list of const types, if the inner type is a struct. The type of each const type must be
//    //   the same as the corresponding struct member type.
    //   the same as the corresponding struct member type.
//    // - A selector (a single value) followed by a const type, if the inner type is an enum. The
    // - A selector (a single value) followed by a const type, if the inner type is an enum. The
//    //   type of the const type must be the same as the corresponding enum variant type.
    //   type of the const type must be the same as the corresponding enum variant type.
//

//    let inner_type = registry.get_type(&info.inner_ty)?;
    let inner_type = registry.get_type(&info.inner_ty)?;
//    let inner_ty = registry.build_type(context, helper, registry, metadata, &info.inner_ty)?;
    let inner_ty = registry.build_type(context, helper, registry, metadata, &info.inner_ty)?;
//

//    match inner_type {
    match inner_type {
//        CoreTypeConcrete::Struct(_) => {
        CoreTypeConcrete::Struct(_) => {
//            let mut fields = Vec::new();
            let mut fields = Vec::new();
//

//            for field in &info.inner_data {
            for field in &info.inner_data {
//                match field {
                match field {
//                    GenericArg::Type(const_field_ty) => {
                    GenericArg::Type(const_field_ty) => {
//                        let field_type = registry.get_type(const_field_ty)?;
                        let field_type = registry.get_type(const_field_ty)?;
//

//                        let const_field_type = match &field_type {
                        let const_field_type = match &field_type {
//                            CoreTypeConcrete::Const(inner) => inner,
                            CoreTypeConcrete::Const(inner) => inner,
//                            _ => unreachable!(),
                            _ => unreachable!(),
//                        };
                        };
//

//                        let field_value = build_const_type_value(
                        let field_value = build_const_type_value(
//                            context,
                            context,
//                            registry,
                            registry,
//                            entry,
                            entry,
//                            location,
                            location,
//                            helper,
                            helper,
//                            metadata,
                            metadata,
//                            const_field_type,
                            const_field_type,
//                        )?;
                        )?;
//                        fields.push(field_value);
                        fields.push(field_value);
//                    }
                    }
//                    _ => return Err(Error::ConstDataMismatch),
                    _ => return Err(Error::ConstDataMismatch),
//                }
                }
//            }
            }
//

//            build_struct_value(
            build_struct_value(
//                context,
                context,
//                registry,
                registry,
//                entry,
                entry,
//                location,
                location,
//                helper,
                helper,
//                metadata,
                metadata,
//                &info.inner_ty,
                &info.inner_ty,
//                &fields,
                &fields,
//            )
            )
//        }
        }
//        CoreTypeConcrete::Enum(_enum_info) => match &info.inner_data[..] {
        CoreTypeConcrete::Enum(_enum_info) => match &info.inner_data[..] {
//            [GenericArg::Value(variant_index), GenericArg::Type(payload_ty)] => {
            [GenericArg::Value(variant_index), GenericArg::Type(payload_ty)] => {
//                let payload_type = registry.get_type(payload_ty)?;
                let payload_type = registry.get_type(payload_ty)?;
//                let const_payload_type = match payload_type {
                let const_payload_type = match payload_type {
//                    CoreTypeConcrete::Const(inner) => inner,
                    CoreTypeConcrete::Const(inner) => inner,
//                    _ => unreachable!(),
                    _ => unreachable!(),
//                };
                };
//

//                let payload_value = build_const_type_value(
                let payload_value = build_const_type_value(
//                    context,
                    context,
//                    registry,
                    registry,
//                    entry,
                    entry,
//                    location,
                    location,
//                    helper,
                    helper,
//                    metadata,
                    metadata,
//                    const_payload_type,
                    const_payload_type,
//                )?;
                )?;
//

//                build_enum_value(
                build_enum_value(
//                    context,
                    context,
//                    registry,
                    registry,
//                    entry,
                    entry,
//                    location,
                    location,
//                    helper,
                    helper,
//                    metadata,
                    metadata,
//                    payload_value,
                    payload_value,
//                    &info.inner_ty,
                    &info.inner_ty,
//                    payload_ty,
                    payload_ty,
//                    variant_index.try_into().unwrap(),
                    variant_index.try_into().unwrap(),
//                )
                )
//            }
            }
//            _ => Err(Error::ConstDataMismatch),
            _ => Err(Error::ConstDataMismatch),
//        },
        },
//        CoreTypeConcrete::NonZero(_) => match &info.inner_data[..] {
        CoreTypeConcrete::NonZero(_) => match &info.inner_data[..] {
//            [GenericArg::Type(_inner)] => {
            [GenericArg::Type(_inner)] => {
//                todo!()
                todo!()
//            }
            }
//            _ => Err(Error::ConstDataMismatch),
            _ => Err(Error::ConstDataMismatch),
//        },
        },
//        inner_type => match &info.inner_data[..] {
        inner_type => match &info.inner_data[..] {
//            [GenericArg::Value(value)] => {
            [GenericArg::Value(value)] => {
//                let mlir_value: Value = match inner_type {
                let mlir_value: Value = match inner_type {
//                    CoreTypeConcrete::Felt252(_) => {
                    CoreTypeConcrete::Felt252(_) => {
//                        let value = if value.sign() == num_bigint::Sign::Minus {
                        let value = if value.sign() == num_bigint::Sign::Minus {
//                            let prime = metadata
                            let prime = metadata
//                                .get::<PrimeModuloMeta<Felt>>()
                                .get::<PrimeModuloMeta<Felt>>()
//                                .ok_or(Error::MissingMetadata)?
                                .ok_or(Error::MissingMetadata)?
//                                .prime();
                                .prime();
//

//                            value + prime.to_bigint().expect("Prime to BigInt shouldn't fail")
                            value + prime.to_bigint().expect("Prime to BigInt shouldn't fail")
//                        } else {
                        } else {
//                            value.clone()
                            value.clone()
//                        };
                        };
//

//                        entry
                        entry
//                            .append_operation(arith::constant(
                            .append_operation(arith::constant(
//                                context,
                                context,
//                                Attribute::parse(context, &format!("{} : {}", value, inner_ty))
                                Attribute::parse(context, &format!("{} : {}", value, inner_ty))
//                                    .unwrap(),
                                    .unwrap(),
//                                location,
                                location,
//                            ))
                            ))
//                            .result(0)?
                            .result(0)?
//                            .into()
                            .into()
//                    }
                    }
//                    // any other int type
                    // any other int type
//                    _ => entry
                    _ => entry
//                        .append_operation(arith::constant(
                        .append_operation(arith::constant(
//                            context,
                            context,
//                            Attribute::parse(context, &format!("{} : {}", value, inner_ty))
                            Attribute::parse(context, &format!("{} : {}", value, inner_ty))
//                                .unwrap(),
                                .unwrap(),
//                            location,
                            location,
//                        ))
                        ))
//                        .result(0)?
                        .result(0)?
//                        .into(),
                        .into(),
//                };
                };
//

//                Ok(mlir_value)
                Ok(mlir_value)
//            }
            }
//            _ => Err(Error::ConstDataMismatch),
            _ => Err(Error::ConstDataMismatch),
//        },
        },
//    }
    }
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//pub mod test {
pub mod test {
//    use crate::{
    use crate::{
//        utils::test::{jit_struct, load_cairo, run_program},
        utils::test::{jit_struct, load_cairo, run_program},
//        values::JitValue,
        values::JitValue,
//    };
    };
//

//    #[test]
    #[test]
//    fn run_const_as_box() {
    fn run_const_as_box() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use core::box::BoxTrait;
            use core::box::BoxTrait;
//

//            struct Hello {
            struct Hello {
//                x: i32,
                x: i32,
//            }
            }
//

//            fn run_test() -> Hello {
            fn run_test() -> Hello {
//                let x = BoxTrait::new(Hello {
                let x = BoxTrait::new(Hello {
//                    x: -2
                    x: -2
//                });
                });
//                x.unbox()
                x.unbox()
//            }
            }
//        );
        );
//

//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//        assert_eq!(result, jit_struct!(JitValue::Sint32(-2)));
        assert_eq!(result, jit_struct!(JitValue::Sint32(-2)));
//    }
    }
//}
}
