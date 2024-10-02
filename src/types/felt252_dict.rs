//! # `Felt` dictionary type
//!
//! A key value storage for values whose type implement Copy. The key is always a felt.
//!
//! This type is represented as a pointer to a tuple of a heap allocated Rust hashmap along with a u64
//! used to count accesses to the dictionary. The type is interacted through the runtime functions to
//! insert, get elements and increment the access counter.

use super::WithSelf;
use crate::{
    error::Result,
    metadata::{
        dup_overrides::DupOverridesMeta, realloc_bindings::ReallocBindingsMeta,
        runtime_bindings::RuntimeBindingsMeta, MetadataStorage,
    },
    types::TypeBuilder,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{func, llvm, ods},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        Block, Location, Module, Region, Type,
    },
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    DupOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `snapshot_take`.

            Ok(Some(build_dup(context, module, registry, metadata, &info)?))
        },
    )?;

    Ok(llvm::r#type::pointer(context, 0))
}

fn build_dup<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: &WithSelf<InfoAndTypeConcreteType>,
) -> Result<Region<'ctx>> {
    let location = Location::unknown(context);
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, module));
    }

    let value_ty = registry.build_type(context, module, registry, metadata, info.self_ty())?;
    let inner_ty = registry.get_type(&info.ty)?;
    let inner_len = inner_ty.layout(registry)?.pad_to_align().size();
    let inner_ty = inner_ty.build(context, module, registry, metadata, &info.ty)?;

    let dup_fn_symbol = format!("dup${}$item", info.self_ty().id);
    {
        let region = Region::new();
        let entry =
            region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));

        let null_ptr =
            entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
        let inner_len = entry.const_int(context, location, inner_len, 64)?;

        let old_ptr = entry.argument(0)?.into();
        let new_ptr = entry.append_op_result(ReallocBindingsMeta::realloc(
            context, null_ptr, inner_len, location,
        ))?;

        let value = entry.load(context, location, old_ptr, inner_ty)?;
        let values = metadata
            .get_or_insert_with(DupOverridesMeta::default)
            .invoke_override(context, &entry, location, &info.ty, value)?;

        entry.store(context, location, old_ptr, values.0)?;
        entry.store(context, location, new_ptr, values.1)?;

        entry.append_operation(llvm::r#return(Some(new_ptr), location));

        module.body().append_operation(llvm::func(
            context,
            StringAttribute::new(context, &dup_fn_symbol),
            TypeAttribute::new(llvm::r#type::function(
                llvm::r#type::pointer(context, 0),
                &[llvm::r#type::pointer(context, 0)],
                false,
            )),
            region,
            &[],
            location,
        ));
    }

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    let dup_fn = entry.append_op_result(
        ods::llvm::mlir_addressof(
            context,
            llvm::r#type::pointer(context, 0),
            FlatSymbolRefAttribute::new(context, &dup_fn_symbol),
            location,
        )
        .into(),
    )?;

    let runtime_bindings_meta = metadata.get_mut::<RuntimeBindingsMeta>().unwrap();
    let value0 = entry.argument(0)?.into();
    let value1 =
        runtime_bindings_meta.dict_dup(context, module, &entry, value0, dup_fn, location)?;

    entry.append_operation(func::r#return(&[value0, value1], location));
    Ok(region)
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_dict, load_cairo, run_program},
        values::Value,
    };
    use pretty_assertions_sorted::assert_eq;
    use starknet_types_core::felt::Felt;
    use std::collections::HashMap;

    #[test]
    fn dict_snapshot_take() {
        let program = load_cairo! {
            fn run_test() -> @Felt252Dict<u32> {
                let mut dict: Felt252Dict<u32> = Default::default();
                            dict.insert(2, 1_u32);

                @dict
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_dict!(
                2 => 1u32
            ),
        );
    }

    #[test]
    fn dict_snapshot_take_complex() {
        let program = load_cairo! {
            fn run_test() -> @Felt252Dict<Nullable<Array<u32>>> {
                let mut dict: Felt252Dict<Nullable<Array<u32>>> = Default::default();
                dict.insert(2, NullableTrait::new(array![3, 4]));

                @dict
            }

        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_dict!(
                2 => Value::Array(vec![3u32.into(), 4u32.into()])
            ),
        );
    }

    #[test]
    fn dict_snapshot_take_compare() {
        let program = load_cairo! {
            fn run_test() -> @Felt252Dict<Nullable<Array<u32>>> {
                let mut dict: Felt252Dict<Nullable<Array<u32>>> = Default::default();
                dict.insert(2, NullableTrait::new(array![3, 4]));

                @dict
            }

        };
        let program2 = load_cairo! {
            fn run_test() -> Felt252Dict<Nullable<Array<u32>>> {
                let mut dict: Felt252Dict<Nullable<Array<u32>>> = Default::default();
                dict.insert(2, NullableTrait::new(array![3, 4]));

                dict
            }

        };
        let result1 = run_program(&program, "run_test", &[]).return_value;
        let result2 = run_program(&program2, "run_test", &[]).return_value;

        assert_eq!(result1, result2);
    }

    /// Ensure that a dictionary of booleans compiles.
    #[test]
    fn dict_type_bool() {
        let program = load_cairo! {
            fn run_program() -> Felt252Dict<bool> {
                let mut x: Felt252Dict<bool> = Default::default();
                x.insert(0, false);
                x.insert(1, true);
                x
            }
        };

        let result = run_program(&program, "run_program", &[]);
        assert_eq!(
            result.return_value,
            Value::Felt252Dict {
                value: HashMap::from([
                    (
                        Felt::ZERO,
                        Value::Enum {
                            tag: 0,
                            value: Box::new(Value::Struct {
                                fields: Vec::new(),
                                debug_name: None
                            }),
                            debug_name: None,
                        },
                    ),
                    (
                        Felt::ONE,
                        Value::Enum {
                            tag: 1,
                            value: Box::new(Value::Struct {
                                fields: Vec::new(),
                                debug_name: None
                            }),
                            debug_name: None,
                        },
                    ),
                ]),
                debug_name: None,
            },
        );
    }

    /// Ensure that a dictionary of felts compiles.
    #[test]
    fn dict_type_felt252() {
        let program = load_cairo! {
            fn run_program() -> Felt252Dict<felt252> {
                let mut x: Felt252Dict<felt252> = Default::default();
                x.insert(0, 0);
                x.insert(1, 1);
                x.insert(2, 2);
                x.insert(3, 3);
                x
            }
        };

        let result = run_program(&program, "run_program", &[]);
        assert_eq!(
            result.return_value,
            Value::Felt252Dict {
                value: HashMap::from([
                    (Felt::ZERO, Value::Felt252(Felt::ZERO)),
                    (Felt::ONE, Value::Felt252(Felt::ONE)),
                    (Felt::TWO, Value::Felt252(Felt::TWO)),
                    (Felt::THREE, Value::Felt252(Felt::THREE)),
                ]),
                debug_name: None,
            },
        );
    }

    /// Ensure that a dictionary of nullables compiles.
    #[test]
    fn dict_type_nullable() {
        let program = load_cairo! {
            #[derive(Drop)]
            struct MyStruct {
                a: u8,
                b: i16,
                c: felt252,
            }

            fn run_program() -> Felt252Dict<Nullable<MyStruct>> {
                let mut x: Felt252Dict<Nullable<MyStruct>> = Default::default();
                x.insert(0, Default::default());
                x.insert(1, NullableTrait::new(MyStruct { a: 0, b: 1, c: 2 }));
                x.insert(2, NullableTrait::new(MyStruct { a: 1, b: -2, c: 3 }));
                x.insert(3, NullableTrait::new(MyStruct { a: 2, b: 3, c: 4 }));
                x
            }
        };

        let result = run_program(&program, "run_program", &[]);
        pretty_assertions_sorted::assert_eq_sorted!(
            result.return_value,
            Value::Felt252Dict {
                value: HashMap::from([
                    (Felt::ZERO, Value::Null),
                    (
                        Felt::ONE,
                        Value::Struct {
                            fields: Vec::from([
                                Value::Uint8(0),
                                Value::Sint16(1),
                                Value::Felt252(2.into()),
                            ]),
                            debug_name: None,
                        },
                    ),
                    (
                        Felt::TWO,
                        Value::Struct {
                            fields: Vec::from([
                                Value::Uint8(1),
                                Value::Sint16(-2),
                                Value::Felt252(3.into()),
                            ]),
                            debug_name: None,
                        },
                    ),
                    (
                        Felt::THREE,
                        Value::Struct {
                            fields: Vec::from([
                                Value::Uint8(2),
                                Value::Sint16(3),
                                Value::Felt252(4.into()),
                            ]),
                            debug_name: None,
                        },
                    ),
                ]),
                debug_name: None,
            },
        );
    }

    /// Ensure that a dictionary of unsigned integers compiles.
    #[test]
    fn dict_type_unsigned() {
        let program = load_cairo! {
            fn run_program() -> Felt252Dict<u128> {
                let mut x: Felt252Dict<u128> = Default::default();
                x.insert(0, 0_u128);
                x.insert(1, 1_u128);
                x.insert(2, 2_u128);
                x.insert(3, 3_u128);
                x
            }
        };

        let result = run_program(&program, "run_program", &[]);
        assert_eq!(
            result.return_value,
            Value::Felt252Dict {
                value: HashMap::from([
                    (Felt::ZERO, Value::Uint128(0)),
                    (Felt::ONE, Value::Uint128(1)),
                    (Felt::TWO, Value::Uint128(2)),
                    (Felt::THREE, Value::Uint128(3)),
                ]),
                debug_name: None,
            },
        );
    }
}
