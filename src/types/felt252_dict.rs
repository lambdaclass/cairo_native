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
    libfuncs::LibfuncHelper,
    metadata::{
        realloc_bindings::ReallocBindingsMeta, runtime_bindings::RuntimeBindingsMeta,
        snapshot_clones::SnapshotClonesMeta, MetadataStorage,
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
    dialect::{
        llvm::{self, r#type::pointer},
        ods, scf,
    },
    ir::{
        attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Region, Type,
        Value,
    },
    Context,
};
use std::cell::Cell;

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
    SnapshotClonesMeta::register_with(metadata, info.self_ty().clone(), |metadata| {
        registry.build_type(context, module, registry, metadata, &info.ty)?;

        Ok(Some((
            snapshot_take,
            InfoAndTypeConcreteType {
                info: info.info.clone(),
                ty: info.ty.clone(),
            },
        )))
    })?;

    Ok(llvm::r#type::pointer(context, 0))
}

#[allow(clippy::too_many_arguments)]
fn snapshot_take<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
    src_value: Value<'ctx, 'this>,
) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let elem_snapshot_take = metadata
        .get::<SnapshotClonesMeta>()
        .and_then(|meta| meta.wrap_invoke(&info.ty));

    let elem_ty = registry.get_type(&info.ty)?;
    let elem_layout = elem_ty.layout(registry)?;
    let elem_ty = elem_ty.build(context, helper, registry, metadata, &info.ty)?;

    let location = Location::name(context, "dict_snapshot_clone", location);

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let len_ptr = helper.init_block().alloca_int(context, location, 64)?;
    let u64_ty = IntegerType::new(context, 64).into();

    let entry_values_type = llvm::r#type::r#struct(
        context,
        &[IntegerType::new(context, 252).into(), pointer(context, 0)], // key, value ptr
        false,
    );

    // ptr to array of entry_values_type
    let entries_ptr = runtime_bindings
        .dict_values(context, helper, src_value, len_ptr, entry, location)?
        .result(0)?
        .into();

    let array_len = entry.load(context, location, len_ptr, u64_ty)?;

    let k0 = entry.const_int(context, location, 0, 64)?;
    let k1 = entry.const_int(context, location, 1, 64)?;
    let elem_stride_bytes =
        entry.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;
    let nullptr = entry.append_op_result(llvm::zero(pointer(context, 0), location))?;

    let cloned_dict_ptr = runtime_bindings
        .dict_alloc_new(context, helper, entry, location)?
        .result(0)?
        .into();

    entry.append_operation(scf::r#for(
        k0,
        array_len,
        k1,
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[(
                IntegerType::new(context, 64).into(),
                location,
            )]));

            let i = block.argument(0)?.into();
            block.append_operation(scf::execute_region(
                &[],
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    let entry_ptr = block.append_op_result(llvm::get_element_ptr_dynamic(
                        context,
                        entries_ptr,
                        &[i],
                        entry_values_type,
                        llvm::r#type::pointer(context, 0),
                        location,
                    ))?;

                    let helper = LibfuncHelper {
                        module: helper.module,
                        init_block: helper.init_block,
                        region: &region,
                        blocks_arena: helper.blocks_arena,
                        last_block: Cell::new(&block),
                        branches: Vec::new(),
                        results: Vec::new(),
                    };

                    let entry_value =
                        block.load(context, location, entry_ptr, entry_values_type)?;

                    let key = block.extract_value(
                        context,
                        location,
                        entry_value,
                        IntegerType::new(context, 252).into(),
                        0,
                    )?;
                    let key_ptr = helper.init_block().alloca_int(context, location, 252)?;
                    block.store(context, location, key_ptr, key)?;
                    let value_ptr = block.extract_value(
                        context,
                        location,
                        entry_value,
                        pointer(context, 0),
                        1,
                    )?;

                    match elem_snapshot_take {
                        Some(elem_snapshot_take) => {
                            let value = block.load(context, location, value_ptr, elem_ty)?;
                            let (block, cloned_value) = elem_snapshot_take(
                                context, registry, &block, location, &helper, metadata, value,
                            )?;

                            let cloned_value_ptr =
                                block.append_op_result(ReallocBindingsMeta::realloc(
                                    context,
                                    nullptr,
                                    elem_stride_bytes,
                                    location,
                                ))?;

                            block.store(context, location, cloned_value_ptr, cloned_value)?;

                            // needed due to mut borrow
                            let runtime_bindings = metadata
                                .get_mut::<RuntimeBindingsMeta>()
                                .expect("Runtime library not available.");
                            runtime_bindings.dict_insert(
                                context,
                                &helper,
                                block,
                                cloned_dict_ptr,
                                key_ptr,
                                cloned_value_ptr,
                                location,
                            )?;
                            block.append_operation(scf::r#yield(&[], location));
                        }
                        None => {
                            let cloned_value_ptr =
                                block.append_op_result(ReallocBindingsMeta::realloc(
                                    context,
                                    nullptr,
                                    elem_stride_bytes,
                                    location,
                                ))?;
                            block.append_operation(
                                ods::llvm::intr_memcpy(
                                    context,
                                    cloned_value_ptr,
                                    value_ptr,
                                    elem_stride_bytes,
                                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                                    location,
                                )
                                .into(),
                            );
                            runtime_bindings.dict_insert(
                                context,
                                &helper,
                                &block,
                                cloned_dict_ptr,
                                key_ptr,
                                cloned_value_ptr,
                                location,
                            )?;
                            block.append_operation(scf::r#yield(&[], location));
                        }
                    }

                    region
                },
                location,
            ));

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        location,
    ));

    Ok((entry, cloned_dict_ptr))
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
