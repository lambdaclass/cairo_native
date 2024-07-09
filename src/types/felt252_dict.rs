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
        runtime_bindings::RuntimeBindingsMeta, snapshot_clones::SnapshotClonesMeta, MetadataStorage,
    },
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{Block, Location, Module, Type, Value},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    metadata
        .get_or_insert_with::<SnapshotClonesMeta>(SnapshotClonesMeta::default)
        .register(
            info.self_ty().clone(),
            snapshot_take,
            InfoAndTypeConcreteType {
                info: info.info.clone(),
                ty: info.ty.clone(),
            },
        );

    Ok(llvm::r#type::pointer(context, 0))
}

#[allow(clippy::too_many_arguments)]
fn snapshot_take<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: WithSelf<InfoAndTypeConcreteType>,
    src_value: Value<'ctx, 'this>,
) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)> {
    // This snapshot take assumes the value types in a felt are always copy and don't need deep cloning.
    // So far (atleast up to 2.6.4) you cannot make a felt dict of array values, so this holds.

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let new_ptr = runtime_bindings
        .dict_clone(context, helper, src_value, entry, location)?
        .result(0)?
        .into();

    Ok((entry, new_ptr))
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_dict, load_cairo, run_program},
        values::JitValue,
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
            JitValue::Felt252Dict {
                value: HashMap::from([
                    (
                        Felt::ZERO,
                        JitValue::Enum {
                            tag: 0,
                            value: Box::new(JitValue::Struct {
                                fields: Vec::new(),
                                debug_name: None
                            }),
                            debug_name: None,
                        },
                    ),
                    (
                        Felt::ONE,
                        JitValue::Enum {
                            tag: 1,
                            value: Box::new(JitValue::Struct {
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
            JitValue::Felt252Dict {
                value: HashMap::from([
                    (Felt::ZERO, JitValue::Felt252(Felt::ZERO)),
                    (Felt::ONE, JitValue::Felt252(Felt::ONE)),
                    (Felt::TWO, JitValue::Felt252(Felt::TWO)),
                    (Felt::THREE, JitValue::Felt252(Felt::THREE)),
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
            JitValue::Felt252Dict {
                value: HashMap::from([
                    (Felt::ZERO, JitValue::Null),
                    (
                        Felt::ONE,
                        JitValue::Struct {
                            fields: Vec::from([
                                JitValue::Uint8(0),
                                JitValue::Sint16(1),
                                JitValue::Felt252(2.into()),
                            ]),
                            debug_name: None,
                        },
                    ),
                    (
                        Felt::TWO,
                        JitValue::Struct {
                            fields: Vec::from([
                                JitValue::Uint8(1),
                                JitValue::Sint16(-2),
                                JitValue::Felt252(3.into()),
                            ]),
                            debug_name: None,
                        },
                    ),
                    (
                        Felt::THREE,
                        JitValue::Struct {
                            fields: Vec::from([
                                JitValue::Uint8(2),
                                JitValue::Sint16(3),
                                JitValue::Felt252(4.into()),
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
            JitValue::Felt252Dict {
                value: HashMap::from([
                    (Felt::ZERO, JitValue::Uint128(0)),
                    (Felt::ONE, JitValue::Uint128(1)),
                    (Felt::TWO, JitValue::Uint128(2)),
                    (Felt::THREE, JitValue::Uint128(3)),
                ]),
                debug_name: None,
            },
        );
    }
}
