////! # `Felt` dictionary type
//! # `Felt` dictionary type
////!
//!
////! A key value storage for values whose type implement Copy. The key is always a felt.
//! A key value storage for values whose type implement Copy. The key is always a felt.
////!
//!
////! This type is represented as a pointer to a tuple of a heap allocated Rust hashmap along with a u64
//! This type is represented as a pointer to a tuple of a heap allocated Rust hashmap along with a u64
////! used to count accesses to the dictionary. The type is interacted through the runtime functions to
//! used to count accesses to the dictionary. The type is interacted through the runtime functions to
////! insert, get elements and increment the access counter.
//! insert, get elements and increment the access counter.
//

//use super::WithSelf;
use super::WithSelf;
//use crate::{error::Result, metadata::MetadataStorage};
use crate::{error::Result, metadata::MetadataStorage};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        types::InfoAndTypeConcreteType,
        types::InfoAndTypeConcreteType,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::llvm,
    dialect::llvm,
//    ir::{Module, Type},
    ir::{Module, Type},
//    Context,
    Context,
//};
};
//

///// Build the MLIR type.
/// Build the MLIR type.
/////
///
///// Check out [the module](self) for more info.
/// Check out [the module](self) for more info.
//pub fn build<'ctx>(
pub fn build<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _module: &Module<'ctx>,
    _module: &Module<'ctx>,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: WithSelf<InfoAndTypeConcreteType>,
    _info: WithSelf<InfoAndTypeConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    Ok(llvm::r#type::pointer(context, 0))
    Ok(llvm::r#type::pointer(context, 0))
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::{
    use crate::{
//        utils::test::{load_cairo, run_program},
        utils::test::{load_cairo, run_program},
//        values::JitValue,
        values::JitValue,
//    };
    };
//    use pretty_assertions_sorted::assert_eq;
    use pretty_assertions_sorted::assert_eq;
//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//    use std::collections::HashMap;
    use std::collections::HashMap;
//

//    /// Ensure that a dictionary of booleans compiles.
    /// Ensure that a dictionary of booleans compiles.
//    #[test]
    #[test]
//    fn dict_type_bool() {
    fn dict_type_bool() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            fn run_program() -> Felt252Dict<bool> {
            fn run_program() -> Felt252Dict<bool> {
//                let mut x: Felt252Dict<bool> = Default::default();
                let mut x: Felt252Dict<bool> = Default::default();
//                x.insert(0, false);
                x.insert(0, false);
//                x.insert(1, true);
                x.insert(1, true);
//                x
                x
//            }
            }
//        };
        };
//

//        let result = run_program(&program, "run_program", &[]);
        let result = run_program(&program, "run_program", &[]);
//        assert_eq!(
        assert_eq!(
//            result.return_value,
            result.return_value,
//            JitValue::Felt252Dict {
            JitValue::Felt252Dict {
//                value: HashMap::from([
                value: HashMap::from([
//                    (
                    (
//                        Felt::ZERO,
                        Felt::ZERO,
//                        JitValue::Enum {
                        JitValue::Enum {
//                            tag: 0,
                            tag: 0,
//                            value: Box::new(JitValue::Struct {
                            value: Box::new(JitValue::Struct {
//                                fields: Vec::new(),
                                fields: Vec::new(),
//                                debug_name: None
                                debug_name: None
//                            }),
                            }),
//                            debug_name: None,
                            debug_name: None,
//                        },
                        },
//                    ),
                    ),
//                    (
                    (
//                        Felt::ONE,
                        Felt::ONE,
//                        JitValue::Enum {
                        JitValue::Enum {
//                            tag: 1,
                            tag: 1,
//                            value: Box::new(JitValue::Struct {
                            value: Box::new(JitValue::Struct {
//                                fields: Vec::new(),
                                fields: Vec::new(),
//                                debug_name: None
                                debug_name: None
//                            }),
                            }),
//                            debug_name: None,
                            debug_name: None,
//                        },
                        },
//                    ),
                    ),
//                ]),
                ]),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    /// Ensure that a dictionary of felts compiles.
    /// Ensure that a dictionary of felts compiles.
//    #[test]
    #[test]
//    fn dict_type_felt252() {
    fn dict_type_felt252() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            fn run_program() -> Felt252Dict<felt252> {
            fn run_program() -> Felt252Dict<felt252> {
//                let mut x: Felt252Dict<felt252> = Default::default();
                let mut x: Felt252Dict<felt252> = Default::default();
//                x.insert(0, 0);
                x.insert(0, 0);
//                x.insert(1, 1);
                x.insert(1, 1);
//                x.insert(2, 2);
                x.insert(2, 2);
//                x.insert(3, 3);
                x.insert(3, 3);
//                x
                x
//            }
            }
//        };
        };
//

//        let result = run_program(&program, "run_program", &[]);
        let result = run_program(&program, "run_program", &[]);
//        assert_eq!(
        assert_eq!(
//            result.return_value,
            result.return_value,
//            JitValue::Felt252Dict {
            JitValue::Felt252Dict {
//                value: HashMap::from([
                value: HashMap::from([
//                    (Felt::ZERO, JitValue::Felt252(Felt::ZERO)),
                    (Felt::ZERO, JitValue::Felt252(Felt::ZERO)),
//                    (Felt::ONE, JitValue::Felt252(Felt::ONE)),
                    (Felt::ONE, JitValue::Felt252(Felt::ONE)),
//                    (Felt::TWO, JitValue::Felt252(Felt::TWO)),
                    (Felt::TWO, JitValue::Felt252(Felt::TWO)),
//                    (Felt::THREE, JitValue::Felt252(Felt::THREE)),
                    (Felt::THREE, JitValue::Felt252(Felt::THREE)),
//                ]),
                ]),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    /// Ensure that a dictionary of nullables compiles.
    /// Ensure that a dictionary of nullables compiles.
//    #[test]
    #[test]
//    fn dict_type_nullable() {
    fn dict_type_nullable() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            #[derive(Drop)]
            #[derive(Drop)]
//            struct MyStruct {
            struct MyStruct {
//                a: u8,
                a: u8,
//                b: i16,
                b: i16,
//                c: felt252,
                c: felt252,
//            }
            }
//

//            fn run_program() -> Felt252Dict<Nullable<MyStruct>> {
            fn run_program() -> Felt252Dict<Nullable<MyStruct>> {
//                let mut x: Felt252Dict<Nullable<MyStruct>> = Default::default();
                let mut x: Felt252Dict<Nullable<MyStruct>> = Default::default();
//                x.insert(0, Default::default());
                x.insert(0, Default::default());
//                x.insert(1, NullableTrait::new(MyStruct { a: 0, b: 1, c: 2 }));
                x.insert(1, NullableTrait::new(MyStruct { a: 0, b: 1, c: 2 }));
//                x.insert(2, NullableTrait::new(MyStruct { a: 1, b: -2, c: 3 }));
                x.insert(2, NullableTrait::new(MyStruct { a: 1, b: -2, c: 3 }));
//                x.insert(3, NullableTrait::new(MyStruct { a: 2, b: 3, c: 4 }));
                x.insert(3, NullableTrait::new(MyStruct { a: 2, b: 3, c: 4 }));
//                x
                x
//            }
            }
//        };
        };
//

//        let result = run_program(&program, "run_program", &[]);
        let result = run_program(&program, "run_program", &[]);
//        pretty_assertions_sorted::assert_eq_sorted!(
        pretty_assertions_sorted::assert_eq_sorted!(
//            result.return_value,
            result.return_value,
//            JitValue::Felt252Dict {
            JitValue::Felt252Dict {
//                value: HashMap::from([
                value: HashMap::from([
//                    (Felt::ZERO, JitValue::Null),
                    (Felt::ZERO, JitValue::Null),
//                    (
                    (
//                        Felt::ONE,
                        Felt::ONE,
//                        JitValue::Struct {
                        JitValue::Struct {
//                            fields: Vec::from([
                            fields: Vec::from([
//                                JitValue::Uint8(0),
                                JitValue::Uint8(0),
//                                JitValue::Sint16(1),
                                JitValue::Sint16(1),
//                                JitValue::Felt252(2.into()),
                                JitValue::Felt252(2.into()),
//                            ]),
                            ]),
//                            debug_name: None,
                            debug_name: None,
//                        },
                        },
//                    ),
                    ),
//                    (
                    (
//                        Felt::TWO,
                        Felt::TWO,
//                        JitValue::Struct {
                        JitValue::Struct {
//                            fields: Vec::from([
                            fields: Vec::from([
//                                JitValue::Uint8(1),
                                JitValue::Uint8(1),
//                                JitValue::Sint16(-2),
                                JitValue::Sint16(-2),
//                                JitValue::Felt252(3.into()),
                                JitValue::Felt252(3.into()),
//                            ]),
                            ]),
//                            debug_name: None,
                            debug_name: None,
//                        },
                        },
//                    ),
                    ),
//                    (
                    (
//                        Felt::THREE,
                        Felt::THREE,
//                        JitValue::Struct {
                        JitValue::Struct {
//                            fields: Vec::from([
                            fields: Vec::from([
//                                JitValue::Uint8(2),
                                JitValue::Uint8(2),
//                                JitValue::Sint16(3),
                                JitValue::Sint16(3),
//                                JitValue::Felt252(4.into()),
                                JitValue::Felt252(4.into()),
//                            ]),
                            ]),
//                            debug_name: None,
                            debug_name: None,
//                        },
                        },
//                    ),
                    ),
//                ]),
                ]),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    /// Ensure that a dictionary of unsigned integers compiles.
    /// Ensure that a dictionary of unsigned integers compiles.
//    #[test]
    #[test]
//    fn dict_type_unsigned() {
    fn dict_type_unsigned() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            fn run_program() -> Felt252Dict<u128> {
            fn run_program() -> Felt252Dict<u128> {
//                let mut x: Felt252Dict<u128> = Default::default();
                let mut x: Felt252Dict<u128> = Default::default();
//                x.insert(0, 0_u128);
                x.insert(0, 0_u128);
//                x.insert(1, 1_u128);
                x.insert(1, 1_u128);
//                x.insert(2, 2_u128);
                x.insert(2, 2_u128);
//                x.insert(3, 3_u128);
                x.insert(3, 3_u128);
//                x
                x
//            }
            }
//        };
        };
//

//        let result = run_program(&program, "run_program", &[]);
        let result = run_program(&program, "run_program", &[]);
//        assert_eq!(
        assert_eq!(
//            result.return_value,
            result.return_value,
//            JitValue::Felt252Dict {
            JitValue::Felt252Dict {
//                value: HashMap::from([
                value: HashMap::from([
//                    (Felt::ZERO, JitValue::Uint128(0)),
                    (Felt::ZERO, JitValue::Uint128(0)),
//                    (Felt::ONE, JitValue::Uint128(1)),
                    (Felt::ONE, JitValue::Uint128(1)),
//                    (Felt::TWO, JitValue::Uint128(2)),
                    (Felt::TWO, JitValue::Uint128(2)),
//                    (Felt::THREE, JitValue::Uint128(3)),
                    (Felt::THREE, JitValue::Uint128(3)),
//                ]),
                ]),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//}
}
