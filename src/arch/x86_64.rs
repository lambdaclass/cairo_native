#![cfg(target_arch = "x86_64")]

use crate::{types::TypeBuilder, utils::get_integer_layout, values::JitValue};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::StarkNetTypeConcrete,
    },
    ids::ConcreteTypeId,
    program_registry::{ProgramRegistry, ProgramRegistryError},
};
use starknet_types_core::felt::Felt;
use std::ptr::null_mut;

pub struct ArgumentMapper<'a> {
    arena: &'a Bump,
    registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,

    invoke_data: Vec<u64>,
}

impl<'a> ArgumentMapper<'a> {
    pub fn new(arena: &'a Bump, registry: &'a ProgramRegistry<CoreType, CoreLibfunc>) -> Self {
        Self {
            arena,
            registry,
            invoke_data: Vec::new(),
        }
    }

    pub fn invoke_data(&self) -> &[u64] {
        &self.invoke_data
    }

    pub fn push_aligned_u64_values(&mut self, align: usize, mut values: &[u64]) {
        assert!(align.is_power_of_two());
        assert!(align <= 16);

        if align == 16 {
            // This works because on both aarch64 and x86_64 the stack is already aligned to
            // 16 bytes when the trampoline starts pushing values.

            // Whenever a value spans across multiple registers, if it's in a position where it would be split between
            // registers and the stack it must be padded so that the entire value is stored within the stack.
            if self.invoke_data.len() >= 6 {
                if self.invoke_data.len() & 1 != 0 {
                    self.invoke_data.push(0);
                }
            } else if self.invoke_data.len() + 1 >= 6 {
                self.invoke_data.push(0);
            } else {
                let new_len = self.invoke_data.len() + values.len();
                if new_len >= 6 && new_len % 2 != 0 {
                    let chunk;
                    (chunk, values) = if values.len() >= 4 {
                        values.split_at(4)
                    } else {
                        (values, [].as_slice())
                    };
                    self.invoke_data.extend(chunk);
                    self.invoke_data.push(0);
                }
            }
        }

        self.invoke_data.extend(values);
    }

    pub fn push(
        &mut self,
        type_id: &ConcreteTypeId,
        type_info: &CoreTypeConcrete,
        value: &JitValue,
    ) -> Result<(), Box<ProgramRegistryError>> {
        match (type_info, value) {
            (CoreTypeConcrete::Array(info), JitValue::Array(values)) => {
                // TODO: Assert that `info.ty` matches all the values' types.

                let type_info = self.registry.get_type(&info.ty)?;
                let type_layout = type_info.layout(self.registry).unwrap().pad_to_align();

                // This needs to be a heap-allocated pointer because it's the actual array data.
                let ptr = if values.is_empty() {
                    null_mut()
                } else {
                    unsafe { libc::realloc(null_mut(), type_layout.size() * values.len()) }
                };

                for (idx, value) in values.iter().enumerate() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            value
                                .to_jit(self.arena, self.registry, &info.ty)
                                .unwrap()
                                .cast()
                                .as_ptr(),
                            (ptr as usize + type_layout.size() * idx) as *mut u8,
                            type_layout.size(),
                        );
                    }
                }

                self.push_aligned(
                    get_integer_layout(64).align(),
                    &[ptr as u64, 0, values.len() as u64, values.len() as u64],
                );
            }
            (CoreTypeConcrete::EcPoint(_), JitValue::EcPoint(a, b)) => {
                let align = get_integer_layout(252).align();
                self.push_aligned(align, &a.to_le_digits());
                self.push_aligned(align, &b.to_le_digits());
            }
            (CoreTypeConcrete::EcState(_), JitValue::EcState(a, b, c, d)) => {
                let align = get_integer_layout(252).align();
                self.push_aligned(align, &a.to_le_digits());
                self.push_aligned(align, &b.to_le_digits());
                self.push_aligned(align, &c.to_le_digits());
                self.push_aligned(align, &d.to_le_digits());
            }
            (CoreTypeConcrete::Enum(info), JitValue::Enum { tag, value, .. }) => {
                if type_info.is_memory_allocated(self.registry) {
                    let (layout, tag_layout, variant_layouts) =
                        crate::types::r#enum::get_layout_for_variants(
                            self.registry,
                            &info.variants,
                        )
                        .unwrap();

                    let ptr = self.arena.alloc_layout(layout);
                    unsafe {
                        match tag_layout.size() {
                            0 => {}
                            1 => *ptr.cast::<u8>().as_mut() = *tag as u8,
                            2 => *ptr.cast::<u16>().as_mut() = *tag as u16,
                            4 => *ptr.cast::<u32>().as_mut() = *tag as u32,
                            8 => *ptr.cast::<u64>().as_mut() = *tag as u64,
                            _ => unreachable!(),
                        }
                    }

                    let offset = tag_layout.extend(variant_layouts[*tag]).unwrap().1;
                    let payload_ptr = value
                        .to_jit(self.arena, self.registry, &info.variants[*tag])
                        .unwrap();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            payload_ptr.cast::<u8>().as_ptr(),
                            ptr.cast::<u8>().as_ptr().add(offset),
                            variant_layouts[*tag].size(),
                        );
                    }

                    self.invoke_data.push(ptr.as_ptr() as u64);
                } else {
                    // Write the tag.
                    match (info.variants.len().next_power_of_two().trailing_zeros() + 7) / 8 {
                        0 => {}
                        _ => self.invoke_data.push(*tag as u64),
                    }

                    // Write the payload.
                    let type_info = self.registry.get_type(&info.variants[*tag]).unwrap();
                    self.push(&info.variants[*tag], type_info, value)?;
                }
            }
            (
                CoreTypeConcrete::Felt252(_)
                | CoreTypeConcrete::StarkNet(
                    StarkNetTypeConcrete::ClassHash(_)
                    | StarkNetTypeConcrete::ContractAddress(_)
                    | StarkNetTypeConcrete::StorageAddress(_)
                    | StarkNetTypeConcrete::StorageBaseAddress(_),
                ),
                JitValue::Felt252(value),
            ) => {
                self.push_aligned(get_integer_layout(252).align(), &value.to_le_digits());
            }
            (CoreTypeConcrete::Bytes31(_), JitValue::Bytes31(value)) => {
                self.push_aligned(
                    get_integer_layout(248).align(),
                    &Felt::from_bytes_be_slice(value).to_le_digits(),
                );
            }
            (CoreTypeConcrete::Felt252Dict(_), JitValue::Felt252Dict { .. }) => {
                #[cfg(not(feature = "with-runtime"))]
                unimplemented!("enable the `with-runtime` feature to use felt252 dicts");

                // TODO: Assert that `info.ty` matches all the values' types.

                self.invoke_data.push(
                    value
                        .to_jit(self.arena, self.registry, type_id)
                        .unwrap()
                        .as_ptr() as u64,
                );
            }
            (CoreTypeConcrete::Struct(info), JitValue::Struct { fields, .. }) => {
                for (field_type_id, field_value) in info.members.iter().zip(fields) {
                    self.push(
                        field_type_id,
                        self.registry.get_type(field_type_id)?,
                        field_value,
                    )?;
                }
            }
            (CoreTypeConcrete::Uint128(_), JitValue::Uint128(value)) => self.push_aligned(
                get_integer_layout(128).align(),
                &[*value as u64, (value >> 64) as u64],
            ),
            (CoreTypeConcrete::Uint64(_), JitValue::Uint64(value)) => {
                self.push_aligned(get_integer_layout(64).align(), &[*value]);
            }
            (CoreTypeConcrete::Uint32(_), JitValue::Uint32(value)) => {
                self.push_aligned(get_integer_layout(32).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Uint16(_), JitValue::Uint16(value)) => {
                self.push_aligned(get_integer_layout(16).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Uint8(_), JitValue::Uint8(value)) => {
                self.push_aligned(get_integer_layout(8).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Sint128(_), JitValue::Sint128(value)) => {
                self.push_aligned(
                    get_integer_layout(128).align(),
                    &[*value as u64, (value >> 64) as u64],
                );
            }
            (CoreTypeConcrete::Sint64(_), JitValue::Sint64(value)) => {
                self.push_aligned(get_integer_layout(64).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Sint32(_), JitValue::Sint32(value)) => {
                self.push_aligned(get_integer_layout(32).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Sint16(_), JitValue::Sint16(value)) => {
                self.push_aligned(get_integer_layout(16).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Sint8(_), JitValue::Sint8(value)) => {
                self.push_aligned(get_integer_layout(8).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::NonZero(info), _) => {
                // TODO: Check that the value is indeed non-zero.
                let type_info = self.registry.get_type(&info.ty)?;
                self.push(&info.ty, type_info, value)?;
            }
            (CoreTypeConcrete::Snapshot(info), _) => {
                let type_info = self.registry.get_type(&info.ty)?;
                self.push(&info.ty, type_info, value)?;
            }
            (
                CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::Secp256Point(_)),
                JitValue::Secp256K1Point { x, y } | JitValue::Secp256R1Point { x, y },
            ) => {
                let x_data = unsafe { std::mem::transmute::<[u128; 2], [u64; 4]>([x.0, x.1]) };
                let y_data = unsafe { std::mem::transmute::<[u128; 2], [u64; 4]>([y.0, y.1]) };

                self.push_aligned(get_integer_layout(252).align(), &x_data);
                self.push_aligned(get_integer_layout(252).align(), &y_data);
            }
            (CoreTypeConcrete::Bitwise(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::BuiltinCosts(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::EcOp(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::Pedersen(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::Poseidon(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::RangeCheck(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::SegmentArena(_), JitValue::Uint64(value)) => {
                self.push_aligned(get_integer_layout(64).align(), &[*value])
            }
            _ => todo!(),
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cairo_lang_sierra::extensions::types::InfoOnlyConcreteType;
    use cairo_lang_sierra::extensions::types::TypeInfo;
    use cairo_lang_sierra::program::ConcreteTypeLongId;
    use cairo_lang_sierra::ProgramParser;

    #[test]
    fn test_argument_mapper_push_sint8() {
        let program = ProgramParser::new().parse("").unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let bump = Bump::new();
        let mut argument_mapper = ArgumentMapper::new(&bump, &registry);

        let type_id = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        let type_info = CoreTypeConcrete::Sint8(InfoOnlyConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
        });

        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint8(12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint8(0));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint8(-12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint8(i8::MIN));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint8(i8::MAX));

        assert_eq!(
            argument_mapper.invoke_data,
            vec![12, 0, (-12_i8) as u64, i8::MIN as u64, i8::MAX as u64]
        );
    }

    #[test]
    fn test_argument_mapper_push_sint16() {
        let program = ProgramParser::new().parse("").unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let bump = Bump::new();
        let mut argument_mapper = ArgumentMapper::new(&bump, &registry);

        let type_id = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        let type_info = CoreTypeConcrete::Sint16(InfoOnlyConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
        });

        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint16(12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint16(0));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint16(-12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint16(i16::MIN));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint16(i16::MAX));

        assert_eq!(
            argument_mapper.invoke_data,
            vec![12, 0, (-12_i16) as u64, i16::MIN as u64, i16::MAX as u64]
        );
    }

    #[test]
    fn test_argument_mapper_push_sint32() {
        let program = ProgramParser::new().parse("").unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let bump = Bump::new();
        let mut argument_mapper = ArgumentMapper::new(&bump, &registry);

        let type_id = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        let type_info = CoreTypeConcrete::Sint32(InfoOnlyConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
        });

        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint32(12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint32(0));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint32(-12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint32(i32::MIN));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint32(i32::MAX));

        assert_eq!(
            argument_mapper.invoke_data,
            vec![12, 0, (-12_i32) as u64, i32::MIN as u64, i32::MAX as u64]
        );
    }

    #[test]
    fn test_argument_mapper_push_sint64() {
        let program = ProgramParser::new().parse("").unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let bump = Bump::new();
        let mut argument_mapper = ArgumentMapper::new(&bump, &registry);

        let type_id = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        let type_info = CoreTypeConcrete::Sint64(InfoOnlyConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
        });

        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint64(12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint64(0));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint64(-12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint64(i64::MIN));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint64(i64::MAX));

        assert_eq!(
            argument_mapper.invoke_data,
            vec![12, 0, (-12_i64) as u64, i64::MIN as u64, i64::MAX as u64]
        );
    }

    #[test]
    fn test_argument_mapper_push_sint128() {
        let program = ProgramParser::new().parse("").unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let bump = Bump::new();
        let mut argument_mapper = ArgumentMapper::new(&bump, &registry);

        let type_id = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        let type_info = CoreTypeConcrete::Sint128(InfoOnlyConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
        });

        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint128(12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint128(0));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint128(-12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint128(i128::MIN));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Sint128(i128::MAX));

        assert_eq!(
            argument_mapper.invoke_data,
            vec![
                12,
                0,
                0,
                0,
                (-12_i128) as u64,
                ((-12_i128) as u128 >> 64) as u64,
                i128::MIN as u64,
                (i128::MIN as u128 >> 64) as u64,
                i128::MAX as u64,
                (i128::MAX as u128 >> 64) as u64,
            ]
        );
    }

    #[test]
    fn test_argument_mapper_push_uint8() {
        let program = ProgramParser::new().parse("").unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let bump = Bump::new();
        let mut argument_mapper = ArgumentMapper::new(&bump, &registry);

        let type_id = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        let type_info = CoreTypeConcrete::Uint8(InfoOnlyConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
        });

        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint8(12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint8(0));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint8(u8::MIN));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint8(u8::MAX));

        assert_eq!(argument_mapper.invoke_data, vec![12, 0, 0, 0xFF]);
    }

    #[test]
    fn test_argument_mapper_push_uint16() {
        let program = ProgramParser::new().parse("").unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let bump = Bump::new();
        let mut argument_mapper = ArgumentMapper::new(&bump, &registry);

        let type_id = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        let type_info = CoreTypeConcrete::Uint16(InfoOnlyConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
        });

        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint16(12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint16(0));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint16(u16::MIN));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint16(u16::MAX));

        assert_eq!(argument_mapper.invoke_data, vec![12, 0, 0, 0xFFFF]);
    }

    #[test]
    fn test_argument_mapper_push_uint32() {
        let program = ProgramParser::new().parse("").unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let bump = Bump::new();
        let mut argument_mapper = ArgumentMapper::new(&bump, &registry);

        let type_id = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        let type_info = CoreTypeConcrete::Uint32(InfoOnlyConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
        });

        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint32(12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint32(0));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint32(u32::MIN));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint32(u32::MAX));

        assert_eq!(argument_mapper.invoke_data, vec![12, 0, 0, 0xFFFFFFFF]);
    }

    #[test]
    fn test_argument_mapper_push_uint64() {
        let program = ProgramParser::new().parse("").unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let bump = Bump::new();
        let mut argument_mapper = ArgumentMapper::new(&bump, &registry);

        let type_id = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        let type_info = CoreTypeConcrete::Uint64(InfoOnlyConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
        });

        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint64(12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint64(0));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint64(u64::MIN));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint64(u64::MAX));

        assert_eq!(
            argument_mapper.invoke_data,
            vec![12, 0, 0, 0xFFFFFFFFFFFFFFFF]
        );
    }

    #[test]
    fn test_argument_mapper_push_uint128() {
        let program = ProgramParser::new().parse("").unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let bump = Bump::new();
        let mut argument_mapper = ArgumentMapper::new(&bump, &registry);

        let type_id = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        let type_info = CoreTypeConcrete::Uint128(InfoOnlyConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
        });

        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint128(12));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint128(0));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint128(u128::MIN));
        let _ = argument_mapper.push(&type_id, &type_info, &JitValue::Uint128(u128::MAX));

        assert_eq!(
            argument_mapper.invoke_data,
            vec![12, 0, 0, 0, 0, 0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF]
        );
    }
}
