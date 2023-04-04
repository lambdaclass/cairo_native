use cairo_lang_sierra::program::GenericArg;
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::Type;
use tracing::debug;

use crate::compiler::{Compiler, SierraType, Storage};

pub const DEFAULT_PRIME: &str =
    "3618502788666131213697322783095070105623107215331596699973092056135872020481";

impl<'ctx> Compiler<'ctx> {
    pub fn process_types(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        for type_decl in &self.program.type_declarations {
            let id = type_decl.id.id;
            let name = type_decl.long_id.generic_id.0.as_str();
            debug!(name, "processing type decl");

            match name {
                "Bitwise" => {
                    let ty = self.bitwise_type();
                    storage.types.insert(id.to_string(), SierraType::Simple(ty));
                }
                "RangeCheck" => {
                    let ty = self.range_check_type();
                    storage.types.insert(id.to_string(), SierraType::Simple(ty));
                }
                "felt252" => {
                    let ty = self.felt_type();
                    storage.types.insert(id.to_string(), SierraType::Simple(ty));
                }
                "NonZero" => {
                    let mut types = vec![];

                    for gen_arg in &type_decl.long_id.generic_args {
                        let gen_arg_ty = match gen_arg {
                            GenericArg::Type(gen_arg_typeid) => storage
                                .types
                                .get(&gen_arg_typeid.id.to_string())
                                .expect("type should exist"),
                            _ => todo!(),
                        };
                        types.push(gen_arg_ty.clone());
                    }

                    if types.len() == 1 {
                        let ty = &types[0];
                        storage.types.insert(id.to_string(), ty.clone());
                    } else {
                        let struct_types = types.iter().map(SierraType::get_type).collect_vec();
                        let struct_type =
                            Type::parse(&self.context, &self.struct_type_string(&struct_types))
                                .unwrap();

                        storage.types.insert(
                            id.to_string(),
                            SierraType::Struct { ty: struct_type, field_types: types },
                        );
                    }
                }
                "Struct" => {
                    let mut types = vec![];

                    let _user_type = match &type_decl.long_id.generic_args[0] {
                        GenericArg::UserType(x) => x,
                        _ => {
                            unreachable!("first arg on struct libfunc should always be a user type")
                        }
                    };

                    for gen_arg in type_decl.long_id.generic_args.iter().skip(1) {
                        let gen_arg_ty = match gen_arg {
                            GenericArg::Type(gen_arg_typeid) => storage
                                .types
                                .get(&gen_arg_typeid.id.to_string())
                                .expect("type should exist"),
                            GenericArg::UserType(user_type_id) => storage
                                .types
                                .get(&user_type_id.id.to_string())
                                .expect("type should exist"),
                            _ => todo!(),
                        };
                        types.push(gen_arg_ty.clone());
                    }

                    let struct_types = types.iter().map(SierraType::get_type).collect_vec();
                    let struct_type =
                        Type::parse(&self.context, &self.struct_type_string(&struct_types))
                            .unwrap();

                    storage.types.insert(
                        id.to_string(),
                        SierraType::Struct { ty: struct_type, field_types: types },
                    );
                }
                "Enum" => {
                    // https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/en/latest/basic-constructs/unions.html#tagged-unions

                    // Get the SierraType for the data of each enum variant
                    let enum_variant_types = type_decl
                        .long_id
                        .generic_args
                        .iter()
                        .skip(1)
                        .map(|gen_arg| {
                            let gen_arg_type_id = match gen_arg {
                                GenericArg::Type(gen_arg_typeid) => gen_arg_typeid.id.to_string(),
                                _ => unreachable!("should always be a type"),
                            };
                            storage.types.get(&gen_arg_type_id).cloned().expect("type should exist")
                        })
                        .collect_vec();

                    // Ceiling division. This is the number of bytes required to store the data of the largest enum variant
                    let data_width =
                        (enum_variant_types.iter().map(SierraType::get_width).max().unwrap_or(0)
                            + 7)
                            / 8;

                    let enum_memory_array =
                        Type::parse(&self.context, &format!("!llvm.array<{data_width} x i8>",))
                            .expect("create the enum storage type.");

                    // for now the tag is a u16 = 65535 variants.
                    // TODO: make the tag size variable?
                    let enum_type = Type::parse(
                        &self.context,
                        &self.struct_type_string(&[self.u16_type(), enum_memory_array]),
                    )
                    .expect("error making enum type");

                    let enum_sierra_type = SierraType::Enum {
                        ty: enum_type,
                        tag_type: self.u16_type(),
                        storage_bytes_len: data_width,
                        storage_type: enum_memory_array,
                        variants_types: enum_variant_types,
                    };

                    storage.types.insert(id.to_string(), enum_sierra_type);
                }
                "Array" => {
                    let array_value_type = match &type_decl.long_id.generic_args[0] {
                        GenericArg::Type(x) => {
                            storage.types.get(&x.id.to_string()).expect("array type should exist")
                        }
                        _ => unreachable!("array type is always a type"),
                    };
                    // array len type is u32 because sierra usize is u32.
                    // we consider the array to be a simple sierra type for now (unless we find something wrong).
                    let sierra_type = SierraType::Simple(self.struct_type(&[
                        self.u32_type(),
                        self.llvm_array_type(0, array_value_type.get_type()),
                    ]));

                    storage.types.insert(id.to_string(), sierra_type);
                }
                "u8" => {
                    let ty = self.u8_type();
                    storage.types.insert(id.to_string(), SierraType::Simple(ty));
                }
                "u16" => {
                    let ty = self.u16_type();
                    storage.types.insert(id.to_string(), SierraType::Simple(ty));
                }
                "u32" => {
                    let ty = self.u32_type();
                    storage.types.insert(id.to_string(), SierraType::Simple(ty));
                }
                "u64" => {
                    let ty = self.u64_type();
                    storage.types.insert(id.to_string(), SierraType::Simple(ty));
                }
                "u128" => {
                    let ty = self.u128_type();
                    storage.types.insert(id.to_string(), SierraType::Simple(ty));
                }
                _ => {
                    todo!("unhandled type: {}", type_decl.id.debug_name.as_ref().unwrap().as_str())
                }
            }
        }

        // debug!(types = ?storage.borrow().types, "processed");
        Ok(())
    }
}
