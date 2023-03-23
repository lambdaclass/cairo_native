use std::{cell::RefCell, rc::Rc};

use cairo_lang_sierra::program::GenericArg;
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::Type;
use tracing::debug;

use crate::compiler::{Compiler, SierraType, Storage};

impl<'ctx> Compiler<'ctx> {
    pub fn process_types(&'ctx self, storage: Rc<RefCell<Storage<'ctx>>>) -> Result<()> {
        for type_decl in &self.program.type_declarations {
            let id = type_decl.id.id;
            let name = type_decl.long_id.generic_id.0.as_str();
            debug!(name, "processing type decl");

            let mut storage = storage.borrow_mut();

            match name {
                "Bitwise" => {
                    let ty = self.bitwise_type();
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
                        let struct_types = types
                            .iter()
                            .map(|ty| match ty {
                                SierraType::Simple(ty) => *ty,
                                SierraType::Struct { ty, field_types: _ } => *ty,
                            })
                            .collect_vec();
                        let struct_type =
                            Type::parse(&self.context, &self.struct_type_string(&struct_types))
                                .unwrap();

                        storage.types.insert(
                            id.to_string(),
                            SierraType::Struct {
                                ty: struct_type,
                                field_types: struct_types,
                            },
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

                    let struct_types = types
                        .iter()
                        .map(|ty| match ty {
                            SierraType::Simple(ty) => *ty,
                            SierraType::Struct { ty, field_types: _ } => *ty,
                        })
                        .collect_vec();
                    let struct_type =
                        Type::parse(&self.context, &self.struct_type_string(&struct_types))
                            .unwrap();

                    storage.types.insert(
                        id.to_string(),
                        SierraType::Struct {
                            ty: struct_type,
                            field_types: struct_types,
                        },
                    );
                }
                _ => debug!(?type_decl, "unhandled type"),
            }
        }

        // debug!(types = ?storage.borrow().types, "processed");
        Ok(())
    }
}
