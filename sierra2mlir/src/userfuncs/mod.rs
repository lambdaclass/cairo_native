use cairo_lang_sierra::program::{GenFunction, GenericArg, Program, StatementIdx, TypeDeclaration};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, Location, Region};

use crate::{
    compiler::{fn_attributes::FnAttributes, mlir_ops::CmpOp, Compiler, Storage},
    libfuncs::lib_func_def::PositionalArg,
    types::is_omitted_builtin_type,
    utility::create_fn_signature,
};

use self::user_func_def::UserFuncDef;

pub mod user_func_def;

impl<'ctx> Compiler<'ctx> {
    pub fn process_functions(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        // First, create the user func defs without internal block information
        self.save_nonflow_function_info_to_storage(self.program, storage);

        // Add a wrapper around the main function to print the felt representation of its returns if the option to do so was passed
        self.create_wrappers_if_necessary(storage)?;

        Ok(())
    }

    pub fn create_wrappers_if_necessary(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        for func in self.program.funcs.iter() {
            let func_name = func.id.debug_name.as_ref().unwrap().as_str();

            if self.main_print && should_create_wrapper(func_name) {
                self.create_felt_representation_wrapper(func_name, func, storage)?;
            }
        }

        Ok(())
    }

    pub fn create_felt_representation_wrapper(
        &'ctx self,
        wrapped_func_name: &str,
        func: &GenFunction<StatementIdx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let userfunc_def = storage.userfuncs.get(wrapped_func_name).unwrap().clone();

        let arg_types = userfunc_def.args.iter().map(|arg| arg.ty.get_type()).collect_vec();
        let ret_types = userfunc_def
            .return_types
            .iter()
            .map(|arg| (arg.ty.clone(), func.signature.ret_types[arg.loc].clone()))
            .collect_vec();

        // We need to collect the sierra type declarations to know how to convert the mlir types to their felt representation
        // This is especially important for enums
        let ret_type_declarations = ret_types
            .iter()
            .map(|(mlir_type, type_id)| {
                (
                    mlir_type,
                    self.program.type_declarations.iter().find(|decl| decl.id == *type_id).unwrap(),
                )
            })
            .collect_vec();

        // Create a list of types for which to generate print functions, with no duplicates
        // For complex types, their components types must be added to the list before them
        let types_to_print = get_all_types_to_print(
            &ret_type_declarations.iter().map(|(_t, decl)| (*decl).clone()).collect_vec(),
            self.program,
        );

        for type_decl in types_to_print {
            let type_category = type_decl.long_id.generic_id.0.as_str();
            match type_category {
                "felt252" => self.create_print_felt(storage)?,
                "NonZero" => todo!("Print box felt representation"),
                "Box" => todo!("Print box felt representation"),
                "Nullable" => {
                    let arg_type = storage
                        .types
                        .get(&type_decl.id.id.to_string())
                        .cloned()
                        .expect("Type should be registered");
                    self.create_print_nullable(&arg_type, type_decl.clone(), storage)?
                }
                "Struct" => {
                    let arg_type = storage
                        .types
                        .get(&type_decl.id.id.to_string())
                        .expect("Type should be registered");
                    self.create_print_struct(arg_type, type_decl.clone())?
                }
                "Enum" => {
                    let arg_type = storage
                        .types
                        .get(&type_decl.id.id.to_string())
                        .expect("Type should be registered")
                        .clone();
                    self.create_print_enum(&arg_type, type_decl.clone(), storage)?
                }
                "Array" => {
                    let arg_type = storage
                        .types
                        .get(&type_decl.id.id.to_string())
                        .expect("Type should be registered");
                    self.create_print_array(arg_type, type_decl.clone())?
                }
                "u8" | "u16" | "u32" | "u64" | "u128" => {
                    let uint_type = storage
                        .types
                        .get(&type_decl.id.id.to_string())
                        .expect("Type should be registered")
                        .clone();
                    self.create_print_uint(&uint_type, type_decl, storage)?
                }
                _ => todo!("Felt representation for {}", type_category),
            }
        }

        let region = Region::new();
        let arg_types_with_locations =
            arg_types.iter().map(|t| (*t, Location::unknown(&self.context))).collect::<Vec<_>>();
        let block = region.append_block(Block::new(&arg_types_with_locations));

        let mut arg_values: Vec<_> = vec![];
        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            arg_values.push(arg.into());
        }
        let arg_values = arg_values;
        let mlir_ret_types = ret_types.iter().map(|(t, _id)| t.get_type()).collect_vec();

        // First, call the wrapped function
        let raw_res = self.op_func_call(&block, wrapped_func_name, &arg_values, &mlir_ret_types)?;

        let success_block = region.append_block(Block::new(&[]));

        let ret_type_name = ret_types[0].1.debug_name.as_ref().unwrap();

        // Then, print whether or not the execution was successful
        if ret_type_name.starts_with("core::PanicResult::<") {
            let panic_enum_type = &ret_types[0].0;
            let panic_value = raw_res.result(0)?.into();
            // Get the tag from the panic enum to determine whether execution was successful or not
            let panic_enum_tag_op =
                self.call_enum_get_tag(&block, panic_value, panic_enum_type, storage)?;
            let panic_enum_tag = panic_enum_tag_op.result(0)?.into();
            let zero_op = self.op_const(&block, "0", panic_enum_type.get_enum_tag_type().unwrap());
            let zero = zero_op.result(0)?.into();
            // If it equals 0 then it's a success
            let is_success_op = self.op_cmp(&block, CmpOp::Equal, panic_enum_tag, zero);
            let is_success = is_success_op.result(0)?.into();

            // Create a block for printing the error message if execution failed
            let panic_block = region.append_block(Block::new(&[]));
            self.op_cond_br(&block, is_success, &success_block, &panic_block, &[], &[]);

            self.call_print_panic_message(
                &panic_block,
                ret_type_name.as_str(),
                panic_value,
                panic_enum_type,
                storage,
            )?;
            self.op_return(&panic_block, &[]);
        } else {
            self.op_br(&block, &success_block, &[]);
        }

        self.call_dprintf(&success_block, "Success\n", &[], storage)?;

        // Finally, print the result if it was, or the error message if not
        for (position, (_, type_decl)) in ret_type_declarations.iter().enumerate() {
            let type_name = type_decl.id.debug_name.as_ref().unwrap().as_str();
            if is_omitted_builtin_type(type_name) {
                continue;
            }
            let result_val = raw_res.result(position)?;
            self.op_func_call(
                &success_block,
                &format!("print_{}", type_name),
                &[result_val.into()],
                &[],
            )?;
        }

        self.op_return(&success_block, &[]);

        // Currently this wrapper is only put on the entrypoint, so we call it main
        // We might want to change this in the future
        let op = self.op_func(
            "main",
            create_fn_signature(&arg_types, &[]).as_str(),
            vec![region],
            FnAttributes { public: true, emit_c_interface: true, ..Default::default() },
        )?;

        self.module.body().append_operation(op);

        Ok(())
    }

    fn save_nonflow_function_info_to_storage(
        &self,
        program: &Program,
        storage: &mut Storage<'ctx>,
    ) {
        for func in program.funcs.iter() {
            let func_name = func.id.debug_name.as_ref().unwrap().to_string();

            let param_types = func
                .params
                .iter()
                .enumerate()
                .filter_map(|(idx, param)| {
                    if is_omitted_builtin_type(&param.ty.debug_name.as_ref().unwrap().to_string()) {
                        None
                    } else {
                        Some(PositionalArg {
                            loc: idx,
                            ty: storage
                                .types
                                .get(&param.ty.id.to_string())
                                .expect("Userfunc arg type should have been registered")
                                .clone(),
                        })
                    }
                })
                .collect_vec();
            let return_types = func
                .signature
                .ret_types
                .iter()
                .enumerate()
                .filter_map(|(idx, ty)| {
                    if is_omitted_builtin_type(&ty.debug_name.as_ref().unwrap().to_string()) {
                        None
                    } else {
                        Some(PositionalArg {
                            loc: idx,
                            ty: storage
                                .types
                                .get(&ty.id.to_string())
                                .expect("Userfunc ret type should have been registered")
                                .clone(),
                        })
                    }
                })
                .collect_vec();

            storage.userfuncs.insert(func_name, UserFuncDef { args: param_types, return_types });
        }
    }
}

// Currently this checks whether the function in question is the main function,
// but it could be switched for anything later
fn should_create_wrapper(raw_func_name: &str) -> bool {
    let parts = raw_func_name.split("::").collect::<Vec<_>>();
    parts.len() == 3 && parts[0] == parts[1] && parts[2] == "main"
}

// Produces an ordered list of all types and component types
fn get_all_types_to_print(
    type_declarations: &[TypeDeclaration],
    program: &Program,
) -> Vec<TypeDeclaration> {
    let mut types_to_print = vec![];
    for type_decl in type_declarations {
        let type_category = type_decl.long_id.generic_id.0.as_str();
        match type_category {
            "felt252" => {
                if !types_to_print.contains(type_decl) {
                    types_to_print.push(type_decl.clone());
                }
            }
            "NonZero" => todo!("Print box felt representation"),
            "Box" => todo!("Print box felt representation"),
            "Struct" => {
                let field_type_declarations = type_decl.long_id.generic_args[1..].iter().map(|member_type| match member_type {
                    GenericArg::Type(type_id) => type_id,
                    _ => panic!("Struct type declaration arguments after the first should all be resolved"),
                }).map(|member_type_id| program.type_declarations.iter().find(|decl| decl.id == *member_type_id).unwrap())
                .map(|component_type_decl| get_all_types_to_print(&[component_type_decl.clone()], program));

                for type_decls in field_type_declarations {
                    for type_decl in type_decls {
                        if !types_to_print.contains(&type_decl) {
                            types_to_print.push(type_decl);
                        }
                    }
                }
                if !types_to_print.contains(type_decl) {
                    types_to_print.push(type_decl.clone());
                }
            }
            "Enum" => {
                let field_type_declarations = type_decl.long_id.generic_args[1..].iter().map(|member_type| match member_type {
                    GenericArg::Type(type_id) => type_id,
                    _ => panic!("Enum type declaration arguments after the first should all be resolved"),
                }).map(|member_type_id| program.type_declarations.iter().find(|decl| decl.id == *member_type_id).unwrap())
                .map(|component_type_decl| get_all_types_to_print(&[component_type_decl.clone()], program));

                for type_decls in field_type_declarations {
                    for type_decl in type_decls {
                        if !types_to_print.contains(&type_decl) {
                            types_to_print.push(type_decl);
                        }
                    }
                }
                if !types_to_print.contains(type_decl) {
                    types_to_print.push(type_decl.clone());
                }
            }
            "Array" => {
                let array_type = match &type_decl.long_id.generic_args[0] {
                    GenericArg::Type(type_id) => type_id,
                    _ => panic!(
                        "Struct type declaration arguments after the first should all be resolved"
                    ),
                };

                for ty in &program.type_declarations {
                    if ty.id == *array_type {
                        let types_to_print_here = get_all_types_to_print(&[ty.clone()], program);
                        for type_decl in types_to_print_here {
                            if !types_to_print.contains(&type_decl) {
                                types_to_print.push(type_decl);
                            }
                        }
                        break;
                    }
                }

                if !types_to_print.contains(type_decl) {
                    types_to_print.push(type_decl.clone());
                }
            }
            "Nullable" => {
                let nullable_type = match &type_decl.long_id.generic_args[0] {
                    GenericArg::Type(type_id) => type_id,
                    _ => panic!(
                        "Struct type declaration arguments after the first should all be resolved"
                    ),
                };

                for ty in &program.type_declarations {
                    if ty.id == *nullable_type {
                        let types_to_print_here = get_all_types_to_print(&[ty.clone()], program);
                        for type_decl in types_to_print_here {
                            if !types_to_print.contains(&type_decl) {
                                types_to_print.push(type_decl);
                            }
                        }
                        break;
                    }
                }

                if !types_to_print.contains(type_decl) {
                    types_to_print.push(type_decl.clone());
                }
            }
            "u8" | "u16" | "u32" | "u64" | "u128" => {
                if !types_to_print.contains(type_decl) {
                    types_to_print.push(type_decl.clone());
                }
            }
            // Specifically omit these types
            "Bitwise" | "Pedersen" | "Poseidon" | "RangeCheck" => {}
            _ => todo!("Felt representation for {}", type_category),
        }
    }
    types_to_print
}
