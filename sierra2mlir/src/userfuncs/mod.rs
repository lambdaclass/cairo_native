use cairo_lang_sierra::program::{GenFunction, GenericArg, Program, StatementIdx, TypeDeclaration};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, Location, Region};

use crate::{
    compiler::{fn_attributes::FnAttributes, mlir_ops::CmpOp, Compiler, Storage},
    libfuncs::lib_func_def::PositionalArg,
    sierra_type::SierraType,
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

        // Then, print whether or not the execution was successful
        if ret_types[0].1.debug_name.as_ref().unwrap().starts_with("core::PanicResult::<") {
            let (tag_type, storage_type, error_msg_type) = if let SierraType::Enum {
                ty: _,
                tag_type,
                storage_bytes_len: _,
                storage_type,
                variants_types,
            } = &ret_types[0].0
            {
                (*tag_type, *storage_type, &variants_types[1])
            } else {
                panic!("PanicResult should be registered as an SierraType::Enum")
            };
            let panic_block = region.append_block(Block::new(&[]));
            let panic_value = raw_res.result(0)?.into();
            let is_panic_flag = self.op_llvm_extractvalue(&block, 0, panic_value, tag_type)?;
            let zero_op = self.op_const(&block, "0", tag_type);
            // Cmp eq to 0 for efficiency
            let is_not_panic_op = self.op_cmp(
                &block,
                CmpOp::Equal,
                is_panic_flag.result(0)?.into(),
                zero_op.result(0)?.into(),
            );
            // Branch to success block if the PanicResult tag equals 0, and to the panic block if it doesn't
            self.op_cond_br(
                &block,
                is_not_panic_op.result(0)?.into(),
                &success_block,
                &panic_block,
                &[],
                &[],
            );

            // In the panic block, extract the error message. This should be an Array<felt252>
            // In order to do this, the enum's data needs to be stored on the stack
            self.call_dprintf(&panic_block, "Program panicked\n", &[], storage)?;
            let panic_data_op =
                self.op_llvm_extractvalue(&panic_block, 1, panic_value, storage_type)?;
            let panic_data = panic_data_op.result(0)?.into();
            let panic_data_ptr_op = self.op_llvm_alloca(&panic_block, storage_type, 1)?;
            let panic_data_ptr = panic_data_ptr_op.result(0)?.into();
            self.op_llvm_store(&panic_block, panic_data, panic_data_ptr)?;

            // Now that the data is on the stack, we can treat the pointer to it as the correct type
            let load_op =
                self.op_llvm_load(&panic_block, panic_data_ptr, error_msg_type.get_type())?;
            let panic_array = load_op.result(0)?.into();
            let (panic_array_len_type, panic_array_element_type) =
                if let SierraType::Array { ty: _, len_type, element_type } = error_msg_type {
                    (*len_type, element_type)
                } else {
                    panic!("Expected panic array type to be an Array Type")
                };
            // Arrays are stored as (len, capacity, data_ptr), so we need to get the data pointer
            let len_op =
                self.op_llvm_extractvalue(&panic_block, 0, panic_array, panic_array_len_type)?;
            let len = len_op.result(0)?.into();
            let zero_op = self.op_const(&panic_block, "0", panic_array_len_type);
            let zero = zero_op.result(0)?.into();
            let one_op = self.op_const(&panic_block, "1", panic_array_len_type);
            let one = one_op.result(0)?.into();
            let data_ptr_op =
                self.op_llvm_extractvalue(&panic_block, 2, panic_array, self.llvm_ptr_type())?;

            // Create a loop to loop through the elements of the array to be printed
            let outer_loop_block = region.append_block(Block::new(&[(
                panic_array_len_type,
                Location::unknown(&self.context),
            )]));
            let done_block = region.append_block(Block::new(&[]));
            self.op_br(&panic_block, &outer_loop_block, &[zero]);
            // At the start of the outer block, check if the loop index is equal to len
            //     If it is, jump to the done block, otherwise the inner block
            let loop_index = outer_loop_block.argument(0)?.into();
            let cmp_op = self.op_cmp(&outer_loop_block, CmpOp::Equal, loop_index, len);
            let cmp = cmp_op.result(0)?.into();
            let inner_loop_start_block = region.append_block(Block::new(&[]));
            self.op_cond_br(&outer_loop_block, cmp, &done_block, &inner_loop_start_block, &[], &[]);

            // In the inner loop start block, get the element at the given index, and pass it to the inner loop print block
            let error_message_data_gep_op = self.op_llvm_gep_dynamic(
                &inner_loop_start_block,
                &[loop_index],
                data_ptr_op.result(0)?.into(),
                panic_array_element_type.get_type(),
            )?;
            let error_message_element_ptr = error_message_data_gep_op.result(0)?.into();
            let error_message_element_op = self.op_llvm_load(
                &inner_loop_start_block,
                error_message_element_ptr,
                panic_array_element_type.get_type(),
            )?;
            let error_message_element = error_message_element_op.result(0)?.into();
            let increment_op = self.op_add(&inner_loop_start_block, loop_index, one);
            let incremented_loop_index = increment_op.result(0)?.into();

            // In the inner loop print block, we're going to print the topmost 8bits of the value, then shift it left by 8 bits, and repeat until it is 0
            let inner_loop_print_block = region.append_block(Block::new(&[(
                panic_array_element_type.get_type(),
                Location::unknown(&self.context),
            )]));
            self.op_br(&inner_loop_start_block, &inner_loop_print_block, &[error_message_element]);
            let print_arg = inner_loop_print_block.argument(0)?.into();
            let zero_op = self.op_felt_const(&inner_loop_print_block, "0");
            let zero = zero_op.result(0)?.into();
            let cmp_op = self.op_cmp(&inner_loop_print_block, CmpOp::Equal, print_arg, zero);
            let cmp = cmp_op.result(0)?.into();
            let shift_amount_op = self.op_felt_const(&inner_loop_print_block, "248");
            let shift_op =
                self.op_shru(&inner_loop_print_block, print_arg, shift_amount_op.result(0)?.into());
            let shift = shift_op.result(0)?.into();
            let trunc_op = self.op_trunc(&inner_loop_print_block, shift, self.u8_type());
            let bits_to_print = trunc_op.result(0)?.into();
            let eight_op = self.op_felt_const(&inner_loop_print_block, "8");
            let eight = eight_op.result(0)?.into();
            let shl_op = self.op_shl(&inner_loop_print_block, print_arg, eight);
            let next_print_arg = shl_op.result(0)?.into();

            let print_block = region.append_block(Block::new(&[]));
            let loop_end_block = region.append_block(Block::new(&[]));
            let bits_to_print_zero_cmp_op =
                self.op_cmp(&inner_loop_print_block, CmpOp::Equal, shift, zero);
            let bits_to_print_zero_cmp = bits_to_print_zero_cmp_op.result(0)?.into();
            self.op_cond_br(
                &inner_loop_print_block,
                bits_to_print_zero_cmp,
                &loop_end_block,
                &print_block,
                &[],
                &[],
            );
            self.call_dprintf(&print_block, "%c", &[bits_to_print], storage)?;
            self.op_br(&print_block, &loop_end_block, &[]);
            self.op_cond_br(
                &loop_end_block,
                cmp,
                &outer_loop_block,
                &inner_loop_print_block,
                &[incremented_loop_index],
                &[next_print_arg],
            );

            self.op_return(&done_block, &[]);
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
            "u8" | "u16" | "u32" | "u64" | "u128" => {
                if !types_to_print.contains(type_decl) {
                    types_to_print.push(type_decl.clone());
                }
            }
            // Specifically omit these types
            "RangeCheck" | "Bitwise" => {}
            _ => todo!("Felt representation for {}", type_category),
        }
    }
    types_to_print
}
