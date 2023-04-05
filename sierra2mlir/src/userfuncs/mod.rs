use cairo_lang_sierra::{
    ids::ConcreteTypeId,
    program::{GenericArg, Program, TypeDeclaration},
};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, Location, Region, Type};

use crate::{
    compiler::{Compiler, FnAttributes, SierraType, Storage, UserFuncDef},
    utility::create_fn_signature,
};

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

            let userfunc_def = storage.userfuncs.get(func_name).unwrap().clone();

            if self.main_print && should_create_wrapper(func_name) {
                let raw_arg_types =
                    userfunc_def.args.iter().map(SierraType::get_type).collect_vec();
                let ret_types = userfunc_def
                    .return_types
                    .iter()
                    .map(SierraType::get_type)
                    .zip_eq(func.signature.ret_types.iter().cloned())
                    .collect_vec();

                self.create_felt_representation_wrapper(
                    func_name,
                    &raw_arg_types,
                    &ret_types,
                    storage,
                )?;
            }
        }

        Ok(())
    }

    pub fn create_felt_representation_wrapper(
        &'ctx self,
        wrapped_func_name: &str,
        arg_types: &[Type],
        ret_types: &[(Type, ConcreteTypeId)],
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
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
                "felt252" => self.create_print_felt()?,
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
                        .expect("Type should be registered");
                    self.create_print_enum(arg_type, type_decl.clone())?
                }
                "u8" | "u16" | "u32" | "u64" | "u128" => {
                    let uint_type = storage
                        .types
                        .get(&type_decl.id.id.to_string())
                        .expect("Type should be registered");
                    self.create_print_uint(uint_type, type_decl)?
                }
                _ => todo!("Felt representation for {}", type_category),
            }
        }

        let region = Region::new();
        let arg_types_with_locations =
            arg_types.iter().map(|t| (*t, Location::unknown(&self.context))).collect::<Vec<_>>();
        let block = Block::new(&arg_types_with_locations);

        let mut arg_values: Vec<_> = vec![];
        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            arg_values.push(arg.into());
        }
        let arg_values = arg_values;
        let mlir_ret_types = ret_types.iter().map(|(t, _id)| *t).collect_vec();

        let raw_res = self.op_func_call(&block, wrapped_func_name, &arg_values, &mlir_ret_types)?;
        for (position, (_, type_decl)) in ret_type_declarations.iter().enumerate() {
            let result_val = raw_res.result(position)?;
            self.op_func_call(
                &block,
                &format!("print_{}", type_decl.id.debug_name.as_ref().unwrap().as_str()),
                &[result_val.into()],
                &[],
            )?;
        }

        self.op_return(&block, &[]);

        region.append_block(block);

        // Currently this wrapper is only put on the entrypoint, so we call it main
        // We might want to change this in the future
        let op = self.op_func(
            "main",
            create_fn_signature(arg_types, &[]).as_str(),
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
                .map(|p| {
                    storage
                        .types
                        .get(&p.ty.id.to_string())
                        .expect("Userfunc arg type should have been registered")
                        .clone()
                })
                .collect_vec();
            let return_types = func
                .signature
                .ret_types
                .iter()
                .map(|t| {
                    storage
                        .types
                        .get(&t.id.to_string())
                        .expect("Userfunc ret type should have been registered")
                        .clone()
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
            "u8" | "u16" | "u32" | "u64" | "u128" => {
                if !types_to_print.contains(type_decl) {
                    types_to_print.push(type_decl.clone());
                }
            }
            _ => todo!("Felt representation for {}", type_category),
        }
    }
    types_to_print
}
