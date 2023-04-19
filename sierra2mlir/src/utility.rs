use std::{
    env,
    ops::Deref,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use itertools::Itertools;

use cairo_lang_sierra::program::{GenericArg, TypeDeclaration};
use melior_next::{
    dialect::cf,
    ir::{
        operation, Block, BlockRef, Location, NamedAttribute, OperationRef, Region, Type, TypeLike,
        Value, ValueLike,
    },
};

use crate::{
    compiler::{Compiler, FnAttributes, SierraType},
    statements::Variable,
};
use color_eyre::Result;

impl<'ctx> Compiler<'ctx> {
    pub fn create_libc_funcs(&self) -> Result<()> {
        let region = Region::new();

        let func = operation::Builder::new("llvm.func", Location::unknown(&self.context))
            .add_attributes(&NamedAttribute::new_parsed_vec(
                &self.context,
                &[
                    ("sym_name", "\"realloc\""),
                    ("function_type", "!llvm.func<ptr (ptr, i64)>"),
                    ("linkage", "#llvm.linkage<external>"),
                ],
            )?)
            .add_regions(vec![region])
            .build();

        self.module.body().append_operation(func);

        let region = Region::new();
        let func = operation::Builder::new("llvm.func", Location::unknown(&self.context))
            .add_attributes(&NamedAttribute::new_parsed_vec(
                &self.context,
                &[
                    ("sym_name", "\"memmove\""),
                    ("function_type", "!llvm.func<ptr (ptr, ptr, i64)>"),
                    ("linkage", "#llvm.linkage<external>"),
                ],
            )?)
            .add_regions(vec![region])
            .build();

        self.module.body().append_operation(func);

        let region = Region::new();
        let func = operation::Builder::new("llvm.func", Location::unknown(&self.context))
            .add_attributes(&NamedAttribute::new_parsed_vec(
                &self.context,
                &[
                    ("sym_name", "\"free\""),
                    ("function_type", "!llvm.func<void (ptr)>"),
                    ("linkage", "#llvm.linkage<external>"),
                ],
            )?)
            .add_regions(vec![region])
            .build();

        self.module.body().append_operation(func);
        Ok(())
    }

    /// Creates the external function definition for printf.
    pub fn create_printf(&'ctx self) -> Result<()> {
        let region = Region::new();

        // needs to be a llvm function due to variadic args.
        let func = operation::Builder::new("llvm.func", Location::unknown(&self.context))
            .add_attributes(&NamedAttribute::new_parsed_vec(
                &self.context,
                &[
                    ("sym_name", "\"dprintf\""),
                    ("function_type", "!llvm.func<i32 (i32, !llvm.ptr, ...)>"),
                    ("linkage", "#llvm.linkage<external>"),
                ],
            )?)
            .add_regions(vec![region])
            .build();

        self.module.body().append_operation(func);
        Ok(())
    }

    /// Utility function to create a printf call.
    /// Null-terminates the string iff it is not already null-terminated
    pub fn call_printf<'a>(&'a self, block: &'a Block, fmt: &str, values: &[Value]) -> Result<()> {
        let terminated_fmt_string = if fmt.ends_with('\0') {
            fmt.to_string()
        } else if fmt.contains('\0') {
            panic!("Format string \"{fmt}\" to printf contains data after \\0")
        } else {
            format!("{fmt}\0")
        };

        let i32_type = Type::integer(&self.context, 32);
        let fmt_len = terminated_fmt_string.as_bytes().len();

        let i8_type = Type::integer(&self.context, 8);
        let arr_ty =
            Type::parse(&self.context, &format!("!llvm.array<{fmt_len} x {i8_type}>")).unwrap();
        let data_op = self.op_llvm_alloca(block, i8_type, fmt_len)?;
        let addr: Value = data_op.result(0)?.into();

        // https://discourse.llvm.org/t/array-globals-in-llvm-dialect/68229
        // To create a constant array, we need to use a dense array attribute, which has a tensor type,
        // which is then interpreted as a llvm.array type.
        let fmt_data = self.op_llvm_const(
            block,
            &format!(
                "dense<[{}]> : tensor<{} x {}>",
                terminated_fmt_string.as_bytes().iter().join(", "),
                fmt_len,
                i8_type
            ),
            arr_ty,
        );

        self.op_llvm_store(block, fmt_data.result(0)?.into(), addr)?;

        let target_fd = self.op_u32_const(block, "1");

        let mut args = vec![target_fd.result(0)?.into(), addr];
        args.extend(values);

        self.op_llvm_call(block, "dprintf", &args, &[i32_type])?;
        Ok(())
    }

    /// value needs to be u64 and is the size in bytes.
    pub fn call_realloc<'a>(
        &'a self,
        block: &'a Block,
        ptr: Value,
        size: Value,
    ) -> Result<OperationRef<'a>> {
        self.op_llvm_call(block, "realloc", &[ptr, size], &[self.llvm_ptr_type()])
    }

    pub fn call_memmove<'a>(
        &'a self,
        block: &'a Block,
        dst: Value,
        src: Value,
        size: Value,
    ) -> Result<OperationRef<'a>> {
        self.op_llvm_call(block, "memmove", &[dst, src, size], &[self.llvm_ptr_type()])
    }

    /// creates the implementation for the print felt method: "print_felt(value: i256) -> ()"
    pub fn create_print_felt(&'ctx self) -> Result<()> {
        let region = Region::new();

        let args_types = [self.felt_type()];
        let args_types_with_locations = [(self.felt_type(), Location::unknown(&self.context))];

        let block = Block::new(&args_types_with_locations);
        let block = region.append_block(block);

        let argument = block.argument(0)?;
        let mut current_value = Variable::Param { argument };
        let mut value_type = self.felt_type();

        let mut bit_width = current_value.get_value().r#type().get_width().unwrap();

        // We need to make sure the bit width is a power of 2.
        let rounded_up_bitwidth = bit_width.next_power_of_two();

        if bit_width != rounded_up_bitwidth {
            value_type = Type::integer(&self.context, rounded_up_bitwidth);
            let res = self.op_zext(&block, current_value.get_value(), value_type);
            current_value = Variable::Local { op: res, result_idx: 0 };
        }

        bit_width = rounded_up_bitwidth;

        while bit_width > 0 {
            let shift_by_constant_op =
                self.op_const(&block, &bit_width.saturating_sub(32).to_string(), value_type);
            let shift_by = shift_by_constant_op.result(0)?.into();

            let shift_op = self.op_shrs(&block, current_value.get_value(), shift_by);
            let shift_result = shift_op.result(0)?.into();

            let truncated_op = self.op_trunc(&block, shift_result, self.i32_type());
            let truncated = truncated_op.result(0)?.into();
            self.call_printf(&block, "%08X\0", &[truncated])?;

            bit_width = bit_width.saturating_sub(32);
        }

        self.call_printf(&block, "\n\0", &[])?;

        self.op_return(&block, &[]);

        let function_type = create_fn_signature(&args_types, &[]);

        let func = self.op_func(
            "print_felt252",
            &function_type,
            vec![region],
            FnAttributes::libfunc(false, false),
        )?;

        self.module.body().append_operation(func);

        Ok(())
    }

    pub fn create_print_struct(
        &'ctx self,
        struct_type: &SierraType,
        sierra_type_declaration: TypeDeclaration,
    ) -> Result<()> {
        let region = Region::new();
        let block = region.append_block(Block::new(&[(
            struct_type.get_type(),
            Location::unknown(&self.context),
        )]));

        let arg = block.argument(0)?;

        let function_type = create_fn_signature(&[struct_type.get_type()], &[]);

        let struct_name = sierra_type_declaration.id.debug_name.unwrap();

        let component_type_ids =
            sierra_type_declaration.long_id.generic_args[1..].iter().map(|member_type| {
                match member_type {
                    GenericArg::Type(type_id) => type_id,
                    _ => panic!(
                        "Struct type declaration arguments after the first should all be resolved"
                    ),
                }
            });

        let field_types = struct_type
            .get_field_types()
            .expect("Attempted to create struct print for simple type");

        for (index, component_type_id) in component_type_ids.enumerate() {
            let component_type_name = component_type_id.debug_name.as_ref().unwrap();
            let component_type = &field_types[index];
            let extract_op =
                self.op_llvm_extractvalue(&block, index, arg.into(), *component_type)?;
            let component_value = extract_op.result(0)?;
            self.op_func_call(
                &block,
                &format!("print_{}", component_type_name),
                &[component_value.into()],
                &[],
            )?;
        }

        self.op_return(&block, &[]);

        let func = self.op_func(
            &format!("print_{}", struct_name.as_str()),
            &function_type,
            vec![region],
            FnAttributes::libfunc(false, false),
        )?;

        self.module.body().append_operation(func);

        Ok(())
    }

    /// like cairo runner, prints the tag value and then the enum value
    pub fn create_print_enum(
        &'ctx self,
        enum_type: &SierraType,
        sierra_type_declaration: TypeDeclaration,
    ) -> Result<()> {
        let region = Region::new();
        let entry_block = region
            .append_block(Block::new(&[(enum_type.get_type(), Location::unknown(&self.context))]));

        let enum_value = entry_block.argument(0)?.into();

        // if its a panic enum wrapper, don't print the tag, as they are meant to be invisible
        let is_panic_enum = match &sierra_type_declaration.long_id.generic_args[0] {
            GenericArg::UserType(x) => {
                x.debug_name.as_ref().unwrap().starts_with("core::PanicResult::")
            }
            _ => unreachable!(),
        };

        let function_type = create_fn_signature(&[enum_type.get_type()], &[]);

        if let SierraType::Enum { tag_type, storage_type, variants_types, .. } = enum_type {
            // create a block for each variant of the enum
            let blocks = variants_types
                .iter()
                .map(|var_ty| (region.append_block(Block::new(&[])), var_ty))
                .collect_vec();

            let default_block = region.append_block(Block::new(&[]));
            self.op_return(&default_block, &[]);

            // type is !llvm.struct<(i16, array<N x i8>)>

            // get the tag
            let tag_value_op = self.op_llvm_extractvalue(&entry_block, 0, enum_value, *tag_type)?;
            let tag_value = tag_value_op.result(0)?.into();

            if !is_panic_enum {
                // Print the tag. Extending it to 32 bits allows for better printf portability
                let tag_32bit = self.op_zext(&entry_block, tag_value, self.u32_type());
                self.call_printf(&entry_block, "%X\n\0", &[tag_32bit.result(0)?.into()])?;
            }

            // Store the enum data on the stack, and create a pointer to it
            // This allows us to interpret a ptr to it as a ptr to any of the variant types
            let data_alloca_op = self.op_llvm_alloca(&entry_block, *storage_type, 1)?;
            let data_ptr = data_alloca_op.result(0)?.into();
            let enum_data_op =
                self.op_llvm_extractvalue(&entry_block, 1, enum_value, *storage_type)?;
            let enum_data = enum_data_op.result(0)?.into();
            self.op_llvm_store(&entry_block, enum_data, data_ptr)?;

            let blockrefs = blocks.iter().map(|x| x.0).collect_vec();
            let case_values = (0..variants_types.len()).map(|x| x.to_string()).collect_vec();

            let blockrefs_with_ops =
                blockrefs.iter().map(|x| (x.deref(), [].as_slice())).collect_vec();
            entry_block.append_operation(cf::switch(
                &self.context,
                &case_values,
                tag_value,
                (&default_block, &[]),
                blockrefs_with_ops.as_slice(),
                Location::unknown(&self.context),
            ));

            // Sierra type id of each variant type, used to work out which print functions to delegate to
            let component_type_ids = sierra_type_declaration.long_id.generic_args[1..]
                .iter()
                .map(|member_type| match member_type {
                    GenericArg::Type(type_id) => type_id,
                    _ => panic!(
                        "Struct type declaration arguments after the first should all be resolved"
                    ),
                })
                .collect_vec();

            let enum_felt_width = enum_type.get_felt_representation_width();

            for (i, (block, var_ty)) in blocks.iter().enumerate() {
                let variant_felt_width = var_ty.get_felt_representation_width();
                let unused_felt_count = enum_felt_width - 1 - variant_felt_width;
                if unused_felt_count != 0 && !is_panic_enum {
                    self.call_printf(block, &"0\n".repeat(unused_felt_count), &[])?;
                }

                let component_type_name = component_type_ids[i].debug_name.as_ref().unwrap();
                let value_op = self.op_llvm_load(block, data_ptr, var_ty.get_type())?;
                let value = value_op.result(0)?.into();
                self.op_func_call(block, &format!("print_{}", component_type_name), &[value], &[])?;
                self.op_return(block, &[]);
            }
        } else {
            panic!("sierra_type_declaration should be a enum")
        }

        let enum_name = sierra_type_declaration.id.debug_name.unwrap();

        let func = self.op_func(
            &format!("print_{}", enum_name.as_str()),
            &function_type,
            vec![region],
            FnAttributes::libfunc(false, false),
        )?;

        self.module.body().append_operation(func);

        Ok(())
    }

    pub fn create_print_array(
        &'ctx self,
        array_type: &SierraType,
        sierra_type_declaration: TypeDeclaration,
    ) -> Result<()> {
        let region = Region::new();
        let entry_block = region
            .append_block(Block::new(&[(array_type.get_type(), Location::unknown(&self.context))]));

        let function_type = create_fn_signature(&[array_type.get_type()], &[]);

        let array_value_type_id = match &sierra_type_declaration.long_id.generic_args[0] {
            GenericArg::Type(type_id) => type_id,
            _ => panic!("Struct type declaration arguments after the first should all be resolved"),
        };

        if let SierraType::Array { ty: _, len_type, element_type } = array_type {
            let array_value = entry_block.argument(0)?.into();

            // get the data pointer
            let data_ptr_op =
                self.op_llvm_extractvalue(&entry_block, 2, array_value, self.llvm_ptr_type())?;
            let data_ptr: Value = data_ptr_op.result(0)?.into();

            let lower_bound_op = self.op_u32_const(&entry_block, "0");
            let lower_bound: Value = lower_bound_op.result(0)?.into();

            let upper_bound_op =
                self.op_llvm_extractvalue(&entry_block, 0, array_value, *len_type)?;
            let upper_bound: Value = upper_bound_op.result(0)?.into();

            let step_op = self.op_u32_const(&entry_block, "1");
            let step: Value = step_op.result(0)?.into();

            let check_block = region
                .append_block(Block::new(&[(self.u32_type(), Location::unknown(&self.context))]));

            let loop_block = region
                .append_block(Block::new(&[(self.u32_type(), Location::unknown(&self.context))]));

            let end_block = region.append_block(Block::new(&[]));

            self.op_br(&entry_block, &check_block, &[lower_bound]);

            let lower_bound = check_block.argument(0)?.into();
            let cmp_op = self.op_cmp(
                &check_block,
                crate::compiler::CmpOp::UnsignedLessThan,
                lower_bound,
                upper_bound,
            );
            let cmp = cmp_op.result(0)?.into();
            self.op_cond_br(
                &check_block,
                cmp,
                &loop_block,
                &end_block,
                &[check_block.argument(0)?.into()],
                &[],
            );

            // for loop body
            let idx = loop_block.argument(0)?.into();

            // get the pointer to the data index
            let value_ptr_op =
                self.op_llvm_gep_dynamic(&loop_block, &[idx], data_ptr, element_type.get_type())?;
            let value_ptr = value_ptr_op.result(0)?.into();
            let value_load_op =
                self.op_llvm_load(&loop_block, value_ptr, element_type.get_type())?;
            let value = value_load_op.result(0)?.into();

            let array_value_type_name = array_value_type_id.debug_name.as_ref().unwrap();
            self.op_func_call(
                &loop_block,
                &format!("print_{}", array_value_type_name),
                &[value],
                &[],
            )?;

            let new_idx_op = self.op_add(&loop_block, idx, step);
            let new_idx = new_idx_op.result(0)?.into();

            self.op_br(&loop_block, &check_block, &[new_idx]);

            self.op_return(&end_block, &[]);
        } else {
            panic!("sierra_type_declaration should be a array")
        }

        let enum_name = sierra_type_declaration.id.debug_name.unwrap();

        let func = self.op_func(
            &format!("print_{}", enum_name.as_str()),
            &function_type,
            vec![region],
            FnAttributes::libfunc(false, false),
        )?;

        self.module.body().append_operation(func);

        Ok(())
    }

    pub fn create_print_uint(
        &'ctx self,
        uint_type: &SierraType,
        sierra_type_declaration: TypeDeclaration,
    ) -> Result<()> {
        let region = Region::new();
        let block = region
            .append_block(Block::new(&[(uint_type.get_type(), Location::unknown(&self.context))]));

        let arg = block.argument(0)?.into();

        let function_type = create_fn_signature(&[uint_type.get_type()], &[]);

        let uint_name = sierra_type_declaration.id.debug_name.unwrap();

        match uint_name.as_str() {
            "u8" => {
                let value_32bit = self.op_zext(&block, arg, self.u32_type());
                self.call_printf(&block, "%X\n", &[value_32bit.result(0)?.into()])?;
            }
            "u16" => {
                let value_32bit = self.op_zext(&block, arg, self.u32_type());
                self.call_printf(&block, "%X\n", &[value_32bit.result(0)?.into()])?;
            }
            "u32" => {
                self.call_printf(&block, "%X\n", &[arg])?;
            }
            "u64" => self.call_printf(&block, "%lX\n", &[arg])?,
            "u128" => {
                let lower = self.op_trunc(&block, arg, self.u64_type());
                let shift_amount = self.op_u128_const(&block, "64");
                let upper_shifted = self.op_shru(&block, arg, shift_amount.result(0)?.into());
                let upper = self.op_trunc(&block, upper_shifted.result(0)?.into(), self.u64_type());
                self.call_printf(
                    &block,
                    "%lX%016lX\n",
                    &[upper.result(0)?.into(), lower.result(0)?.into()],
                )?;
            }
            _ => unreachable!("Encountered unexpected type {} when creating uint print", uint_name),
        }

        self.op_return(&block, &[]);

        let func = self.op_func(
            &format!("print_{}", uint_name.as_str()),
            &function_type,
            vec![region],
            FnAttributes::libfunc(false, false),
        )?;

        self.module.body().append_operation(func);

        Ok(())
    }

    /// Utility method to create a print_felt call.
    pub fn call_print_felt(&'ctx self, block: BlockRef<'ctx>, value: Value) -> Result<()> {
        self.op_func_call(&block, "print_felt", &[value], &[])?;
        Ok(())
    }
}

pub fn create_fn_signature(params: &[Type], return_types: &[Type]) -> String {
    format!(
        "({}) -> {}",
        params.iter().map(|x| x.to_string()).join(", "),
        &format!("({})", return_types.iter().map(|x| x.to_string()).join(", ")),
    )
}

fn find_mlir_prefix() -> PathBuf {
    env::var_os("MLIR_SYS_160_PREFIX").map_or_else(
        || {
            let cmd_output = Command::new("../scripts/find-llvm.sh")
                .stdout(Stdio::piped())
                .spawn()
                .unwrap()
                .wait_with_output()
                .unwrap();

            PathBuf::from(String::from_utf8(cmd_output.stdout).unwrap().trim())
        },
        |x| Path::new(x.to_str().unwrap()).to_owned(),
    )
}

pub fn run_llvm_config(args: &[&str]) -> String {
    let prefix = find_mlir_prefix();
    let llvm_config = prefix.join("bin/llvm-config");

    let output = Command::new(llvm_config)
        .args(args)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();

    String::from_utf8(output.stdout).unwrap()
}
