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
    ir::{Block, Location, Region, Type, TypeLike, Value, ValueLike},
};

use crate::{
    compiler::{fn_attributes::FnAttributes, mlir_ops::CmpOp, Compiler, Storage},
    sierra_type::SierraType,
    statements::Variable,
};
use color_eyre::Result;

impl<'ctx> Compiler<'ctx> {
    /// Packages the given blocks in a function and adds it to module.body()
    /// `blocks` must not be empty
    pub fn create_function(
        &self,
        name: &str,
        blocks: Vec<Block>,
        ret_types: &[Type],
        attributes: FnAttributes,
    ) -> Result<()> {
        // Need to first collect the arguments, as r#type does not outlive the argument
        let mut args = vec![];
        for arg_index in 0..blocks[0].argument_count() {
            args.push(blocks[0].argument(arg_index)?);
        }
        let args = args;
        let arg_types = args.iter().map(|arg| arg.r#type()).collect_vec();

        let function_signature = create_fn_signature(&arg_types, ret_types);

        let region = Region::new();
        for block in blocks {
            region.append_block(block);
        }

        let func = self.op_func(name, &function_signature, vec![region], attributes)?;
        self.module.body().append_operation(func);

        Ok(())
    }

    /// creates the implementation for the print felt method: "print_felt(value: i256) -> ()"
    pub fn create_print_felt(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        let arg_types = [self.felt_type()];
        let block = self.new_block(&arg_types);

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
            self.call_dprintf(&block, "%08X\0", &[truncated], storage)?;

            bit_width = bit_width.saturating_sub(32);
        }

        self.call_dprintf(&block, "\n\0", &[], storage)?;

        self.op_return(&block, &[]);

        self.create_function("print_felt252", vec![block], &[], FnAttributes::libfunc(false, false))
    }

    pub fn create_print_struct(
        &'ctx self,
        struct_type: &SierraType,
        sierra_type_declaration: TypeDeclaration,
    ) -> Result<()> {
        let block = self.new_block(&[struct_type.get_type()]);

        let arg = block.argument(0)?;

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

        self.create_function(
            &format!("print_{}", struct_name.as_str()),
            vec![block],
            &[],
            FnAttributes::libfunc(false, false),
        )
    }

    pub fn create_print_nullable(
        &'ctx self,
        struct_type: &SierraType,
        sierra_type_declaration: TypeDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let region = Region::new();
        let block = region.append_block(Block::new(&[(
            struct_type.get_type(),
            Location::unknown(&self.context),
        )]));

        let arg = block.argument(0)?.into();

        let function_type = create_fn_signature(&[struct_type.get_type()], &[]);

        let value_type_name = sierra_type_declaration.id.debug_name.unwrap();

        let is_null_op = self.op_llvm_extractvalue(&block, 1, arg, self.bool_type())?;
        let is_null = is_null_op.result(0)?.into();

        let is_null_block = region.append_block(Block::new(&[]));
        let is_nonnull_block = region.append_block(Block::new(&[]));

        self.op_cond_br(&block, is_null, &is_nonnull_block, &is_null_block, &[], &[]);

        self.call_dprintf(&is_null_block, "0\n", &[], storage)?;
        self.op_return(&is_null_block, &[]);

        let component_type_id = match &sierra_type_declaration.long_id.generic_args[0] {
            GenericArg::Type(x) => x.debug_name.as_ref().unwrap().as_str(),
            _ => unreachable!(),
        };
        // 0 is the element type
        let value_op = self.op_llvm_extractvalue(
            &is_nonnull_block,
            0,
            arg,
            struct_type.get_field_types().unwrap()[0],
        )?;
        let value = value_op.result(0)?.into();
        self.op_func_call(
            &is_nonnull_block,
            &format!("print_{}", component_type_id),
            &[value],
            &[],
        )?;
        self.op_return(&is_nonnull_block, &[]);

        let func = self.op_func(
            &format!("print_{}", value_type_name.as_str()),
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
        enum_sierra_type: &SierraType,
        sierra_type_declaration: TypeDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let enum_type = enum_sierra_type.get_type();

        let entry_block = self.new_block(&[enum_type]);

        let enum_value: Value = entry_block.argument(0)?.into();

        // if its a panic enum wrapper, don't print the tag, as they are meant to be invisible
        let is_panic_enum = match &sierra_type_declaration.long_id.generic_args[0] {
            GenericArg::UserType(x) => {
                x.debug_name.as_ref().unwrap().starts_with("core::PanicResult::")
            }
            _ => unreachable!(),
        };

        let variants_types = if let SierraType::Enum { variants_types, .. } = enum_sierra_type {
            variants_types
        } else {
            panic!("sierra_type_declaration should be a enum")
        };

        // get the tag
        let tag_op = self.call_enum_get_tag(&entry_block, enum_value, enum_sierra_type, storage)?;
        let tag = tag_op.result(0)?.into();

        // create a block for each variant of the enum
        let variant_blocks =
            variants_types.iter().map(|var_ty| (self.new_block(&[]), var_ty)).collect_vec();

        // default block
        let default_block = self.new_block(&[]);
        self.op_return(&default_block, &[]);

        if !is_panic_enum {
            // Print the tag. Extending it to 32 bits allows for better printf portability
            let tag_32bit = self.op_zext(&entry_block, tag, self.u32_type());
            self.call_dprintf(&entry_block, "%X\n\0", &[tag_32bit.result(0)?.into()], storage)?;
        }

        let blockrefs = variant_blocks.iter().map(|x| &x.0).collect_vec();
        let case_values = (0..variants_types.len()).map(|x| x.to_string()).collect_vec();

        let blockrefs_with_ops = blockrefs.iter().map(|x| (x.deref(), [].as_slice())).collect_vec();
        entry_block.append_operation(cf::switch(
            &self.context,
            &case_values,
            tag,
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

        let enum_felt_width = enum_sierra_type.get_felt_representation_width();

        let enum_name = sierra_type_declaration.id.debug_name.unwrap();

        for (i, (block, var_ty)) in variant_blocks.iter().enumerate() {
            let variant_felt_width = var_ty.get_felt_representation_width();
            let unused_felt_count = enum_felt_width - 1 - variant_felt_width;
            if unused_felt_count != 0 && !is_panic_enum {
                self.call_dprintf(block, &"0\n".repeat(unused_felt_count), &[], storage)?;
            }

            let component_type_name = component_type_ids[i].debug_name.as_ref().unwrap();

            let value_op = self.call_enum_get_data_as_variant_type(
                block,
                &enum_name,
                enum_value,
                enum_sierra_type,
                i,
                storage,
            )?;
            let value = value_op.result(0)?.into();
            self.op_func_call(block, &format!("print_{}", component_type_name), &[value], &[])?;
            self.op_return(block, &[]);
        }

        let mut all_blocks = vec![entry_block];
        all_blocks.extend(variant_blocks.into_iter().map(|(b, _)| b));
        all_blocks.push(default_block);

        self.create_function(
            &format!("print_{}", enum_name.as_str()),
            all_blocks,
            &[],
            FnAttributes::libfunc(false, false),
        )
    }

    pub fn create_print_array(
        &'ctx self,
        array_type: &SierraType,
        sierra_type_declaration: TypeDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let entry_block = self.new_block(&[array_type.get_type()]);

        let array_value_type_id = match &sierra_type_declaration.long_id.generic_args[0] {
            GenericArg::Type(type_id) => type_id,
            _ => panic!("Struct type declaration arguments after the first should all be resolved"),
        };

        let array_value = entry_block.argument(0)?.into();

        let lower_bound_op = self.op_u32_const(&entry_block, "0");
        let lower_bound: Value = lower_bound_op.result(0)?.into();

        let upper_bound_op =
            self.call_array_len_impl(&entry_block, array_value, array_type, storage)?;
        let upper_bound: Value = upper_bound_op.result(0)?.into();

        let step_op = self.op_u32_const(&entry_block, "1");
        let step: Value = step_op.result(0)?.into();

        let check_block = self.new_block(&[self.u32_type()]);
        let loop_block = self.new_block(&[self.u32_type()]);
        let end_block = self.new_block(&[]);

        self.op_br(&entry_block, &check_block, &[lower_bound]);

        let lower_bound = check_block.argument(0)?.into();
        let cmp_op = self.op_cmp(&check_block, CmpOp::UnsignedLessThan, lower_bound, upper_bound);
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

        // get the data at the given index
        let element_op =
            self.call_array_get_unchecked(&loop_block, array_value, idx, array_type, storage)?;
        let element = element_op.result(0)?.into();

        let array_value_type_name = array_value_type_id.debug_name.as_ref().unwrap();
        self.op_func_call(
            &loop_block,
            &format!("print_{}", array_value_type_name),
            &[element],
            &[],
        )?;

        let new_idx_op = self.op_add(&loop_block, idx, step);
        let new_idx = new_idx_op.result(0)?.into();

        self.op_br(&loop_block, &check_block, &[new_idx]);

        self.op_return(&end_block, &[]);

        let array_type_name = sierra_type_declaration.id.debug_name.unwrap();

        self.create_function(
            &format!("print_{}", array_type_name.as_str()),
            vec![entry_block, check_block, loop_block, end_block],
            &[],
            FnAttributes::libfunc(false, false),
        )
    }

    pub fn create_print_uint(
        &'ctx self,
        uint_type: &SierraType,
        sierra_type_declaration: TypeDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let block = self.new_block(&[uint_type.get_type()]);

        let arg = block.argument(0)?.into();

        let uint_name = sierra_type_declaration.id.debug_name.unwrap();

        match uint_name.as_str() {
            "u8" => {
                let value_32bit = self.op_zext(&block, arg, self.u32_type());
                self.call_dprintf(&block, "%X\n", &[value_32bit.result(0)?.into()], storage)?;
            }
            "u16" => {
                let value_32bit = self.op_zext(&block, arg, self.u32_type());
                self.call_dprintf(&block, "%X\n", &[value_32bit.result(0)?.into()], storage)?;
            }
            "u32" => {
                self.call_dprintf(&block, "%X\n", &[arg], storage)?;
            }
            "u64" => self.call_dprintf(&block, "%lX\n", &[arg], storage)?,
            "u128" => {
                let lower = self.op_trunc(&block, arg, self.u64_type());
                let shift_amount = self.op_u128_const(&block, "64");
                let upper_shifted = self.op_shru(&block, arg, shift_amount.result(0)?.into());
                let upper = self.op_trunc(&block, upper_shifted.result(0)?.into(), self.u64_type());
                self.call_dprintf(
                    &block,
                    "%lX%016lX\n",
                    &[upper.result(0)?.into(), lower.result(0)?.into()],
                    storage,
                )?;
            }
            _ => unreachable!("Encountered unexpected type {} when creating uint print", uint_name),
        }

        self.op_return(&block, &[]);

        self.create_function(
            &format!("print_{}", uint_name.as_str()),
            vec![block],
            &[],
            FnAttributes::libfunc(false, false),
        )
    }

    /// Utility method to create a print_felt call.
    pub fn call_print_felt(&'ctx self, block: &Block<'ctx>, value: Value) -> Result<()> {
        self.op_func_call(block, "print_felt", &[value], &[])?;
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

pub fn get_type_id(generic_arg: &GenericArg) -> String {
    match generic_arg {
        GenericArg::Type(type_id) => type_id.id.to_string(),
        _ => unreachable!("Generic argument expected to be a Type, found {:?}", generic_arg),
    }
}
