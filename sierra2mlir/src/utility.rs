use itertools::Itertools;

use cairo_lang_sierra::program::{GenericArg, TypeDeclaration};
use melior_next::ir::{
    operation, Block, BlockRef, Location, NamedAttribute, Region, Type, TypeLike, Value, ValueLike,
};

use crate::{
    compiler::{Compiler, SierraType},
    statements::Variable,
};
use color_eyre::Result;

impl<'ctx> Compiler<'ctx> {
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
    pub fn call_printf(
        &'ctx self,
        block: BlockRef<'ctx>,
        fmt: &str,
        values: &[Value],
    ) -> Result<()> {
        let i32_type = Type::integer(&self.context, 32);
        let fmt_len = fmt.as_bytes().len();

        let i8_type = Type::integer(&self.context, 8);
        let arr_ty =
            Type::parse(&self.context, &format!("!llvm.array<{fmt_len} x {i8_type}>")).unwrap();
        let data_op = self.op_llvm_alloca(&block, i8_type, fmt_len)?;
        let addr: Value = data_op.result(0)?.into();

        // https://discourse.llvm.org/t/array-globals-in-llvm-dialect/68229
        // To create a constant array, we need to use a dense array attribute, which has a tensor type,
        // which is then interpreted as a llvm.array type.
        let fmt_data = self.op_llvm_const(
            &block,
            &format!(
                "dense<[{}]> : tensor<{} x {}>",
                fmt.as_bytes().iter().join(", "),
                fmt_len,
                i8_type
            ),
            arr_ty,
        );

        self.op_llvm_store(&block, fmt_data.result(0)?.into(), addr)?;

        let target_fd = self.op_u32_const(&block, "1");

        let mut args = vec![target_fd.result(0)?.into(), addr];
        args.extend(values);

        self.op_llvm_call(&block, "dprintf", &args, &[i32_type])?;
        Ok(())
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
            self.call_printf(block, "%08X\0", &[truncated])?;

            bit_width = bit_width.saturating_sub(32);
        }

        self.call_printf(block, "\n\0", &[])?;

        self.op_return(&block, &[]);

        let function_type = create_fn_signature(&args_types, &[]);

        let func = self.op_func("print_felt252", &function_type, vec![region], false, false)?;

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
            false,
            false,
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

        let enum_value = entry_block.argument(0)?;

        let function_type = create_fn_signature(&[enum_type.get_type()], &[]);

        if let SierraType::Enum {
            ty,
            tag_type,
            storage_bytes_len: _,
            storage_type: _,
            variants_types,
        } = enum_type
        {
            // create a block for each variant of the enum
            let mut blocks = vec![];

            for var_ty in variants_types {
                let block = region.append_block(Block::new(&[]));
                blocks.push((block, var_ty));
            }

            let default_block = region.append_block(Block::new(&[]));
            self.op_return(&default_block, &[]);

            // type is !llvm.struct<(i16, array<N x i8>)>

            // get the tag
            let tag_value_op =
                self.op_llvm_extractvalue(&entry_block, 0, enum_value.into(), *tag_type)?;
            let tag_value = tag_value_op.result(0)?.into();

            self.call_printf(entry_block, "%X\n\0", &[tag_value])?;

            // put the enum in a alloca for easier interpreting

            let enum_alloca_op = self.op_llvm_alloca(&entry_block, *ty, 1)?;
            let enum_ptr = enum_alloca_op.result(0)?.into();

            self.op_llvm_store(&entry_block, enum_value.into(), enum_ptr)?;

            let value_ptr_op = self.op_llvm_gep(&entry_block, 1, enum_ptr, *ty)?;
            let value_ptr = value_ptr_op.result(0)?.into();

            let blockrefs = blocks.iter().map(|x| x.0).collect_vec();
            let case_values =
                variants_types.iter().enumerate().map(|x| x.0.to_string()).collect_vec();

            self.op_switch(&entry_block, &case_values, tag_value, default_block, &blockrefs)?;

            let component_type_ids = sierra_type_declaration.long_id.generic_args[1..]
                .iter()
                .map(|member_type| match member_type {
                    GenericArg::Type(type_id) => type_id,
                    _ => panic!(
                        "Struct type declaration arguments after the first should all be resolved"
                    ),
                })
                .collect_vec();

            for (i, (block, var_ty)) in blocks.iter().enumerate() {
                let component_type_name = component_type_ids[i].debug_name.as_ref().unwrap();
                let value_op = self.op_llvm_load(block, value_ptr, var_ty.get_type())?;
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
            false,
            false,
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

        let arg = block.argument(0)?;

        let function_type = create_fn_signature(&[uint_type.get_type()], &[]);

        let uint_name = sierra_type_declaration.id.debug_name.unwrap();

        self.call_printf(block, "%X\n\0", &[arg.into()])?;

        self.op_return(&block, &[]);

        let func = self.op_func(
            &format!("print_{}", uint_name.as_str()),
            &function_type,
            vec![region],
            false,
            false,
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
