use itertools::Itertools;

use melior_next::ir::{
    operation, Block, Location, NamedAttribute, OperationRef, Region, Type, Value,
};

use crate::compiler::Compiler;
use color_eyre::Result;

use super::Storage;

// Function creation
impl<'ctx> Compiler<'ctx> {
    /// Creates the external function definition for printf.
    fn create_dprintf(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        if storage.helperfuncs.contains("dprintf") {
            return Ok(());
        }

        storage.helperfuncs.insert("dprintf".to_string());

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

    // Free is not used yet, but should be used down the line
    #[allow(dead_code)]
    fn create_free(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        if storage.helperfuncs.contains("free") {
            return Ok(());
        }

        storage.helperfuncs.insert("free".to_string());

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

    fn create_realloc(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        if storage.helperfuncs.contains("realloc") {
            return Ok(());
        }

        storage.helperfuncs.insert("realloc".to_string());

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

        Ok(())
    }

    fn create_memmove(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        if storage.helperfuncs.contains("memmove") {
            return Ok(());
        }

        storage.helperfuncs.insert("memmove".to_string());

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

        Ok(())
    }

    pub fn create_utils(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        if storage.helperfuncs.contains("sierra2mlir_util_ec_point_from_x_nz") {
            return Ok(());
        }

        storage.helperfuncs.insert("sierra2mlir_util_ec_point_from_x_nz".to_string());

        let region = Region::new();
        let func = operation::Builder::new("func.func", Location::unknown(&self.context))
            .add_attributes(&[
                NamedAttribute::new_parsed(
                    &self.context,
                    "sym_name",
                    "\"sierra2mlir_util_ec_point_from_x_nz\"",
                )
                .unwrap(),
                NamedAttribute::new_parsed(
                    &self.context,
                    "function_type",
                    "(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()",
                )
                .unwrap(),
                NamedAttribute::new_parsed(&self.context, "sym_visibility", "\"private\"").unwrap(),
            ])
            .add_regions(vec![region])
            .build();

        self.module.body().append_operation(func);
        Ok(())
    }
}

// Function use
impl<'ctx> Compiler<'ctx> {
    /// Utility function to create a printf call.
    /// Null-terminates the string iff it is not already null-terminated
    pub fn call_dprintf<'block>(
        &'ctx self,
        block: &'block Block,
        fmt: &str,
        values: &[Value],
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        self.create_dprintf(storage)?;

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
    pub fn call_realloc<'block>(
        &'ctx self,
        block: &'block Block,
        ptr: Value,
        size: Value,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        self.create_realloc(storage)?;
        self.op_llvm_call(block, "realloc", &[ptr, size], &[self.llvm_ptr_type()])
    }

    pub fn call_memmove<'block>(
        &'ctx self,
        block: &'block Block,
        dst: Value,
        src: Value,
        size: Value,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        self.create_memmove(storage)?;
        self.op_llvm_call(block, "memmove", &[dst, src, size], &[self.llvm_ptr_type()])
    }
}
