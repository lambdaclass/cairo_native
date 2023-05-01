use crate::{
    compiler::{fn_attributes::FnAttributes, Compiler, Storage},
    sierra_type::SierraType,
};
use color_eyre::Result;

impl<'ctx> Compiler<'ctx> {
    pub fn create_nullable_is_null(
        &'ctx self,
        nullable_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let func_name =
            format!("nullable_is_null<{}>", nullable_type.get_field_types().unwrap()[0]);

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[nullable_type.get_type()]);
        let nullable_value = block.argument(0)?.into();
        let is_null_op = self.op_llvm_extractvalue(&block, 1, nullable_value, self.bool_type())?;
        let is_null = is_null_op.result(0)?.into();
        self.op_return(&block, &[is_null]);

        self.create_function(
            &func_name,
            vec![block],
            &[self.bool_type()],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }

    pub fn create_nullable_unwrap_unsafe(
        &'ctx self,
        nullable_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let wrapped_type = nullable_type.get_field_types().unwrap()[0];
        let func_name = format!("nullable_unwrap_unsafe<{}>", wrapped_type);

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[nullable_type.get_type()]);
        let nullable_value = block.argument(0)?.into();
        let is_null_op = self.op_llvm_extractvalue(&block, 0, nullable_value, wrapped_type)?;
        let is_null = is_null_op.result(0)?.into();
        self.op_return(&block, &[is_null]);

        self.create_function(
            &func_name,
            vec![block],
            &[wrapped_type],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }
}
