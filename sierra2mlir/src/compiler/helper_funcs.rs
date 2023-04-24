use super::{Compiler, Storage};
use color_eyre::Result;
use melior_next::ir::{
    operation, Block, BlockRef, Location, Module, NamedAttribute, OperationRef, Value,
};
use std::ops::Deref;

impl<'ctx> Compiler<'ctx> {
    fn create_helper_felt_pow(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        if storage.helperfuncs.contains("dprintf") {
            return Ok(());
        }

        let module =
            Module::parse(&self.context, include_str!("helper_funcs/felt_pow.mlir")).unwrap();

        // Transfer its contents to the target block.
        fn push_recursive(m: BlockRef, op: OperationRef) {
            m.append_operation(op.deref().to_owned());
            if let Some(next_op) = op.next_in_block() {
                push_recursive(m, next_op);
            }
        }

        push_recursive(self.module.body(), module.body().first_operation().unwrap());
        storage.helperfuncs.insert("helper_felt_pow".to_string());
        Ok(())
    }
}

impl<'ctx> Compiler<'ctx> {
    pub fn call_helper_felt_pow<'a>(
        &'ctx self,
        block: &'a Block,
        base: Value,
        exponent: Value,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'a>> {
        self.create_helper_felt_pow(storage)?;

        let op = block.append_operation(
            operation::Builder::new("func.call", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "callee",
                    "@helper_felt_pow",
                )?])
                .add_operands(&[base, exponent])
                .add_results(&[self.felt_type()])
                .build(),
        );

        Ok(op)
    }
}
