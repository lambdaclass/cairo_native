use color_eyre::Result;
use melior_next::ir::{Block, OperationRef, Value};

use crate::compiler::{mlir_ops::CmpOp, Compiler};

/// Gas type is u128.
pub static GAS_COUNTER_SYMBOL: &str = "__gas_counter";

impl<'ctx> Compiler<'ctx> {
    /// Add the gas counter global
    pub fn create_gas_global(&self) -> Result<()> {
        let gas = self.gas.as_ref().expect("gas should exist when using gas helper functions");
        self.op_llvm_global(
            &self.module.body(),
            GAS_COUNTER_SYMBOL,
            self.u128_type(),
            &gas.available_gas.to_string(),
        )?;
        Ok(())
    }

    /// Returns the address of the counter and the loaded value operations.
    /// (address_op, load_op)
    pub fn call_get_gas_counter<'block>(
        &'ctx self,
        block: &'block Block,
    ) -> Result<(OperationRef<'block>, OperationRef<'block>)> {
        assert!(self.gas.is_some(), "gas should exist when using gas helper functions");
        let addr_op = self.op_llvm_addressof(block, GAS_COUNTER_SYMBOL)?;
        let addr = addr_op.result(0)?.into();
        Ok((addr_op, self.op_llvm_load(block, addr, self.u128_type())?))
    }

    /// Value must be u128_type.
    pub fn call_decrease_gas_counter<'block>(
        &'ctx self,
        block: &'block Block,
        value: Value,
    ) -> Result<()> {
        assert!(self.gas.is_some(), "gas should exist when using gas helper functions");
        let (addr_op, current_value_op) = self.call_get_gas_counter(block)?;
        let addr = addr_op.result(0)?.into();
        let current_value = current_value_op.result(0)?.into();
        let new_val_op = self.op_sub(block, current_value, value);
        let new_val = new_val_op.result(0)?.into();
        self.op_llvm_store(block, new_val, addr)?;
        Ok(())
    }

    /// Value must be u128.
    pub fn call_increase_gas_counter<'block>(
        &'ctx self,
        block: &'block Block,
        value: Value,
    ) -> Result<()> {
        assert!(self.gas.is_some(), "gas should exist when using gas helper functions");
        let (addr_op, current_value_op) = self.call_get_gas_counter(block)?;
        let addr = addr_op.result(0)?.into();
        let current_value = current_value_op.result(0)?.into();
        let new_val_op = self.op_add(block, current_value, value);
        let new_val = new_val_op.result(0)?.into();
        self.op_llvm_store(block, new_val, addr)?;
        Ok(())
    }

    /// Returns the resulting cmp operation comparing current gas to the given value.
    ///
    /// `value <= current_gas`
    ///
    /// Value should be u128.
    pub fn call_has_enough_gas<'block>(
        &'ctx self,
        block: &'block Block,
        value: Value,
    ) -> Result<OperationRef<'block>> {
        assert!(self.gas.is_some(), "gas should exist when using gas helper functions");
        let (_, gas_value_op) = self.call_get_gas_counter(block)?;
        let gas_value = gas_value_op.result(0)?.into();
        let cmp_op = self.op_cmp(block, CmpOp::UnsignedLessThanEqual, value, gas_value);
        Ok(cmp_op)
    }
}
