use std::collections::BTreeMap;

use serde::Serialize;

/// A set of compilation statistics gathered during the compilation.
/// It should be completely filled at the end of the compilation.
#[derive(Default, Serialize)]
pub struct Statistics {
    pub sierra_type_count: Option<usize>,
    pub sierra_libfunc_count: Option<usize>,
    pub sierra_statement_count: Option<usize>,
    pub sierra_func_count: Option<usize>,
    pub sierra_libfunc_frequency: BTreeMap<String, u128>,
    pub mlir_operation_count: Option<u128>,
    pub mlir_operations_by_libfunc: BTreeMap<String, u128>,
    pub llvmir_instruction_count: Option<u128>,
    pub llvmir_virtual_register_count: Option<u128>,
    pub llvmir_opcode_frequency: BTreeMap<String, u128>,
    pub compilation_total_time_ms: Option<u128>,
    pub compilation_sierra_to_mlir_time_ms: Option<u128>,
    pub compilation_mlir_passes_time_ms: Option<u128>,
    pub compilation_mlir_to_llvm_time_ms: Option<u128>,
    pub compilation_llvm_passes_time_ms: Option<u128>,
    pub compilation_llvm_to_object_time_ms: Option<u128>,
    pub compilation_linking_time_ms: Option<u128>,
    pub object_size_bytes: Option<usize>,
}

impl Statistics {
    pub fn validate(&self) -> bool {
        self.sierra_type_count.is_some()
            && self.sierra_libfunc_count.is_some()
            && self.sierra_statement_count.is_some()
            && self.sierra_func_count.is_some()
            && !self.sierra_libfunc_frequency.is_empty()
            && self.mlir_operation_count.is_some()
            && !self.mlir_operations_by_libfunc.is_empty()
            && self.llvmir_instruction_count.is_some()
            && self.llvmir_virtual_register_count.is_some()
            && !self.llvmir_opcode_frequency.is_empty()
            && self.compilation_total_time_ms.is_some()
            && self.compilation_sierra_to_mlir_time_ms.is_some()
            && self.compilation_mlir_passes_time_ms.is_some()
            && self.compilation_mlir_to_llvm_time_ms.is_some()
            && self.compilation_llvm_passes_time_ms.is_some()
            && self.compilation_llvm_to_object_time_ms.is_some()
            && self.compilation_linking_time_ms.is_some()
            && self.object_size_bytes.is_some()
    }
}

/// Clones a variable of type `Option<&mut T>` without consuming self
///
/// # Example
///
/// The following example would fail to compile otherwise.
///
/// ```
/// # use cairo_native::clone_option_mut;
/// fn consume(v: Option<&mut Vec<u8>>) {}
///
/// let mut vec = Vec::new();
/// let option = Some(&mut vec);
/// consume(clone_option_mut!(option));
/// consume(option);
/// ```
#[macro_export]
macro_rules! clone_option_mut {
    ( $var:ident ) => {
        match $var {
            None => None,
            Some(&mut ref mut s) => Some(s),
        }
    };
}
