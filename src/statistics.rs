use std::collections::BTreeMap;

use serde::Serialize;

#[derive(Default, Serialize)]
pub struct Statistics {
    pub sierra_type_count: Option<usize>,
    pub sierra_libfunc_count: Option<usize>,
    pub sierra_statement_count: Option<usize>,
    pub sierra_func_count: Option<usize>,
    pub sierra_libfunc_frequency: BTreeMap<String, u128>,
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
