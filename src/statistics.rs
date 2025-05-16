use serde::Serialize;

#[derive(Default, Serialize)]
pub struct Statistics {
    pub sierra_type_count: usize,
    pub sierra_libfunc_count: usize,
    pub sierra_statement_count: usize,
    pub sierra_func_count: usize,
    // pub compilation_total_time_ms: u128,
    pub compilation_sierra_to_mlir_time_ms: u128,
    pub compilation_mlir_passes_time_ms: u128,
    // pub compilation_mlir_to_llvm_time_ms: u128,
    // pub compilation_llvm_to_object_time_ms: u128,
    pub compilation_linking_time_ms: u128,
}

impl Statistics {
    pub fn builder() -> StatisticsBuilder {
        StatisticsBuilder::default()
    }
}

#[derive(Default)]
pub struct StatisticsBuilder {
    pub sierra_type_count: Option<usize>,
    pub sierra_libfunc_count: Option<usize>,
    pub sierra_statement_count: Option<usize>,
    pub sierra_func_count: Option<usize>,
    pub compilation_total_time_ms: Option<u128>,
    pub compilation_sierra_to_mlir_time_ms: Option<u128>,
    pub compilation_mlir_passes_time_ms: Option<u128>,
    pub compilation_mlir_to_llvm_time_ms: Option<u128>,
    pub compilation_llvm_to_object_time_ms: Option<u128>,
    pub compilation_linking_time_ms: Option<u128>,
}

impl StatisticsBuilder {
    pub fn build(self) -> Option<Statistics> {
        Some(Statistics {
            sierra_type_count: self.sierra_type_count?,
            sierra_libfunc_count: self.sierra_libfunc_count?,
            sierra_statement_count: self.sierra_statement_count?,
            sierra_func_count: self.sierra_func_count?,
            // compilation_total_time_ms: self.compilation_total_time_ms?,
            compilation_sierra_to_mlir_time_ms: self.compilation_sierra_to_mlir_time_ms?,
            compilation_mlir_passes_time_ms: self.compilation_mlir_passes_time_ms?,
            // compilation_mlir_to_llvm_time_ms: self.compilation_mlir_to_llvm_time_ms?,
            // compilation_llvm_to_object_time_ms: self.compilation_llvm_to_object_time_ms?,
            compilation_linking_time_ms: self.compilation_linking_time_ms?,
        })
    }
}
