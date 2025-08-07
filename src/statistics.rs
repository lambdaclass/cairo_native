use cairo_lang_sierra::{
    extensions::{
        circuit::CircuitInfo,
        core::{CoreLibfunc, CoreType},
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::Serialize;
use std::collections::BTreeMap;

use crate::types::TypeBuilder;

/// A set of compilation statistics gathered during the compilation.
/// It should be completely filled at the end of the compilation.
#[derive(Default, Serialize)]
pub struct Statistics {
    /// Number of types defined in the Sierra code.
    pub sierra_type_count: Option<usize>,
    /// Number of libfuncs defined in the Sierra code.
    pub sierra_libfunc_count: Option<usize>,
    /// Number of statements contained in the Sierra code.
    pub sierra_statement_count: Option<usize>,
    /// Number of user functions defined in the Sierra code.
    pub sierra_func_count: Option<usize>,
    /// Stats of the declared types in Sierra.
    pub sierra_declared_types_stats: BTreeMap<String, SierraDeclaredTypeStats>,
    /// Stats about params and return types of each Sierra function.
    pub sierra_func_stats: BTreeMap<String, SierraFuncStats>,
    /// Number of statements for each distinct libfunc.
    pub sierra_libfunc_frequency: BTreeMap<String, u128>,
    /// Number of times each circuit gate is used.
    sierra_circuit_gates_count: CircuitGatesStats,
    /// Number of MLIR operations generated.
    pub mlir_operation_count: Option<u128>,
    /// Number of MLIR operations generated for each distinct libfunc.
    pub mlir_operations_by_libfunc: BTreeMap<String, u128>,
    /// Number of LLVMIR instructions generated.
    pub llvmir_instruction_count: Option<u128>,
    /// Number of LLVMIR virtual registers defined.
    pub llvmir_virtual_register_count: Option<u128>,
    /// Number of LLVMIR instructions for each distinct opcode.
    pub llvmir_opcode_frequency: BTreeMap<String, u128>,
    /// Total compilation time.
    pub compilation_total_time_ms: Option<u128>,
    /// Time spent at Sierra to MLIR.
    pub compilation_sierra_to_mlir_time_ms: Option<u128>,
    /// Time spent at MLIR passes.
    pub compilation_mlir_passes_time_ms: Option<u128>,
    /// Time spent at MLIR to LLVMIR translation.
    pub compilation_mlir_to_llvm_time_ms: Option<u128>,
    /// Time spent at LLVM passes.
    pub compilation_llvm_passes_time_ms: Option<u128>,
    /// Time spent at LLVM to object compilation.
    pub compilation_llvm_to_object_time_ms: Option<u128>,
    /// Time spent at linking the shared library.
    pub compilation_linking_time_ms: Option<u128>,
    /// Size of the compiled object.
    pub object_size_bytes: Option<usize>,
}

/// Contains the stats about a Sierra function:
/// - params_quant: Quantity of params
/// - params_total_size: Total size of all the params
/// - return_types_quant: Quantity of return types
/// - return_types_total_size: Total size of all the params
#[derive(Debug, Default, Serialize)]
pub struct SierraFuncStats {
    pub params_quant: usize,
    pub params_total_size: usize,
    pub return_types_quant: usize,
    pub return_types_total_size: usize,
}

/// Contains the stats for each Sierra declared type:
/// - concrete_type: The concrete type (e.g Struct)
/// - size: Layout size of the whole type
/// - as_param_count: Number of times the type is used as a param in a libfunc
#[derive(Debug, Default, Serialize)]
pub struct SierraDeclaredTypeStats {
    pub concrete_type: String,
    pub size: usize,
    pub as_param_count: usize,
}

/// Contains the quantity of each circuit gate
/// in a contract
#[derive(Debug, Default, Serialize)]
struct CircuitGatesStats {
    add_gate: usize,
    sub_gate: usize,
    mul_gate: usize,
    inverse_gate: usize,
}

impl Statistics {
    pub fn validate(&self) -> bool {
        self.sierra_type_count.is_some()
            && self.sierra_libfunc_count.is_some()
            && self.sierra_statement_count.is_some()
            && self.sierra_func_count.is_some()
            && self.mlir_operation_count.is_some()
            && self.llvmir_instruction_count.is_some()
            && self.llvmir_virtual_register_count.is_some()
            && self.compilation_total_time_ms.is_some()
            && self.compilation_sierra_to_mlir_time_ms.is_some()
            && self.compilation_mlir_passes_time_ms.is_some()
            && self.compilation_mlir_to_llvm_time_ms.is_some()
            && self.compilation_llvm_passes_time_ms.is_some()
            && self.compilation_llvm_to_object_time_ms.is_some()
            && self.compilation_linking_time_ms.is_some()
            && self.object_size_bytes.is_some()
    }

    /// Gets the size of the full set of params of a Sierra function
    pub fn get_types_total_size(
        &self,
        types_ids: &[ConcreteTypeId],
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    ) -> usize {
        types_ids
            .iter()
            .fold(0, |accum, type_id| match registry.get_type(type_id) {
                Ok(concrete_type) => accum + concrete_type.layout(registry).unwrap().size(),
                Err(_) => accum,
            })
    }

    /// Counts the gates in a circuit
    pub fn add_circuit_gates(&mut self, info: &CircuitInfo) {
        for gate_offset in &info.add_offsets {
            if gate_offset.lhs > gate_offset.output {
                // SUB
                self.sierra_circuit_gates_count.sub_gate += 1;
            } else {
                // ADD
                self.sierra_circuit_gates_count.add_gate += 1;
            }
        }

        for gate_offset in &info.mul_offsets {
            if gate_offset.lhs > gate_offset.output {
                // INVERSE
                self.sierra_circuit_gates_count.inverse_gate += 1;
            } else {
                // MUL
                self.sierra_circuit_gates_count.mul_gate += 1;
            }
        }
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
