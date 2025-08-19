use cairo_lang_sierra::extensions::circuit::{CircuitInfo, GateOffsets};
use serde::Serialize;
use std::collections::BTreeMap;

use crate::{error::Result, native_panic};

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

/// Contains the following stats about a Sierra function:
/// - params_total_size: Total size of all the params
/// - return_types_total_size: Total size of all the params
#[derive(Debug, Default, Serialize)]
pub struct SierraFuncStats {
    pub params_total_size: usize,
    pub return_types_total_size: usize,
    pub times_used: usize,
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
/// in a program
#[derive(Debug, Default, Serialize)]
struct CircuitGatesStats {
    add_gate_count: usize,
    sub_gate_count: usize,
    mul_gate_count: usize,
    inverse_gate_count: usize,
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

    /// Counts the gates in a circuit. It uses the same algorithm used
    /// to evaluate the gates on a circuit when evaluating it.
    pub fn add_circuit_gates(&mut self, info: &CircuitInfo) -> Result<()> {
        let mut known_gates = vec![false; 1 + info.n_inputs + info.values.len()];
        known_gates[0] = true;
        for i in 0..info.n_inputs {
            known_gates[i + 1] = true;
        }

        let mut add_offsets = info.add_offsets.iter().peekable();
        let mut mul_offsets = info.mul_offsets.iter();

        loop {
            while let Some(&add_gate_offset) = add_offsets.peek() {
                let lhs = known_gates[add_gate_offset.lhs].to_owned();
                let rhs = known_gates[add_gate_offset.rhs].to_owned();
                let output = known_gates[add_gate_offset.output].to_owned();

                match (lhs, rhs, output) {
                    (true, true, false) => {
                        // ADD
                        self.sierra_circuit_gates_count.add_gate_count += 1;
                        known_gates[add_gate_offset.output] = true;
                    }
                    (false, true, true) => {
                        // SUB
                        self.sierra_circuit_gates_count.sub_gate_count += 1;
                        known_gates[add_gate_offset.lhs] = true;
                    }
                    _ => break,
                }
                add_offsets.next();
            }

            if let Some(&GateOffsets { lhs, rhs, output }) = mul_offsets.next() {
                let lhs_value = known_gates[lhs];
                let rhs_value = known_gates[rhs];
                let output_value = known_gates[output];

                match (lhs_value, rhs_value, output_value) {
                    (true, true, false) => {
                        // MUL
                        self.sierra_circuit_gates_count.mul_gate_count += 1;
                        known_gates[output] = true;
                    }
                    (false, true, true) => {
                        self.sierra_circuit_gates_count.inverse_gate_count += 1;
                        known_gates[lhs] = true;
                    }
                    _ => native_panic!("Imposible circuit"), // It should never reach this point, since it would have failed in the compilation before
                }
            } else {
                break;
            }
        }
        Ok(())
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
