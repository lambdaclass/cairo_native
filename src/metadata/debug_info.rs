use mlir_sys::MlirAttribute;

#[derive(Clone, Debug)]
pub struct FunctionDebugInfo {
    pub file: MlirAttribute,
    pub scope: MlirAttribute,
    pub subprogram: MlirAttribute,
}
