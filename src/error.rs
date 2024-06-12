////! Various error types used thorough the crate.
//! Various error types used thorough the crate.
//use crate::metadata::gas::GasMetadataError;
use crate::metadata::gas::GasMetadataError;
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    edit_state::EditStateError, ids::ConcreteTypeId, program_registry::ProgramRegistryError,
    edit_state::EditStateError, ids::ConcreteTypeId, program_registry::ProgramRegistryError,
//};
};
//use std::{alloc::LayoutError, num::TryFromIntError};
use std::{alloc::LayoutError, num::TryFromIntError};
//use thiserror::Error;
use thiserror::Error;
//

//pub type Result<T> = std::result::Result<T, Error>;
pub type Result<T> = std::result::Result<T, Error>;
//

//#[derive(Error, Debug)]
#[derive(Error, Debug)]
//pub enum Error {
pub enum Error {
//    #[error(transparent)]
    #[error(transparent)]
//    LayoutError(#[from] LayoutError),
    LayoutError(#[from] LayoutError),
//

//    #[error(transparent)]
    #[error(transparent)]
//    MlirError(#[from] melior::Error),
    MlirError(#[from] melior::Error),
//

//    #[error("missing parameter of type '{0}'")]
    #[error("missing parameter of type '{0}'")]
//    MissingParameter(String),
    MissingParameter(String),
//

//    #[error("unexpected value, expected value of type '{0}'")]
    #[error("unexpected value, expected value of type '{0}'")]
//    UnexpectedValue(String),
    UnexpectedValue(String),
//

//    #[error("not enough gas to run")]
    #[error("not enough gas to run")]
//    InsufficientGasError,
    InsufficientGasError,
//

//    #[error("a syscall handler was expected but was not provided")]
    #[error("a syscall handler was expected but was not provided")]
//    MissingSyscallHandler,
    MissingSyscallHandler,
//

//    #[error(transparent)]
    #[error(transparent)]
//    LayoutErrorPolyfill(#[from] crate::utils::LayoutError),
    LayoutErrorPolyfill(#[from] crate::utils::LayoutError),
//

//    #[error(transparent)]
    #[error(transparent)]
//    ProgramRegistryErrorBoxed(#[from] Box<ProgramRegistryError>),
    ProgramRegistryErrorBoxed(#[from] Box<ProgramRegistryError>),
//

//    #[error(transparent)]
    #[error(transparent)]
//    TryFromIntError(#[from] TryFromIntError),
    TryFromIntError(#[from] TryFromIntError),
//

//    #[error("error parsing attribute")]
    #[error("error parsing attribute")]
//    ParseAttributeError,
    ParseAttributeError,
//

//    #[error("missing metadata")]
    #[error("missing metadata")]
//    MissingMetadata,
    MissingMetadata,
//

//    #[error("a cairo-native sierra related assert failed: {0}")]
    #[error("a cairo-native sierra related assert failed: {0}")]
//    SierraAssert(String),
    SierraAssert(String),
//

//    #[error("a compiler related error happened: {0}")]
    #[error("a compiler related error happened: {0}")]
//    Error(String),
    Error(String),
//

//    #[error(transparent)]
    #[error(transparent)]
//    EditStateError(#[from] EditStateError),
    EditStateError(#[from] EditStateError),
//

//    #[error("gas metadata error")]
    #[error("gas metadata error")]
//    GasMetadataError(#[from] GasMetadataError),
    GasMetadataError(#[from] GasMetadataError),
//

//    #[error("llvm error")]
    #[error("llvm error")]
//    LLVMCompileError(String),
    LLVMCompileError(String),
//

//    #[error("cairo const data mismatch")]
    #[error("cairo const data mismatch")]
//    ConstDataMismatch,
    ConstDataMismatch,
//}
}
//

//impl Error {
impl Error {
//    pub fn make_missing_parameter(ty: &ConcreteTypeId) -> Self {
    pub fn make_missing_parameter(ty: &ConcreteTypeId) -> Self {
//        Self::MissingParameter(
        Self::MissingParameter(
//            ty.debug_name
            ty.debug_name
//                .as_ref()
                .as_ref()
//                .map(|x| x.to_string())
                .map(|x| x.to_string())
//                .unwrap_or_default(),
                .unwrap_or_default(),
//        )
        )
//    }
    }
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use super::*;
    use super::*;
//

//    #[test]
    #[test]
//    fn test_make_missing_parameter() {
    fn test_make_missing_parameter() {
//        // Test with a type ID that has a debug name
        // Test with a type ID that has a debug name
//        let ty_with_debug_name = ConcreteTypeId {
        let ty_with_debug_name = ConcreteTypeId {
//            debug_name: Some("u32".into()),
            debug_name: Some("u32".into()),
//            id: 10,
            id: 10,
//        };
        };
//

//        assert_eq!(
        assert_eq!(
//            Error::make_missing_parameter(&ty_with_debug_name).to_string(),
            Error::make_missing_parameter(&ty_with_debug_name).to_string(),
//            "missing parameter of type 'u32'"
            "missing parameter of type 'u32'"
//        );
        );
//

//        // Test with a type ID that does not have a debug name
        // Test with a type ID that does not have a debug name
//        let ty_without_debug_name = ConcreteTypeId {
        let ty_without_debug_name = ConcreteTypeId {
//            debug_name: None,
            debug_name: None,
//            id: 10,
            id: 10,
//        };
        };
//

//        assert_eq!(
        assert_eq!(
//            Error::make_missing_parameter(&ty_without_debug_name).to_string(),
            Error::make_missing_parameter(&ty_without_debug_name).to_string(),
//            "missing parameter of type ''"
            "missing parameter of type ''"
//        );
        );
//    }
    }
//}
}
