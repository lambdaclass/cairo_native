//! Various error types used thorough the crate.
use crate::metadata::gas::GasMetadataError;
use cairo_lang_sierra::extensions::modules::utils::Range;
use cairo_lang_sierra::{
    edit_state::EditStateError, ids::ConcreteTypeId, program_registry::ProgramRegistryError,
};
use std::{alloc::LayoutError, num::TryFromIntError};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    LayoutError(#[from] LayoutError),

    #[error(transparent)]
    MlirError(#[from] melior::Error),

    #[error("missing parameter of type '{0}'")]
    MissingParameter(String),

    #[error("unexpected value, expected value of type '{0}'")]
    UnexpectedValue(String),

    #[error("not enough gas to run")]
    InsufficientGasError,

    #[error("a syscall handler was expected but was not provided")]
    MissingSyscallHandler,

    #[error(transparent)]
    LayoutErrorPolyfill(#[from] crate::utils::LayoutError),

    #[error(transparent)]
    ProgramRegistryErrorBoxed(#[from] Box<ProgramRegistryError>),

    #[error(transparent)]
    TryFromIntError(#[from] TryFromIntError),

    #[error("error parsing attribute")]
    ParseAttributeError,

    #[error("missing metadata")]
    MissingMetadata,

    #[error(transparent)]
    SierraAssert(SierraAssertError),

    #[error("a compiler related error happened: {0}")]
    Error(String),

    #[error(transparent)]
    EditStateError(#[from] EditStateError),

    #[error("gas metadata error")]
    GasMetadataError(#[from] GasMetadataError),

    #[error("llvm error")]
    LLVMCompileError(String),

    #[error("cairo const data mismatch")]
    ConstDataMismatch,
}

impl Error {
    pub fn make_missing_parameter(ty: &ConcreteTypeId) -> Self {
        Self::MissingParameter(
            ty.debug_name
                .as_ref()
                .map(|x| x.to_string())
                .unwrap_or_default(),
        )
    }
}

#[derive(Error, Debug)]
pub enum SierraAssertError {
    #[error("casts always happen between numerical types")]
    Cast,
    #[error("range should always intersect, from {ranges.0:?} to {ranges.1:?}")]
    Range {
        ranges: Box<(Range, Range)>,
    },
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_make_missing_parameter() {
        // Test with a type ID that has a debug name
        let ty_with_debug_name = ConcreteTypeId {
            debug_name: Some("u32".into()),
            id: 10,
        };

        assert_eq!(
            Error::make_missing_parameter(&ty_with_debug_name).to_string(),
            "missing parameter of type 'u32'"
        );

        // Test with a type ID that does not have a debug name
        let ty_without_debug_name = ConcreteTypeId {
            debug_name: None,
            id: 10,
        };

        assert_eq!(
            Error::make_missing_parameter(&ty_without_debug_name).to_string(),
            "missing parameter of type ''"
        );
    }
}
