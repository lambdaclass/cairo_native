//! # MLIR builder errors.

use cairo_lang_sierra::program_registry::ProgramRegistryError;
use std::{alloc::LayoutError, fmt, num::TryFromIntError, ops::Deref};
use thiserror::Error;

/// A [`Result`](std::result::Result) alias with the error type fixed to [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Wrapper for the error type and the error's origin backtrace (soon™).
#[derive(Debug, Error)]
pub struct Error {
    // TODO: Enable once it stabilizes.
    // pub backtrace: Backtrace,
    /// The actual error.
    pub source: Box<ErrorImpl>,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.source, f)
    }
}

impl Deref for Error {
    type Target = ErrorImpl;

    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl<E> From<E> for Error
where
    ErrorImpl: From<E>,
{
    fn from(error: E) -> Self {
        Self {
            // backtrace: Backtrace::capture(),
            source: Box::new(error.into()),
        }
    }
}

/// A [`TypeBuilder`](crate::types::TypeBuilder) or
/// [`LibfuncBuilder`](crate::libfuncs::LibfuncBuilder) error.
#[derive(Debug, Error)]
pub enum ErrorImpl {
    /// A [Layout](std::alloc::Layout) error.
    #[error(transparent)]
    LayoutError(#[from] LayoutError),
    /// An MLIR error (from [melior](melior::Error)).
    #[error(transparent)]
    MlirError(#[from] melior::Error),
    /// A Sierra program registry error. This should mean an invalid Sierra has been provided to the
    /// compiler.
    #[error(transparent)]
    ProgramRegistryError(#[from] Box<ProgramRegistryError>),
    /// An integer conversion error.
    #[error(transparent)]
    TryFromIntError(#[from] TryFromIntError),
    /// An MLIR attribute parser error.
    #[error("error parsing attribute")]
    ParseAttributeError,
    /// Some required metadata is missing.
    #[error("missing metadata")]
    MissingMetadata,
}
