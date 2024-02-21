//! # Executor errors.

use super::BuilderError;
use cairo_lang_sierra::{ids::ConcreteTypeId, program_registry::ProgramRegistryError};
use std::{alloc::LayoutError, fmt, ops::Deref};
use thiserror::Error;

/// A [`Result`](std::result::Result) alias with the error type fixed to [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Wrapper for the error type and the error's origin backtrace (soon™).
#[derive(Error)]
pub struct Error {
    // TODO: Enable once it stabilizes.
    // pub backtrace: Backtrace,
    /// The actual error.
    pub source: ErrorImpl,
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
            source: error.into(),
        }
    }
}

impl<E> From<E> for Box<Error>
where
    ErrorImpl: From<E>,
{
    fn from(error: E) -> Self {
        Self::new(Error::from(error))
    }
}

// Manual implementation necessary because `#[derive(Debug)]` requires that `TType` and `TLibfunc`
// both implement `Debug`, which isn't the case.
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            // .field("backtrace", &self.backtrace)
            .field("source", &self.source)
            .finish()
    }
}

/// An executor error.
#[derive(Error)]
pub enum ErrorImpl {
    /// An invalid layout was generated. This should never happen.
    #[error(transparent)]
    LayoutError(#[from] LayoutError),
    /// An MLIR error has occurred.
    #[error(transparent)]
    MlirError(#[from] melior::Error),
    /// The program registry returned an error. This should mean an invalid Sierra has been provided
    /// to the compiler.
    #[error(transparent)]
    ProgramRegistryError(#[from] Box<ProgramRegistryError>),

    /// A [TypeBuilder](crate::types::TypeBuilder) error.
    #[error("Error building type '{type_id}': {error}")]
    TypeBuilderError {
        /// The type which caused an error.
        type_id: ConcreteTypeId,
        /// The actual error.
        error: BuilderError,
    },

    /// There's not enough parameters to invoke an entrypoint.
    #[error("missing parameter of type '{0}'")]
    MissingParameter(String),

    /// Found a value of an unexpected type.
    #[error("unexpected value, expected value of type '{0}'")]
    UnexpectedType(String),

    /// There's not enough gas to run the program.
    #[error("not enough gas to run, needed '{needed}' had '{have}'")]
    InsufficientGasError {
        /// The required gas amount.
        needed: u128,
        /// The current gas amount.
        have: u128,
    },
    /// The syscall handler is required, but has not been provided.
    #[error("a syscall handler was expected but was not provided")]
    MissingSyscallHandler,
}

impl fmt::Debug for ErrorImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayoutError(arg0) => f.debug_tuple("LayoutError").field(arg0).finish(),
            Self::MlirError(arg0) => f.debug_tuple("MlirError").field(arg0).finish(),
            Self::MissingParameter(arg0) => f.debug_tuple("MissingParameter").field(arg0).finish(),
            Self::ProgramRegistryError(arg0) => {
                f.debug_tuple("ProgramRegistryError").field(arg0).finish()
            }
            Self::TypeBuilderError { type_id, error } => f
                .debug_struct("BuilderError")
                .field("type_id", type_id)
                .field("error", error)
                .finish(),
            Self::UnexpectedType(arg0) => f.debug_tuple("UnexpectedValue").field(arg0).finish(),
            Self::MissingSyscallHandler => f.debug_struct("MissingSyscallHandler").finish(),
            Self::InsufficientGasError { needed, have } => f
                .debug_struct("InsufficientGasError")
                .field("needed", needed)
                .field("have", have)
                .finish(),
        }
    }
}

pub(crate) fn make_type_builder_error(
    id: &ConcreteTypeId,
) -> impl '_ + FnOnce(BuilderError) -> Error {
    move |source| {
        ErrorImpl::TypeBuilderError {
            type_id: id.clone(),
            error: source,
        }
        .into()
    }
}
