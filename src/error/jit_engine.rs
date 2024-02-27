use cairo_lang_sierra::{ids::ConcreteTypeId, program_registry::ProgramRegistryError};
use std::{alloc::LayoutError, fmt, ops::Deref};
use thiserror::Error;

pub type RunnerError = Box<Error>;

#[derive(Error, Debug)]
pub struct Error {
    // TODO: enable once its stable in rust
    // pub backtrace: Backtrace,
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

#[derive(Error, Debug)]
pub enum ErrorImpl {
    #[error(transparent)]
    LayoutError(#[from] LayoutError),
    #[error(transparent)]
    MlirError(#[from] melior::Error),
    #[error(transparent)]
    ProgramRegistryError(#[from] Box<ProgramRegistryError>),

    #[error("Error building type '{type_id}': {error}")]
    TypeBuilderError {
        type_id: ConcreteTypeId,
        error: crate::error::types::Error,
    },

    #[error("missing parameter of type '{0}'")]
    MissingParameter(String),

    #[error("unexpected value, expected value of type '{0}'")]
    UnexpectedValue(String),

    #[error("not enough gas to run")]
    InsufficientGasError,

    #[error("a syscall handler was expected but was not provided")]
    MissingSyscallHandler,
}

pub fn make_unexpected_value_error(expected: String) -> Error {
    ErrorImpl::UnexpectedValue(expected).into()
}

pub fn make_type_builder_error(
    id: &ConcreteTypeId,
) -> impl '_ + FnOnce(crate::error::types::Error) -> Error {
    move |source| {
        ErrorImpl::TypeBuilderError {
            type_id: id.clone(),
            error: source,
        }
        .into()
    }
}

pub fn make_missing_parameter(ty: &ConcreteTypeId) -> Error {
    ErrorImpl::MissingParameter(
        ty.debug_name
            .as_ref()
            .map(|x| x.to_string())
            .unwrap_or_default(),
    )
    .into()
}
