use crate::types::TypeBuilder;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        GenericType,
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistryError,
};
use std::{alloc::LayoutError, fmt, ops::Deref};
use thiserror::Error;

pub type RunnerError = Error;

#[derive(Error)]
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

#[derive(Error)]
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
        error: <<CoreType as GenericType>::Concrete as TypeBuilder<CoreType, CoreLibfunc>>::Error,
    },

    #[error("missing parameter of type '{0}'")]
    MissingParameter(String),

    #[error("unexpected value, expected value of type '{0}'")]
    UnexpectedValue(String),

    #[error("not enough gas to run, needed '{needed}' had '{have}'")]
    InsufficientGasError { needed: u128, have: u128 },

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
                .debug_struct("TypeBuilderError")
                .field("type_id", type_id)
                .field("error", error)
                .finish(),
            Self::UnexpectedValue(arg0) => f.debug_tuple("UnexpectedValue").field(arg0).finish(),
            Self::MissingSyscallHandler => f.debug_struct("MissingSyscallHandler").finish(),
            Self::InsufficientGasError { needed, have } => f
                .debug_struct("InsufficientGasError")
                .field("needed", needed)
                .field("have", have)
                .finish(),
        }
    }
}

pub fn make_unexpected_value_error(expected: String) -> Error {
    ErrorImpl::UnexpectedValue(expected).into()
}

pub fn make_type_builder_error(
    id: &ConcreteTypeId,
) -> impl '_
       + FnOnce(
    <<CoreType as GenericType>::Concrete as TypeBuilder<CoreType, CoreLibfunc>>::Error,
) -> Error {
    move |source| {
        ErrorImpl::TypeBuilderError {
            type_id: id.clone(),
            error: source,
        }
        .into()
    }
}

pub fn make_insufficient_gas_error(needed: u128, have: u128) -> Error {
    ErrorImpl::InsufficientGasError { needed, have }.into()
}

pub fn make_missing_parameter(ty: &ConcreteTypeId) -> Error {
    ErrorImpl::MissingParameter(
        ty.debug_name
            .as_ref()
            .map(|x| x.to_string())
            .unwrap_or_else(String::new),
    )
    .into()
}
