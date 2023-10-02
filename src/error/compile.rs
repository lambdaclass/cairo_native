use crate::{libfuncs::LibfuncBuilder, types::TypeBuilder};
use cairo_lang_sierra::{
    edit_state::EditStateError,
    extensions::{
        core::{CoreLibfunc, CoreType},
        GenericLibfunc, GenericType,
    },
    ids::{ConcreteLibfuncId, ConcreteTypeId},
    program_registry::ProgramRegistryError,
};
use std::{fmt, ops::Deref};
use thiserror::Error;

pub type CompileError = Box<Error<CoreType, CoreLibfunc>>;

#[derive(Error)]
pub struct Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    // TODO: enable once its stable in rust
    // pub backtrace: Backtrace,
    pub source: ErrorImpl<TType, TLibfunc>,
}

impl<TType, TLibfunc> fmt::Display for Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.source, f)
    }
}

impl<TType, TLibfunc> Deref for Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    type Target = ErrorImpl<TType, TLibfunc>;

    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl<TType, TLibfunc, E> From<E> for Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    ErrorImpl<TType, TLibfunc>: From<E>,
{
    fn from(error: E) -> Self {
        Self {
            // backtrace: Backtrace::capture(),
            source: error.into(),
        }
    }
}

impl<TType, TLibfunc, E> From<E> for Box<Error<TType, TLibfunc>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    ErrorImpl<TType, TLibfunc>: From<E>,
{
    fn from(error: E) -> Self {
        Self::new(Error::from(error))
    }
}

// Manual implementation necessary because `#[derive(Debug)]` requires that `TType` and `TLibfunc`
// both implement `Debug`, which isn't the case.
impl<TType, TLibfunc> fmt::Debug for Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            // .field("backtrace", &self.backtrace)
            .field("source", &self.source)
            .finish()
    }
}

#[derive(Error)]
pub enum ErrorImpl<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    #[error(transparent)]
    EditStateError(#[from] EditStateError),
    #[error(transparent)]
    MlirError(#[from] melior::Error),
    #[error(transparent)]
    ProgramRegistryError(#[from] Box<ProgramRegistryError>),

    #[error("Error building type '{type_id}': {error}")]
    TypeBuilderError {
        type_id: ConcreteTypeId,
        error: <<TType as GenericType>::Concrete as TypeBuilder<TType, TLibfunc>>::Error,
    },
    #[error("Error building type '{libfunc_id}': {error}")]
    LibfuncBuilderError {
        libfunc_id: ConcreteLibfuncId,
        error: <<TLibfunc as GenericLibfunc>::Concrete as LibfuncBuilder<TType, TLibfunc>>::Error,
    },
}

// Manual implementation necessary because `#[derive(Debug)]` requires that `TType` and `TLibfunc`
// both implement `Debug`, which isn't the case.
impl<TType, TLibfunc> fmt::Debug for ErrorImpl<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EditStateError(arg0) => f.debug_tuple("EditStateError").field(arg0).finish(),
            Self::MlirError(arg0) => f.debug_tuple("MlirError").field(arg0).finish(),
            Self::ProgramRegistryError(arg0) => {
                f.debug_tuple("ProgramRegistryError").field(arg0).finish()
            }
            Self::TypeBuilderError { type_id, error } => f
                .debug_struct("TypeBuilderError")
                .field("type_id", type_id)
                .field("error", error)
                .finish(),
            Self::LibfuncBuilderError { libfunc_id, error } => f
                .debug_struct("LibfuncBuilderError")
                .field("libfunc_id", libfunc_id)
                .field("error", error)
                .finish(),
        }
    }
}

pub fn make_type_builder_error<TType, TLibfunc>(
    id: &ConcreteTypeId,
) -> impl '_
       + FnOnce(
    <<TType as GenericType>::Concrete as TypeBuilder<TType, TLibfunc>>::Error,
) -> Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    move |source| {
        ErrorImpl::TypeBuilderError {
            type_id: id.clone(),
            error: source,
        }
        .into()
    }
}

pub fn make_libfunc_builder_error<TType, TLibfunc>(
    id: &ConcreteLibfuncId,
) -> impl '_
       + FnOnce(
    <<TLibfunc as GenericLibfunc>::Concrete as LibfuncBuilder<TType, TLibfunc>>::Error,
) -> Error<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    move |source| {
        ErrorImpl::LibfuncBuilderError {
            libfunc_id: id.clone(),
            error: source,
        }
        .into()
    }
}
