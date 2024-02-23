use crate::metadata::MetadataStorage;
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program_registry::ProgramRegistry,
};
use melior::ir::Module;
use std::{any::Any, fmt::Debug};

/// A MLIR module in the context of Cairo Native.
/// It is conformed by the MLIR module, the Sierra program registry
/// and the program metadata.
pub struct NativeModule<'m> {
    pub(crate) module: Module<'m>,
    pub(crate) registry: ProgramRegistry<CoreType, CoreLibfunc>,
    pub(crate) metadata: MetadataStorage,
}

impl<'m> NativeModule<'m> {
    pub fn new(
        module: Module<'m>,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: MetadataStorage,
    ) -> Self {
        Self {
            module,
            registry,
            metadata,
        }
    }

    /// Insert some metadata for the program execution and return a mutable reference to it.
    ///
    /// The insertion will fail, if there is already some metadata with the same type, in which case
    /// it'll return `None`.
    pub fn insert_metadata<T>(&mut self, meta: T) -> Option<&mut T>
    where
        T: Any,
    {
        self.metadata.insert(meta)
    }

    /// Removes metadata
    pub fn remove_metadata<T>(&mut self) -> Option<T>
    where
        T: Any,
    {
        self.metadata.remove()
    }

    /// Retrieve a reference to some stored metadata.
    ///
    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
    /// return `None`.
    pub fn get_metadata<T>(&self) -> Option<&T>
    where
        T: Any,
    {
        self.metadata.get::<T>()
    }

    pub fn metadata(&self) -> &MetadataStorage {
        &self.metadata
    }

    pub fn module(&self) -> &Module {
        &self.module
    }

    pub fn program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        &self.registry
    }
}

impl Debug for NativeModule<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.module.as_operation().to_string())
    }
}
