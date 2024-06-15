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

// PLT: since we already use Educe, this could be replaced by an Educe derive.
impl Debug for NativeModule<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.module.as_operation().to_string())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::context::NativeContext;
    use cairo_lang_sierra::ProgramParser;
    use melior::ir::Location;
    use starknet_types_core::felt::Felt;

    // PLT: make a test helper to reduce boilerplate.
    // E.g., it could create the context, location and module.
    #[test]
    fn test_insert_metadata() {
        // Create a new context for MLIR operations
        let native_context = NativeContext::new();
        let context = native_context.context();

        // Create an unknown location in the context
        let location = Location::unknown(context);
        // Create a new MLIR module with the unknown location
        let module = Module::new(location);

        // Parse a simple program to create a Program instance
        let program = ProgramParser::new()
            .parse("type felt252 = felt252;")
            .unwrap();

        // Create a ProgramRegistry based on the parsed program
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Create a new NativeModule instance with the module, registry, and MetadataStorage
        let mut module = NativeModule::new(module, registry, MetadataStorage::new());

        // Insert metadata of type u32 into the module
        module.insert_metadata(42u32);
        // Assert that the inserted metadata of type u32 is retrieved correctly
        assert_eq!(module.get_metadata::<u32>(), Some(&42u32));

        // Insert metadata of type Felt into the module
        module.insert_metadata(Felt::from(43));
        // Assert that the inserted metadata of type Felt is retrieved correctly
        assert_eq!(module.get_metadata::<Felt>(), Some(&Felt::from(43)));

        // Insert metadata of type u64 into the module
        module.insert_metadata(44u64);
        // Assert that the inserted metadata of type u64 is retrieved correctly
        assert_eq!(module.metadata().get::<u64>(), Some(&44u64));
    }

    #[test]
    fn test_remove_metadata() {
        // Create a new context for MLIR operations
        let native_context = NativeContext::new();
        let context = native_context.context();

        // Create an unknown location in the context
        let location = Location::unknown(context);
        // Create a new MLIR module with the unknown location
        let module = Module::new(location);

        // Parse a simple program to create a Program instance
        let program = ProgramParser::new()
            .parse("type felt252 = felt252;")
            .unwrap();

        // Create a ProgramRegistry based on the parsed program
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Create a new NativeModule instance with the module, registry, and MetadataStorage
        let mut module = NativeModule::new(module, registry, MetadataStorage::new());

        // Insert metadata of type u32 into the module
        module.insert_metadata(42u32);
        // Assert that the inserted metadata of type u32 is retrieved correctly
        assert_eq!(module.get_metadata::<u32>(), Some(&42u32));

        // Insert metadata of type Felt into the module
        module.insert_metadata(Felt::from(43));
        // Assert that the inserted metadata of type Felt is retrieved correctly
        assert_eq!(module.get_metadata::<Felt>(), Some(&Felt::from(43)));

        // Remove metadata of type u32 from the module
        module.remove_metadata::<u32>();
        // Assert that the metadata of type u32 is removed from the module
        assert!(module.get_metadata::<u32>().is_none());
        // Assert that the metadata of type Felt is still present in the module
        assert_eq!(module.get_metadata::<Felt>(), Some(&Felt::from(43)));

        // Insert metadata of type u32 into the module again
        module.insert_metadata(44u32);
        // Assert that the re-inserted metadata of type u32 is retrieved correctly
        assert_eq!(module.get_metadata::<u32>(), Some(&44u32));
    }
}
// PLT: ACK
