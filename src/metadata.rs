//! # Code generation metadata
//!
//! The metadata is used for various stuff that would be otherwise difficult or plain impossible
//! with the current design, such as:
//!   - Pass compile-time constants that would otherwise have to be hardcoded in various places (ex.
//!     [PrimeModuloMeta](self::prime_modulo)).
//!   - Declare FFI bindings to external libraries (ex.
//!     [ReallocBindingsMeta](self::realloc_bindings)).
//!   - Pass extra compilation info to the libfunc generators (ex.
//!     [TailRecursionMeta](self::tail_recursion)).

use std::{
    any::{Any, TypeId},
    collections::{hash_map::Entry, HashMap},
};

pub mod debug_utils;
pub mod enum_snapshot_variants;
pub mod gas;
pub mod prime_modulo;
pub mod realloc_bindings;
pub mod runtime_bindings;
pub mod snapshot_clones;
pub mod tail_recursion;

/// Metadata container.
#[cfg_attr(not(feature = "with-debug-utils"), derive(Default))]
#[derive(Debug)]
pub struct MetadataStorage {
    // PLT: do we really need the `Box<dyn Any>`? Why not impl a trait that returns
    // metadata for anything that implements Any instead, and store the result?
    // PLT: second round: it seems what we really build here is singletons for some
    // metafunctionality, rather than metadata about types.
    entries: HashMap<TypeId, Box<dyn Any>>,
}

impl MetadataStorage {
    /// Create an empty metadata container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert some metadata and return a mutable reference.
    ///
    /// The insertion will fail if there is already some metadata with the same type, in which case
    /// it'll return `None`.
    pub fn insert<T>(&mut self, meta: T) -> Option<&mut T>
    where
        T: Any,
    {
        if let Entry::Vacant(e) = self.entries.entry(TypeId::of::<T>()) {
            e.insert(Box::new(meta));
            self.get_mut::<T>()
        } else {
            None
        }
    }

    /// Remove some metadata and return its last value.
    ///
    /// The removal will fail if there is no metadata with the requested type, in which case it'll
    /// return `None`.
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: Any,
    {
        self.entries
            .remove(&TypeId::of::<T>())
            .map(|meta| *(Box::<(dyn Any + 'static)>::downcast::<T>(meta).unwrap()))
    }

    /// Retrieve a reference to some metadata.
    ///
    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
    /// return `None`.
    pub fn get<T>(&self) -> Option<&T>
    where
        T: Any,
    {
        self.entries
            .get(&TypeId::of::<T>())
            .map(|meta| meta.downcast_ref::<T>().unwrap())
    }

    /// Retrieve a mutable reference to some metadata.
    ///
    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
    /// return `None`.
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Any,
    {
        self.entries
            .get_mut(&TypeId::of::<T>())
            .map(|meta| meta.downcast_mut::<T>().unwrap())
    }

    pub fn get_or_insert_with<T>(&mut self, meta_gen: impl FnOnce() -> T) -> &mut T
    where
        T: Any,
    {
        self.entries
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(meta_gen()))
            .downcast_mut::<T>()
            .unwrap()
    }
}

#[cfg(feature = "with-debug-utils")]
impl Default for MetadataStorage {
    fn default() -> Self {
        let mut metadata = Self {
            entries: Default::default(),
        };

        metadata.insert(debug_utils::DebugUtils::default());

        metadata
    }
}

#[cfg(test)]
mod test {
    use super::{runtime_bindings::RuntimeBindingsMeta, *};

    // PLT: missing tests for most of the behavior
    #[test]
    fn runtime_library_insert_works() {
        let mut metadata = MetadataStorage::new();
        let ret = metadata.insert(RuntimeBindingsMeta::default());

        assert!(ret.is_some());
    }
}

// PLT: badly named store for singletons?
// PLT: ACK
