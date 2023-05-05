//! Collection of type processors.

use crate::compiler::{TypeFactory, TypeLayout};
use cairo_lang_sierra::program::TypeDeclaration;
use std::{
    collections::{btree_map::Entry, BTreeMap},
    ops::Deref,
};

/// A type processor.
pub type TypeProcessor = dyn Fn(&TypeFactory, &TypeDeclaration) -> TypeLayout;

/// A collection of type processors.
pub struct TypeDatabase(BTreeMap<String, Box<TypeProcessor>>);

impl TypeDatabase {
    /// Create an empty type database.
    #[must_use]
    pub fn new() -> TypeDatabase {
        Self(BTreeMap::default())
    }

    /// Register a type into the database.
    pub fn register(
        &mut self,
        id: impl Into<String>,
        processor: impl 'static + Fn(&TypeFactory, &TypeDeclaration) -> TypeLayout,
    ) {
        match self.0.entry(id.into()) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(processor));
            }
            Entry::Occupied(_) => todo!(),
        }
    }
}

impl Default for TypeDatabase {
    fn default() -> Self {
        use crate::types;

        let mut database = TypeDatabase::new();
        database.register("Array", Box::new(types::array));
        database.register("felt252", Box::new(types::felt252));
        database.register("Struct", Box::new(types::r#struct));
        database
    }
}

impl Deref for TypeDatabase {
    type Target = BTreeMap<String, Box<TypeProcessor>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
