//! Collection of libfunc processors.

use crate::compiler::{LibfuncImpl, TypeFactory};
use cairo_lang_sierra::program::LibfuncDeclaration;
use std::{
    collections::{btree_map::Entry, BTreeMap},
    ops::Deref,
};

/// A libfunc processor.
pub type LibfuncProcessor = dyn Fn(&TypeFactory, &LibfuncDeclaration) -> LibfuncImpl;

/// A collection of libfunc processors.
pub struct LibfuncDatabase(BTreeMap<String, Box<LibfuncProcessor>>);

impl LibfuncDatabase {
    /// Create an empty libfunc database.
    #[must_use]
    pub fn new() -> LibfuncDatabase {
        Self(BTreeMap::default())
    }

    /// Create a type into the database.
    pub fn register(
        &mut self,
        id: impl Into<String>,
        processor: impl 'static + Fn(&TypeFactory, &LibfuncDeclaration) -> LibfuncImpl,
    ) {
        match self.0.entry(id.into()) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(processor));
            }
            Entry::Occupied(_) => todo!(),
        }
    }
}

impl Default for LibfuncDatabase {
    fn default() -> Self {
        use crate::libfuncs;

        let mut database = LibfuncDatabase::new();
        database.register("felt252_const", Box::new(libfuncs::felt252_const));
        database.register("store_temp", Box::new(libfuncs::store_temp));
        database
    }
}

impl Deref for LibfuncDatabase {
    type Target = BTreeMap<String, Box<LibfuncProcessor>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
