use crate::{Context, Error};

use super::{Attribute, Identifier};

/// Helper type. A named attribute, needed on all operations that use attributes.
#[derive(Clone, Copy)]
pub struct NamedAttribute<'c> {
    pub identifier: Identifier<'c>,
    pub attribute: Attribute<'c>,
}

impl<'c> NamedAttribute<'c> {
    /// Creates a new named attribute, parsing the given attribute.
    pub fn new_parsed(context: &'c Context, name: &str, attribute: &str) -> Result<Self, Error> {
        Ok(Self {
            identifier: Identifier::new(context, name),
            attribute: Attribute::parse(context, attribute)
                .ok_or_else(|| Error::NamedAttributeParse(attribute.to_string()))?,
        })
    }

    /// Creates a new vector of named attribute from the given pairs.
    pub fn new_parsed_vec(
        context: &'c Context,
        ident_attr_pairs: &[(&str, &str)],
    ) -> Result<Vec<Self>, Error> {
        ident_attr_pairs.iter().map(|x| NamedAttribute::new_parsed(context, x.0, x.1)).collect()
    }

    pub const fn new(identifier: Identifier<'c>, attribute: Attribute<'c>) -> Result<Self, Error> {
        Ok(Self { identifier, attribute })
    }
}
