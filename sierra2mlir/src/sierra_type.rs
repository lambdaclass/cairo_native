use itertools::Itertools;
use melior_next::{
    ir::{Location, Type, TypeLike},
    Context,
};

use crate::compiler::Compiler;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SierraType<'ctx> {
    Simple(Type<'ctx>),
    // We represent a struct as a contiguous list of types, like sierra does, for now.
    Struct {
        ty: Type<'ctx>,
        field_types: Vec<Self>,
    },
    Enum {
        ty: Type<'ctx>,
        tag_type: Type<'ctx>,
        storage_bytes_len: u32,
        storage_type: Type<'ctx>, // the array
        variants_types: Vec<Self>,
    },
    Array {
        /// (u32, u32, ptr)
        ///
        /// (length, capacity, data)
        ty: Type<'ctx>,
        len_type: Type<'ctx>, // type of length and capacity: u32
        element_type: Box<Self>,
    },
    Dictionary {
        /// (u32, ptr)
        /// where ptr is an array of (key (always felt), value (T), is_used (bool))
        ///
        /// (length, data)
        ty: Type<'ctx>,
        len_type: Type<'ctx>, // type of length and capacity: u32
        element_type: Box<Self>,
    },
}

impl<'ctx> SierraType<'ctx> {
    /// gets the sierra array type for the given element
    pub fn create_array_type<'c>(
        compiler: &'c Compiler<'c>,
        element: SierraType<'c>,
    ) -> SierraType<'c> {
        SierraType::Array {
            ty: compiler.llvm_struct_type(
                &[compiler.u32_type(), compiler.u32_type(), compiler.llvm_ptr_type()],
                false,
            ),
            len_type: compiler.u32_type(),
            element_type: Box::new(element),
        }
    }

    pub fn create_dict_type<'c>(
        compiler: &'c Compiler<'c>,
        element: SierraType<'c>,
    ) -> SierraType<'c> {
        SierraType::Dictionary {
            ty: compiler.llvm_struct_type(&[compiler.u32_type(), compiler.llvm_ptr_type()], false),
            len_type: compiler.u32_type(),
            element_type: Box::new(element),
        }
    }

    /// gets the sierra nullable type for the given type
    pub fn create_nullable_type<'c>(
        compiler: &'c Compiler<'c>,
        element: SierraType<'c>,
    ) -> SierraType<'c> {
        SierraType::Struct {
            ty: compiler.llvm_struct_type(&[element.get_type(), compiler.bool_type()], false),
            field_types: vec![element, SierraType::Simple(compiler.bool_type())],
        }
    }

    /// creates the sierra U128MulGuaranteeType type.
    pub fn create_u128_guarantee_type<'c>(compiler: &'c Compiler<'c>) -> SierraType<'c> {
        let u128_ty = compiler.u128_type();
        let ty = compiler.llvm_struct_type(&[u128_ty, u128_ty, u128_ty, u128_ty], false);
        SierraType::Struct {
            ty,
            field_types: vec![
                SierraType::Simple(u128_ty),
                SierraType::Simple(u128_ty),
                SierraType::Simple(u128_ty),
                SierraType::Simple(u128_ty),
            ],
        }
    }

    /// Returns the width in bits of the mlir representation of the type
    pub fn get_width(&self) -> u32 {
        match self {
            SierraType::Simple(ty) => {
                ty.get_width().expect("Type size should be calculable for Simple SierraTypes")
            }
            SierraType::Struct { ty: _, field_types } => {
                let mut width = 0;
                for ty in field_types {
                    width += ty.get_width();
                }
                width
            }
            SierraType::Enum {
                ty: _,
                tag_type,
                storage_bytes_len: storage_type_len,
                storage_type: _,
                variants_types: _,
            } => tag_type.get_width().unwrap() + (storage_type_len * 8),
            SierraType::Array { ty: _, len_type, element_type: _ } => {
                // 64 is the pointer size, assuming here
                // TODO: find a better way to find the pointer size? it would require getting the context here
                // NOTE: This should at least be safe, since overestimating type sizes is generally okay, it just means extra space may be allocated
                len_type.get_width().unwrap() * 2 + 64
            }
            SierraType::Dictionary { ty: _, len_type, element_type: _ } => {
                // 64 is the pointer size, assuming here
                // TODO: find a better way to find the pointer size? it would require getting the context here
                // NOTE: This should at least be safe, since overestimating type sizes is generally okay, it just means extra space may be allocated
                len_type.get_width().unwrap() + 64
            }
        }
    }

    /// Returns the width in felts of the casm representation of the type
    /// e.g. felt => 1, u8 => 1, (felt, felt, felt) => 3
    /// Arrays have a felt width of 2, because their casm representation is two pointers
    pub fn get_felt_representation_width(&self) -> usize {
        match self {
            SierraType::Simple(_) => 1,
            SierraType::Struct { ty: _, field_types } => {
                field_types.iter().map(Self::get_felt_representation_width).sum()
            }
            SierraType::Enum {
                ty: _,
                tag_type: _,
                storage_bytes_len: _,
                storage_type: _,
                variants_types,
            } => {
                1 + variants_types
                    .iter()
                    .map(Self::get_felt_representation_width)
                    .max()
                    .unwrap_or(0)
            }
            SierraType::Array { .. } => 2,
            SierraType::Dictionary { .. } => 2, // TODO: check
        }
    }

    /// Returns the MLIR type of this sierra type
    pub const fn get_type(&self) -> Type<'ctx> {
        match self {
            Self::Simple(ty) => *ty,
            Self::Struct { ty, field_types: _ } => *ty,
            Self::Enum {
                ty,
                tag_type: _,
                storage_bytes_len: _,
                storage_type: _,
                variants_types: _,
            } => *ty,
            Self::Array { ty, .. } => *ty,
            Self::Dictionary { ty, .. } => *ty,
        }
    }

    /// Returns the mlir representation of the type, paired with an unknown Location. Used for block arguments
    pub fn get_type_location(&self, context: &'ctx Context) -> (Type<'ctx>, Location<'ctx>) {
        (
            match self {
                Self::Simple(ty) => *ty,
                Self::Struct { ty, field_types: _ } => *ty,
                Self::Enum {
                    ty,
                    tag_type: _,
                    storage_bytes_len: _,
                    storage_type: _,
                    variants_types: _,
                } => *ty,
                Self::Array { ty, .. } => *ty,
                Self::Dictionary { ty, .. } => *ty,
            },
            Location::unknown(context),
        )
    }

    /// Returns a vec of field types if this is a struct type.
    pub fn get_field_types(&self) -> Option<Vec<Type<'ctx>>> {
        match self {
            SierraType::Struct { ty: _, field_types } => {
                Some(field_types.iter().map(|x| x.get_type()).collect_vec())
            }
            _ => None,
        }
    }

    /// Returns the sierra types for the struct members if this is a struct type
    pub fn get_field_sierra_types(&self) -> Option<&[Self]> {
        match self {
            SierraType::Struct { ty: _, field_types } => Some(field_types),
            _ => None,
        }
    }

    pub const fn get_enum_tag_type(&self) -> Option<Type> {
        match self {
            SierraType::Enum {
                ty: _,
                tag_type,
                storage_bytes_len: _,
                storage_type: _,
                variants_types: _,
            } => Some(*tag_type),
            _ => None,
        }
    }
}
