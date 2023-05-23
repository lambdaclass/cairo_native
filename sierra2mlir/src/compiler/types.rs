use melior_next::{dialect::llvm, ir::Type};

use super::Compiler;

impl<'ctx> Compiler<'ctx> {
    pub fn llvm_ptr_type(&self) -> Type {
        llvm::r#type::opaque_pointer(&self.context)
    }

    pub fn felt_type(&self) -> Type {
        Type::integer(&self.context, 256)
    }

    pub fn double_felt_type(&self) -> Type {
        Type::integer(&self.context, 512)
    }

    pub fn i32_type(&self) -> Type {
        Type::integer(&self.context, 32)
    }

    pub fn bool_type(&self) -> Type {
        Type::integer(&self.context, 1)
    }

    pub fn u8_type(&self) -> Type {
        Type::integer(&self.context, 8)
    }

    pub fn u16_type(&self) -> Type {
        Type::integer(&self.context, 16)
    }

    pub fn u32_type(&self) -> Type {
        Type::integer(&self.context, 32)
    }

    pub fn u64_type(&self) -> Type {
        Type::integer(&self.context, 64)
    }

    pub fn u128_type(&self) -> Type {
        Type::integer(&self.context, 128)
    }

    pub fn u256_type(&self) -> Type {
        Type::integer(&self.context, 256)
    }

    /// The enum struct type. Needed due to some libfuncs using it.
    ///
    /// The tag value is the boolean value: 0, 1
    ///
    /// Sierra: type core::bool = Enum<ut@core::bool, Unit, Unit>;
    pub fn boolean_enum_type(&self) -> Type {
        self.llvm_struct_type(&[self.u16_type(), self.llvm_array_type(self.u8_type(), 0)], false)
    }

    pub fn llvm_struct_type<'c>(&'c self, fields: &[Type<'c>], _packed: bool) -> Type {
        llvm::r#type::r#struct(&self.context, fields, true)
    }

    pub fn llvm_array_type<'c>(&'c self, element_type: Type<'c>, len: u32) -> Type {
        llvm::r#type::array(element_type, len)
    }
}
