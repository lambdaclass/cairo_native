////! Trait that extends the melior Block type to aid in codegen and consistency.
//! Trait that extends the melior Block type to aid in codegen and consistency.
//

//use melior::{
use melior::{
//    dialect::{llvm::r#type::pointer, ods},
    dialect::{llvm::r#type::pointer, ods},
//    ir::{
    ir::{
//        attribute::{DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute},
        attribute::{DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute},
//        r#type::IntegerType,
        r#type::IntegerType,
//        Attribute, Block, Location, Operation, Type, Value, ValueLike,
        Attribute, Block, Location, Operation, Type, Value, ValueLike,
//    },
    },
//    Context,
    Context,
//};
};
//use num_bigint::BigInt;
use num_bigint::BigInt;
//

//use crate::{error::Error, utils::get_integer_layout};
use crate::{error::Error, utils::get_integer_layout};
//

//pub trait BlockExt<'ctx> {
pub trait BlockExt<'ctx> {
//    /// Appends the operation and returns the first result.
    /// Appends the operation and returns the first result.
//    fn append_op_result(&self, operation: Operation<'ctx>) -> Result<Value<'ctx, '_>, Error>;
    fn append_op_result(&self, operation: Operation<'ctx>) -> Result<Value<'ctx, '_>, Error>;
//

//    /// Creates a constant of the given integer bit width. Do not use for felt252.
    /// Creates a constant of the given integer bit width. Do not use for felt252.
//    fn const_int<T>(
    fn const_int<T>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        value: T,
        value: T,
//        bits: u32,
        bits: u32,
//    ) -> Result<Value<'ctx, '_>, Error>
    ) -> Result<Value<'ctx, '_>, Error>
//    where
    where
//        T: Into<BigInt>;
        T: Into<BigInt>;
//

//    /// Creates a constant of the given integer type. Do not use for felt252.
    /// Creates a constant of the given integer type. Do not use for felt252.
//    fn const_int_from_type<T>(
    fn const_int_from_type<T>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        value: T,
        value: T,
//        int_type: Type<'ctx>,
        int_type: Type<'ctx>,
//    ) -> Result<Value<'ctx, '_>, Error>
    ) -> Result<Value<'ctx, '_>, Error>
//    where
    where
//        T: Into<BigInt>;
        T: Into<BigInt>;
//

//    /// Uses a llvm::extract_value operation to return the value at the given index of a container (e.g struct).
    /// Uses a llvm::extract_value operation to return the value at the given index of a container (e.g struct).
//    fn extract_value(
    fn extract_value(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        container: Value<'ctx, '_>,
        container: Value<'ctx, '_>,
//        value_type: Type<'ctx>,
        value_type: Type<'ctx>,
//        index: usize,
        index: usize,
//    ) -> Result<Value<'ctx, '_>, Error>;
    ) -> Result<Value<'ctx, '_>, Error>;
//

//    /// Uses a llvm::insert_value operation to insert the value at the given index of a container (e.g struct),
    /// Uses a llvm::insert_value operation to insert the value at the given index of a container (e.g struct),
//    /// the result is the container with the value.
    /// the result is the container with the value.
//    fn insert_value(
    fn insert_value(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        container: Value<'ctx, '_>,
        container: Value<'ctx, '_>,
//        value: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
//        index: usize,
        index: usize,
//    ) -> Result<Value<'ctx, '_>, Error>;
    ) -> Result<Value<'ctx, '_>, Error>;
//

//    /// Uses a llvm::insert_value operation to insert the values starting from index 0 into a container (e.g struct),
    /// Uses a llvm::insert_value operation to insert the values starting from index 0 into a container (e.g struct),
//    /// the result is the container with the values.
    /// the result is the container with the values.
//    fn insert_values<'block>(
    fn insert_values<'block>(
//        &'block self,
        &'block self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        container: Value<'ctx, 'block>,
        container: Value<'ctx, 'block>,
//        values: &[Value<'ctx, 'block>],
        values: &[Value<'ctx, 'block>],
//    ) -> Result<Value<'ctx, 'block>, Error>;
    ) -> Result<Value<'ctx, 'block>, Error>;
//

//    /// Loads a value from the given addr.
    /// Loads a value from the given addr.
//    fn load(
    fn load(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        addr: Value<'ctx, '_>,
        addr: Value<'ctx, '_>,
//        value_type: Type<'ctx>,
        value_type: Type<'ctx>,
//        align: Option<usize>,
        align: Option<usize>,
//    ) -> Result<Value<'ctx, '_>, Error>;
    ) -> Result<Value<'ctx, '_>, Error>;
//

//    /// Allocates the given number of elements of type in memory on the stack, returning a opaque pointer.
    /// Allocates the given number of elements of type in memory on the stack, returning a opaque pointer.
//    fn alloca(
    fn alloca(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        elem_type: Type<'ctx>,
        elem_type: Type<'ctx>,
//        num_elems: Value<'ctx, '_>,
        num_elems: Value<'ctx, '_>,
//        align: Option<usize>,
        align: Option<usize>,
//    ) -> Result<Value<'ctx, '_>, Error>;
    ) -> Result<Value<'ctx, '_>, Error>;
//

//    /// Allocates one element of the given type in memory on the stack, returning a opaque pointer.
    /// Allocates one element of the given type in memory on the stack, returning a opaque pointer.
//    fn alloca1(
    fn alloca1(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        elem_type: Type<'ctx>,
        elem_type: Type<'ctx>,
//        align: Option<usize>,
        align: Option<usize>,
//    ) -> Result<Value<'ctx, '_>, Error>;
    ) -> Result<Value<'ctx, '_>, Error>;
//

//    /// Allocates one integer of the given bit width.
    /// Allocates one integer of the given bit width.
//    fn alloca_int(
    fn alloca_int(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        bits: u32,
        bits: u32,
//    ) -> Result<Value<'ctx, '_>, Error>;
    ) -> Result<Value<'ctx, '_>, Error>;
//

//    /// Stores a value at the given addr.
    /// Stores a value at the given addr.
//    fn store(
    fn store(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        addr: Value<'ctx, '_>,
        addr: Value<'ctx, '_>,
//        value: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
//        align: Option<usize>,
        align: Option<usize>,
//    );
    );
//

//    /// Creates a memcpy operation.
    /// Creates a memcpy operation.
//    fn memcpy(
    fn memcpy(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        src: Value<'ctx, '_>,
        src: Value<'ctx, '_>,
//        dst: Value<'ctx, '_>,
        dst: Value<'ctx, '_>,
//        len_bytes: Value<'ctx, '_>,
        len_bytes: Value<'ctx, '_>,
//    );
    );
//}
}
//

//impl<'ctx> BlockExt<'ctx> for Block<'ctx> {
impl<'ctx> BlockExt<'ctx> for Block<'ctx> {
//    fn const_int<T>(
    fn const_int<T>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        value: T,
        value: T,
//        bits: u32,
        bits: u32,
//    ) -> Result<Value<'ctx, '_>, Error>
    ) -> Result<Value<'ctx, '_>, Error>
//    where
    where
//        T: Into<BigInt>,
        T: Into<BigInt>,
//    {
    {
//        let ty = IntegerType::new(context, bits).into();
        let ty = IntegerType::new(context, bits).into();
//        Ok(self
        Ok(self
//            .append_operation(
            .append_operation(
//                ods::arith::constant(
                ods::arith::constant(
//                    context,
                    context,
//                    ty,
                    ty,
//                    Attribute::parse(context, &format!("{} : {}", value.into(), ty))
                    Attribute::parse(context, &format!("{} : {}", value.into(), ty))
//                        .ok_or(Error::ParseAttributeError)?,
                        .ok_or(Error::ParseAttributeError)?,
//                    location,
                    location,
//                )
                )
//                .into(),
                .into(),
//            )
            )
//            .result(0)?
            .result(0)?
//            .into())
            .into())
//    }
    }
//

//    fn const_int_from_type<T>(
    fn const_int_from_type<T>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        value: T,
        value: T,
//        ty: Type<'ctx>,
        ty: Type<'ctx>,
//    ) -> Result<Value<'ctx, '_>, Error>
    ) -> Result<Value<'ctx, '_>, Error>
//    where
    where
//        T: Into<BigInt>,
        T: Into<BigInt>,
//    {
    {
//        Ok(self
        Ok(self
//            .append_operation(
            .append_operation(
//                ods::arith::constant(
                ods::arith::constant(
//                    context,
                    context,
//                    ty,
                    ty,
//                    Attribute::parse(context, &format!("{} : {}", value.into(), ty))
                    Attribute::parse(context, &format!("{} : {}", value.into(), ty))
//                        .ok_or(Error::ParseAttributeError)?,
                        .ok_or(Error::ParseAttributeError)?,
//                    location,
                    location,
//                )
                )
//                .into(),
                .into(),
//            )
            )
//            .result(0)?
            .result(0)?
//            .into())
            .into())
//    }
    }
//

//    fn extract_value(
    fn extract_value(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        container: Value<'ctx, '_>,
        container: Value<'ctx, '_>,
//        value_type: Type<'ctx>,
        value_type: Type<'ctx>,
//        index: usize,
        index: usize,
//    ) -> Result<Value<'ctx, '_>, Error> {
    ) -> Result<Value<'ctx, '_>, Error> {
//        Ok(self
        Ok(self
//            .append_operation(
            .append_operation(
//                ods::llvm::extractvalue(
                ods::llvm::extractvalue(
//                    context,
                    context,
//                    value_type,
                    value_type,
//                    container,
                    container,
//                    DenseI64ArrayAttribute::new(context, &[index.try_into().unwrap()]).into(),
                    DenseI64ArrayAttribute::new(context, &[index.try_into().unwrap()]).into(),
//                    location,
                    location,
//                )
                )
//                .into(),
                .into(),
//            )
            )
//            .result(0)?
            .result(0)?
//            .into())
            .into())
//    }
    }
//

//    fn insert_value(
    fn insert_value(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        container: Value<'ctx, '_>,
        container: Value<'ctx, '_>,
//        value: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
//        index: usize,
        index: usize,
//    ) -> Result<Value<'ctx, '_>, Error> {
    ) -> Result<Value<'ctx, '_>, Error> {
//        Ok(self
        Ok(self
//            .append_operation(
            .append_operation(
//                ods::llvm::insertvalue(
                ods::llvm::insertvalue(
//                    context,
                    context,
//                    container.r#type(),
                    container.r#type(),
//                    container,
                    container,
//                    value,
                    value,
//                    DenseI64ArrayAttribute::new(context, &[index.try_into().unwrap()]).into(),
                    DenseI64ArrayAttribute::new(context, &[index.try_into().unwrap()]).into(),
//                    location,
                    location,
//                )
                )
//                .into(),
                .into(),
//            )
            )
//            .result(0)?
            .result(0)?
//            .into())
            .into())
//    }
    }
//

//    fn insert_values<'block>(
    fn insert_values<'block>(
//        &'block self,
        &'block self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        mut container: Value<'ctx, 'block>,
        mut container: Value<'ctx, 'block>,
//        values: &[Value<'ctx, 'block>],
        values: &[Value<'ctx, 'block>],
//    ) -> Result<Value<'ctx, 'block>, Error> {
    ) -> Result<Value<'ctx, 'block>, Error> {
//        for (i, value) in values.iter().enumerate() {
        for (i, value) in values.iter().enumerate() {
//            container = self.insert_value(context, location, container, *value, i)?;
            container = self.insert_value(context, location, container, *value, i)?;
//        }
        }
//        Ok(container)
        Ok(container)
//    }
    }
//

//    fn store(
    fn store(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        addr: Value<'ctx, '_>,
        addr: Value<'ctx, '_>,
//        value: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
//        align: Option<usize>,
        align: Option<usize>,
//    ) {
    ) {
//        let mut op = ods::llvm::store(context, value, addr, location);
        let mut op = ods::llvm::store(context, value, addr, location);
//

//        if let Some(align) = align {
        if let Some(align) = align {
//            op.set_alignment(IntegerAttribute::new(
            op.set_alignment(IntegerAttribute::new(
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                align as i64,
                align as i64,
//            ));
            ));
//        }
        }
//

//        self.append_operation(op.into());
        self.append_operation(op.into());
//    }
    }
//

//    // Use this only when returning the result. Otherwise, append_operation is fine.
    // Use this only when returning the result. Otherwise, append_operation is fine.
//    #[inline]
    #[inline]
//    fn append_op_result(&self, operation: Operation<'ctx>) -> Result<Value<'ctx, '_>, Error> {
    fn append_op_result(&self, operation: Operation<'ctx>) -> Result<Value<'ctx, '_>, Error> {
//        Ok(self.append_operation(operation).result(0)?.into())
        Ok(self.append_operation(operation).result(0)?.into())
//    }
    }
//

//    fn load(
    fn load(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        addr: Value<'ctx, '_>,
        addr: Value<'ctx, '_>,
//        value_type: Type<'ctx>,
        value_type: Type<'ctx>,
//        align: Option<usize>,
        align: Option<usize>,
//    ) -> Result<Value<'ctx, '_>, Error> {
    ) -> Result<Value<'ctx, '_>, Error> {
//        let mut op = ods::llvm::load(context, value_type, addr, location);
        let mut op = ods::llvm::load(context, value_type, addr, location);
//

//        if let Some(align) = align {
        if let Some(align) = align {
//            op.set_alignment(IntegerAttribute::new(
            op.set_alignment(IntegerAttribute::new(
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                align as i64,
                align as i64,
//            ));
            ));
//        }
        }
//

//        self.append_op_result(op.into())
        self.append_op_result(op.into())
//    }
    }
//

//    fn memcpy(
    fn memcpy(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        src: Value<'ctx, '_>,
        src: Value<'ctx, '_>,
//        dst: Value<'ctx, '_>,
        dst: Value<'ctx, '_>,
//        len_bytes: Value<'ctx, '_>,
        len_bytes: Value<'ctx, '_>,
//    ) {
    ) {
//        self.append_operation(
        self.append_operation(
//            ods::llvm::intr_memcpy(
            ods::llvm::intr_memcpy(
//                context,
                context,
//                dst,
                dst,
//                src,
                src,
//                len_bytes,
                len_bytes,
//                IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
//                location,
                location,
//            )
            )
//            .into(),
            .into(),
//        );
        );
//    }
    }
//

//    fn alloca(
    fn alloca(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        elem_type: Type<'ctx>,
        elem_type: Type<'ctx>,
//        num_elems: Value<'ctx, '_>,
        num_elems: Value<'ctx, '_>,
//        align: Option<usize>,
        align: Option<usize>,
//    ) -> Result<Value<'ctx, '_>, Error> {
    ) -> Result<Value<'ctx, '_>, Error> {
//        let mut op = ods::llvm::alloca(
        let mut op = ods::llvm::alloca(
//            context,
            context,
//            pointer(context, 0),
            pointer(context, 0),
//            num_elems,
            num_elems,
//            TypeAttribute::new(elem_type),
            TypeAttribute::new(elem_type),
//            location,
            location,
//        );
        );
//

//        op.set_elem_type(TypeAttribute::new(elem_type));
        op.set_elem_type(TypeAttribute::new(elem_type));
//

//        if let Some(align) = align {
        if let Some(align) = align {
//            op.set_alignment(IntegerAttribute::new(
            op.set_alignment(IntegerAttribute::new(
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                align.try_into().unwrap(),
                align.try_into().unwrap(),
//            ));
            ));
//        }
        }
//

//        self.append_op_result(op.into())
        self.append_op_result(op.into())
//    }
    }
//

//    fn alloca1(
    fn alloca1(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        elem_type: Type<'ctx>,
        elem_type: Type<'ctx>,
//        align: Option<usize>,
        align: Option<usize>,
//    ) -> Result<Value<'ctx, '_>, Error> {
    ) -> Result<Value<'ctx, '_>, Error> {
//        let num_elems = self.const_int(context, location, 1, 64)?;
        let num_elems = self.const_int(context, location, 1, 64)?;
//        self.alloca(context, location, elem_type, num_elems, align)
        self.alloca(context, location, elem_type, num_elems, align)
//    }
    }
//

//    fn alloca_int(
    fn alloca_int(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        bits: u32,
        bits: u32,
//    ) -> Result<Value<'ctx, '_>, Error> {
    ) -> Result<Value<'ctx, '_>, Error> {
//        let num_elems = self.const_int(context, location, 1, 64)?;
        let num_elems = self.const_int(context, location, 1, 64)?;
//        self.alloca(
        self.alloca(
//            context,
            context,
//            location,
            location,
//            IntegerType::new(context, bits).into(),
            IntegerType::new(context, bits).into(),
//            num_elems,
            num_elems,
//            Some(get_integer_layout(bits).align()),
            Some(get_integer_layout(bits).align()),
//        )
        )
//    }
    }
//}
}
