//! Trait that extends the melior Block type to aid in codegen and consistency.

use melior::{
    dialect::{llvm::r#type::opaque_pointer, ods},
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute},
        r#type::IntegerType,
        Attribute, Block, Location, Operation, Type, Value, ValueLike,
    },
    Context,
};
use num_bigint::BigInt;

use crate::error::Error;

pub trait BlockExt<'ctx> {
    /// Appends the operation and returns the first result.
    fn append_op_result(&self, operation: Operation<'ctx>) -> Result<Value<'ctx, '_>, Error>;

    /// Creates a constant of the given integer bit width. Do not use for felt252.
    fn const_int<T>(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        value: T,
        bits: u32,
    ) -> Result<Value<'ctx, '_>, Error>
    where
        T: Into<BigInt>;

    /// Creates a constant of the given integer type. Do not use for felt252.
    fn const_int_from_type<T>(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        value: T,
        int_type: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>
    where
        T: Into<BigInt>;

    /// Uses a llvm::extract_value operation to return the value at the given index of a container (e.g struct).
    fn extract_value(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, '_>,
        value_type: Type<'ctx>,
        index: usize,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Uses a llvm::insert_value operation to insert the value at the given index of a container (e.g struct),
    /// the result is the container with the value.
    fn insert_value(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
        index: usize,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Uses a llvm::insert_value operation to insert the values starting from index 0 into a container (e.g struct),
    /// the result is the container with the values.
    fn insert_values<'block>(
        &'block self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, 'block>,
        values: &[Value<'ctx, 'block>],
    ) -> Result<Value<'ctx, 'block>, Error>;

    /// Loads a value from the given addr.
    fn load(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value_type: Type<'ctx>,
        align: Option<usize>,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Allocates the given number of elements of type in memory on the stack, returning a opaque pointer.
    fn alloca(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        elem_type: Type<'ctx>,
        num_elems: Value<'ctx, '_>,
        align: Option<usize>,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Allocates one element of the given type in memory on the stack, returning a opaque pointer.
    fn alloca1(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        elem_type: Type<'ctx>,
        align: Option<usize>,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Stores a value at the given addr.
    fn store(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
        align: Option<usize>,
    );

    /// Creates a memcpy operation.
    fn memcpy(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        src: Value<'ctx, '_>,
        dst: Value<'ctx, '_>,
        len_bytes: Value<'ctx, '_>,
    );
}

impl<'ctx> BlockExt<'ctx> for Block<'ctx> {
    fn const_int<T>(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        value: T,
        bits: u32,
    ) -> Result<Value<'ctx, '_>, Error>
    where
        T: Into<BigInt>,
    {
        let ty = IntegerType::new(context, bits).into();
        Ok(self
            .append_operation(
                ods::arith::constant(
                    context,
                    ty,
                    Attribute::parse(context, &format!("{} : {}", value.into(), ty))
                        .ok_or(Error::ParseAttributeError)?,
                    location,
                )
                .into(),
            )
            .result(0)?
            .into())
    }

    fn const_int_from_type<T>(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        value: T,
        ty: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>
    where
        T: Into<BigInt>,
    {
        Ok(self
            .append_operation(
                ods::arith::constant(
                    context,
                    ty,
                    Attribute::parse(context, &format!("{} : {}", value.into(), ty))
                        .ok_or(Error::ParseAttributeError)?,
                    location,
                )
                .into(),
            )
            .result(0)?
            .into())
    }

    fn extract_value(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, '_>,
        value_type: Type<'ctx>,
        index: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
        Ok(self
            .append_operation(
                ods::llvm::extractvalue(
                    context,
                    value_type,
                    container,
                    DenseI64ArrayAttribute::new(context, &[index.try_into().unwrap()]).into(),
                    location,
                )
                .into(),
            )
            .result(0)?
            .into())
    }

    fn insert_value(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
        index: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
        Ok(self
            .append_operation(
                ods::llvm::insertvalue(
                    context,
                    container.r#type(),
                    container,
                    value,
                    DenseI64ArrayAttribute::new(context, &[index.try_into().unwrap()]).into(),
                    location,
                )
                .into(),
            )
            .result(0)?
            .into())
    }

    fn insert_values<'block>(
        &'block self,
        context: &'ctx Context,
        location: Location<'ctx>,
        mut container: Value<'ctx, 'block>,
        values: &[Value<'ctx, 'block>],
    ) -> Result<Value<'ctx, 'block>, Error> {
        for (i, value) in values.iter().enumerate() {
            container = self.insert_value(context, location, container, *value, i)?;
        }
        Ok(container)
    }

    fn store(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
        align: Option<usize>,
    ) {
        let mut op = ods::llvm::store(context, value, addr, location);

        if let Some(align) = align {
            op.set_alignment(IntegerAttribute::new(
                IntegerType::new(context, 64).into(),
                align as i64,
            ));
        }

        self.append_operation(op.into());
    }

    #[inline]
    fn append_op_result(&self, operation: Operation<'ctx>) -> Result<Value<'ctx, '_>, Error> {
        Ok(self.append_operation(operation).result(0)?.into())
    }

    fn load(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value_type: Type<'ctx>,
        align: Option<usize>,
    ) -> Result<Value<'ctx, '_>, Error> {
        let mut op = ods::llvm::load(context, value_type, addr, location);

        if let Some(align) = align {
            op.set_alignment(IntegerAttribute::new(
                IntegerType::new(context, 64).into(),
                align as i64,
            ));
        }

        self.append_op_result(op.into())
    }

    fn memcpy(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        src: Value<'ctx, '_>,
        dst: Value<'ctx, '_>,
        len_bytes: Value<'ctx, '_>,
    ) {
        self.append_operation(
            ods::llvm::intr_memcpy(
                context,
                dst,
                src,
                len_bytes,
                IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                location,
            )
            .into(),
        );
    }

    fn alloca(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        elem_type: Type<'ctx>,
        num_elems: Value<'ctx, '_>,
        align: Option<usize>,
    ) -> Result<Value<'ctx, '_>, Error> {
        let mut op = ods::llvm::alloca(context, opaque_pointer(context), num_elems, location);

        op.set_elem_type(TypeAttribute::new(elem_type));

        if let Some(align) = align {
            op.set_alignment(IntegerAttribute::new(
                IntegerType::new(context, 64).into(),
                align.try_into().unwrap(),
            ));
        }

        self.append_op_result(op.into())
    }

    fn alloca1(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        elem_type: Type<'ctx>,
        align: Option<usize>,
    ) -> Result<Value<'ctx, '_>, Error> {
        let num_elems = self.const_int(context, location, 1, 64)?;
        self.alloca(context, location, elem_type, num_elems, align)
    }
}
