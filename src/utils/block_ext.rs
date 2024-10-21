//! Trait that extends the melior Block type to aid in codegen and consistency.

use crate::{error::Error, utils::get_integer_layout};
use melior::{
    dialect::{llvm::r#type::pointer, ods},
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute,
        },
        r#type::IntegerType,
        Attribute, Block, Location, Operation, Type, Value, ValueLike,
    },
    Context,
};
use num_bigint::BigInt;

/// Index types for LLVM GEP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GepIndex<'c, 'a> {
    /// A compile time known index.
    Const(i32),
    /// A runtime value index.
    Value(Value<'c, 'a>),
}

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
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Allocates the given number of elements of type in memory on the stack, returning a opaque pointer.
    fn alloca(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        elem_type: Type<'ctx>,
        num_elems: Value<'ctx, '_>,
        align: usize,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Allocates one element of the given type in memory on the stack, returning a opaque pointer.
    fn alloca1(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        elem_type: Type<'ctx>,
        align: usize,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Allocates one integer of the given bit width.
    fn alloca_int(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        bits: u32,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Stores a value at the given addr.
    fn store(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
    ) -> Result<(), Error>;

    /// Creates a memcpy operation.
    fn memcpy(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        src: Value<'ctx, '_>,
        dst: Value<'ctx, '_>,
        len_bytes: Value<'ctx, '_>,
    );

    /// Creates a getelementptr operation. Returns a pointer to the indexed element.
    /// This method allows combining both compile time indexes and runtime value indexes.
    ///
    /// See:
    /// - https://llvm.org/docs/LangRef.html#getelementptr-instruction
    /// - https://llvm.org/docs/GetElementPtr.html
    ///
    /// Get Element Pointer is used to index into pointers, it uses the given
    /// element type to compute the offsets, it allows offsetting deep into a structure (field of field of a ptr for example),
    /// this is why it accepts a array of indexes, it indexes through the list, offsetting depending on the element type,
    /// it also knows when you index into a struct, so the next index uses the struct field for offsets, etc.
    /// Address computation is done at compile time.
    ///
    /// Note: This GEP sets the inbounds attribute, all GEPs we do in native should be inbounds, llvm inbounds requires the following:
    ///
    /// The base pointer has an in bounds address of the allocated object that it is based on. This means that it points into that allocated object, or to its end. Note that the object does not have to be live anymore; being in-bounds of a deallocated object is sufficient.
    ///
    /// During the successive addition of offsets to the address, the resulting pointer must remain in bounds of the allocated object at each step.
    fn gep(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        ptr: Value<'ctx, '_>,
        indexes: &[GepIndex<'ctx, '_>],
        elem_type: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;
}

impl<'ctx> BlockExt<'ctx> for Block<'ctx> {
    #[inline]
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
        self.append_op_result(
            ods::arith::constant(
                context,
                ty,
                Attribute::parse(context, &format!("{} : {}", value.into(), ty))
                    .ok_or(Error::ParseAttributeError)?,
                location,
            )
            .into(),
        )
    }

    #[inline]
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
        self.append_op_result(
            ods::arith::constant(
                context,
                ty,
                Attribute::parse(context, &format!("{} : {}", value.into(), ty))
                    .ok_or(Error::ParseAttributeError)?,
                location,
            )
            .into(),
        )
    }

    #[inline]
    fn extract_value(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, '_>,
        value_type: Type<'ctx>,
        index: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(
            ods::llvm::extractvalue(
                context,
                value_type,
                container,
                DenseI64ArrayAttribute::new(
                    context,
                    &[index.try_into().map_err(|_| Error::IntegerConversion)?],
                )
                .into(),
                location,
            )
            .into(),
        )
    }

    #[inline]
    fn insert_value(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
        index: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(
            ods::llvm::insertvalue(
                context,
                container.r#type(),
                container,
                value,
                DenseI64ArrayAttribute::new(
                    context,
                    &[index.try_into().map_err(|_| Error::IntegerConversion)?],
                )
                .into(),
                location,
            )
            .into(),
        )
    }

    #[inline]
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

    #[inline]
    fn store(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
    ) -> Result<(), Error> {
        self.append_operation(ods::llvm::store(context, value, addr, location).into());
        Ok(())
    }

    // Use this only when returning the result. Otherwise, append_operation is fine.
    #[inline]
    fn append_op_result(&self, operation: Operation<'ctx>) -> Result<Value<'ctx, '_>, Error> {
        Ok(self.append_operation(operation).result(0)?.into())
    }

    #[inline]
    fn load(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value_type: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(ods::llvm::load(context, value_type, addr, location).into())
    }

    #[inline]
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

    #[inline]
    fn alloca(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        elem_type: Type<'ctx>,
        num_elems: Value<'ctx, '_>,
        align: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
        let mut op = ods::llvm::alloca(
            context,
            pointer(context, 0),
            num_elems,
            TypeAttribute::new(elem_type),
            location,
        );

        op.set_elem_type(TypeAttribute::new(elem_type));
        op.set_alignment(IntegerAttribute::new(
            IntegerType::new(context, 64).into(),
            align.try_into().map_err(|_| Error::IntegerConversion)?,
        ));

        self.append_op_result(op.into())
    }

    #[inline]
    fn alloca1(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        elem_type: Type<'ctx>,
        align: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
        let num_elems = self.const_int(context, location, 1, 64)?;
        self.alloca(context, location, elem_type, num_elems, align)
    }

    #[inline]
    fn alloca_int(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        bits: u32,
    ) -> Result<Value<'ctx, '_>, Error> {
        let num_elems = self.const_int(context, location, 1, 64)?;
        self.alloca(
            context,
            location,
            IntegerType::new(context, bits).into(),
            num_elems,
            get_integer_layout(bits).align(),
        )
    }

    #[inline]
    fn gep(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        ptr: Value<'ctx, '_>,
        indexes: &[GepIndex<'ctx, '_>],
        elem_type: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        let mut dynamic_indices = Vec::new();
        let mut raw_constant_indices = Vec::new();

        for index in indexes {
            match index {
                GepIndex::Const(idx) => raw_constant_indices.push(*idx),
                GepIndex::Value(value) => {
                    dynamic_indices.push(*value);
                    raw_constant_indices.push(i32::MIN); // marker for dynamic index
                }
            }
        }

        let mut op = ods::llvm::getelementptr(
            context,
            pointer(context, 0),
            ptr,
            &dynamic_indices,
            DenseI32ArrayAttribute::new(context, &raw_constant_indices),
            TypeAttribute::new(elem_type),
            location,
        );
        op.set_inbounds(Attribute::unit(context));

        self.append_op_result(op.into())
    }
}
