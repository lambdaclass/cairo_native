//! # Runtime library bindings
//!
//! This metadata ensures that the bindings to the runtime functions exist in the current
//! compilation context.

use crate::{
    error::{Error, Result},
    libfuncs::LibfuncHelper,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, llvm, ods,
    },
    helpers::{ArithBlockExt, BuiltinBlockExt, LlvmBlockExt},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, BlockLike, Identifier, Location, Module, OperationRef, Region, Type,
        Value,
    },
    Context,
};
use std::{
    alloc::Layout,
    collections::HashSet,
    ffi::{c_int, c_void},
};

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
enum RuntimeBinding {
    Pedersen,
    HadesPermutation,
    EcStateTryFinalizeNz,
    EcStateAddMul,
    EcStateInit,
    EcStateAdd,
    EcPointTryNewNz,
    EcPointFromXNz,
    DictNew,
    DictGet,
    DictSquash,
    DictDrop,
    DictDup,
    GetCostsBuiltin,
    DebugPrint,
    ExtendedEuclideanAlgorithm,
    #[cfg(feature = "with-cheatcode")]
    VtableCheatcode,
}

impl RuntimeBinding {
    const fn symbol(self) -> &'static str {
        match self {
            RuntimeBinding::DebugPrint => "cairo_native__libfunc__debug__print",
            RuntimeBinding::Pedersen => "cairo_native__libfunc__pedersen",
            RuntimeBinding::HadesPermutation => "cairo_native__libfunc__hades_permutation",
            RuntimeBinding::EcStateTryFinalizeNz => {
                "cairo_native__libfunc__ec__ec_state_try_finalize_nz"
            }
            RuntimeBinding::EcStateAddMul => "cairo_native__libfunc__ec__ec_state_add_mul",
            RuntimeBinding::EcStateInit => "cairo_native__libfunc__ec__ec_state_init",
            RuntimeBinding::EcStateAdd => "cairo_native__libfunc__ec__ec_state_add",
            RuntimeBinding::EcPointTryNewNz => "cairo_native__libfunc__ec__ec_point_try_new_nz",
            RuntimeBinding::EcPointFromXNz => "cairo_native__libfunc__ec__ec_point_from_x_nz",
            RuntimeBinding::DictNew => "cairo_native__dict_new",
            RuntimeBinding::DictGet => "cairo_native__dict_get",
            RuntimeBinding::DictSquash => "cairo_native__dict_squash",
            RuntimeBinding::DictDrop => "cairo_native__dict_drop",
            RuntimeBinding::DictDup => "cairo_native__dict_dup",
            RuntimeBinding::GetCostsBuiltin => "cairo_native__get_costs_builtin",
            RuntimeBinding::ExtendedEuclideanAlgorithm => {
                "cairo_native__extended_euclidean_algorithm"
            }
            #[cfg(feature = "with-cheatcode")]
            RuntimeBinding::VtableCheatcode => "cairo_native__vtable_cheatcode",
        }
    }

    const fn function_ptr(self) -> *const () {
        match self {
            RuntimeBinding::DebugPrint => {
                crate::runtime::cairo_native__libfunc__debug__print as *const ()
            }
            RuntimeBinding::Pedersen => {
                crate::runtime::cairo_native__libfunc__pedersen as *const ()
            }
            RuntimeBinding::HadesPermutation => {
                crate::runtime::cairo_native__libfunc__hades_permutation as *const ()
            }
            RuntimeBinding::EcStateTryFinalizeNz => {
                crate::runtime::cairo_native__libfunc__ec__ec_state_try_finalize_nz as *const ()
            }
            RuntimeBinding::EcStateAddMul => {
                crate::runtime::cairo_native__libfunc__ec__ec_state_add_mul as *const ()
            }
            RuntimeBinding::EcStateInit => {
                crate::runtime::cairo_native__libfunc__ec__ec_state_init as *const ()
            }
            RuntimeBinding::EcStateAdd => {
                crate::runtime::cairo_native__libfunc__ec__ec_state_add as *const ()
            }
            RuntimeBinding::EcPointTryNewNz => {
                crate::runtime::cairo_native__libfunc__ec__ec_point_try_new_nz as *const ()
            }
            RuntimeBinding::EcPointFromXNz => {
                crate::runtime::cairo_native__libfunc__ec__ec_point_from_x_nz as *const ()
            }
            RuntimeBinding::DictNew => crate::runtime::cairo_native__dict_new as *const (),
            RuntimeBinding::DictGet => crate::runtime::cairo_native__dict_get as *const (),
            RuntimeBinding::DictSquash => crate::runtime::cairo_native__dict_squash as *const (),
            RuntimeBinding::DictDrop => crate::runtime::cairo_native__dict_drop as *const (),
            RuntimeBinding::DictDup => crate::runtime::cairo_native__dict_dup as *const (),
            RuntimeBinding::GetCostsBuiltin => {
                crate::runtime::cairo_native__get_costs_builtin as *const ()
            }
            RuntimeBinding::ExtendedEuclideanAlgorithm => unreachable!(),
            #[cfg(feature = "with-cheatcode")]
            RuntimeBinding::VtableCheatcode => {
                crate::starknet::cairo_native__vtable_cheatcode as *const ()
            }
        }
    }
}

/// Runtime library bindings metadata.
#[derive(Debug, Default)]
pub struct RuntimeBindingsMeta {
    active_map: HashSet<RuntimeBinding>,
}

impl RuntimeBindingsMeta {
    /// Register the global for the given binding, if not yet registered, and return
    /// a pointer to the stored function.
    ///
    /// For the function to be available, `setup_runtime` must be called before running the module
    fn build_function<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
        binding: RuntimeBinding,
    ) -> Result<Value<'c, 'a>> {
        if self.active_map.insert(binding) {
            module.body().append_operation(
                ods::llvm::mlir_global(
                    context,
                    Region::new(),
                    TypeAttribute::new(llvm::r#type::pointer(context, 0)),
                    StringAttribute::new(context, binding.symbol()),
                    Attribute::parse(context, "#llvm.linkage<weak>")
                        .ok_or(Error::ParseAttributeError)?,
                    location,
                )
                .into(),
            );
        }

        let global_address = block.append_op_result(
            ods::llvm::mlir_addressof(
                context,
                llvm::r#type::pointer(context, 0),
                FlatSymbolRefAttribute::new(context, binding.symbol()),
                location,
            )
            .into(),
        )?;

        Ok(block.load(
            context,
            location,
            global_address,
            llvm::r#type::pointer(context, 0),
        )?)
    }

    /// Build if necessary the extended euclidean algorithm used in circuit inverse gates
    pub fn extended_euclidean_algorithm<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
        rhs_value: Value<'c, '_>,
        circuit_modulus: Value<'c, '_>,
    ) -> Result<Value<'c, 'a>>
    where
        'c: 'a,
    {
        let func_symbol = RuntimeBinding::ExtendedEuclideanAlgorithm.symbol();
        if self
            .active_map
            .insert(RuntimeBinding::ExtendedEuclideanAlgorithm)
        {
            build_egcd_function(module, context, location, func_symbol)?;
        }
        let integer_type: Type = IntegerType::new(context, 384 * 2).into();
        // The struct returned by the function that contains both of the results
        let return_type = llvm::r#type::r#struct(context, &[integer_type, integer_type], false);
        Ok(block
            .append_operation(
                OperationBuilder::new("llvm.call", location)
                    .add_attributes(&[(
                        Identifier::new(context, "callee"),
                        FlatSymbolRefAttribute::new(context, func_symbol).into(),
                    )])
                    .add_operands(&[rhs_value, circuit_modulus])
                    .add_results(&[return_type])
                    .build()?,
            )
            .result(0)?
            .into())
    }

    /// Register if necessary, then invoke the `debug::print()` function.
    #[allow(clippy::too_many_arguments)]
    pub fn libfunc_debug_print<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        target_fd: Value<'c, '_>,
        values_ptr: Value<'c, '_>,
        values_len: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>>
    where
        'c: 'a,
    {
        let function =
            self.build_function(context, module, block, location, RuntimeBinding::DebugPrint)?;

        Ok(block
            .append_operation(
                OperationBuilder::new("llvm.call", location)
                    .add_operands(&[function])
                    .add_operands(&[target_fd, values_ptr, values_len])
                    .add_results(&[IntegerType::new(context, 32).into()])
                    .build()?,
            )
            .result(0)?
            .into())
    }

    /// Register if necessary, then invoke the `pedersen()` function.
    #[allow(clippy::too_many_arguments)]
    pub fn libfunc_pedersen<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        dst_ptr: Value<'c, '_>,
        lhs_ptr: Value<'c, '_>,
        rhs_ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function =
            self.build_function(context, module, block, location, RuntimeBinding::Pedersen)?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[dst_ptr, lhs_ptr, rhs_ptr])
                .build()?,
        ))
    }

    /// Register if necessary, then invoke the `poseidon()` function.
    /// The passed pointers serve both as in/out pointers. I.E results are stored in the given pointers.
    #[allow(clippy::too_many_arguments)]
    pub fn libfunc_hades_permutation<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        op0_ptr: Value<'c, '_>,
        op1_ptr: Value<'c, '_>,
        op2_ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function = self.build_function(
            context,
            module,
            block,
            location,
            RuntimeBinding::HadesPermutation,
        )?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[op0_ptr, op1_ptr, op2_ptr])
                .build()?,
        ))
    }

    /// Register if necessary, then invoke the `ec_point_from_x_nz()` function.
    pub fn libfunc_ec_point_from_x_nz<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        point_ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function = self.build_function(
            context,
            module,
            block,
            location,
            RuntimeBinding::EcPointFromXNz,
        )?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[point_ptr])
                .add_results(&[IntegerType::new(context, 1).into()])
                .build()?,
        ))
    }

    /// Register if necessary, then invoke the `ec_point_try_new_nz()` function.
    pub fn libfunc_ec_point_try_new_nz<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        point_ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function = self.build_function(
            context,
            module,
            block,
            location,
            RuntimeBinding::EcPointTryNewNz,
        )?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[point_ptr])
                .add_results(&[IntegerType::new(context, 1).into()])
                .build()?,
        ))
    }

    /// Register if necessary, then invoke the `ec_state_init()` function.
    pub fn libfunc_ec_state_init<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        state_ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function = self.build_function(
            context,
            module,
            block,
            location,
            RuntimeBinding::EcStateInit,
        )?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[state_ptr])
                .build()?,
        ))
    }

    /// Register if necessary, then invoke the `ec_state_add()` function.
    pub fn libfunc_ec_state_add<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        state_ptr: Value<'c, '_>,
        point_ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function =
            self.build_function(context, module, block, location, RuntimeBinding::EcStateAdd)?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[state_ptr, point_ptr])
                .build()?,
        ))
    }

    /// Register if necessary, then invoke the `ec_state_add_mul()` function.
    #[allow(clippy::too_many_arguments)]
    pub fn libfunc_ec_state_add_mul<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        state_ptr: Value<'c, '_>,
        scalar_ptr: Value<'c, '_>,
        point_ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function = self.build_function(
            context,
            module,
            block,
            location,
            RuntimeBinding::EcStateAddMul,
        )?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[state_ptr, scalar_ptr, point_ptr])
                .build()?,
        ))
    }

    pub fn libfunc_ec_state_try_finalize_nz<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        point_ptr: Value<'c, '_>,
        state_ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function = self.build_function(
            context,
            module,
            block,
            location,
            RuntimeBinding::EcStateTryFinalizeNz,
        )?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[point_ptr, state_ptr])
                .add_results(&[IntegerType::new(context, 1).into()])
                .build()?,
        ))
    }

    /// Register if necessary, then invoke the `dict_alloc_new()` function.
    ///
    /// Returns a opaque pointer as the result.
    #[allow(clippy::too_many_arguments)]
    pub fn dict_new<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
        drop_fn: Option<Value<'c, 'a>>,
        layout: Layout,
    ) -> Result<Value<'c, 'a>>
    where
        'c: 'a,
    {
        let function =
            self.build_function(context, module, block, location, RuntimeBinding::DictNew)?;

        let i64_ty = IntegerType::new(context, 64).into();
        let size = block.const_int_from_type(context, location, layout.size(), i64_ty)?;
        let align = block.const_int_from_type(context, location, layout.align(), i64_ty)?;

        let drop_fn = match drop_fn {
            Some(x) => x,
            None => {
                block.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?
            }
        };

        Ok(block.append_op_result(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[size, align, drop_fn])
                .add_results(&[llvm::r#type::pointer(context, 0)])
                .build()?,
        )?)
    }

    /// Register if necessary, then invoke the `dict_alloc_new()` function.
    ///
    /// Returns a opaque pointer as the result.
    #[allow(clippy::too_many_arguments)]
    pub fn dict_drop<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        ptr: Value<'c, 'a>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function =
            self.build_function(context, module, block, location, RuntimeBinding::DictDrop)?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[ptr])
                .build()?,
        ))
    }

    /// Register if necessary, then invoke the `dict_alloc_new()` function.
    ///
    /// Returns a opaque pointer as the result.
    #[allow(clippy::too_many_arguments)]
    pub fn dict_dup<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        ptr: Value<'c, 'a>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>>
    where
        'c: 'a,
    {
        let function =
            self.build_function(context, module, block, location, RuntimeBinding::DictDup)?;

        Ok(block.append_op_result(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[ptr])
                .add_results(&[llvm::r#type::pointer(context, 0)])
                .build()?,
        )?)
    }

    /// Register if necessary, then invoke the `dict_get()` function.
    ///
    /// Gets the value for a given key, the returned pointer is null if not found.
    ///
    /// Returns a opaque pointer as the result.
    #[allow(clippy::too_many_arguments)]
    pub fn dict_get<'c, 'a>(
        &mut self,
        context: &'c Context,
        helper: &LibfuncHelper<'c, 'a>,
        block: &'a Block<'c>,
        dict_ptr: Value<'c, 'a>, // ptr to the dict
        key_ptr: Value<'c, 'a>,  // key must be a ptr to Felt
        location: Location<'c>,
    ) -> Result<(Value<'c, 'a>, Value<'c, 'a>)>
    where
        'c: 'a,
    {
        let function =
            self.build_function(context, helper, block, location, RuntimeBinding::DictGet)?;

        let value_ptr = helper.init_block().alloca1(
            context,
            location,
            llvm::r#type::pointer(context, 0),
            align_of::<*mut ()>(),
        )?;

        let is_present = block.append_op_result(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[dict_ptr, key_ptr, value_ptr])
                .add_results(&[IntegerType::new(context, c_int::BITS).into()])
                .build()?,
        )?;

        let value_ptr = block.load(
            context,
            location,
            value_ptr,
            llvm::r#type::pointer(context, 0),
        )?;

        Ok((is_present, value_ptr))
    }

    /// Register if necessary, then invoke the `dict_gas_refund()` function.
    ///
    /// Compute the total gas refund for the dictionary.
    ///
    /// Returns a u64 of the result.
    #[allow(clippy::too_many_arguments)]
    pub fn dict_squash<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        dict_ptr: Value<'c, 'a>,        // ptr to the dict
        range_check_ptr: Value<'c, 'a>, // ptr to range check
        gas_ptr: Value<'c, 'a>,         // ptr to gas
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function =
            self.build_function(context, module, block, location, RuntimeBinding::DictSquash)?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[dict_ptr, range_check_ptr, gas_ptr])
                .add_results(&[IntegerType::new(context, 64).into()])
                .build()?,
        ))
    }

    // Register if necessary, then invoke the `get_costs_builtin()` function.
    #[allow(clippy::too_many_arguments)]
    pub fn get_costs_builtin<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function = self.build_function(
            context,
            module,
            block,
            location,
            RuntimeBinding::GetCostsBuiltin,
        )?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_results(&[llvm::r#type::pointer(context, 0)])
                .build()?,
        ))
    }

    /// Register if necessary, then invoke the `vtable_cheatcode()` runtime function.
    ///
    /// Calls the cheatcode syscall with the given arguments.
    ///
    /// The result is stored in `result_ptr`.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "with-cheatcode")]
    pub fn vtable_cheatcode<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
        result_ptr: Value<'c, 'a>,
        selector_ptr: Value<'c, 'a>,
        args: Value<'c, 'a>,
    ) -> Result<OperationRef<'c, 'a>>
    where
        'c: 'a,
    {
        let function = self.build_function(
            context,
            module,
            block,
            location,
            RuntimeBinding::VtableCheatcode,
        )?;

        Ok(block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[result_ptr, selector_ptr, args])
                .build()?,
        ))
    }
}

pub fn setup_runtime(find_symbol_ptr: impl Fn(&str) -> Option<*mut c_void>) {
    for binding in [
        RuntimeBinding::DebugPrint,
        RuntimeBinding::Pedersen,
        RuntimeBinding::HadesPermutation,
        RuntimeBinding::EcStateTryFinalizeNz,
        RuntimeBinding::EcStateAddMul,
        RuntimeBinding::EcStateInit,
        RuntimeBinding::EcStateAdd,
        RuntimeBinding::EcPointTryNewNz,
        RuntimeBinding::EcPointFromXNz,
        RuntimeBinding::DictNew,
        RuntimeBinding::DictGet,
        RuntimeBinding::DictSquash,
        RuntimeBinding::DictDrop,
        RuntimeBinding::DictDup,
        RuntimeBinding::GetCostsBuiltin,
        RuntimeBinding::DebugPrint,
        #[cfg(feature = "with-cheatcode")]
        RuntimeBinding::VtableCheatcode,
    ] {
        if let Some(global) = find_symbol_ptr(binding.symbol()) {
            let global = global.cast::<*const ()>();
            unsafe { *global = binding.function_ptr() };
        }
    }
}

/// The extended euclidean algorithm calculates the greatest common divisor (gcd) of two integers a and b,
/// as well as the bezout coefficients x and y such that ax+by=gcd(a,b)
/// if gcd(a,b) = 1, then x is the modular multiplicative inverse of a modulo b.
/// See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
///
/// Given two numbers a, b. It returns a block with gcd(a, b) and the bezout coefficient x.
/// Both of these numbers are received as arguments in the entry block.
fn build_egcd_function<'ctx>(
    module: &Module,
    context: &'ctx Context,
    location: Location<'ctx>,
    func_symbol: &str,
) -> Result<()> {
    let integer_type: Type = IntegerType::new(context, 384 * 2).into();
    let region = Region::new();

    let entry_block = region.append_block(Block::new(&[
        (integer_type, location),
        (integer_type, location),
    ]));

    let a = entry_block.arg(0)?;
    let b = entry_block.arg(1)?;
    // The egcd algorithm works by calculating a series of remainders, each the remainder of dividing the previous two
    // For the initial setup, r0 = b, r1 = a.
    // This order is chosen because if we reverse them, then the first iteration will just swap them
    let remainder = a;
    let prev_remainder = b;

    // Similarly we'll calculate another series which starts 0,1,... and from which we
    // will retrieve themodular inverse of a
    let prev_inverse = entry_block.const_int_from_type(context, location, 0, integer_type)?;
    let inverse = entry_block.const_int_from_type(context, location, 1, integer_type)?;

    let loop_block = region.append_block(Block::new(&[
        (integer_type, location),
        (integer_type, location),
        (integer_type, location),
        (integer_type, location),
    ]));
    let end_block = region.append_block(Block::new(&[
        (integer_type, location),
        (integer_type, location),
    ]));

    entry_block.append_operation(cf::br(
        &loop_block,
        &[prev_remainder, remainder, prev_inverse, inverse],
        location,
    ));

    // -- Loop body --
    // Arguments are rem_(i-1), rem, inv_(i-1), inv
    let prev_remainder = loop_block.arg(0)?;
    let remainder = loop_block.arg(1)?;
    let prev_inverse = loop_block.arg(2)?;
    let inverse = loop_block.arg(3)?;

    // First calculate q = rem_(i-1)/rem_i, rounded down
    let quotient =
        loop_block.append_op_result(arith::divui(prev_remainder, remainder, location))?;

    // Then r_(i+1) = r_(i-1) - q * r_i, and inv_(i+1) = inv_(i-1) - q * inv_i
    let rem_times_quo = loop_block.muli(remainder, quotient, location)?;
    let inv_times_quo = loop_block.muli(inverse, quotient, location)?;
    let next_remainder =
        loop_block.append_op_result(arith::subi(prev_remainder, rem_times_quo, location))?;
    let next_inverse =
        loop_block.append_op_result(arith::subi(prev_inverse, inv_times_quo, location))?;

    // Check if r_(i+1) is 0
    // If true, then:
    // - r_i is the gcd of a and b
    // - inv_i is the bezout coefficient x
    let zero = loop_block.const_int_from_type(context, location, 0, integer_type)?;
    let next_remainder_eq_zero =
        loop_block.cmpi(context, CmpiPredicate::Eq, next_remainder, zero, location)?;
    loop_block.append_operation(cf::cond_br(
        context,
        next_remainder_eq_zero,
        &end_block,
        &loop_block,
        &[remainder, inverse],
        &[remainder, next_remainder, inverse, next_inverse],
        location,
    ));

    // Create the struct that will contain the results
    let results = end_block.append_op_result(llvm::undef(
        llvm::r#type::r#struct(context, &[integer_type, integer_type], false),
        location,
    ))?;
    let results = end_block.insert_values(
        context,
        location,
        results,
        &[end_block.arg(0)?, end_block.arg(1)?],
    )?;
    end_block.append_operation(llvm::r#return(Some(results), location));

    let func_name = StringAttribute::new(context, func_symbol);
    module.body().append_operation(llvm::func(
        context,
        func_name,
        TypeAttribute::new(llvm::r#type::function(
            llvm::r#type::r#struct(context, &[integer_type, integer_type], false),
            &[integer_type, integer_type],
            false,
        )),
        region,
        &[(
            Identifier::new(context, "no_inline"),
            Attribute::unit(context),
        )],
        location,
    ));
    Ok(())
}
