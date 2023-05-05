pub use self::ffi::*;

#[allow(clippy::module_inception)]
#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("sierra2mlir/cpp/mlir.hpp");

        type Attribute;
        type Block;
        type Location;
        type MLIRContext;
        type ModuleOp;
        type OpBuilder;
        type Operation;
        type OperationState;
        type Region;
        type Type;
        type Value;

        unsafe fn make_Block(builder: &OpBuilder, region: *mut Region) -> *mut Block;
        fn make_MLIRContext() -> UniquePtr<MLIRContext>;
        fn make_ModuleOp(location: &Location) -> UniquePtr<ModuleOp>;
        fn make_OpBuilder(context: &MLIRContext) -> UniquePtr<OpBuilder>;
        fn make_OperationState(location: &Location, name: &str) -> UniquePtr<OperationState>;
        fn make_Region() -> UniquePtr<Region>;

        unsafe fn proxy_Block_getArgument(block: *mut Block, index: u32) -> UniquePtr<Value>;
        unsafe fn proxy_Block_push_back(block: *mut Block, op: *mut Operation);
        fn proxy_ModuleOp_getBodyRegion(module: &ModuleOp) -> *mut Region;
        fn proxy_OpBuilder_getFunctionType(
            builder: &OpBuilder,
            args: &[*mut Type],
            rets: &[*mut Type],
        ) -> UniquePtr<Type>;
        fn proxy_OpBuilder_getIntegerType(builder: &OpBuilder, width: u32) -> UniquePtr<Type>;
        fn proxy_OpBuilder_getStringAttr(builder: &OpBuilder, value: &str) -> UniquePtr<Attribute>;
        fn proxy_OpBuilder_getUnknownLoc(builder: &OpBuilder) -> UniquePtr<Location>;
        unsafe fn proxy_Operation_getNumResults(op: *mut Operation) -> u32;
        unsafe fn proxy_Operation_getResult(op: *mut Operation, index: u32) -> UniquePtr<Value>;
        fn proxy_OperationState_addAttribute(state: &OperationState, name: &str, attr: &Attribute);
        fn proxy_OperationState_addOperand(state: &OperationState, ret_type: &Value);
        fn proxy_OperationState_addRegion(state: &OperationState, region: UniquePtr<Region>);
        unsafe fn proxy_OperationState_addSuccessor(state: &OperationState, successor: *mut Block);
        fn proxy_OperationState_addType(state: &OperationState, ret_type: &Type);

        unsafe fn aux_Block_createAndPush(
            block: *mut Block,
            state: &OperationState,
        ) -> *mut Operation;
        fn aux_ModuleOp_print(module: &ModuleOp) -> UniquePtr<CxxString>;
        unsafe fn aux_Region_getFirstBlock(region: *mut Region) -> *mut Block;
        fn aux_TypeAttr_get(attr_type: &Type) -> UniquePtr<Attribute>;
    }
}
