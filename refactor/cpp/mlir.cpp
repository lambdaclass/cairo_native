#include "mlir.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Support/LLVM.h>


Block *make_Block(const OpBuilder &builder, Region *region)
{
    return const_cast<OpBuilder &>(builder).createBlock(region);
}

std::unique_ptr<MLIRContext> make_MLIRContext()
{
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    return std::make_unique<MLIRContext>(registry);
}

std::unique_ptr<ModuleOp> make_ModuleOp(const Location &location)
{
    return std::make_unique<ModuleOp>(ModuleOp::create(location));
}

std::unique_ptr<OpBuilder> make_OpBuilder(const MLIRContext &context)
{
    return std::make_unique<OpBuilder>(const_cast<MLIRContext *>(&context));
}

std::unique_ptr<OperationState> make_OperationState(const Location &location, rust::Str name)
{
    return std::make_unique<OperationState>(location, mlir::StringRef{name.data(), name.size()});
}

std::unique_ptr<Region> make_Region()
{
    return std::make_unique<Region>();
}


std::unique_ptr<Value> proxy_Block_getArgument(Block *block, std::uint32_t index)
{
    return std::make_unique<Value>(block->getArgument(index));
}

void proxy_Block_push_back(Block *block, Operation *op)
{
    block->push_back(op);
}

Region *proxy_ModuleOp_getBodyRegion(const ModuleOp &module)
{
    return &const_cast<ModuleOp &>(module).getBodyRegion();
}

std::unique_ptr<Type> proxy_OpBuilder_getFunctionType(
    const OpBuilder & builder,
    rust::Slice<Type *const> args,
    rust::Slice<Type *const> rets
)
{
    std::vector<Type> args_vec;
    std::vector<Type> rets_vec;

    for (Type *arg : args)
        args_vec.push_back(*arg);
    for (Type *ret : rets)
        rets_vec.push_back(*ret);

    return std::make_unique<Type>(
        const_cast<OpBuilder &>(builder)
            .getFunctionType(args_vec, rets_vec)
    );
}

std::unique_ptr<Type> proxy_OpBuilder_getIntegerType(
    const OpBuilder &builder,
    std::uint32_t width
)
{
    return std::make_unique<Type>(const_cast<OpBuilder &>(builder).getIntegerType(width));
}

std::unique_ptr<Attribute> proxy_OpBuilder_getStringAttr(const OpBuilder &builder, rust::Str value)
{
    return std::make_unique<Attribute>(
        const_cast<OpBuilder &>(builder)
            .getStringAttr(mlir::StringRef(value.data(), value.size()))
    );
}

std::unique_ptr<Location> proxy_OpBuilder_getUnknownLoc(const OpBuilder &builder)
{
    return std::make_unique<Location>(const_cast<OpBuilder &>(builder).getUnknownLoc());
}

std::uint32_t proxy_Operation_getNumResults(Operation *op)
{
    return op->getNumResults();
}

std::unique_ptr<Value> proxy_Operation_getResult(Operation *op, std::uint32_t index)
{
    return std::make_unique<Value>(op->getResult(index));
}

void proxy_OperationState_addAttribute(
    const OperationState &state,
    rust::Str name,
    const Attribute &attr
)
{
    const_cast<OperationState &>(state).addAttribute(
        mlir::StringRef(name.data(), name.size()),
        attr
    );
}

void proxy_OperationState_addOperand(const OperationState &state, const Value &operand)
{
    const_cast<OperationState &>(state).addOperands(operand);
}

void proxy_OperationState_addRegion(const OperationState &state, std::unique_ptr<Region> region)
{
    const_cast<OperationState &>(state).addRegion(std::move(region));
}

void proxy_OperationState_addSuccessor(const OperationState &state, Block *successor)
{
    const_cast<OperationState &>(state).addSuccessors(successor);
}

void proxy_OperationState_addType(const OperationState &state, const Type &type)
{
    const_cast<OperationState &>(state).addTypes(type);
}


Operation *aux_Block_createAndPush(Block *block, const OperationState &state)
{
    Operation *op = Operation::create(state);
    block->push_back(op);
    return op;
}

std::unique_ptr<std::string> aux_ModuleOp_print(const ModuleOp &module)
{
    std::string output;
    llvm::raw_string_ostream stream(output);
    const_cast<ModuleOp &>(module).print(stream);
    return std::make_unique<std::string>(output);
}

Block *aux_Region_getFirstBlock(Region *region)
{
    return &region->getBlocks().front();
}

std::unique_ptr<Attribute> aux_TypeAttr_get(const Type &type)
{
    return std::make_unique<Attribute>(mlir::TypeAttr::get(type));
}
