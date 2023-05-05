#pragma once

#include <cstdint>
#include <memory>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <rust/cxx.h>


using mlir::Attribute;
using mlir::Block;
using mlir::Location;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::Region;
using mlir::Type;
using mlir::Value;

Block *make_Block(const OpBuilder &, Region *);
std::unique_ptr<MLIRContext> make_MLIRContext();
std::unique_ptr<OpBuilder> make_OpBuilder(const MLIRContext &);
std::unique_ptr<ModuleOp> make_ModuleOp(const Location &);
std::unique_ptr<OperationState> make_OperationState(const Location &, rust::Str);
std::unique_ptr<Region> make_Region();

std::unique_ptr<Value> proxy_Block_getArgument(Block *, std::uint32_t);
void proxy_Block_push_back(Block *, Operation *);
Region *proxy_ModuleOp_getBodyRegion(const ModuleOp &);
std::unique_ptr<Type> proxy_OpBuilder_getFunctionType(
    const OpBuilder &,
    rust::Slice<Type *const>,
    rust::Slice<Type *const>
);
std::unique_ptr<Type> proxy_OpBuilder_getIntegerType(const OpBuilder &, std::uint32_t);
std::unique_ptr<Attribute> proxy_OpBuilder_getStringAttr(const OpBuilder &, rust::Str);
std::unique_ptr<Location> proxy_OpBuilder_getUnknownLoc(const OpBuilder &);
std::uint32_t proxy_Operation_getNumResults(Operation *);
std::unique_ptr<Value> proxy_Operation_getResult(Operation *, std::uint32_t);
void proxy_OperationState_addAttribute(const OperationState &, rust::Str, const Attribute &);
void proxy_OperationState_addOperand(const OperationState &, const Value &);
void proxy_OperationState_addRegion(const OperationState &, std::unique_ptr<Region>);
void proxy_OperationState_addSuccessor(const OperationState &, Block *);
void proxy_OperationState_addType(const OperationState &, const Type &);

Operation *aux_Block_createAndPush(Block *, const OperationState &);
std::unique_ptr<std::string> aux_ModuleOp_print(const ModuleOp &);
Block *aux_Region_getFirstBlock(Region *);
std::unique_ptr<Attribute> aux_TypeAttr_get(const Type &);
