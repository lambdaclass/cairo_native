#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>


extern "C" unsigned Type_getABIAlignment(const void *mod_ptr, const void *ty_ptr)
{
    mlir::ModuleOp mod = mlir::ModuleOp::getFromOpaquePointer(mod_ptr);
    mlir::Type ty = mlir::Type::getFromOpaquePointer(ty_ptr);

    mlir::DataLayout data_layout(mod);
    return data_layout.getTypeABIAlignment(ty);
}

extern "C" unsigned Type_getPreferredAlignment(const void *mod_ptr, const void *ty_ptr)
{
    mlir::ModuleOp mod = mlir::ModuleOp::getFromOpaquePointer(mod_ptr);
    mlir::Type ty = mlir::Type::getFromOpaquePointer(ty_ptr);

    mlir::DataLayout data_layout(mod);
    return data_layout.getTypePreferredAlignment(ty);
}

extern "C" unsigned Type_getSize(const void *mod_ptr, const void *ty_ptr)
{
    mlir::ModuleOp mod = mlir::ModuleOp::getFromOpaquePointer(mod_ptr);
    mlir::Type ty = mlir::Type::getFromOpaquePointer(ty_ptr);

    mlir::DataLayout data_layout(mod);
    return data_layout.getTypeSize(ty);
}

extern "C" unsigned Type_getSizeInBits(const void *mod_ptr, const void *ty_ptr)
{
    mlir::ModuleOp mod = mlir::ModuleOp::getFromOpaquePointer(mod_ptr);
    mlir::Type ty = mlir::Type::getFromOpaquePointer(ty_ptr);

    mlir::DataLayout data_layout(mod);
    return data_layout.getTypeSizeInBits(ty);
}

extern "C" const void *LLVMStructType_getFieldTypeAt(const void *ty_ptr, unsigned index)
{
    mlir::LLVM::LLVMStructType type = mlir::LLVM::LLVMStructType::getFromOpaquePointer(ty_ptr);

    return type.getBody()[index].getAsOpaquePointer();
}

extern "C" const void *MemRefType_getElementType(const void *ty_ptr)
{
    mlir::MemRefType type = mlir::MemRefType::getFromOpaquePointer(ty_ptr);

    return type.getElementType().getAsOpaquePointer();
}
