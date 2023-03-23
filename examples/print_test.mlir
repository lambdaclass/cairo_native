module attributes {llvm.data_layout = ""} {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func internal @print_felt(%arg0: i256) {
    %0 = llvm.mlir.constant(224 : i256) : i256
    %1 = llvm.ashr %arg0, %0  : i256
    %2 = llvm.trunc %1 : i256 to i32
    %3 = llvm.mlir.constant(5 : i64) : i64
    %4 = llvm.alloca %3 x i8 : (i64) -> !llvm.ptr
    %5 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %5, %4 : !llvm.array<5 x i8>, !llvm.ptr
    %6 = llvm.call @printf(%4, %2) : (!llvm.ptr, i32) -> i32
    %7 = llvm.mlir.constant(192 : i256) : i256
    %8 = llvm.ashr %arg0, %7  : i256
    %9 = llvm.trunc %8 : i256 to i32
    %10 = llvm.mlir.constant(5 : i64) : i64
    %11 = llvm.alloca %10 x i8 : (i64) -> !llvm.ptr
    %12 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %12, %11 : !llvm.array<5 x i8>, !llvm.ptr
    %13 = llvm.call @printf(%11, %9) : (!llvm.ptr, i32) -> i32
    %14 = llvm.mlir.constant(160 : i256) : i256
    %15 = llvm.ashr %arg0, %14  : i256
    %16 = llvm.trunc %15 : i256 to i32
    %17 = llvm.mlir.constant(5 : i64) : i64
    %18 = llvm.alloca %17 x i8 : (i64) -> !llvm.ptr
    %19 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %19, %18 : !llvm.array<5 x i8>, !llvm.ptr
    %20 = llvm.call @printf(%18, %16) : (!llvm.ptr, i32) -> i32
    %21 = llvm.mlir.constant(128 : i256) : i256
    %22 = llvm.ashr %arg0, %21  : i256
    %23 = llvm.trunc %22 : i256 to i32
    %24 = llvm.mlir.constant(5 : i64) : i64
    %25 = llvm.alloca %24 x i8 : (i64) -> !llvm.ptr
    %26 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %26, %25 : !llvm.array<5 x i8>, !llvm.ptr
    %27 = llvm.call @printf(%25, %23) : (!llvm.ptr, i32) -> i32
    %28 = llvm.mlir.constant(96 : i256) : i256
    %29 = llvm.ashr %arg0, %28  : i256
    %30 = llvm.trunc %29 : i256 to i32
    %31 = llvm.mlir.constant(5 : i64) : i64
    %32 = llvm.alloca %31 x i8 : (i64) -> !llvm.ptr
    %33 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %33, %32 : !llvm.array<5 x i8>, !llvm.ptr
    %34 = llvm.call @printf(%32, %30) : (!llvm.ptr, i32) -> i32
    %35 = llvm.mlir.constant(64 : i256) : i256
    %36 = llvm.ashr %arg0, %35  : i256
    %37 = llvm.trunc %36 : i256 to i32
    %38 = llvm.mlir.constant(5 : i64) : i64
    %39 = llvm.alloca %38 x i8 : (i64) -> !llvm.ptr
    %40 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %40, %39 : !llvm.array<5 x i8>, !llvm.ptr
    %41 = llvm.call @printf(%39, %37) : (!llvm.ptr, i32) -> i32
    %42 = llvm.mlir.constant(32 : i256) : i256
    %43 = llvm.ashr %arg0, %42  : i256
    %44 = llvm.trunc %43 : i256 to i32
    %45 = llvm.mlir.constant(5 : i64) : i64
    %46 = llvm.alloca %45 x i8 : (i64) -> !llvm.ptr
    %47 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %47, %46 : !llvm.array<5 x i8>, !llvm.ptr
    %48 = llvm.call @printf(%46, %44) : (!llvm.ptr, i32) -> i32
    %49 = llvm.mlir.constant(0 : i256) : i256
    %50 = llvm.trunc %arg0 : i256 to i32
    %51 = llvm.mlir.constant(5 : i64) : i64
    %52 = llvm.alloca %51 x i8 : (i64) -> !llvm.ptr
    %53 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %53, %52 : !llvm.array<5 x i8>, !llvm.ptr
    %54 = llvm.call @printf(%52, %50) : (!llvm.ptr, i32) -> i32
    %55 = llvm.mlir.constant(2 : i64) : i64
    %56 = llvm.alloca %55 x i8 : (i64) -> !llvm.ptr
    %57 = llvm.mlir.constant(dense<[10, 0]> : tensor<2xi8>) : !llvm.array<2 x i8>
    llvm.store %57, %56 : !llvm.array<2 x i8>, !llvm.ptr
    %58 = llvm.call @printf(%56) : (!llvm.ptr) -> i32
    llvm.return
  }
  llvm.func internal @"store_temp<felt252>"(%arg0: i256) -> i256 {
    llvm.return %arg0 : i256
  }
  llvm.func @main() -> i256 attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(24 : i256) : i256
    %1 = llvm.call @"store_temp<felt252>"(%0) : (i256) -> i256
    llvm.call @print_felt(%1) : (i256) -> ()
    llvm.return %1 : i256
  }
  llvm.func @_mlir_ciface_main() -> i256 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @main() : () -> i256
    llvm.return %0 : i256
  }
}
