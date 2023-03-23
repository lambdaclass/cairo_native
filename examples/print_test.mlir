module attributes {llvm.data_layout = ""} {
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  llvm.func internal @print_felt(%arg0: i256) {
    %0 = llvm.mlir.constant(224 : i256) : i256
    %1 = llvm.ashr %arg0, %0  : i256
    %2 = llvm.trunc %1 : i256 to i32
    %3 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.ptr<i8>
    %4 = llvm.call @printf(%3, %2) : (!llvm.ptr<i8>, i32) -> i32
    %5 = llvm.mlir.constant(192 : i256) : i256
    %6 = llvm.ashr %arg0, %5  : i256
    %7 = llvm.trunc %6 : i256 to i32
    %8 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.ptr<i8>
    %9 = llvm.call @printf(%8, %7) : (!llvm.ptr<i8>, i32) -> i32
    %10 = llvm.mlir.constant(160 : i256) : i256
    %11 = llvm.ashr %arg0, %10  : i256
    %12 = llvm.trunc %11 : i256 to i32
    %13 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.ptr<i8>
    %14 = llvm.call @printf(%13, %12) : (!llvm.ptr<i8>, i32) -> i32
    %15 = llvm.mlir.constant(128 : i256) : i256
    %16 = llvm.ashr %arg0, %15  : i256
    %17 = llvm.trunc %16 : i256 to i32
    %18 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.ptr<i8>
    %19 = llvm.call @printf(%18, %17) : (!llvm.ptr<i8>, i32) -> i32
    %20 = llvm.mlir.constant(96 : i256) : i256
    %21 = llvm.ashr %arg0, %20  : i256
    %22 = llvm.trunc %21 : i256 to i32
    %23 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.ptr<i8>
    %24 = llvm.call @printf(%23, %22) : (!llvm.ptr<i8>, i32) -> i32
    %25 = llvm.mlir.constant(64 : i256) : i256
    %26 = llvm.ashr %arg0, %25  : i256
    %27 = llvm.trunc %26 : i256 to i32
    %28 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.ptr<i8>
    %29 = llvm.call @printf(%28, %27) : (!llvm.ptr<i8>, i32) -> i32
    %30 = llvm.mlir.constant(32 : i256) : i256
    %31 = llvm.ashr %arg0, %30  : i256
    %32 = llvm.trunc %31 : i256 to i32
    %33 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.ptr<i8>
    %34 = llvm.call @printf(%33, %32) : (!llvm.ptr<i8>, i32) -> i32
    %35 = llvm.mlir.constant(0 : i256) : i256
    %36 = llvm.trunc %arg0 : i256 to i32
    %37 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.ptr<i8>
    %38 = llvm.call @printf(%37, %36) : (!llvm.ptr<i8>, i32) -> i32
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
