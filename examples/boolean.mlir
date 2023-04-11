module attributes {llvm.data_layout = ""} {
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func internal @"enum_init<core::bool, 1>"(%arg0: !llvm.struct<()>) -> !llvm.struct<(i16, array<0 x i8>)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<0 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<0 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<0 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<0 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func internal @"enum_init<core::bool, 0>"(%arg0: !llvm.struct<()>) -> !llvm.struct<(i16, array<0 x i8>)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<0 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<0 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<0 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<0 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func internal @bool_or_impl(%arg0: !llvm.struct<(i16, array<0 x i8>)>, %arg1: !llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)> {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %1 = llvm.extractvalue %arg1[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %2 = llvm.or %0, %1  : i16
    %3 = llvm.mlir.undef : !llvm.struct<(i16, array<0 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(i16, array<0 x i8>)> 
    llvm.return %4 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func internal @bool_not_impl(%arg0: !llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)> {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %1 = llvm.mlir.constant(1 : i16) : i16
    %2 = llvm.xor %0, %1  : i16
    %3 = llvm.mlir.undef : !llvm.struct<(i16, array<0 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(i16, array<0 x i8>)> 
    llvm.return %4 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func internal @bool_and_impl(%arg0: !llvm.struct<(i16, array<0 x i8>)>, %arg1: !llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)> {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %1 = llvm.extractvalue %arg1[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %2 = llvm.and %0, %1  : i16
    %3 = llvm.mlir.undef : !llvm.struct<(i16, array<0 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(i16, array<0 x i8>)> 
    llvm.return %4 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func internal @bool_xor_impl(%arg0: !llvm.struct<(i16, array<0 x i8>)>, %arg1: !llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)> {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %1 = llvm.extractvalue %arg1[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %2 = llvm.xor %0, %1  : i16
    %3 = llvm.mlir.undef : !llvm.struct<(i16, array<0 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(i16, array<0 x i8>)> 
    llvm.return %4 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func internal @bool_to_felt252(%arg0: !llvm.struct<(i16, array<0 x i8>)>) -> i256 {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %1 = llvm.zext %0 : i16 to i256
    llvm.return %1 : i256
  }
  llvm.func internal @print_felt252(%arg0: i256) {
    %0 = llvm.mlir.constant(224 : i256) : i256
    %1 = llvm.ashr %arg0, %0  : i256
    %2 = llvm.trunc %1 : i256 to i32
    %3 = llvm.mlir.constant(5 : i64) : i64
    %4 = llvm.alloca %3 x i8 : (i64) -> !llvm.ptr
    %5 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %5, %4 : !llvm.array<5 x i8>, !llvm.ptr
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.call @dprintf(%6, %4, %2) : (i32, !llvm.ptr, i32) -> i32
    %8 = llvm.mlir.constant(192 : i256) : i256
    %9 = llvm.ashr %arg0, %8  : i256
    %10 = llvm.trunc %9 : i256 to i32
    %11 = llvm.mlir.constant(5 : i64) : i64
    %12 = llvm.alloca %11 x i8 : (i64) -> !llvm.ptr
    %13 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %13, %12 : !llvm.array<5 x i8>, !llvm.ptr
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.call @dprintf(%14, %12, %10) : (i32, !llvm.ptr, i32) -> i32
    %16 = llvm.mlir.constant(160 : i256) : i256
    %17 = llvm.ashr %arg0, %16  : i256
    %18 = llvm.trunc %17 : i256 to i32
    %19 = llvm.mlir.constant(5 : i64) : i64
    %20 = llvm.alloca %19 x i8 : (i64) -> !llvm.ptr
    %21 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %21, %20 : !llvm.array<5 x i8>, !llvm.ptr
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = llvm.call @dprintf(%22, %20, %18) : (i32, !llvm.ptr, i32) -> i32
    %24 = llvm.mlir.constant(128 : i256) : i256
    %25 = llvm.ashr %arg0, %24  : i256
    %26 = llvm.trunc %25 : i256 to i32
    %27 = llvm.mlir.constant(5 : i64) : i64
    %28 = llvm.alloca %27 x i8 : (i64) -> !llvm.ptr
    %29 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %29, %28 : !llvm.array<5 x i8>, !llvm.ptr
    %30 = llvm.mlir.constant(1 : i32) : i32
    %31 = llvm.call @dprintf(%30, %28, %26) : (i32, !llvm.ptr, i32) -> i32
    %32 = llvm.mlir.constant(96 : i256) : i256
    %33 = llvm.ashr %arg0, %32  : i256
    %34 = llvm.trunc %33 : i256 to i32
    %35 = llvm.mlir.constant(5 : i64) : i64
    %36 = llvm.alloca %35 x i8 : (i64) -> !llvm.ptr
    %37 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %37, %36 : !llvm.array<5 x i8>, !llvm.ptr
    %38 = llvm.mlir.constant(1 : i32) : i32
    %39 = llvm.call @dprintf(%38, %36, %34) : (i32, !llvm.ptr, i32) -> i32
    %40 = llvm.mlir.constant(64 : i256) : i256
    %41 = llvm.ashr %arg0, %40  : i256
    %42 = llvm.trunc %41 : i256 to i32
    %43 = llvm.mlir.constant(5 : i64) : i64
    %44 = llvm.alloca %43 x i8 : (i64) -> !llvm.ptr
    %45 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %45, %44 : !llvm.array<5 x i8>, !llvm.ptr
    %46 = llvm.mlir.constant(1 : i32) : i32
    %47 = llvm.call @dprintf(%46, %44, %42) : (i32, !llvm.ptr, i32) -> i32
    %48 = llvm.mlir.constant(32 : i256) : i256
    %49 = llvm.ashr %arg0, %48  : i256
    %50 = llvm.trunc %49 : i256 to i32
    %51 = llvm.mlir.constant(5 : i64) : i64
    %52 = llvm.alloca %51 x i8 : (i64) -> !llvm.ptr
    %53 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %53, %52 : !llvm.array<5 x i8>, !llvm.ptr
    %54 = llvm.mlir.constant(1 : i32) : i32
    %55 = llvm.call @dprintf(%54, %52, %50) : (i32, !llvm.ptr, i32) -> i32
    %56 = llvm.mlir.constant(0 : i256) : i256
    %57 = llvm.trunc %arg0 : i256 to i32
    %58 = llvm.mlir.constant(5 : i64) : i64
    %59 = llvm.alloca %58 x i8 : (i64) -> !llvm.ptr
    %60 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %60, %59 : !llvm.array<5 x i8>, !llvm.ptr
    %61 = llvm.mlir.constant(1 : i32) : i32
    %62 = llvm.call @dprintf(%61, %59, %57) : (i32, !llvm.ptr, i32) -> i32
    %63 = llvm.mlir.constant(2 : i64) : i64
    %64 = llvm.alloca %63 x i8 : (i64) -> !llvm.ptr
    %65 = llvm.mlir.constant(dense<[10, 0]> : tensor<2xi8>) : !llvm.array<2 x i8>
    llvm.store %65, %64 : !llvm.array<2 x i8>, !llvm.ptr
    %66 = llvm.mlir.constant(1 : i32) : i32
    %67 = llvm.call @dprintf(%66, %64) : (i32, !llvm.ptr) -> i32
    llvm.return
  }
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"boolean::boolean::main"() : () -> i256
    llvm.call @print_felt252(%0) : (i256) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"boolean::boolean::main"() -> i256 attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    %1 = llvm.call @"enum_init<core::bool, 1>"(%0) : (!llvm.struct<()>) -> !llvm.struct<(i16, array<0 x i8>)>
    %2 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    %3 = llvm.call @"enum_init<core::bool, 0>"(%2) : (!llvm.struct<()>) -> !llvm.struct<(i16, array<0 x i8>)>
    %4 = llvm.call @bool_or_impl(%1, %3) : (!llvm.struct<(i16, array<0 x i8>)>, !llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)>
    %5 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    %6 = llvm.call @"enum_init<core::bool, 0>"(%5) : (!llvm.struct<()>) -> !llvm.struct<(i16, array<0 x i8>)>
    %7 = llvm.call @bool_not_impl(%6) : (!llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)>
    %8 = llvm.call @bool_not_impl(%7) : (!llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)>
    %9 = llvm.call @bool_and_impl(%4, %8) : (!llvm.struct<(i16, array<0 x i8>)>, !llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)>
    %10 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    %11 = llvm.call @"enum_init<core::bool, 0>"(%10) : (!llvm.struct<()>) -> !llvm.struct<(i16, array<0 x i8>)>
    %12 = llvm.call @bool_xor_impl(%9, %11) : (!llvm.struct<(i16, array<0 x i8>)>, !llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)>
    %13 = llvm.call @bool_to_felt252(%12) : (!llvm.struct<(i16, array<0 x i8>)>) -> i256
    llvm.return %13 : i256
  }
  llvm.func @"_mlir_ciface_boolean::boolean::main"() -> i256 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"boolean::boolean::main"() : () -> i256
    llvm.return %0 : i256
  }
}
