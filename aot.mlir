module attributes {llvm.data_layout = ""} {
  llvm.mlir.global private constant @assert_msg_5(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_4(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_3(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_2(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_1(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.func @abort()
  llvm.func @puts(!llvm.ptr)
  llvm.mlir.global private constant @assert_msg_0(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @"hello_starknet::hello_starknet::Echo::__wrapper__echo"(%arg0: !llvm.array<0 x i8>, %arg1: i128, %arg2: !llvm.ptr, %arg3: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) -> !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(256 : i64) : i64
    %1 = llvm.mlir.constant(485748461484230571791265682659113160264223489397539653310998840191492913 : i252) : i252
    %2 = llvm.mlir.constant(375233589013918064796019 : i252) : i252
    %3 = llvm.mlir.constant(0 : i128) : i128
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(32 : i64) : i64
    %7 = llvm.mlir.constant(8 : i32) : i32
    %8 = llvm.mlir.constant(7733229381460288120802334208475838166080759535023995805565484692595 : i252) : i252
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(false) : i1
    %11 = llvm.mlir.constant(4070 : i128) : i128
    %12 = llvm.mlir.constant(1 : i64) : i64
    %13 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %14 = llvm.alloca %12 x !llvm.struct<(i1, array<15 x i8>, i252, array<0 x i8>)> {alignment = 16 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>>
    %15 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %16 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %17 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %18 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %19 = llvm.icmp "uge" %arg1, %11 : i128
    %20 = llvm.call_intrinsic "llvm.usub.sat"(%arg1, %11) : (i128, i128) -> i128  {intrin = "llvm.usub.sat"}
    llvm.cond_br %19, ^bb1(%arg3 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>), ^bb5(%2, %13, %13, %20 : i252, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, i128)
  ^bb1(%21: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb0
    %22 = llvm.call @"core::Felt252Serde::deserialize"(%21) : (!llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) -> !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)>
    %23 = llvm.extractvalue %22[0] : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)> 
    %24 = llvm.extractvalue %22[1] : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)> 
    llvm.store %24, %14 {alignment = 16 : i64} : !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>>
    %25 = llvm.extractvalue %24[0] : !llvm.struct<(i1, array<15 x i8>, i252, array<0 x i8>)> 
    llvm.switch %25 : i1, ^bb2 [
      0: ^bb4,
      1: ^bb5(%1, %15, %15, %20 : i252, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, i128)
    ]
  ^bb2:  // pred: ^bb1
    llvm.cond_br %10, ^bb3, ^bb8
  ^bb3:  // pred: ^bb2
    llvm.unreachable
  ^bb4:  // pred: ^bb1
    %26 = llvm.bitcast %14 : !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<15 x i8>, i252)>>
    %27 = llvm.load %26 : !llvm.ptr<struct<(i1, array<15 x i8>, i252)>>
    %28 = llvm.extractvalue %27[2] : !llvm.struct<(i1, array<15 x i8>, i252)> 
    %29 = llvm.extractvalue %23[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)> 
    %30 = llvm.extractvalue %29[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %31 = llvm.icmp "eq" %30, %9 : i32
    llvm.cond_br %31, ^bb6, ^bb5(%8, %18, %18, %20 : i252, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, i128)
  ^bb5(%32: i252, %33: !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, %34: !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, %35: i128):  // 4 preds: ^bb0, ^bb1, ^bb4, ^bb6
    %36 = llvm.mlir.null : !llvm.ptr<i252>
    %37 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %38 = llvm.insertvalue %36, %37[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %39 = llvm.insertvalue %9, %38[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %40 = llvm.insertvalue %9, %39[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %41 = llvm.bitcast %36 : !llvm.ptr<i252> to !llvm.ptr
    %42 = llvm.call @realloc(%41, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %43 = llvm.bitcast %42 : !llvm.ptr to !llvm.ptr<i252>
    %44 = llvm.insertvalue %43, %40[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %45 = llvm.insertvalue %7, %44[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %46 = llvm.getelementptr %43[0] : (!llvm.ptr<i252>) -> !llvm.ptr, i252
    llvm.store %32, %46 : i252, !llvm.ptr
    %47 = llvm.insertvalue %5, %45[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %48 = llvm.mlir.undef : !llvm.struct<()>
    %49 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %50 = llvm.insertvalue %48, %49[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %51 = llvm.insertvalue %47, %50[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %52 = llvm.mlir.undef : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %53 = llvm.insertvalue %4, %52[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %54 = llvm.insertvalue %51, %53[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %55 = llvm.bitcast %33 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %54, %55 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %56 = llvm.load %34 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %57 = llvm.mlir.undef : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %58 = llvm.insertvalue %arg0, %57[0] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %59 = llvm.insertvalue %35, %58[1] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %60 = llvm.insertvalue %arg2, %59[2] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %61 = llvm.insertvalue %56, %60[3] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.return %61 : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
  ^bb6:  // pred: ^bb4
    %62 = llvm.icmp "uge" %20, %3 : i128
    %63 = llvm.call_intrinsic "llvm.usub.sat"(%20, %3) : (i128, i128) -> i128  {intrin = "llvm.usub.sat"}
    llvm.cond_br %62, ^bb7, ^bb5(%2, %16, %16, %63 : i252, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, i128)
  ^bb7:  // pred: ^bb6
    %64 = llvm.mlir.undef : !llvm.struct<()>
    %65 = llvm.mlir.undef : !llvm.struct<(struct<()>)>
    %66 = llvm.insertvalue %64, %65[0] : !llvm.struct<(struct<()>)> 
    %67 = llvm.call @"hello_starknet::hello_starknet::Echo::echo"(%66, %28) : (!llvm.struct<(struct<()>)>, i252) -> !llvm.struct<(struct<(struct<()>)>, i252)>
    %68 = llvm.extractvalue %67[0] : !llvm.struct<(struct<(struct<()>)>, i252)> 
    %69 = llvm.extractvalue %67[1] : !llvm.struct<(struct<(struct<()>)>, i252)> 
    %70 = llvm.mlir.null : !llvm.ptr<i252>
    %71 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %73 = llvm.insertvalue %9, %72[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %74 = llvm.insertvalue %9, %73[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %75 = llvm.call @"core::Felt252Serde::serialize"(%69, %74) : (i252, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>
    %76 = llvm.extractvalue %75[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %77 = llvm.extractvalue %75[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %78 = llvm.extractvalue %76[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %79 = llvm.extractvalue %76[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %80 = llvm.zext %79 : i32 to i64
    %81 = llvm.mul %80, %6  : i64
    %82 = llvm.mlir.null : !llvm.ptr
    %83 = llvm.call @realloc(%82, %81) : (!llvm.ptr, i64) -> !llvm.ptr
    %84 = llvm.bitcast %83 : !llvm.ptr to !llvm.ptr<i252>
    llvm.call_intrinsic "llvm.memcpy"(%84, %78, %81, %10) : (!llvm.ptr<i252>, !llvm.ptr<i252>, i64, i1) -> ()  {intrin = "llvm.memcpy"}
    %85 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %86 = llvm.insertvalue %84, %85[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %87 = llvm.insertvalue %79, %86[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %88 = llvm.insertvalue %79, %87[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %89 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>
    %90 = llvm.insertvalue %88, %89[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)> 
    %91 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>
    %92 = llvm.insertvalue %90, %91[0] : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)> 
    %93 = llvm.mlir.undef : !llvm.struct<(i1, array<7 x i8>, struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>)>
    %94 = llvm.insertvalue %10, %93[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>)> 
    %95 = llvm.insertvalue %92, %94[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>)> 
    %96 = llvm.bitcast %17 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>)>>
    llvm.store %95, %96 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>)>>
    %97 = llvm.load %17 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %98 = llvm.mlir.undef : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %99 = llvm.insertvalue %arg0, %98[0] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %100 = llvm.insertvalue %63, %99[1] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %101 = llvm.insertvalue %arg2, %100[2] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %102 = llvm.insertvalue %97, %101[3] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.return %102 : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
  ^bb8:  // pred: ^bb2
    %103 = llvm.mlir.addressof @assert_msg_0 : !llvm.ptr
    llvm.call @puts(%103) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_hello_starknet::hello_starknet::Echo::__wrapper__echo"(%arg0: !llvm.ptr, %arg1: !llvm.array<0 x i8>, %arg2: i128, %arg3: !llvm.ptr, %arg4: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"hello_starknet::hello_starknet::Echo::__wrapper__echo"(%arg1, %arg2, %arg3, %arg4) : (!llvm.array<0 x i8>, i128, !llvm.ptr, !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) -> !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"hello_starknet::hello_starknet::Echo::__wrapper__constructor"(%arg0: !llvm.array<0 x i8>, %arg1: i128, %arg2: !llvm.ptr, %arg3: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) -> !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(256 : i64) : i64
    %2 = llvm.mlir.constant(485748461484230571791265682659113160264223489397539653310998840191492913 : i252) : i252
    %3 = llvm.mlir.constant(375233589013918064796019 : i252) : i252
    %4 = llvm.mlir.constant(0 : i128) : i128
    %5 = llvm.mlir.constant(true) : i1
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(8 : i32) : i32
    %8 = llvm.mlir.constant(7733229381460288120802334208475838166080759535023995805565484692595 : i252) : i252
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(false) : i1
    %11 = llvm.mlir.constant(17470 : i128) : i128
    %12 = llvm.mlir.constant(1 : i64) : i64
    %13 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %14 = llvm.alloca %12 x !llvm.struct<(i1, array<15 x i8>, i252, array<0 x i8>)> {alignment = 16 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>>
    %15 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %16 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %17 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %18 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %19 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %20 = llvm.alloca %12 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %21 = llvm.icmp "uge" %arg1, %11 : i128
    %22 = llvm.call_intrinsic "llvm.usub.sat"(%arg1, %11) : (i128, i128) -> i128  {intrin = "llvm.usub.sat"}
    llvm.cond_br %21, ^bb1(%arg3 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>), ^bb5(%3, %13, %13, %22 : i252, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, i128)
  ^bb1(%23: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb0
    %24 = llvm.call @"core::Felt252Serde::deserialize"(%23) : (!llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) -> !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)>
    %25 = llvm.extractvalue %24[0] : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)> 
    %26 = llvm.extractvalue %24[1] : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)> 
    llvm.store %26, %14 {alignment = 16 : i64} : !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>>
    %27 = llvm.extractvalue %26[0] : !llvm.struct<(i1, array<15 x i8>, i252, array<0 x i8>)> 
    llvm.switch %27 : i1, ^bb2 [
      0: ^bb4,
      1: ^bb5(%2, %15, %15, %22 : i252, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, i128)
    ]
  ^bb2:  // 2 preds: ^bb1, ^bb7
    llvm.cond_br %10, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.unreachable
  ^bb4:  // pred: ^bb1
    %28 = llvm.bitcast %14 : !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<15 x i8>, i252)>>
    %29 = llvm.load %28 : !llvm.ptr<struct<(i1, array<15 x i8>, i252)>>
    %30 = llvm.extractvalue %29[2] : !llvm.struct<(i1, array<15 x i8>, i252)> 
    %31 = llvm.extractvalue %25[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)> 
    %32 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %33 = llvm.icmp "eq" %32, %9 : i32
    llvm.cond_br %33, ^bb6, ^bb5(%8, %20, %20, %22 : i252, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, i128)
  ^bb5(%34: i252, %35: !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, %36: !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, %37: i128):  // 4 preds: ^bb0, ^bb1, ^bb4, ^bb6
    %38 = llvm.mlir.null : !llvm.ptr<i252>
    %39 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %41 = llvm.insertvalue %9, %40[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %42 = llvm.insertvalue %9, %41[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %43 = llvm.bitcast %38 : !llvm.ptr<i252> to !llvm.ptr
    %44 = llvm.call @realloc(%43, %1) : (!llvm.ptr, i64) -> !llvm.ptr
    %45 = llvm.bitcast %44 : !llvm.ptr to !llvm.ptr<i252>
    %46 = llvm.insertvalue %45, %42[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %47 = llvm.insertvalue %7, %46[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %48 = llvm.getelementptr %45[0] : (!llvm.ptr<i252>) -> !llvm.ptr, i252
    llvm.store %34, %48 : i252, !llvm.ptr
    %49 = llvm.insertvalue %6, %47[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %50 = llvm.mlir.undef : !llvm.struct<()>
    %51 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %52 = llvm.insertvalue %50, %51[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %53 = llvm.insertvalue %49, %52[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %54 = llvm.mlir.undef : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %55 = llvm.insertvalue %5, %54[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %56 = llvm.insertvalue %53, %55[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %57 = llvm.bitcast %35 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %56, %57 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %58 = llvm.load %36 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %59 = llvm.mlir.undef : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %60 = llvm.insertvalue %arg0, %59[0] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %61 = llvm.insertvalue %37, %60[1] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %62 = llvm.insertvalue %arg2, %61[2] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %63 = llvm.insertvalue %58, %62[3] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.return %63 : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
  ^bb6:  // pred: ^bb4
    %64 = llvm.icmp "uge" %22, %4 : i128
    %65 = llvm.call_intrinsic "llvm.usub.sat"(%22, %4) : (i128, i128) -> i128  {intrin = "llvm.usub.sat"}
    llvm.cond_br %64, ^bb7, ^bb5(%3, %16, %16, %65 : i252, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>, i128)
  ^bb7:  // pred: ^bb6
    %66 = llvm.mlir.undef : !llvm.struct<()>
    %67 = llvm.mlir.undef : !llvm.struct<(struct<()>)>
    %68 = llvm.insertvalue %66, %67[0] : !llvm.struct<(struct<()>)> 
    %69 = llvm.call @"hello_starknet::hello_starknet::Echo::constructor"(%65, %arg2, %68, %30) : (i128, !llvm.ptr, !llvm.struct<(struct<()>)>, i252) -> !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %70 = llvm.extractvalue %69[0] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %71 = llvm.extractvalue %69[1] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %72 = llvm.extractvalue %69[2] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.store %72, %17 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %73 = llvm.extractvalue %72[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> 
    llvm.switch %73 : i1, ^bb2 [
      0: ^bb8,
      1: ^bb9
    ]
  ^bb8:  // pred: ^bb7
    %74 = llvm.mlir.null : !llvm.ptr<i252>
    %75 = llvm.mlir.null : !llvm.ptr
    %76 = llvm.call @realloc(%75, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %77 = llvm.bitcast %76 : !llvm.ptr to !llvm.ptr<i252>
    llvm.call_intrinsic "llvm.memcpy"(%77, %74, %0, %10) : (!llvm.ptr<i252>, !llvm.ptr<i252>, i64, i1) -> ()  {intrin = "llvm.memcpy"}
    %78 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %79 = llvm.insertvalue %77, %78[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %80 = llvm.insertvalue %9, %79[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %81 = llvm.insertvalue %9, %80[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %82 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>
    %83 = llvm.insertvalue %81, %82[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)> 
    %84 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>
    %85 = llvm.insertvalue %83, %84[0] : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)> 
    %86 = llvm.mlir.undef : !llvm.struct<(i1, array<7 x i8>, struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>)>
    %87 = llvm.insertvalue %10, %86[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>)> 
    %88 = llvm.insertvalue %85, %87[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>)> 
    %89 = llvm.bitcast %19 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>)>>
    llvm.store %88, %89 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<(struct<(ptr<i252>, i32, i32)>)>)>)>>
    %90 = llvm.load %19 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %91 = llvm.mlir.undef : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %92 = llvm.insertvalue %arg0, %91[0] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %93 = llvm.insertvalue %70, %92[1] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %94 = llvm.insertvalue %71, %93[2] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %95 = llvm.insertvalue %90, %94[3] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.return %95 : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
  ^bb9:  // pred: ^bb7
    %96 = llvm.bitcast %17 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %97 = llvm.load %96 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %98 = llvm.extractvalue %97[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %99 = llvm.mlir.undef : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %100 = llvm.insertvalue %5, %99[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %101 = llvm.insertvalue %98, %100[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %102 = llvm.bitcast %18 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %101, %102 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %103 = llvm.load %18 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %104 = llvm.mlir.undef : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %105 = llvm.insertvalue %arg0, %104[0] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %106 = llvm.insertvalue %70, %105[1] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %107 = llvm.insertvalue %71, %106[2] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %108 = llvm.insertvalue %103, %107[3] : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.return %108 : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
  ^bb10:  // pred: ^bb2
    %109 = llvm.mlir.addressof @assert_msg_1 : !llvm.ptr
    llvm.call @puts(%109) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_hello_starknet::hello_starknet::Echo::__wrapper__constructor"(%arg0: !llvm.ptr, %arg1: !llvm.array<0 x i8>, %arg2: i128, %arg3: !llvm.ptr, %arg4: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"hello_starknet::hello_starknet::Echo::__wrapper__constructor"(%arg1, %arg2, %arg3, %arg4) : (!llvm.array<0 x i8>, i128, !llvm.ptr, !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) -> !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::Felt252Serde::deserialize"(%arg0: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) -> !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(32 : i64) : i64
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x !llvm.struct<(i1, array<7 x i8>, ptr, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>>
    %7 = llvm.alloca %5 x !llvm.struct<(i1, array<7 x i8>, ptr, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>>
    %8 = llvm.alloca %5 x !llvm.struct<(i1, array<15 x i8>, i252, array<0 x i8>)> {alignment = 16 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>>
    %9 = llvm.alloca %5 x !llvm.struct<(i1, array<15 x i8>, i252, array<0 x i8>)> {alignment = 16 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>>
    %10 = llvm.alloca %5 x !llvm.struct<(i1, array<7 x i8>, ptr, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>>
    %11 = llvm.extractvalue %arg0[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)> 
    %12 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %13 = llvm.icmp "eq" %12, %4 : i32
    llvm.cond_br %13, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %14 = llvm.extractvalue %11[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %15 = llvm.getelementptr %14[0] : (!llvm.ptr<i252>) -> !llvm.ptr, i252
    %16 = llvm.mlir.null : !llvm.ptr
    %17 = llvm.call @realloc(%16, %3) : (!llvm.ptr, i64) -> !llvm.ptr
    %18 = llvm.load %15 {alignment = 16 : i64} : !llvm.ptr -> i252
    llvm.store %18, %17 {alignment = 16 : i64} : i252, !llvm.ptr
    %19 = llvm.getelementptr %14[1] : (!llvm.ptr<i252>) -> !llvm.ptr, i252
    %20 = llvm.sub %12, %2  : i32
    %21 = llvm.zext %20 : i32 to i64
    %22 = llvm.mul %21, %3  : i64
    %23 = llvm.bitcast %14 : !llvm.ptr<i252> to !llvm.ptr
    "llvm.intr.memmove"(%23, %19, %22) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %24 = llvm.insertvalue %20, %11[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %25 = llvm.mlir.undef : !llvm.struct<(i1, array<7 x i8>, ptr)>
    %26 = llvm.insertvalue %1, %25[0] : !llvm.struct<(i1, array<7 x i8>, ptr)> 
    %27 = llvm.insertvalue %17, %26[2] : !llvm.struct<(i1, array<7 x i8>, ptr)> 
    %28 = llvm.bitcast %10 : !llvm.ptr<struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, ptr)>>
    llvm.store %27, %28 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, ptr)>>
    %29 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>>
    llvm.br ^bb3(%24, %29 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>)
  ^bb2:  // pred: ^bb0
    %30 = llvm.mlir.undef : !llvm.struct<()>
    %31 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>, struct<()>)>
    %32 = llvm.insertvalue %0, %31[0] : !llvm.struct<(i1, array<0 x i8>, struct<()>)> 
    %33 = llvm.insertvalue %30, %32[2] : !llvm.struct<(i1, array<0 x i8>, struct<()>)> 
    %34 = llvm.bitcast %6 : !llvm.ptr<struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>>
    llvm.store %33, %34 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>>
    %35 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>>
    llvm.br ^bb3(%11, %35 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>)
  ^bb3(%36: !llvm.struct<(ptr<i252>, i32, i32)>, %37: !llvm.struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>):  // 2 preds: ^bb1, ^bb2
    %38 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>
    %39 = llvm.insertvalue %36, %38[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>)> 
    llvm.store %37, %7 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>>
    %40 = llvm.extractvalue %37[0] : !llvm.struct<(i1, array<7 x i8>, ptr, array<0 x i8>)> 
    llvm.switch %40 : i1, ^bb4 [
      0: ^bb6,
      1: ^bb7
    ]
  ^bb4:  // pred: ^bb3
    llvm.cond_br %1, ^bb5, ^bb8
  ^bb5:  // pred: ^bb4
    llvm.unreachable
  ^bb6:  // pred: ^bb3
    %41 = llvm.bitcast %7 : !llvm.ptr<struct<(i1, array<7 x i8>, ptr, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, ptr)>>
    %42 = llvm.load %41 : !llvm.ptr<struct<(i1, array<7 x i8>, ptr)>>
    %43 = llvm.extractvalue %42[2] : !llvm.struct<(i1, array<7 x i8>, ptr)> 
    %44 = llvm.load %43 {alignment = 16 : i64} : !llvm.ptr -> i252
    %45 = llvm.mlir.undef : !llvm.struct<(i1, array<15 x i8>, i252)>
    %46 = llvm.insertvalue %1, %45[0] : !llvm.struct<(i1, array<15 x i8>, i252)> 
    %47 = llvm.insertvalue %44, %46[2] : !llvm.struct<(i1, array<15 x i8>, i252)> 
    %48 = llvm.bitcast %9 : !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<15 x i8>, i252)>>
    llvm.store %47, %48 {alignment = 16 : i64} : !llvm.ptr<struct<(i1, array<15 x i8>, i252)>>
    %49 = llvm.load %9 {alignment = 16 : i64} : !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>>
    %50 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)>
    %51 = llvm.insertvalue %39, %50[0] : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)> 
    %52 = llvm.insertvalue %49, %51[1] : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)> 
    llvm.return %52 : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)>
  ^bb7:  // pred: ^bb3
    %53 = llvm.mlir.undef : !llvm.struct<()>
    %54 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>, struct<()>)>
    %55 = llvm.insertvalue %0, %54[0] : !llvm.struct<(i1, array<0 x i8>, struct<()>)> 
    %56 = llvm.insertvalue %53, %55[2] : !llvm.struct<(i1, array<0 x i8>, struct<()>)> 
    %57 = llvm.bitcast %8 : !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>>
    llvm.store %56, %57 {alignment = 16 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>>
    %58 = llvm.load %8 {alignment = 16 : i64} : !llvm.ptr<struct<(i1, array<15 x i8>, i252, array<0 x i8>)>>
    %59 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)>
    %60 = llvm.insertvalue %39, %59[0] : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)> 
    %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)> 
    llvm.return %61 : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)>
  ^bb8:  // pred: ^bb4
    %62 = llvm.mlir.addressof @assert_msg_2 : !llvm.ptr
    llvm.call @puts(%62) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_core::Felt252Serde::deserialize"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::Felt252Serde::deserialize"(%arg1) : (!llvm.struct<(struct<(ptr<i252>, i32, i32)>)>) -> !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(struct<(struct<(ptr<i252>, i32, i32)>)>, struct<(i1, array<15 x i8>, i252, array<0 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"hello_starknet::hello_starknet::Echo::echo"(%arg0: !llvm.struct<(struct<()>)>, %arg1: i252) -> !llvm.struct<(struct<(struct<()>)>, i252)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.undef : !llvm.struct<(struct<(struct<()>)>, i252)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(struct<(struct<()>)>, i252)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(struct<(struct<()>)>, i252)> 
    llvm.return %2 : !llvm.struct<(struct<(struct<()>)>, i252)>
  }
  llvm.func @"_mlir_ciface_hello_starknet::hello_starknet::Echo::echo"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(struct<()>)>, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"hello_starknet::hello_starknet::Echo::echo"(%arg1, %arg2) : (!llvm.struct<(struct<()>)>, i252) -> !llvm.struct<(struct<(struct<()>)>, i252)>
    llvm.store %0, %arg0 : !llvm.struct<(struct<(struct<()>)>, i252)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::Felt252Serde::serialize"(%arg0: i252, %arg1: !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(32 : i64) : i64
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.extractvalue %arg1[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %4 = llvm.extractvalue %arg1[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %5 = llvm.extractvalue %arg1[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %6 = llvm.icmp "uge" %4, %5 : i32
    llvm.cond_br %6, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %7 = llvm.add %5, %5  : i32
    %8 = llvm.intr.umax(%7, %2)  : (i32, i32) -> i32
    %9 = llvm.zext %8 : i32 to i64
    %10 = llvm.mul %9, %1  : i64
    %11 = llvm.bitcast %3 : !llvm.ptr<i252> to !llvm.ptr
    %12 = llvm.call @realloc(%11, %10) : (!llvm.ptr, i64) -> !llvm.ptr
    %13 = llvm.bitcast %12 : !llvm.ptr to !llvm.ptr<i252>
    %14 = llvm.insertvalue %13, %arg1[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %15 = llvm.insertvalue %8, %14[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    llvm.br ^bb3(%15, %13 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%arg1, %3 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>)
  ^bb3(%16: !llvm.struct<(ptr<i252>, i32, i32)>, %17: !llvm.ptr<i252>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %18 = llvm.getelementptr %17[%4] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %arg0, %18 : i252, !llvm.ptr
    %19 = llvm.add %4, %0  : i32
    %20 = llvm.insertvalue %19, %16[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %21 = llvm.mlir.undef : !llvm.struct<()>
    %22 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>
    %23 = llvm.insertvalue %20, %22[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %24 = llvm.insertvalue %21, %23[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    llvm.return %24 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>
  }
  llvm.func @"_mlir_ciface_core::Felt252Serde::serialize"(%arg0: !llvm.ptr, %arg1: i252, %arg2: !llvm.struct<(ptr<i252>, i32, i32)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::Felt252Serde::serialize"(%arg1, %arg2) : (i252, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>
    llvm.store %0, %arg0 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"hello_starknet::hello_starknet::Echo::constructor"(%arg0: i128, %arg1: !llvm.ptr, %arg2: !llvm.struct<(struct<()>)>, %arg3: i252) -> !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %4 = llvm.alloca %2 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %5 = llvm.alloca %2 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %6 = llvm.extractvalue %arg2[0] : !llvm.struct<(struct<()>)> 
    %7 = llvm.call @"hello_starknet::hello_starknet::Echo::balance::InternalContractMemberStateImpl::write"(%arg0, %arg1, %6, %arg3) : (i128, !llvm.ptr, !llvm.struct<()>, i252) -> !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %8 = llvm.extractvalue %7[0] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %9 = llvm.extractvalue %7[1] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %10 = llvm.extractvalue %7[2] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.store %10, %3 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> 
    llvm.switch %11 : i1, ^bb1 [
      0: ^bb3,
      1: ^bb4
    ]
  ^bb1:  // pred: ^bb0
    llvm.cond_br %1, ^bb2, ^bb5
  ^bb2:  // pred: ^bb1
    llvm.unreachable
  ^bb3:  // pred: ^bb0
    %12 = llvm.bitcast %3 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<(struct<()>, struct<()>)>)>>
    %13 = llvm.load %12 : !llvm.ptr<struct<(i1, array<0 x i8>, struct<(struct<()>, struct<()>)>)>>
    %14 = llvm.extractvalue %13[2] : !llvm.struct<(i1, array<0 x i8>, struct<(struct<()>, struct<()>)>)> 
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(struct<()>, struct<()>)> 
    %16 = llvm.mlir.undef : !llvm.struct<()>
    %17 = llvm.mlir.undef : !llvm.struct<(struct<()>)>
    %18 = llvm.insertvalue %15, %17[0] : !llvm.struct<(struct<()>)> 
    %19 = llvm.mlir.undef : !llvm.struct<(struct<(struct<()>)>, struct<()>)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(struct<(struct<()>)>, struct<()>)> 
    %21 = llvm.insertvalue %16, %20[1] : !llvm.struct<(struct<(struct<()>)>, struct<()>)> 
    %22 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>, struct<(struct<(struct<()>)>, struct<()>)>)>
    %23 = llvm.insertvalue %1, %22[0] : !llvm.struct<(i1, array<0 x i8>, struct<(struct<(struct<()>)>, struct<()>)>)> 
    %24 = llvm.insertvalue %21, %23[2] : !llvm.struct<(i1, array<0 x i8>, struct<(struct<(struct<()>)>, struct<()>)>)> 
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<(struct<(struct<()>)>, struct<()>)>)>>
    llvm.store %24, %25 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<(struct<(struct<()>)>, struct<()>)>)>>
    %26 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %27 = llvm.mlir.undef : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %28 = llvm.insertvalue %8, %27[0] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %29 = llvm.insertvalue %9, %28[1] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %30 = llvm.insertvalue %26, %29[2] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.return %30 : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
  ^bb4:  // pred: ^bb0
    %31 = llvm.bitcast %3 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %32 = llvm.load %31 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %33 = llvm.extractvalue %32[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %34 = llvm.mlir.undef : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %35 = llvm.insertvalue %0, %34[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %36 = llvm.insertvalue %33, %35[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %37 = llvm.bitcast %4 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %36, %37 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %38 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %39 = llvm.mlir.undef : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %40 = llvm.insertvalue %8, %39[0] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %41 = llvm.insertvalue %9, %40[1] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %42 = llvm.insertvalue %38, %41[2] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.return %42 : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
  ^bb5:  // pred: ^bb1
    %43 = llvm.mlir.addressof @assert_msg_3 : !llvm.ptr
    llvm.call @puts(%43) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_hello_starknet::hello_starknet::Echo::constructor"(%arg0: !llvm.ptr, %arg1: i128, %arg2: !llvm.ptr, %arg3: !llvm.struct<(struct<()>)>, %arg4: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"hello_starknet::hello_starknet::Echo::constructor"(%arg1, %arg2, %arg3, %arg4) : (i128, !llvm.ptr, !llvm.struct<(struct<()>)>, i252) -> !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"hello_starknet::hello_starknet::Echo::balance::InternalContractMemberStateImpl::write"(%arg0: i128, %arg1: !llvm.ptr, %arg2: !llvm.struct<()>, %arg3: i252) -> !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(916907772491729262376534102982219947830828984996257231353398618781993312401 : i252) : i252
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.struct<(i1, array<23 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %6 = llvm.alloca %4 x i128 {alignment = 8 : i64} : (i64) -> !llvm.ptr<i128>
    %7 = llvm.alloca %4 x i252 {alignment = 16 : i64} : (i64) -> !llvm.ptr<i252>
    %8 = llvm.alloca %4 x i252 {alignment = 16 : i64} : (i64) -> !llvm.ptr<i252>
    %9 = llvm.alloca %4 x !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>>
    %10 = llvm.alloca %4 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %11 = llvm.alloca %4 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %12 = llvm.alloca %4 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %13 = llvm.alloca %4 x !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>>
    %14 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
    "llvm.intr.debugtrap"() : () -> ()
    llvm.store %arg0, %6 : !llvm.ptr<i128>
    llvm.store %3, %7 : !llvm.ptr<i252>
    llvm.store %arg3, %8 : !llvm.ptr<i252>
    %15 = llvm.getelementptr %arg1[8] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %16 = llvm.load %15 : !llvm.ptr -> !llvm.ptr<func<void (ptr, ptr, ptr<i128>, i32, ptr<i252>, ptr<i252>)>>
    "llvm.intr.debugtrap"() : () -> ()
    llvm.call %16(%5, %14, %6, %2, %7, %8) : !llvm.ptr<func<void (ptr, ptr, ptr<i128>, i32, ptr<i252>, ptr<i252>)>>, (!llvm.ptr, !llvm.ptr, !llvm.ptr<i128>, i32, !llvm.ptr<i252>, !llvm.ptr<i252>) -> ()
    "llvm.intr.debugtrap"() : () -> ()
    %17 = llvm.load %5 : !llvm.ptr -> !llvm.struct<(i1, array<23 x i8>)>
    %18 = llvm.extractvalue %17[0] : !llvm.struct<(i1, array<23 x i8>)> 
    %19 = llvm.getelementptr %5[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %20 = llvm.load %19 : !llvm.ptr -> !llvm.struct<(ptr<i252>, i32, i32)>
    %21 = llvm.load %6 : !llvm.ptr<i128>
    llvm.cond_br %18, ^bb2(%20 : !llvm.struct<(ptr<i252>, i32, i32)>), ^bb1
  ^bb1:  // pred: ^bb0
    %22 = llvm.mlir.undef : !llvm.struct<()>
    %23 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>, struct<()>)>
    %24 = llvm.insertvalue %1, %23[0] : !llvm.struct<(i1, array<0 x i8>, struct<()>)> 
    %25 = llvm.insertvalue %22, %24[2] : !llvm.struct<(i1, array<0 x i8>, struct<()>)> 
    %26 = llvm.bitcast %13 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>>
    llvm.store %25, %26 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>>
    %27 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>>
    llvm.br ^bb3(%arg2, %21, %arg1, %27 : !llvm.struct<()>, i128, !llvm.ptr, !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>)
  ^bb2(%28: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb0
    %29 = llvm.mlir.undef : !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>)>
    %30 = llvm.insertvalue %0, %29[0] : !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>)> 
    %31 = llvm.insertvalue %28, %30[2] : !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>)> 
    %32 = llvm.bitcast %9 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>)>>
    llvm.store %31, %32 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>)>>
    %33 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>>
    llvm.br ^bb3(%arg2, %21, %arg1, %33 : !llvm.struct<()>, i128, !llvm.ptr, !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>)
  ^bb3(%34: !llvm.struct<()>, %35: i128, %36: !llvm.ptr, %37: !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>):  // 2 preds: ^bb1, ^bb2
    %38 = llvm.call @"core::starknet::SyscallResultTraitImpl::<()>::unwrap_syscall"(%37) : (!llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>) -> !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>
    llvm.store %38, %10 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %39 = llvm.extractvalue %38[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> 
    llvm.switch %39 : i1, ^bb4 [
      0: ^bb6,
      1: ^bb7
    ]
  ^bb4:  // pred: ^bb3
    llvm.cond_br %1, ^bb5, ^bb8
  ^bb5:  // pred: ^bb4
    llvm.unreachable
  ^bb6:  // pred: ^bb3
    %40 = llvm.bitcast %10 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<(struct<()>)>)>>
    %41 = llvm.load %40 : !llvm.ptr<struct<(i1, array<0 x i8>, struct<(struct<()>)>)>>
    %42 = llvm.extractvalue %41[2] : !llvm.struct<(i1, array<0 x i8>, struct<(struct<()>)>)> 
    %43 = llvm.extractvalue %42[0] : !llvm.struct<(struct<()>)> 
    %44 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<()>)>
    %45 = llvm.insertvalue %34, %44[0] : !llvm.struct<(struct<()>, struct<()>)> 
    %46 = llvm.insertvalue %43, %45[1] : !llvm.struct<(struct<()>, struct<()>)> 
    %47 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>, struct<(struct<()>, struct<()>)>)>
    %48 = llvm.insertvalue %1, %47[0] : !llvm.struct<(i1, array<0 x i8>, struct<(struct<()>, struct<()>)>)> 
    %49 = llvm.insertvalue %46, %48[2] : !llvm.struct<(i1, array<0 x i8>, struct<(struct<()>, struct<()>)>)> 
    %50 = llvm.bitcast %12 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<(struct<()>, struct<()>)>)>>
    llvm.store %49, %50 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<(struct<()>, struct<()>)>)>>
    %51 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %52 = llvm.mlir.undef : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %53 = llvm.insertvalue %35, %52[0] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %54 = llvm.insertvalue %36, %53[1] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %55 = llvm.insertvalue %51, %54[2] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.return %55 : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
  ^bb7:  // pred: ^bb3
    %56 = llvm.bitcast %10 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %57 = llvm.load %56 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %58 = llvm.extractvalue %57[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %59 = llvm.mlir.undef : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %60 = llvm.insertvalue %0, %59[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %61 = llvm.insertvalue %58, %60[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %62 = llvm.bitcast %11 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %61, %62 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %63 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %64 = llvm.mlir.undef : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    %65 = llvm.insertvalue %35, %64[0] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %66 = llvm.insertvalue %36, %65[1] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    %67 = llvm.insertvalue %63, %66[2] : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)> 
    llvm.return %67 : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
  ^bb8:  // pred: ^bb4
    %68 = llvm.mlir.addressof @assert_msg_4 : !llvm.ptr
    llvm.call @puts(%68) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_hello_starknet::hello_starknet::Echo::balance::InternalContractMemberStateImpl::write"(%arg0: !llvm.ptr, %arg1: i128, %arg2: !llvm.ptr, %arg3: !llvm.struct<()>, %arg4: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"hello_starknet::hello_starknet::Echo::balance::InternalContractMemberStateImpl::write"(%arg1, %arg2, %arg3, %arg4) : (i128, !llvm.ptr, !llvm.struct<()>, i252) -> !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::starknet::SyscallResultTraitImpl::<()>::unwrap_syscall"(%arg0: !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>) -> !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>>
    %4 = llvm.alloca %2 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    %5 = llvm.alloca %2 x !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    llvm.store %arg0, %3 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>>
    %6 = llvm.extractvalue %arg0[0] : !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)> 
    llvm.switch %6 : i1, ^bb1 [
      0: ^bb3,
      1: ^bb4
    ]
  ^bb1:  // pred: ^bb0
    llvm.cond_br %1, ^bb2, ^bb5
  ^bb2:  // pred: ^bb1
    llvm.unreachable
  ^bb3:  // pred: ^bb0
    %7 = llvm.bitcast %3 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>>
    %8 = llvm.load %7 : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>>
    %9 = llvm.extractvalue %8[2] : !llvm.struct<(i1, array<0 x i8>, struct<()>)> 
    %10 = llvm.mlir.undef : !llvm.struct<(struct<()>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(struct<()>)> 
    %12 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>, struct<(struct<()>)>)>
    %13 = llvm.insertvalue %1, %12[0] : !llvm.struct<(i1, array<0 x i8>, struct<(struct<()>)>)> 
    %14 = llvm.insertvalue %11, %13[2] : !llvm.struct<(i1, array<0 x i8>, struct<(struct<()>)>)> 
    %15 = llvm.bitcast %5 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<(struct<()>)>)>>
    llvm.store %14, %15 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<(struct<()>)>)>>
    %16 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    llvm.return %16 : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>
  ^bb4:  // pred: ^bb0
    %17 = llvm.bitcast %3 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>)>>
    %18 = llvm.load %17 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>)>>
    %19 = llvm.extractvalue %18[2] : !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>)> 
    %20 = llvm.mlir.undef : !llvm.struct<()>
    %21 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %23 = llvm.insertvalue %19, %22[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %24 = llvm.mlir.undef : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %25 = llvm.insertvalue %0, %24[0] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %26 = llvm.insertvalue %23, %25[2] : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %27 = llvm.bitcast %4 : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %26, %27 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %28 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>>
    llvm.return %28 : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>
  ^bb5:  // pred: ^bb1
    %29 = llvm.mlir.addressof @assert_msg_5 : !llvm.ptr
    llvm.call @puts(%29) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_core::starknet::SyscallResultTraitImpl::<()>::unwrap_syscall"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::starknet::SyscallResultTraitImpl::<()>::unwrap_syscall"(%arg1) : (!llvm.struct<(i1, array<7 x i8>, struct<(ptr<i252>, i32, i32)>, array<0 x i8>)>) -> !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>
    llvm.store %0, %arg0 : !llvm.struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>, !llvm.ptr
    llvm.return
  }
}
