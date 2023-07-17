#loc = loc(unknown)
module {
  func.func private @cairo_native__alloc_dict() -> !llvm.ptr loc(#loc)
  func.func private @realloc(!llvm.ptr, i64) -> !llvm.ptr loc(#loc)
  func.func private @cairo_native__dict_get(!llvm.ptr, !llvm.ptr<i252>) -> !llvm.ptr loc(#loc)
  func.func private @cairo_native__dict_insert(!llvm.ptr, !llvm.ptr<i252>, !llvm.ptr) -> !llvm.ptr loc(#loc)
  func.func public @"ex_dict::ex_dict::main"(%arg0: !llvm.array<0 x i8> loc(unknown), %arg1: !llvm.array<0 x i8> loc(unknown), %arg2: i64 loc(unknown)) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, i32) attributes {llvm.emit_c_interface} {
    cf.br ^bb1(%arg0, %arg1, %arg2 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64) loc(#loc)
  ^bb1(%0: !llvm.array<0 x i8> loc(unknown), %1: !llvm.array<0 x i8> loc(unknown), %2: i64 loc(unknown)):  // pred: ^bb0
    cf.br ^bb2 loc(#loc)
  ^bb2:  // pred: ^bb1
    %3 = llvm.mlir.undef : i32 loc(#loc1)
    cf.br ^bb3 loc(#loc1)
  ^bb3:  // pred: ^bb2
    cf.br ^bb4 loc(#loc2)
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%1 : !llvm.array<0 x i8>) loc(#loc3)
  ^bb5(%4: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb4
    %5 = call @cairo_native__alloc_dict() : () -> !llvm.ptr loc(#loc)
    cf.br ^bb6 loc(#loc)
  ^bb6:  // pred: ^bb5
    %c2_i252 = arith.constant 2 : i252 loc(#loc)
    cf.br ^bb7(%5 : !llvm.ptr) loc(#loc)
  ^bb7(%6: !llvm.ptr loc(unknown)):  // pred: ^bb6
    cf.br ^bb8(%c2_i252 : i252) loc(#loc4)
  ^bb8(%7: i252 loc(unknown)):  // pred: ^bb7
    cf.br ^bb9(%6, %7 : !llvm.ptr, i252) loc(#loc5)
  ^bb9(%8: !llvm.ptr loc(unknown), %9: i252 loc(unknown)):  // pred: ^bb8
    %10 = llvm.alloca %9 x i252 {alignment = 8 : i64} : (i252) -> !llvm.ptr<i252> loc(#loc)
    llvm.store %9, %10 {alignment = 8 : i64} : !llvm.ptr<i252> loc(#loc)
    %11 = call @cairo_native__dict_get(%8, %10) : (!llvm.ptr, !llvm.ptr<i252>) -> !llvm.ptr loc(#loc)
    %12 = llvm.mlir.null : !llvm.ptr loc(#loc)
    %13 = llvm.icmp "eq" %11, %12 : !llvm.ptr loc(#loc)
    cf.cond_br %13, ^bb10, ^bb11 loc(#loc)
  ^bb10:  // pred: ^bb9
    %c4_i64 = arith.constant 4 : i64 loc(#loc)
    %14 = call @realloc(%11, %c4_i64) : (!llvm.ptr, i64) -> !llvm.ptr loc(#loc)
    %15 = llvm.mlir.undef : i32 loc(#loc)
    cf.br ^bb12(%14, %15 : !llvm.ptr, i32) loc(#loc)
  ^bb11:  // pred: ^bb9
    %16 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32 loc(#loc)
    cf.br ^bb12(%11, %16 : !llvm.ptr, i32) loc(#loc)
  ^bb12(%17: !llvm.ptr loc(unknown), %18: i32 loc(unknown)):  // 2 preds: ^bb10, ^bb11
    %19 = llvm.mlir.undef : !llvm.struct<(i252, ptr, ptr)> loc(#loc)
    %20 = llvm.insertvalue %9, %19[0] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    %21 = llvm.insertvalue %17, %20[1] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    %22 = llvm.insertvalue %8, %21[2] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    cf.br ^bb13(%18 : i32) loc(#loc)
  ^bb13(%23: i32 loc(unknown)):  // pred: ^bb12
    cf.br ^bb14 loc(#loc)
  ^bb14:  // pred: ^bb13
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    cf.br ^bb15(%c1_i32 : i32) loc(#loc)
  ^bb15(%24: i32 loc(unknown)):  // pred: ^bb14
    cf.br ^bb16(%22, %24 : !llvm.struct<(i252, ptr, ptr)>, i32) loc(#loc6)
  ^bb16(%25: !llvm.struct<(i252, ptr, ptr)> loc(unknown), %26: i32 loc(unknown)):  // pred: ^bb15
    %27 = llvm.extractvalue %25[0] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    %28 = llvm.extractvalue %25[1] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    %29 = llvm.extractvalue %25[2] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    llvm.store %26, %28 {alignment = 4 : i64} : i32, !llvm.ptr loc(#loc)
    %30 = llvm.alloca %27 x i252 {alignment = 8 : i64} : (i252) -> !llvm.ptr<i252> loc(#loc)
    llvm.store %27, %30 {alignment = 8 : i64} : !llvm.ptr<i252> loc(#loc)
    %31 = call @cairo_native__dict_insert(%29, %30, %28) : (!llvm.ptr, !llvm.ptr<i252>, !llvm.ptr) -> !llvm.ptr loc(#loc)
    cf.br ^bb17 loc(#loc)
  ^bb17:  // pred: ^bb16
    %c2_i252_0 = arith.constant 2 : i252 loc(#loc)
    cf.br ^bb18(%c2_i252_0 : i252) loc(#loc)
  ^bb18(%32: i252 loc(unknown)):  // pred: ^bb17
    cf.br ^bb19(%29, %32 : !llvm.ptr, i252) loc(#loc)
  ^bb19(%33: !llvm.ptr loc(unknown), %34: i252 loc(unknown)):  // pred: ^bb18
    %35 = llvm.alloca %34 x i252 {alignment = 8 : i64} : (i252) -> !llvm.ptr<i252> loc(#loc)
    llvm.store %34, %35 {alignment = 8 : i64} : !llvm.ptr<i252> loc(#loc)
    %36 = call @cairo_native__dict_get(%33, %35) : (!llvm.ptr, !llvm.ptr<i252>) -> !llvm.ptr loc(#loc)
    %37 = llvm.mlir.null : !llvm.ptr loc(#loc)
    %38 = llvm.icmp "eq" %36, %37 : !llvm.ptr loc(#loc)
    cf.cond_br %38, ^bb20, ^bb21 loc(#loc)
  ^bb20:  // pred: ^bb19
    %c4_i64_1 = arith.constant 4 : i64 loc(#loc)
    %39 = call @realloc(%36, %c4_i64_1) : (!llvm.ptr, i64) -> !llvm.ptr loc(#loc)
    %40 = llvm.mlir.undef : i32 loc(#loc)
    cf.br ^bb22(%39, %40 : !llvm.ptr, i32) loc(#loc)
  ^bb21:  // pred: ^bb19
    %41 = llvm.load %36 {alignment = 4 : i64} : !llvm.ptr -> i32 loc(#loc)
    cf.br ^bb22(%36, %41 : !llvm.ptr, i32) loc(#loc)
  ^bb22(%42: !llvm.ptr loc(unknown), %43: i32 loc(unknown)):  // 2 preds: ^bb20, ^bb21
    %44 = llvm.mlir.undef : !llvm.struct<(i252, ptr, ptr)> loc(#loc)
    %45 = llvm.insertvalue %34, %44[0] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    %46 = llvm.insertvalue %42, %45[1] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    %47 = llvm.insertvalue %33, %46[2] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    cf.br ^bb23(%3, %43 : i32, i32) loc(#loc)
  ^bb23(%48: i32 loc(unknown), %49: i32 loc(unknown)):  // pred: ^bb22
    cf.br ^bb24(%49 : i32) loc(#loc7)
  ^bb24(%50: i32 loc(unknown)):  // pred: ^bb23
    cf.br ^bb25(%47, %50 : !llvm.struct<(i252, ptr, ptr)>, i32) loc(#loc)
  ^bb25(%51: !llvm.struct<(i252, ptr, ptr)> loc(unknown), %52: i32 loc(unknown)):  // pred: ^bb24
    %53 = llvm.extractvalue %51[0] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    %54 = llvm.extractvalue %51[1] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    %55 = llvm.extractvalue %51[2] : !llvm.struct<(i252, ptr, ptr)>  loc(#loc)
    llvm.store %52, %54 {alignment = 4 : i64} : i32, !llvm.ptr loc(#loc)
    %56 = llvm.alloca %53 x i252 {alignment = 8 : i64} : (i252) -> !llvm.ptr<i252> loc(#loc)
    llvm.store %53, %56 {alignment = 8 : i64} : !llvm.ptr<i252> loc(#loc)
    %57 = call @cairo_native__dict_insert(%55, %56, %54) : (!llvm.ptr, !llvm.ptr<i252>, !llvm.ptr) -> !llvm.ptr loc(#loc)
    cf.br ^bb26(%0 : !llvm.array<0 x i8>) loc(#loc)
  ^bb26(%58: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb25
    cf.br ^bb27(%4 : !llvm.array<0 x i8>) loc(#loc8)
  ^bb27(%59: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb26
    cf.br ^bb28(%2 : i64) loc(#loc8)
  ^bb28(%60: i64 loc(unknown)):  // pred: ^bb27
    cf.br ^bb29(%55 : !llvm.ptr) loc(#loc)
  ^bb29(%61: !llvm.ptr loc(unknown)):  // pred: ^bb28
    cf.br ^bb30(%58, %59, %60, %61 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) loc(#loc)
  ^bb30(%62: !llvm.array<0 x i8> loc(unknown), %63: !llvm.array<0 x i8> loc(unknown), %64: i64 loc(unknown), %65: !llvm.ptr loc(unknown)):  // pred: ^bb29
    %66:4 = call @"core::dict::Felt252DictDestruct::<core::integer::u32, core::integer::u32Drop, core::integer::U32Felt252DictValue>::destruct"(%62, %63, %64, %65) : (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.struct<()>) loc(#loc)
    cf.br ^bb31(%66#3 : !llvm.struct<()>) loc(#loc)
  ^bb31(%67: !llvm.struct<()> loc(unknown)):  // pred: ^bb30
    cf.br ^bb32(%66#0 : !llvm.array<0 x i8>) loc(#loc)
  ^bb32(%68: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb31
    cf.br ^bb33(%66#1 : !llvm.array<0 x i8>) loc(#loc)
  ^bb33(%69: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb32
    cf.br ^bb34(%66#2 : i64) loc(#loc)
  ^bb34(%70: i64 loc(unknown)):  // pred: ^bb33
    cf.br ^bb35(%50 : i32) loc(#loc)
  ^bb35(%71: i32 loc(unknown)):  // pred: ^bb34
    cf.br ^bb36(%68, %69, %70, %71 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, i32) loc(#loc)
  ^bb36(%72: !llvm.array<0 x i8> loc(unknown), %73: !llvm.array<0 x i8> loc(unknown), %74: i64 loc(unknown), %75: i32 loc(unknown)):  // pred: ^bb35
    return %68, %69, %70, %71 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, i32 loc(#loc)
  } loc(#loc)
  func.func public @"core::dict::Felt252DictDestruct::<core::integer::u32, core::integer::u32Drop, core::integer::U32Felt252DictValue>::destruct"(%arg0: !llvm.array<0 x i8> loc(unknown), %arg1: !llvm.array<0 x i8> loc(unknown), %arg2: i64 loc(unknown), %arg3: !llvm.ptr loc(unknown)) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.struct<()>) attributes {llvm.emit_c_interface} {
    cf.br ^bb1(%arg0, %arg1, %arg2, %arg3 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) loc(#loc)
  ^bb1(%0: !llvm.array<0 x i8> loc(unknown), %1: !llvm.array<0 x i8> loc(unknown), %2: i64 loc(unknown), %3: !llvm.ptr loc(unknown)):  // pred: ^bb0
    cf.br ^bb2 loc(#loc)
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%0 : !llvm.array<0 x i8>) loc(#loc9)
  ^bb3(%4: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb2
    cf.br ^bb4(%1 : !llvm.array<0 x i8>) loc(#loc)
  ^bb4(%5: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb3
    cf.br ^bb5(%2 : i64) loc(#loc)
  ^bb5(%6: i64 loc(unknown)):  // pred: ^bb4
    cf.br ^bb6(%3 : !llvm.ptr) loc(#loc)
  ^bb6(%7: !llvm.ptr loc(unknown)):  // pred: ^bb5
    cf.br ^bb7(%4, %5, %6, %7 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) loc(#loc)
  ^bb7(%8: !llvm.array<0 x i8> loc(unknown), %9: !llvm.array<0 x i8> loc(unknown), %10: i64 loc(unknown), %11: !llvm.ptr loc(unknown)):  // pred: ^bb6
    %12:4 = call @"core::dict::Felt252DictImpl::<core::integer::u32, core::integer::U32Felt252DictValue>::squash"(%8, %9, %10, %11) : (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) loc(#loc)
    cf.br ^bb8(%12#3 : !llvm.ptr) loc(#loc)
  ^bb8(%13: !llvm.ptr loc(unknown)):  // pred: ^bb7
    cf.br ^bb9 loc(#loc)
  ^bb9:  // pred: ^bb8
    %14 = llvm.mlir.undef : !llvm.struct<()> loc(#loc)
    cf.br ^bb10(%12#0 : !llvm.array<0 x i8>) loc(#loc)
  ^bb10(%15: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb9
    cf.br ^bb11(%12#1 : !llvm.array<0 x i8>) loc(#loc)
  ^bb11(%16: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb10
    cf.br ^bb12(%12#2 : i64) loc(#loc)
  ^bb12(%17: i64 loc(unknown)):  // pred: ^bb11
    cf.br ^bb13(%14 : !llvm.struct<()>) loc(#loc)
  ^bb13(%18: !llvm.struct<()> loc(unknown)):  // pred: ^bb12
    cf.br ^bb14(%15, %16, %17, %18 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.struct<()>) loc(#loc)
  ^bb14(%19: !llvm.array<0 x i8> loc(unknown), %20: !llvm.array<0 x i8> loc(unknown), %21: i64 loc(unknown), %22: !llvm.struct<()> loc(unknown)):  // pred: ^bb13
    return %15, %16, %17, %18 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.struct<()> loc(#loc)
  } loc(#loc)
  func.func public @"core::dict::Felt252DictImpl::<core::integer::u32, core::integer::U32Felt252DictValue>::squash"(%arg0: !llvm.array<0 x i8> loc(unknown), %arg1: !llvm.array<0 x i8> loc(unknown), %arg2: i64 loc(unknown), %arg3: !llvm.ptr loc(unknown)) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) attributes {llvm.emit_c_interface} {
    cf.br ^bb1(%arg0, %arg1, %arg2, %arg3 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) loc(#loc)
  ^bb1(%0: !llvm.array<0 x i8> loc(unknown), %1: !llvm.array<0 x i8> loc(unknown), %2: i64 loc(unknown), %3: !llvm.ptr loc(unknown)):  // pred: ^bb0
    cf.br ^bb2 loc(#loc)
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%0, %2, %1, %3 : !llvm.array<0 x i8>, i64, !llvm.array<0 x i8>, !llvm.ptr) loc(#loc10)
  ^bb3(%4: !llvm.array<0 x i8> loc(unknown), %5: i64 loc(unknown), %6: !llvm.array<0 x i8> loc(unknown), %7: !llvm.ptr loc(unknown)):  // pred: ^bb2
    cf.br ^bb4(%4 : !llvm.array<0 x i8>) loc(#loc)
  ^bb4(%8: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb3
    cf.br ^bb5(%6 : !llvm.array<0 x i8>) loc(#loc)
  ^bb5(%9: !llvm.array<0 x i8> loc(unknown)):  // pred: ^bb4
    cf.br ^bb6(%5 : i64) loc(#loc)
  ^bb6(%10: i64 loc(unknown)):  // pred: ^bb5
    cf.br ^bb7(%7 : !llvm.ptr) loc(#loc)
  ^bb7(%11: !llvm.ptr loc(unknown)):  // pred: ^bb6
    cf.br ^bb8(%8, %9, %10, %11 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) loc(#loc)
  ^bb8(%12: !llvm.array<0 x i8> loc(unknown), %13: !llvm.array<0 x i8> loc(unknown), %14: i64 loc(unknown), %15: !llvm.ptr loc(unknown)):  // pred: ^bb7
    return %8, %9, %10, %11 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("programs/examples/ex_dict.cairo":5:16)
#loc2 = loc("programs/examples/ex_dict.cairo":5:19)
#loc3 = loc("/data1/edgar/work/cairo_sierra_2_MLIR/corelib/src/dict.cairo":78:8)
#loc4 = loc("programs/examples/ex_dict.cairo":6:13)
#loc5 = loc("/data1/edgar/work/cairo_sierra_2_MLIR/corelib/src/dict.cairo":41:35)
#loc6 = loc("/data1/edgar/work/cairo_sierra_2_MLIR/corelib/src/dict.cairo":42:15)
#loc7 = loc("/data1/edgar/work/cairo_sierra_2_MLIR/corelib/src/dict.cairo":47:34)
#loc8 = loc("/data1/edgar/work/cairo_sierra_2_MLIR/corelib/src/dict.cairo":49:15)
#loc9 = loc("/data1/edgar/work/cairo_sierra_2_MLIR/corelib/src/dict.cairo":87:8)
#loc10 = loc("/data1/edgar/work/cairo_sierra_2_MLIR/corelib/src/dict.cairo":55:8)
