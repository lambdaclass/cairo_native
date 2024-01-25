"builtin.module"() ({
  "func.func"() <{function_type = (i128, !llvm.ptr, !llvm.struct<(i128, i128)>, !llvm.struct<(i128, i128)>) -> (i128, !llvm.ptr, !llvm.struct<(i128, array<80 x i8>)>), sym_name = "secp256::secp256::secp256k1_new(f0)", sym_visibility = "public"}> ({
  ^bb0(%arg0: i128, %arg1: !llvm.ptr, %arg2: !llvm.struct<(i128, i128)>, %arg3: !llvm.struct<(i128, i128)>):
    %0 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %1 = "llvm.alloca"(%0) <{alignment = 16 : i64, elem_type = !llvm.struct<(i1, array<95 x i8>)>}> : (i64) -> !llvm.ptr
    %2 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<i128>
    %3 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<struct<(i128, i128)>>
    %4 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<struct<(i128, i128)>>
    %5 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %6 = "llvm.alloca"(%5) <{alignment = 16 : i64, elem_type = !llvm.struct<(i128, array<80 x i8>)>}> : (i64) -> !llvm.ptr
    %7 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %8 = "llvm.alloca"(%7) <{alignment = 16 : i64, elem_type = !llvm.struct<(i128, array<80 x i8>)>}> : (i64) -> !llvm.ptr
    "cf.br"(%arg0, %arg1, %arg2, %arg3)[^bb1] : (i128, !llvm.ptr, !llvm.struct<(i128, i128)>, !llvm.struct<(i128, i128)>) -> ()
  ^bb1(%9: i128, %10: !llvm.ptr, %11: !llvm.struct<(i128, i128)>, %12: !llvm.struct<(i128, i128)>):  // pred: ^bb0
    "cf.br"(%9, %10, %11, %12)[^bb2] : (i128, !llvm.ptr, !llvm.struct<(i128, i128)>, !llvm.struct<(i128, i128)>) -> ()
  ^bb2(%13: i128, %14: !llvm.ptr, %15: !llvm.struct<(i128, i128)>, %16: !llvm.struct<(i128, i128)>):  // pred: ^bb1
    %17 = "llvm.load"(%14) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.ptr
    "llvm.store"(%13, %2) <{ordering = 0 : i64}> : (i128, !llvm.ptr<i128>) -> ()
    "llvm.store"(%15, %3) <{ordering = 0 : i64}> : (!llvm.struct<(i128, i128)>, !llvm.ptr<struct<(i128, i128)>>) -> ()
    "llvm.store"(%16, %4) <{ordering = 0 : i64}> : (!llvm.struct<(i128, i128)>, !llvm.ptr<struct<(i128, i128)>>) -> ()
    %18 = "llvm.getelementptr"(%14) <{elem_type = !llvm.ptr, rawConstantIndices = array<i32: 12>}> : (!llvm.ptr) -> !llvm.ptr
    %19 = "llvm.load"(%18) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.ptr<func<void (ptr, ptr, ptr<i128>, ptr<struct<(i128, i128)>>, ptr<struct<(i128, i128)>>)>>
    "llvm.call"(%19, %1, %17, %2, %3, %4) <{fastmathFlags = #llvm.fastmath<none>}> : (!llvm.ptr<func<void (ptr, ptr, ptr<i128>, ptr<struct<(i128, i128)>>, ptr<struct<(i128, i128)>>)>>, !llvm.ptr, !llvm.ptr, !llvm.ptr<i128>, !llvm.ptr<struct<(i128, i128)>>, !llvm.ptr<struct<(i128, i128)>>) -> ()
    %20 = "llvm.load"(%1) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i1, array<95 x i8>)>
    %21 = "llvm.extractvalue"(%20) <{position = array<i64: 0>}> : (!llvm.struct<(i1, array<95 x i8>)>) -> i1
    %22 = "llvm.getelementptr"(%1) <{elem_type = i8, rawConstantIndices = array<i32: 16>}> : (!llvm.ptr) -> !llvm.ptr
    %23 = "llvm.getelementptr"(%1) <{elem_type = i8, rawConstantIndices = array<i32: 8>}> : (!llvm.ptr) -> !llvm.ptr
    %24 = "llvm.load"(%23) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(ptr<i252>, i32, i32)>
    %25 = "llvm.load"(%2) <{ordering = 0 : i64}> : (!llvm.ptr<i128>) -> i128
    "cf.cond_br"(%21)[^bb9, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb3:  // pred: ^bb2
    "cf.br"(%22)[^bb4] : (!llvm.ptr) -> ()
  ^bb4(%26: !llvm.ptr):  // pred: ^bb3
    %27 = "arith.constant"() <{value = false}> : () -> i1
    %28 = "llvm.load"(%26) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i128, array<64 x i8>)>
    %29 = "llvm.mlir.undef"() : () -> !llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>
    %30 = "llvm.insertvalue"(%29, %27) <{position = array<i64: 0>}> : (!llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>, i1) -> !llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>
    %31 = "llvm.insertvalue"(%30, %28) <{position = array<i64: 1>}> : (!llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>, !llvm.struct<(i128, array<64 x i8>)>) -> !llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>
    "llvm.store"(%31, %8) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>, !llvm.ptr) -> ()
    "cf.br"(%25)[^bb5] : (i128) -> ()
  ^bb5(%32: i128):  // pred: ^bb4
    "cf.br"(%14)[^bb6] : (!llvm.ptr) -> ()
  ^bb6(%33: !llvm.ptr):  // pred: ^bb5
    "cf.br"(%8)[^bb7] : (!llvm.ptr) -> ()
  ^bb7(%34: !llvm.ptr):  // pred: ^bb6
    "cf.br"()[^bb8] : () -> ()
  ^bb8:  // pred: ^bb7
    "cf.br"(%32, %33, %34)[^bb14] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb9:  // pred: ^bb2
    "cf.br"(%24)[^bb10] : (!llvm.struct<(ptr<i252>, i32, i32)>) -> ()
  ^bb10(%35: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb9
    %36 = "arith.constant"() <{value = true}> : () -> i1
    %37 = "llvm.mlir.undef"() : () -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    %38 = "llvm.insertvalue"(%37, %36) <{position = array<i64: 0>}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, i1) -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    %39 = "llvm.insertvalue"(%38, %35) <{position = array<i64: 1>}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    "llvm.store"(%39, %6) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, !llvm.ptr) -> ()
    "cf.br"(%25)[^bb11] : (i128) -> ()
  ^bb11(%40: i128):  // pred: ^bb10
    "cf.br"(%14)[^bb12] : (!llvm.ptr) -> ()
  ^bb12(%41: !llvm.ptr):  // pred: ^bb11
    "cf.br"(%6)[^bb13] : (!llvm.ptr) -> ()
  ^bb13(%42: !llvm.ptr):  // pred: ^bb12
    "cf.br"(%40, %41, %42)[^bb14] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb14(%43: i128, %44: !llvm.ptr, %45: !llvm.ptr):  // 2 preds: ^bb8, ^bb13
    "cf.br"(%43, %44, %45)[^bb15] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb15(%46: i128, %47: !llvm.ptr, %48: !llvm.ptr):  // pred: ^bb14
    %49 = "llvm.load"(%45) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i128, array<80 x i8>)>
    "func.return"(%43, %44, %49) : (i128, !llvm.ptr, !llvm.struct<(i128, array<80 x i8>)>) -> ()
  }) {llvm.emit_c_interface} : () -> ()
  "func.func"() <{function_type = (i128, !llvm.ptr, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>) -> (i128, !llvm.ptr, !llvm.struct<(i128, array<64 x i8>)>), sym_name = "secp256::secp256::secp256k1_add(f1)", sym_visibility = "public"}> ({
  ^bb0(%arg0: i128, %arg1: !llvm.ptr, %arg2: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, %arg3: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>):
    %0 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %1 = "llvm.alloca"(%0) <{alignment = 16 : i64, elem_type = !llvm.struct<(i1, array<79 x i8>)>}> : (i64) -> !llvm.ptr
    %2 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<i128>
    %3 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>
    %4 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>
    %5 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %6 = "llvm.alloca"(%5) <{alignment = 16 : i64, elem_type = !llvm.struct<(i128, array<64 x i8>)>}> : (i64) -> !llvm.ptr
    %7 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %8 = "llvm.alloca"(%7) <{alignment = 16 : i64, elem_type = !llvm.struct<(i128, array<64 x i8>)>}> : (i64) -> !llvm.ptr
    "cf.br"(%arg0, %arg1, %arg2, %arg3)[^bb1] : (i128, !llvm.ptr, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>) -> ()
  ^bb1(%9: i128, %10: !llvm.ptr, %11: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, %12: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>):  // pred: ^bb0
    "cf.br"(%9, %10, %11, %12)[^bb2] : (i128, !llvm.ptr, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>) -> ()
  ^bb2(%13: i128, %14: !llvm.ptr, %15: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, %16: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>):  // pred: ^bb1
    %17 = "llvm.load"(%14) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.ptr
    "llvm.store"(%13, %2) <{ordering = 0 : i64}> : (i128, !llvm.ptr<i128>) -> ()
    "llvm.store"(%15, %3) <{ordering = 0 : i64}> : (!llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, !llvm.ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>) -> ()
    "llvm.store"(%16, %4) <{ordering = 0 : i64}> : (!llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, !llvm.ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>) -> ()
    %18 = "llvm.getelementptr"(%14) <{elem_type = !llvm.ptr, rawConstantIndices = array<i32: 13>}> : (!llvm.ptr) -> !llvm.ptr
    %19 = "llvm.load"(%18) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.ptr<func<void (ptr, ptr, ptr<i128>, ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>, ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>)>>
    "llvm.call"(%19, %1, %17, %2, %3, %4) <{fastmathFlags = #llvm.fastmath<none>}> : (!llvm.ptr<func<void (ptr, ptr, ptr<i128>, ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>, ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>)>>, !llvm.ptr, !llvm.ptr, !llvm.ptr<i128>, !llvm.ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>, !llvm.ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>) -> ()
    %20 = "llvm.load"(%1) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i1, array<79 x i8>)>
    %21 = "llvm.extractvalue"(%20) <{position = array<i64: 0>}> : (!llvm.struct<(i1, array<79 x i8>)>) -> i1
    %22 = "llvm.getelementptr"(%1) <{elem_type = i8, rawConstantIndices = array<i32: 16>}> : (!llvm.ptr) -> !llvm.ptr
    %23 = "llvm.load"(%22) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>
    %24 = "llvm.getelementptr"(%1) <{elem_type = i8, rawConstantIndices = array<i32: 8>}> : (!llvm.ptr) -> !llvm.ptr
    %25 = "llvm.load"(%24) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(ptr<i252>, i32, i32)>
    %26 = "llvm.load"(%2) <{ordering = 0 : i64}> : (!llvm.ptr<i128>) -> i128
    "cf.cond_br"(%21)[^bb9, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb3:  // pred: ^bb2
    "cf.br"(%23)[^bb4] : (!llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>) -> ()
  ^bb4(%27: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>):  // pred: ^bb3
    %28 = "arith.constant"() <{value = false}> : () -> i1
    %29 = "llvm.mlir.undef"() : () -> !llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>
    %30 = "llvm.insertvalue"(%29, %28) <{position = array<i64: 0>}> : (!llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>, i1) -> !llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>
    %31 = "llvm.insertvalue"(%30, %27) <{position = array<i64: 1>}> : (!llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>) -> !llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>
    "llvm.store"(%31, %8) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>, !llvm.ptr) -> ()
    "cf.br"(%26)[^bb5] : (i128) -> ()
  ^bb5(%32: i128):  // pred: ^bb4
    "cf.br"(%14)[^bb6] : (!llvm.ptr) -> ()
  ^bb6(%33: !llvm.ptr):  // pred: ^bb5
    "cf.br"(%8)[^bb7] : (!llvm.ptr) -> ()
  ^bb7(%34: !llvm.ptr):  // pred: ^bb6
    "cf.br"()[^bb8] : () -> ()
  ^bb8:  // pred: ^bb7
    "cf.br"(%32, %33, %34)[^bb14] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb9:  // pred: ^bb2
    "cf.br"(%25)[^bb10] : (!llvm.struct<(ptr<i252>, i32, i32)>) -> ()
  ^bb10(%35: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb9
    %36 = "arith.constant"() <{value = true}> : () -> i1
    %37 = "llvm.mlir.undef"() : () -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    %38 = "llvm.insertvalue"(%37, %36) <{position = array<i64: 0>}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, i1) -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    %39 = "llvm.insertvalue"(%38, %35) <{position = array<i64: 1>}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    "llvm.store"(%39, %6) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, !llvm.ptr) -> ()
    "cf.br"(%26)[^bb11] : (i128) -> ()
  ^bb11(%40: i128):  // pred: ^bb10
    "cf.br"(%14)[^bb12] : (!llvm.ptr) -> ()
  ^bb12(%41: !llvm.ptr):  // pred: ^bb11
    "cf.br"(%6)[^bb13] : (!llvm.ptr) -> ()
  ^bb13(%42: !llvm.ptr):  // pred: ^bb12
    "cf.br"(%40, %41, %42)[^bb14] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb14(%43: i128, %44: !llvm.ptr, %45: !llvm.ptr):  // 2 preds: ^bb8, ^bb13
    "cf.br"(%43, %44, %45)[^bb15] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb15(%46: i128, %47: !llvm.ptr, %48: !llvm.ptr):  // pred: ^bb14
    %49 = "llvm.load"(%45) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i128, array<64 x i8>)>
    "func.return"(%43, %44, %49) : (i128, !llvm.ptr, !llvm.struct<(i128, array<64 x i8>)>) -> ()
  }) {llvm.emit_c_interface} : () -> ()
  "func.func"() <{function_type = (i128, !llvm.ptr, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, !llvm.struct<(i128, i128)>) -> (i128, !llvm.ptr, !llvm.struct<(i128, array<64 x i8>)>), sym_name = "secp256::secp256::secp256k1_mul(f2)", sym_visibility = "public"}> ({
  ^bb0(%arg0: i128, %arg1: !llvm.ptr, %arg2: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, %arg3: !llvm.struct<(i128, i128)>):
    %0 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %1 = "llvm.alloca"(%0) <{alignment = 16 : i64, elem_type = !llvm.struct<(i1, array<79 x i8>)>}> : (i64) -> !llvm.ptr
    %2 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<i128>
    %3 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>
    %4 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<struct<(i128, i128)>>
    %5 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %6 = "llvm.alloca"(%5) <{alignment = 16 : i64, elem_type = !llvm.struct<(i128, array<64 x i8>)>}> : (i64) -> !llvm.ptr
    %7 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %8 = "llvm.alloca"(%7) <{alignment = 16 : i64, elem_type = !llvm.struct<(i128, array<64 x i8>)>}> : (i64) -> !llvm.ptr
    "cf.br"(%arg0, %arg1, %arg2, %arg3)[^bb1] : (i128, !llvm.ptr, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, !llvm.struct<(i128, i128)>) -> ()
  ^bb1(%9: i128, %10: !llvm.ptr, %11: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, %12: !llvm.struct<(i128, i128)>):  // pred: ^bb0
    "cf.br"(%9, %10, %11, %12)[^bb2] : (i128, !llvm.ptr, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, !llvm.struct<(i128, i128)>) -> ()
  ^bb2(%13: i128, %14: !llvm.ptr, %15: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, %16: !llvm.struct<(i128, i128)>):  // pred: ^bb1
    %17 = "llvm.load"(%14) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.ptr
    "llvm.store"(%13, %2) <{ordering = 0 : i64}> : (i128, !llvm.ptr<i128>) -> ()
    "llvm.store"(%15, %3) <{ordering = 0 : i64}> : (!llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>, !llvm.ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>) -> ()
    "llvm.store"(%16, %4) <{ordering = 0 : i64}> : (!llvm.struct<(i128, i128)>, !llvm.ptr<struct<(i128, i128)>>) -> ()
    %18 = "llvm.getelementptr"(%14) <{elem_type = !llvm.ptr, rawConstantIndices = array<i32: 14>}> : (!llvm.ptr) -> !llvm.ptr
    %19 = "llvm.load"(%18) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.ptr<func<void (ptr, ptr, ptr<i128>, ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>, ptr<struct<(i128, i128)>>)>>
    "llvm.call"(%19, %1, %17, %2, %3, %4) <{fastmathFlags = #llvm.fastmath<none>}> : (!llvm.ptr<func<void (ptr, ptr, ptr<i128>, ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>, ptr<struct<(i128, i128)>>)>>, !llvm.ptr, !llvm.ptr, !llvm.ptr<i128>, !llvm.ptr<struct<(struct<(i128, i128)>, struct<(i128, i128)>)>>, !llvm.ptr<struct<(i128, i128)>>) -> ()
    %20 = "llvm.load"(%1) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i1, array<79 x i8>)>
    %21 = "llvm.extractvalue"(%20) <{position = array<i64: 0>}> : (!llvm.struct<(i1, array<79 x i8>)>) -> i1
    %22 = "llvm.getelementptr"(%1) <{elem_type = i8, rawConstantIndices = array<i32: 16>}> : (!llvm.ptr) -> !llvm.ptr
    %23 = "llvm.load"(%22) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>
    %24 = "llvm.getelementptr"(%1) <{elem_type = i8, rawConstantIndices = array<i32: 8>}> : (!llvm.ptr) -> !llvm.ptr
    %25 = "llvm.load"(%24) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(ptr<i252>, i32, i32)>
    %26 = "llvm.load"(%2) <{ordering = 0 : i64}> : (!llvm.ptr<i128>) -> i128
    "cf.cond_br"(%21)[^bb9, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb3:  // pred: ^bb2
    "cf.br"(%23)[^bb4] : (!llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>) -> ()
  ^bb4(%27: !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>):  // pred: ^bb3
    %28 = "arith.constant"() <{value = false}> : () -> i1
    %29 = "llvm.mlir.undef"() : () -> !llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>
    %30 = "llvm.insertvalue"(%29, %28) <{position = array<i64: 0>}> : (!llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>, i1) -> !llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>
    %31 = "llvm.insertvalue"(%30, %27) <{position = array<i64: 1>}> : (!llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>, !llvm.struct<(struct<(i128, i128)>, struct<(i128, i128)>)>) -> !llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>
    "llvm.store"(%31, %8) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.struct<(i1, struct<(struct<(i128, i128)>, struct<(i128, i128)>)>)>, !llvm.ptr) -> ()
    "cf.br"(%26)[^bb5] : (i128) -> ()
  ^bb5(%32: i128):  // pred: ^bb4
    "cf.br"(%14)[^bb6] : (!llvm.ptr) -> ()
  ^bb6(%33: !llvm.ptr):  // pred: ^bb5
    "cf.br"(%8)[^bb7] : (!llvm.ptr) -> ()
  ^bb7(%34: !llvm.ptr):  // pred: ^bb6
    "cf.br"()[^bb8] : () -> ()
  ^bb8:  // pred: ^bb7
    "cf.br"(%32, %33, %34)[^bb14] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb9:  // pred: ^bb2
    "cf.br"(%25)[^bb10] : (!llvm.struct<(ptr<i252>, i32, i32)>) -> ()
  ^bb10(%35: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb9
    %36 = "arith.constant"() <{value = true}> : () -> i1
    %37 = "llvm.mlir.undef"() : () -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    %38 = "llvm.insertvalue"(%37, %36) <{position = array<i64: 0>}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, i1) -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    %39 = "llvm.insertvalue"(%38, %35) <{position = array<i64: 1>}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    "llvm.store"(%39, %6) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, !llvm.ptr) -> ()
    "cf.br"(%26)[^bb11] : (i128) -> ()
  ^bb11(%40: i128):  // pred: ^bb10
    "cf.br"(%14)[^bb12] : (!llvm.ptr) -> ()
  ^bb12(%41: !llvm.ptr):  // pred: ^bb11
    "cf.br"(%6)[^bb13] : (!llvm.ptr) -> ()
  ^bb13(%42: !llvm.ptr):  // pred: ^bb12
    "cf.br"(%40, %41, %42)[^bb14] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb14(%43: i128, %44: !llvm.ptr, %45: !llvm.ptr):  // 2 preds: ^bb8, ^bb13
    "cf.br"(%43, %44, %45)[^bb15] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb15(%46: i128, %47: !llvm.ptr, %48: !llvm.ptr):  // pred: ^bb14
    %49 = "llvm.load"(%45) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i128, array<64 x i8>)>
    "func.return"(%43, %44, %49) : (i128, !llvm.ptr, !llvm.struct<(i128, array<64 x i8>)>) -> ()
  }) {llvm.emit_c_interface} : () -> ()
  "func.func"() <{function_type = (i128, !llvm.ptr, !llvm.struct<(i128, i128)>, !llvm.struct<(i1, array<0 x i8>)>) -> (i128, !llvm.ptr, !llvm.struct<(i128, array<80 x i8>)>), sym_name = "secp256::secp256::secp256k1_get_point_from_x(f3)", sym_visibility = "public"}> ({
  ^bb0(%arg0: i128, %arg1: !llvm.ptr, %arg2: !llvm.struct<(i128, i128)>, %arg3: !llvm.struct<(i1, array<0 x i8>)>):
    %0 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %1 = "llvm.alloca"(%0) <{alignment = 16 : i64, elem_type = !llvm.struct<(i1, array<95 x i8>)>}> : (i64) -> !llvm.ptr
    %2 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<i128>
    %3 = "llvm.alloca"(%0) <{alignment = 16 : i64}> : (i64) -> !llvm.ptr<struct<(i128, i128)>>
    %4 = "llvm.alloca"(%0) : (i64) -> !llvm.ptr<i1>
    %5 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %6 = "llvm.alloca"(%5) <{alignment = 16 : i64, elem_type = !llvm.struct<(i128, array<80 x i8>)>}> : (i64) -> !llvm.ptr
    %7 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %8 = "llvm.alloca"(%7) <{alignment = 16 : i64, elem_type = !llvm.struct<(i128, array<80 x i8>)>}> : (i64) -> !llvm.ptr
    "cf.br"(%arg0, %arg1, %arg2, %arg3)[^bb1] : (i128, !llvm.ptr, !llvm.struct<(i128, i128)>, !llvm.struct<(i1, array<0 x i8>)>) -> ()
  ^bb1(%9: i128, %10: !llvm.ptr, %11: !llvm.struct<(i128, i128)>, %12: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb0
    "cf.br"(%9, %10, %11, %12)[^bb2] : (i128, !llvm.ptr, !llvm.struct<(i128, i128)>, !llvm.struct<(i1, array<0 x i8>)>) -> ()
  ^bb2(%13: i128, %14: !llvm.ptr, %15: !llvm.struct<(i128, i128)>, %16: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb1
    %17 = "llvm.load"(%14) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.ptr
    "llvm.store"(%13, %2) <{ordering = 0 : i64}> : (i128, !llvm.ptr<i128>) -> ()
    "llvm.store"(%15, %3) <{ordering = 0 : i64}> : (!llvm.struct<(i128, i128)>, !llvm.ptr<struct<(i128, i128)>>) -> ()
    "llvm.store"(%16, %4) <{ordering = 0 : i64}> : (!llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr<i1>) -> ()
    %18 = "llvm.getelementptr"(%14) <{elem_type = !llvm.ptr, rawConstantIndices = array<i32: 15>}> : (!llvm.ptr) -> !llvm.ptr
    %19 = "llvm.load"(%18) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.ptr<func<void (ptr, ptr, ptr<i128>, ptr<struct<(i128, i128)>>, ptr<i1>)>>
    "llvm.call"(%19, %1, %17, %2, %3, %4) <{fastmathFlags = #llvm.fastmath<none>}> : (!llvm.ptr<func<void (ptr, ptr, ptr<i128>, ptr<struct<(i128, i128)>>, ptr<i1>)>>, !llvm.ptr, !llvm.ptr, !llvm.ptr<i128>, !llvm.ptr<struct<(i128, i128)>>, !llvm.ptr<i1>) -> ()
    %20 = "llvm.load"(%1) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i1, array<95 x i8>)>
    %21 = "llvm.extractvalue"(%20) <{position = array<i64: 0>}> : (!llvm.struct<(i1, array<95 x i8>)>) -> i1
    %22 = "llvm.getelementptr"(%1) <{elem_type = i8, rawConstantIndices = array<i32: 16>}> : (!llvm.ptr) -> !llvm.ptr
    %23 = "llvm.load"(%22) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i128, array<64 x i8>)>
    %24 = "llvm.getelementptr"(%1) <{elem_type = i8, rawConstantIndices = array<i32: 8>}> : (!llvm.ptr) -> !llvm.ptr
    %25 = "llvm.load"(%24) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(ptr<i252>, i32, i32)>
    %26 = "llvm.load"(%2) <{ordering = 0 : i64}> : (!llvm.ptr<i128>) -> i128
    "cf.cond_br"(%21)[^bb9, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb3:  // pred: ^bb2
    "cf.br"(%23)[^bb4] : (!llvm.struct<(i128, array<64 x i8>)>) -> ()
  ^bb4(%27: !llvm.ptr):  // pred: ^bb3
    %28 = "arith.constant"() <{value = false}> : () -> i1
    %29 = "llvm.load"(%27) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i128, array<64 x i8>)>
    %30 = "llvm.mlir.undef"() : () -> !llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>
    %31 = "llvm.insertvalue"(%30, %28) <{position = array<i64: 0>}> : (!llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>, i1) -> !llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>
    %32 = "llvm.insertvalue"(%31, %29) <{position = array<i64: 1>}> : (!llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>, !llvm.struct<(i128, array<64 x i8>)>) -> !llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>
    "llvm.store"(%32, %8) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.struct<(i1, struct<(i128, array<64 x i8>)>)>, !llvm.ptr) -> ()
    "cf.br"(%26)[^bb5] : (i128) -> ()
  ^bb5(%33: i128):  // pred: ^bb4
    "cf.br"(%14)[^bb6] : (!llvm.ptr) -> ()
  ^bb6(%34: !llvm.ptr):  // pred: ^bb5
    "cf.br"(%8)[^bb7] : (!llvm.ptr) -> ()
  ^bb7(%35: !llvm.ptr):  // pred: ^bb6
    "cf.br"()[^bb8] : () -> ()
  ^bb8:  // pred: ^bb7
    "cf.br"(%33, %34, %35)[^bb14] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb9:  // pred: ^bb2
    "cf.br"(%25)[^bb10] : (!llvm.struct<(ptr<i252>, i32, i32)>) -> ()
  ^bb10(%36: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb9
    %37 = "arith.constant"() <{value = true}> : () -> i1
    %38 = "llvm.mlir.undef"() : () -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    %39 = "llvm.insertvalue"(%38, %37) <{position = array<i64: 0>}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, i1) -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    %40 = "llvm.insertvalue"(%39, %36) <{position = array<i64: 1>}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>
    "llvm.store"(%40, %6) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.struct<(i1, struct<(ptr<i252>, i32, i32)>)>, !llvm.ptr) -> ()
    "cf.br"(%26)[^bb11] : (i128) -> ()
  ^bb11(%41: i128):  // pred: ^bb10
    "cf.br"(%14)[^bb12] : (!llvm.ptr) -> ()
  ^bb12(%42: !llvm.ptr):  // pred: ^bb11
    "cf.br"(%6)[^bb13] : (!llvm.ptr) -> ()
  ^bb13(%43: !llvm.ptr):  // pred: ^bb12
    "cf.br"(%41, %42, %43)[^bb14] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb14(%44: i128, %45: !llvm.ptr, %46: !llvm.ptr):  // 2 preds: ^bb8, ^bb13
    "cf.br"(%44, %45, %46)[^bb15] : (i128, !llvm.ptr, !llvm.ptr) -> ()
  ^bb15(%47: i128, %48: !llvm.ptr, %49: !llvm.ptr):  // pred: ^bb14
    %50 = "llvm.load"(%46) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(i128, array<80 x i8>)>
    "func.return"(%44, %45, %50) : (i128, !llvm.ptr, !llvm.struct<(i128, array<80 x i8>)>) -> ()
  }) {llvm.emit_c_interface} : () -> ()
}) : () -> ()
