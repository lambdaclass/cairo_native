module {
  func.func private @realloc(!llvm.ptr, i64) -> !llvm.ptr
  func.func private @free(!llvm.ptr)
  func.func public @"program::program::special_casts::test_felt252_downcasts(f20)"(%arg0: i64, %arg1: i128) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_1 = arith.constant 1 : i64
    %2 = llvm.alloca %c1_i64_1 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_2 = arith.constant 1 : i64
    %3 = llvm.alloca %c1_i64_2 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_3 = arith.constant 1 : i64
    %4 = llvm.alloca %c1_i64_3 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_4 = arith.constant 1 : i64
    %5 = llvm.alloca %c1_i64_4 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_5 = arith.constant 1 : i64
    %6 = llvm.alloca %c1_i64_5 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_6 = arith.constant 1 : i64
    %7 = llvm.alloca %c1_i64_6 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_7 = arith.constant 1 : i64
    %8 = llvm.alloca %c1_i64_7 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_8 = arith.constant 1 : i64
    %9 = llvm.alloca %c1_i64_8 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_9 = arith.constant 1 : i64
    %10 = llvm.alloca %c1_i64_9 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_10 = arith.constant 1 : i64
    %11 = llvm.alloca %c1_i64_10 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_11 = arith.constant 1 : i64
    %12 = llvm.alloca %c1_i64_11 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_12 = arith.constant 1 : i64
    %13 = llvm.alloca %c1_i64_12 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_13 = arith.constant 1 : i64
    %14 = llvm.alloca %c1_i64_13 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_14 = arith.constant 1 : i64
    %15 = llvm.alloca %c1_i64_14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_15 = arith.constant 1 : i64
    %16 = llvm.alloca %c1_i64_15 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_16 = arith.constant 1 : i64
    %17 = llvm.alloca %c1_i64_16 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_17 = arith.constant 1 : i64
    %18 = llvm.alloca %c1_i64_17 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_18 = arith.constant 1 : i64
    %19 = llvm.alloca %c1_i64_18 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_19 = arith.constant 1 : i64
    %20 = llvm.alloca %c1_i64_19 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_20 = arith.constant 1 : i64
    %21 = llvm.alloca %c1_i64_20 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_21 = arith.constant 1 : i64
    %22 = llvm.alloca %c1_i64_21 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_22 = arith.constant 1 : i64
    %23 = llvm.alloca %c1_i64_22 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_23 = arith.constant 1 : i64
    %24 = llvm.alloca %c1_i64_23 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_24 = arith.constant 1 : i64
    %25 = llvm.alloca %c1_i64_24 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_25 = arith.constant 1 : i64
    %26 = llvm.alloca %c1_i64_25 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_26 = arith.constant 1 : i64
    %27 = llvm.alloca %c1_i64_26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_27 = arith.constant 1 : i64
    %28 = llvm.alloca %c1_i64_27 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_28 = arith.constant 1 : i64
    %29 = llvm.alloca %c1_i64_28 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_29 = arith.constant 1 : i64
    %30 = llvm.alloca %c1_i64_29 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_30 = arith.constant 1 : i64
    %31 = llvm.alloca %c1_i64_30 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_31 = arith.constant 1 : i64
    %32 = llvm.alloca %c1_i64_31 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_32 = arith.constant 1 : i64
    %33 = llvm.alloca %c1_i64_32 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_33 = arith.constant 1 : i64
    %34 = llvm.alloca %c1_i64_33 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_34 = arith.constant 1 : i64
    %35 = llvm.alloca %c1_i64_34 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_35 = arith.constant 1 : i64
    %36 = llvm.alloca %c1_i64_35 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_36 = arith.constant 1 : i64
    %37 = llvm.alloca %c1_i64_36 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_37 = arith.constant 1 : i64
    %38 = llvm.alloca %c1_i64_37 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_38 = arith.constant 1 : i64
    %39 = llvm.alloca %c1_i64_38 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_39 = arith.constant 1 : i64
    %40 = llvm.alloca %c1_i64_39 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_40 = arith.constant 1 : i64
    %41 = llvm.alloca %c1_i64_40 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_41 = arith.constant 1 : i64
    %42 = llvm.alloca %c1_i64_41 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_42 = arith.constant 1 : i64
    %43 = llvm.alloca %c1_i64_42 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_43 = arith.constant 1 : i64
    %44 = llvm.alloca %c1_i64_43 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_44 = arith.constant 1 : i64
    %45 = llvm.alloca %c1_i64_44 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_45 = arith.constant 1 : i64
    %46 = llvm.alloca %c1_i64_45 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_46 = arith.constant 1 : i64
    %47 = llvm.alloca %c1_i64_46 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_47 = arith.constant 1 : i64
    %48 = llvm.alloca %c1_i64_47 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_48 = arith.constant 1 : i64
    %49 = llvm.alloca %c1_i64_48 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_49 = arith.constant 1 : i64
    %50 = llvm.alloca %c1_i64_49 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_50 = arith.constant 1 : i64
    %51 = llvm.alloca %c1_i64_50 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_51 = arith.constant 1 : i64
    %52 = llvm.alloca %c1_i64_51 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_52 = arith.constant 1 : i64
    %53 = llvm.alloca %c1_i64_52 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_53 = arith.constant 1 : i64
    %54 = llvm.alloca %c1_i64_53 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_54 = arith.constant 1 : i64
    %55 = llvm.alloca %c1_i64_54 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1 : i64, i128)
  ^bb1(%56: i64, %57: i128):  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    cf.br ^bb3
  ^bb3:  // pred: ^bb2
    %c1_i252 = arith.constant 1 : i252
    cf.br ^bb4(%56 : i64)
  ^bb4(%58: i64):  // pred: ^bb3
    cf.br ^bb5(%c1_i252 : i252)
  ^bb5(%59: i252):  // pred: ^bb4
    cf.br ^bb6(%58, %59 : i64, i252)
  ^bb6(%60: i64, %61: i252):  // pred: ^bb5
    %62:2 = call @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<0, 0>>(f17)"(%60, %61) : (i64, i252) -> (i64, !llvm.struct<(i1, array<0 x i8>)>)
    cf.br ^bb7(%62#1 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb7(%63: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb6
    %64 = llvm.extractvalue %63[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.switch %64 : i1, [
      default: ^bb8,
      0: ^bb9,
      1: ^bb10
    ]
  ^bb8:  // pred: ^bb7
    %false = arith.constant false
    cf.assert %false, "Invalid enum tag."
    llvm.unreachable
  ^bb9:  // pred: ^bb7
    %65 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb11
  ^bb10:  // pred: ^bb7
    %66 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb123
  ^bb11:  // pred: ^bb9
    cf.br ^bb12(%65 : !llvm.struct<()>)
  ^bb12(%67: !llvm.struct<()>):  // pred: ^bb11
    cf.br ^bb13
  ^bb13:  // pred: ^bb12
    %68 = call @"core::fmt::FormatterDefault::default(f12)"() : () -> !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
    cf.br ^bb14
  ^bb14:  // pred: ^bb13
    %c172180977190876322177717838039515195832848434332511767082422530228238249590_i252 = arith.constant 172180977190876322177717838039515195832848434332511767082422530228238249590 : i252
    cf.br ^bb15
  ^bb15:  // pred: ^bb14
    %c31_i32 = arith.constant 31 : i32
    cf.br ^bb16(%68 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>)
  ^bb16(%69: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>):  // pred: ^bb15
    %70 = llvm.extractvalue %69[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)> 
    cf.br ^bb17(%62#0 : i64)
  ^bb17(%71: i64):  // pred: ^bb16
    cf.br ^bb18(%70 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb18(%72: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb17
    cf.br ^bb19(%c172180977190876322177717838039515195832848434332511767082422530228238249590_i252 : i252)
  ^bb19(%73: i252):  // pred: ^bb18
    cf.br ^bb20(%c31_i32 : i32)
  ^bb20(%74: i32):  // pred: ^bb19
    cf.br ^bb21(%71, %72, %73, %74 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb21(%75: i64, %76: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %77: i252, %78: i32):  // pred: ^bb20
    %79:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%75, %76, %77, %78) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %79#1, %45 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb22(%45 : !llvm.ptr)
  ^bb22(%80: !llvm.ptr):  // pred: ^bb21
    %81 = llvm.load %80 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %81 : i1, [
      default: ^bb23,
      0: ^bb24,
      1: ^bb25
    ]
  ^bb23:  // pred: ^bb22
    %false_55 = arith.constant false
    cf.assert %false_55, "Invalid enum tag."
    llvm.unreachable
  ^bb24:  // pred: ^bb22
    %82 = llvm.load %80 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %83 = llvm.extractvalue %82[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb26
  ^bb25:  // pred: ^bb22
    %84 = llvm.load %80 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %85 = llvm.extractvalue %84[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb117
  ^bb26:  // pred: ^bb24
    cf.br ^bb27(%83 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb27(%86: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb26
    %87 = llvm.extractvalue %86[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %88 = llvm.extractvalue %86[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb28(%88 : !llvm.struct<()>)
  ^bb28(%89: !llvm.struct<()>):  // pred: ^bb27
    cf.br ^bb29
  ^bb29:  // pred: ^bb28
    %c172132395238539276156095731494591481422342884673560379810407770160236339248_i252 = arith.constant 172132395238539276156095731494591481422342884673560379810407770160236339248 : i252
    cf.br ^bb30
  ^bb30:  // pred: ^bb29
    %c31_i32_56 = arith.constant 31 : i32
    cf.br ^bb31(%79#0 : i64)
  ^bb31(%90: i64):  // pred: ^bb30
    cf.br ^bb32(%87 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb32(%91: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb31
    cf.br ^bb33(%c172132395238539276156095731494591481422342884673560379810407770160236339248_i252 : i252)
  ^bb33(%92: i252):  // pred: ^bb32
    cf.br ^bb34(%c31_i32_56 : i32)
  ^bb34(%93: i32):  // pred: ^bb33
    cf.br ^bb35(%90, %91, %92, %93 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb35(%94: i64, %95: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %96: i252, %97: i32):  // pred: ^bb34
    %98:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%94, %95, %96, %97) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %98#1, %47 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb36(%47 : !llvm.ptr)
  ^bb36(%99: !llvm.ptr):  // pred: ^bb35
    %100 = llvm.load %99 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %100 : i1, [
      default: ^bb37,
      0: ^bb38,
      1: ^bb39
    ]
  ^bb37:  // pred: ^bb36
    %false_57 = arith.constant false
    cf.assert %false_57, "Invalid enum tag."
    llvm.unreachable
  ^bb38:  // pred: ^bb36
    %101 = llvm.load %99 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %102 = llvm.extractvalue %101[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb40
  ^bb39:  // pred: ^bb36
    %103 = llvm.load %99 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %104 = llvm.extractvalue %103[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb111
  ^bb40:  // pred: ^bb38
    cf.br ^bb41(%102 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb41(%105: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb40
    %106 = llvm.extractvalue %105[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %107 = llvm.extractvalue %105[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb42(%107 : !llvm.struct<()>)
  ^bb42(%108: !llvm.struct<()>):  // pred: ^bb41
    cf.br ^bb43
  ^bb43:  // pred: ^bb42
    %c17519790900469806_i252 = arith.constant 17519790900469806 : i252
    cf.br ^bb44
  ^bb44:  // pred: ^bb43
    %c7_i32 = arith.constant 7 : i32
    cf.br ^bb45(%98#0 : i64)
  ^bb45(%109: i64):  // pred: ^bb44
    cf.br ^bb46(%106 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb46(%110: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb45
    cf.br ^bb47(%c17519790900469806_i252 : i252)
  ^bb47(%111: i252):  // pred: ^bb46
    cf.br ^bb48(%c7_i32 : i32)
  ^bb48(%112: i32):  // pred: ^bb47
    cf.br ^bb49(%109, %110, %111, %112 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb49(%113: i64, %114: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %115: i252, %116: i32):  // pred: ^bb48
    %117:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%113, %114, %115, %116) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %117#1, %49 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb50(%49 : !llvm.ptr)
  ^bb50(%118: !llvm.ptr):  // pred: ^bb49
    %119 = llvm.load %118 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %119 : i1, [
      default: ^bb51,
      0: ^bb52,
      1: ^bb53
    ]
  ^bb51:  // pred: ^bb50
    %false_58 = arith.constant false
    cf.assert %false_58, "Invalid enum tag."
    llvm.unreachable
  ^bb52:  // pred: ^bb50
    %120 = llvm.load %118 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %121 = llvm.extractvalue %120[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb54
  ^bb53:  // pred: ^bb50
    %122 = llvm.load %118 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %123 = llvm.extractvalue %122[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb105
  ^bb54:  // pred: ^bb52
    cf.br ^bb55
  ^bb55:  // pred: ^bb54
    %124 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb56(%124 : !llvm.struct<()>)
  ^bb56(%125: !llvm.struct<()>):  // pred: ^bb55
    %false_59 = arith.constant false
    %126 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %127 = llvm.insertvalue %false_59, %126[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb57(%127 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb57(%128: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb56
    cf.br ^bb58(%128 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb58(%129: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb57
    call @"core::result::ResultTraitImpl::<(), core::fmt::Error>::unwrap::<core::fmt::ErrorDrop>(f1)"(%51, %129) : (!llvm.ptr, !llvm.struct<(i1, array<0 x i8>)>) -> ()
    %130 = llvm.getelementptr %51[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb59(%130 : !llvm.ptr)
  ^bb59(%131: !llvm.ptr):  // pred: ^bb58
    %132 = llvm.load %131 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %132 : i1, [
      default: ^bb60,
      0: ^bb61,
      1: ^bb62
    ]
  ^bb60:  // pred: ^bb59
    %false_60 = arith.constant false
    cf.assert %false_60, "Invalid enum tag."
    llvm.unreachable
  ^bb61:  // pred: ^bb59
    %133 = llvm.load %131 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>)>)>
    %134 = llvm.extractvalue %133[1] : !llvm.struct<(i1, struct<(struct<()>)>)> 
    cf.br ^bb63
  ^bb62:  // pred: ^bb59
    %135 = llvm.load %131 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %136 = llvm.extractvalue %135[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb98
  ^bb63:  // pred: ^bb61
    cf.br ^bb64(%134 : !llvm.struct<(struct<()>)>)
  ^bb64(%137: !llvm.struct<(struct<()>)>):  // pred: ^bb63
    cf.br ^bb65
  ^bb65:  // pred: ^bb64
    %138 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32 = arith.constant 0 : i32
    %139 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %140 = llvm.insertvalue %138, %139[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %141 = llvm.insertvalue %c0_i32, %140[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %142 = llvm.insertvalue %c0_i32, %141[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb66
  ^bb66:  // pred: ^bb65
    %c1997209042069643135709344952807065910992472029923670688473712229447419591075_i252 = arith.constant 1997209042069643135709344952807065910992472029923670688473712229447419591075 : i252
    cf.br ^bb67(%c1997209042069643135709344952807065910992472029923670688473712229447419591075_i252 : i252)
  ^bb67(%143: i252):  // pred: ^bb66
    cf.br ^bb68(%142, %143 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb68(%144: !llvm.struct<(ptr<i252>, i32, i32)>, %145: i252):  // pred: ^bb67
    %146 = llvm.extractvalue %144[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %147 = llvm.extractvalue %144[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %148 = llvm.extractvalue %144[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %149 = arith.cmpi uge, %147, %148 : i32
    %150:2 = scf.if %149 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32_146 = arith.constant 8 : i32
      %1047 = arith.addi %148, %148 : i32
      %1048 = arith.maxui %c8_i32_146, %1047 : i32
      %1049 = arith.extui %1048 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1050 = arith.muli %1049, %c32_i64 : i64
      %1051 = llvm.bitcast %146 : !llvm.ptr<i252> to !llvm.ptr
      %1052 = func.call @realloc(%1051, %1050) : (!llvm.ptr, i64) -> !llvm.ptr
      %1053 = llvm.bitcast %1052 : !llvm.ptr to !llvm.ptr<i252>
      %1054 = llvm.insertvalue %1053, %144[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1055 = llvm.insertvalue %1048, %1054[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1055, %1053 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %144, %146 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %151 = llvm.getelementptr %150#1[%147] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %145, %151 : i252, !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %152 = arith.addi %147, %c1_i32 : i32
    %153 = llvm.insertvalue %152, %150#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb69(%121 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb69(%154: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb68
    %155 = llvm.extractvalue %154[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %156 = llvm.extractvalue %154[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb70(%156 : !llvm.struct<()>)
  ^bb70(%157: !llvm.struct<()>):  // pred: ^bb69
    cf.br ^bb71(%155 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb71(%158: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb70
    cf.br ^bb72(%158 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb72(%159: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb71
    cf.br ^bb73(%117#0 : i64)
  ^bb73(%160: i64):  // pred: ^bb72
    cf.br ^bb74(%57 : i128)
  ^bb74(%161: i128):  // pred: ^bb73
    cf.br ^bb75(%158 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb75(%162: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb74
    cf.br ^bb76(%153 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb76(%163: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb75
    cf.br ^bb77(%160, %161, %162, %163 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb77(%164: i64, %165: i128, %166: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %167: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb76
    %168:3 = call @"core::byte_array::ByteArraySerde::serialize(f0)"(%164, %165, %166, %167) : (i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %168#2, %53 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb78(%53 : !llvm.ptr)
  ^bb78(%169: !llvm.ptr):  // pred: ^bb77
    %170 = llvm.load %169 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %170 : i1, [
      default: ^bb79,
      0: ^bb80,
      1: ^bb81
    ]
  ^bb79:  // pred: ^bb78
    %false_61 = arith.constant false
    cf.assert %false_61, "Invalid enum tag."
    llvm.unreachable
  ^bb80:  // pred: ^bb78
    %171 = llvm.load %169 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %172 = llvm.extractvalue %171[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    cf.br ^bb82
  ^bb81:  // pred: ^bb78
    %173 = llvm.load %169 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %174 = llvm.extractvalue %173[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb92
  ^bb82:  // pred: ^bb80
    cf.br ^bb83(%172 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)
  ^bb83(%175: !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>):  // pred: ^bb82
    %176 = llvm.extractvalue %175[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %177 = llvm.extractvalue %175[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    cf.br ^bb84(%177 : !llvm.struct<()>)
  ^bb84(%178: !llvm.struct<()>):  // pred: ^bb83
    cf.br ^bb85
  ^bb85:  // pred: ^bb84
    %179 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb86(%179, %176 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb86(%180: !llvm.struct<()>, %181: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb85
    %182 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %183 = llvm.insertvalue %180, %182[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %184 = llvm.insertvalue %181, %183[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb87(%184 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb87(%185: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb86
    %true = arith.constant true
    %186 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %187 = llvm.insertvalue %true, %186[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %188 = llvm.insertvalue %185, %187[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %188, %55 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb88(%168#0 : i64)
  ^bb88(%189: i64):  // pred: ^bb87
    cf.br ^bb89(%168#1 : i128)
  ^bb89(%190: i128):  // pred: ^bb88
    cf.br ^bb90(%55 : !llvm.ptr)
  ^bb90(%191: !llvm.ptr):  // pred: ^bb89
    cf.br ^bb91(%189, %190, %191 : i64, i128, !llvm.ptr)
  ^bb91(%192: i64, %193: i128, %194: !llvm.ptr):  // pred: ^bb90
    %195 = llvm.load %191 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %189, %190, %195 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb92:  // pred: ^bb81
    cf.br ^bb93(%174 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb93(%196: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb92
    %true_62 = arith.constant true
    %197 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %198 = llvm.insertvalue %true_62, %197[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %199 = llvm.insertvalue %196, %198[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %199, %54 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb94(%168#0 : i64)
  ^bb94(%200: i64):  // pred: ^bb93
    cf.br ^bb95(%168#1 : i128)
  ^bb95(%201: i128):  // pred: ^bb94
    cf.br ^bb96(%54 : !llvm.ptr)
  ^bb96(%202: !llvm.ptr):  // pred: ^bb95
    cf.br ^bb97(%200, %201, %202 : i64, i128, !llvm.ptr)
  ^bb97(%203: i64, %204: i128, %205: !llvm.ptr):  // pred: ^bb96
    %206 = llvm.load %202 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %200, %201, %206 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb98:  // pred: ^bb62
    cf.br ^bb99(%121 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb99(%207: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb98
    cf.br ^bb100(%136 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb100(%208: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb99
    %true_63 = arith.constant true
    %209 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %210 = llvm.insertvalue %true_63, %209[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %211 = llvm.insertvalue %208, %210[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %211, %52 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb101(%117#0 : i64)
  ^bb101(%212: i64):  // pred: ^bb100
    cf.br ^bb102(%57 : i128)
  ^bb102(%213: i128):  // pred: ^bb101
    cf.br ^bb103(%52 : !llvm.ptr)
  ^bb103(%214: !llvm.ptr):  // pred: ^bb102
    cf.br ^bb104(%212, %213, %214 : i64, i128, !llvm.ptr)
  ^bb104(%215: i64, %216: i128, %217: !llvm.ptr):  // pred: ^bb103
    %218 = llvm.load %214 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %212, %213, %218 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb105:  // pred: ^bb53
    cf.br ^bb106(%123 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb106(%219: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb105
    %true_64 = arith.constant true
    %220 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %221 = llvm.insertvalue %true_64, %220[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %222 = llvm.insertvalue %219, %221[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %222, %50 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb107(%117#0 : i64)
  ^bb107(%223: i64):  // pred: ^bb106
    cf.br ^bb108(%57 : i128)
  ^bb108(%224: i128):  // pred: ^bb107
    cf.br ^bb109(%50 : !llvm.ptr)
  ^bb109(%225: !llvm.ptr):  // pred: ^bb108
    cf.br ^bb110(%223, %224, %225 : i64, i128, !llvm.ptr)
  ^bb110(%226: i64, %227: i128, %228: !llvm.ptr):  // pred: ^bb109
    %229 = llvm.load %225 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %223, %224, %229 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb111:  // pred: ^bb39
    cf.br ^bb112(%104 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb112(%230: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb111
    %true_65 = arith.constant true
    %231 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %232 = llvm.insertvalue %true_65, %231[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %233 = llvm.insertvalue %230, %232[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %233, %48 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb113(%98#0 : i64)
  ^bb113(%234: i64):  // pred: ^bb112
    cf.br ^bb114(%57 : i128)
  ^bb114(%235: i128):  // pred: ^bb113
    cf.br ^bb115(%48 : !llvm.ptr)
  ^bb115(%236: !llvm.ptr):  // pred: ^bb114
    cf.br ^bb116(%234, %235, %236 : i64, i128, !llvm.ptr)
  ^bb116(%237: i64, %238: i128, %239: !llvm.ptr):  // pred: ^bb115
    %240 = llvm.load %236 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %234, %235, %240 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb117:  // pred: ^bb25
    cf.br ^bb118(%85 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb118(%241: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb117
    %true_66 = arith.constant true
    %242 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %243 = llvm.insertvalue %true_66, %242[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %244 = llvm.insertvalue %241, %243[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %244, %46 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb119(%79#0 : i64)
  ^bb119(%245: i64):  // pred: ^bb118
    cf.br ^bb120(%57 : i128)
  ^bb120(%246: i128):  // pred: ^bb119
    cf.br ^bb121(%46 : !llvm.ptr)
  ^bb121(%247: !llvm.ptr):  // pred: ^bb120
    cf.br ^bb122(%245, %246, %247 : i64, i128, !llvm.ptr)
  ^bb122(%248: i64, %249: i128, %250: !llvm.ptr):  // pred: ^bb121
    %251 = llvm.load %247 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %245, %246, %251 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb123:  // pred: ^bb10
    cf.br ^bb124(%66 : !llvm.struct<()>)
  ^bb124(%252: !llvm.struct<()>):  // pred: ^bb123
    cf.br ^bb125
  ^bb125:  // pred: ^bb124
    %c0_i252 = arith.constant 0 : i252
    cf.br ^bb126(%62#0 : i64)
  ^bb126(%253: i64):  // pred: ^bb125
    cf.br ^bb127(%c0_i252 : i252)
  ^bb127(%254: i252):  // pred: ^bb126
    cf.br ^bb128(%253, %254 : i64, i252)
  ^bb128(%255: i64, %256: i252):  // pred: ^bb127
    %257:2 = call @"program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<0, 0>>(f18)"(%255, %256) : (i64, i252) -> (i64, !llvm.struct<(i1, array<0 x i8>)>)
    cf.br ^bb129(%257#1 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb129(%258: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb128
    %259 = llvm.extractvalue %258[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.switch %259 : i1, [
      default: ^bb130,
      0: ^bb131,
      1: ^bb132
    ]
  ^bb130:  // pred: ^bb129
    %false_67 = arith.constant false
    cf.assert %false_67, "Invalid enum tag."
    llvm.unreachable
  ^bb131:  // pred: ^bb129
    %260 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb133
  ^bb132:  // pred: ^bb129
    %261 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb245
  ^bb133:  // pred: ^bb131
    cf.br ^bb134(%260 : !llvm.struct<()>)
  ^bb134(%262: !llvm.struct<()>):  // pred: ^bb133
    cf.br ^bb135
  ^bb135:  // pred: ^bb134
    %263 = call @"core::fmt::FormatterDefault::default(f12)"() : () -> !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
    cf.br ^bb136
  ^bb136:  // pred: ^bb135
    %c172180977190876322177717838039515195832848434333118596004974948384901134190_i252 = arith.constant 172180977190876322177717838039515195832848434333118596004974948384901134190 : i252
    cf.br ^bb137
  ^bb137:  // pred: ^bb136
    %c31_i32_68 = arith.constant 31 : i32
    cf.br ^bb138(%263 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>)
  ^bb138(%264: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>):  // pred: ^bb137
    %265 = llvm.extractvalue %264[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)> 
    cf.br ^bb139(%257#0 : i64)
  ^bb139(%266: i64):  // pred: ^bb138
    cf.br ^bb140(%265 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb140(%267: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb139
    cf.br ^bb141(%c172180977190876322177717838039515195832848434333118596004974948384901134190_i252 : i252)
  ^bb141(%268: i252):  // pred: ^bb140
    cf.br ^bb142(%c31_i32_68 : i32)
  ^bb142(%269: i32):  // pred: ^bb141
    cf.br ^bb143(%266, %267, %268, %269 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb143(%270: i64, %271: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %272: i252, %273: i32):  // pred: ^bb142
    %274:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%270, %271, %272, %273) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %274#1, %34 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb144(%34 : !llvm.ptr)
  ^bb144(%275: !llvm.ptr):  // pred: ^bb143
    %276 = llvm.load %275 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %276 : i1, [
      default: ^bb145,
      0: ^bb146,
      1: ^bb147
    ]
  ^bb145:  // pred: ^bb144
    %false_69 = arith.constant false
    cf.assert %false_69, "Invalid enum tag."
    llvm.unreachable
  ^bb146:  // pred: ^bb144
    %277 = llvm.load %275 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %278 = llvm.extractvalue %277[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb148
  ^bb147:  // pred: ^bb144
    %279 = llvm.load %275 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %280 = llvm.extractvalue %279[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb239
  ^bb148:  // pred: ^bb146
    cf.br ^bb149(%278 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb149(%281: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb148
    %282 = llvm.extractvalue %281[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %283 = llvm.extractvalue %281[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb150(%283 : !llvm.struct<()>)
  ^bb150(%284: !llvm.struct<()>):  // pred: ^bb149
    cf.br ^bb151
  ^bb151:  // pred: ^bb150
    %c175590441458062252655054000563808860724590867505131080439014981759796133416_i252 = arith.constant 175590441458062252655054000563808860724590867505131080439014981759796133416 : i252
    cf.br ^bb152
  ^bb152:  // pred: ^bb151
    %c31_i32_70 = arith.constant 31 : i32
    cf.br ^bb153(%274#0 : i64)
  ^bb153(%285: i64):  // pred: ^bb152
    cf.br ^bb154(%282 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb154(%286: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb153
    cf.br ^bb155(%c175590441458062252655054000563808860724590867505131080439014981759796133416_i252 : i252)
  ^bb155(%287: i252):  // pred: ^bb154
    cf.br ^bb156(%c31_i32_70 : i32)
  ^bb156(%288: i32):  // pred: ^bb155
    cf.br ^bb157(%285, %286, %287, %288 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb157(%289: i64, %290: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %291: i252, %292: i32):  // pred: ^bb156
    %293:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%289, %290, %291, %292) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %293#1, %36 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb158(%36 : !llvm.ptr)
  ^bb158(%294: !llvm.ptr):  // pred: ^bb157
    %295 = llvm.load %294 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %295 : i1, [
      default: ^bb159,
      0: ^bb160,
      1: ^bb161
    ]
  ^bb159:  // pred: ^bb158
    %false_71 = arith.constant false
    cf.assert %false_71, "Invalid enum tag."
    llvm.unreachable
  ^bb160:  // pred: ^bb158
    %296 = llvm.load %294 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %297 = llvm.extractvalue %296[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb162
  ^bb161:  // pred: ^bb158
    %298 = llvm.load %294 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %299 = llvm.extractvalue %298[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb233
  ^bb162:  // pred: ^bb160
    cf.br ^bb163(%297 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb163(%300: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb162
    %301 = llvm.extractvalue %300[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %302 = llvm.extractvalue %300[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb164(%302 : !llvm.struct<()>)
  ^bb164(%303: !llvm.struct<()>):  // pred: ^bb163
    cf.br ^bb165
  ^bb165:  // pred: ^bb164
    %c808017966_i252 = arith.constant 808017966 : i252
    cf.br ^bb166
  ^bb166:  // pred: ^bb165
    %c4_i32 = arith.constant 4 : i32
    cf.br ^bb167(%293#0 : i64)
  ^bb167(%304: i64):  // pred: ^bb166
    cf.br ^bb168(%301 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb168(%305: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb167
    cf.br ^bb169(%c808017966_i252 : i252)
  ^bb169(%306: i252):  // pred: ^bb168
    cf.br ^bb170(%c4_i32 : i32)
  ^bb170(%307: i32):  // pred: ^bb169
    cf.br ^bb171(%304, %305, %306, %307 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb171(%308: i64, %309: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %310: i252, %311: i32):  // pred: ^bb170
    %312:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%308, %309, %310, %311) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %312#1, %38 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb172(%38 : !llvm.ptr)
  ^bb172(%313: !llvm.ptr):  // pred: ^bb171
    %314 = llvm.load %313 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %314 : i1, [
      default: ^bb173,
      0: ^bb174,
      1: ^bb175
    ]
  ^bb173:  // pred: ^bb172
    %false_72 = arith.constant false
    cf.assert %false_72, "Invalid enum tag."
    llvm.unreachable
  ^bb174:  // pred: ^bb172
    %315 = llvm.load %313 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %316 = llvm.extractvalue %315[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb176
  ^bb175:  // pred: ^bb172
    %317 = llvm.load %313 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %318 = llvm.extractvalue %317[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb227
  ^bb176:  // pred: ^bb174
    cf.br ^bb177
  ^bb177:  // pred: ^bb176
    %319 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb178(%319 : !llvm.struct<()>)
  ^bb178(%320: !llvm.struct<()>):  // pred: ^bb177
    %false_73 = arith.constant false
    %321 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %322 = llvm.insertvalue %false_73, %321[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb179(%322 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb179(%323: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb178
    cf.br ^bb180(%323 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb180(%324: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb179
    call @"core::result::ResultTraitImpl::<(), core::fmt::Error>::unwrap::<core::fmt::ErrorDrop>(f1)"(%40, %324) : (!llvm.ptr, !llvm.struct<(i1, array<0 x i8>)>) -> ()
    %325 = llvm.getelementptr %40[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb181(%325 : !llvm.ptr)
  ^bb181(%326: !llvm.ptr):  // pred: ^bb180
    %327 = llvm.load %326 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %327 : i1, [
      default: ^bb182,
      0: ^bb183,
      1: ^bb184
    ]
  ^bb182:  // pred: ^bb181
    %false_74 = arith.constant false
    cf.assert %false_74, "Invalid enum tag."
    llvm.unreachable
  ^bb183:  // pred: ^bb181
    %328 = llvm.load %326 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>)>)>
    %329 = llvm.extractvalue %328[1] : !llvm.struct<(i1, struct<(struct<()>)>)> 
    cf.br ^bb185
  ^bb184:  // pred: ^bb181
    %330 = llvm.load %326 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %331 = llvm.extractvalue %330[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb220
  ^bb185:  // pred: ^bb183
    cf.br ^bb186(%329 : !llvm.struct<(struct<()>)>)
  ^bb186(%332: !llvm.struct<(struct<()>)>):  // pred: ^bb185
    cf.br ^bb187
  ^bb187:  // pred: ^bb186
    %333 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_75 = arith.constant 0 : i32
    %334 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %335 = llvm.insertvalue %333, %334[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %336 = llvm.insertvalue %c0_i32_75, %335[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %337 = llvm.insertvalue %c0_i32_75, %336[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb188
  ^bb188:  // pred: ^bb187
    %c1997209042069643135709344952807065910992472029923670688473712229447419591075_i252_76 = arith.constant 1997209042069643135709344952807065910992472029923670688473712229447419591075 : i252
    cf.br ^bb189(%c1997209042069643135709344952807065910992472029923670688473712229447419591075_i252_76 : i252)
  ^bb189(%338: i252):  // pred: ^bb188
    cf.br ^bb190(%337, %338 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb190(%339: !llvm.struct<(ptr<i252>, i32, i32)>, %340: i252):  // pred: ^bb189
    %341 = llvm.extractvalue %339[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %342 = llvm.extractvalue %339[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %343 = llvm.extractvalue %339[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %344 = arith.cmpi uge, %342, %343 : i32
    %345:2 = scf.if %344 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32_146 = arith.constant 8 : i32
      %1047 = arith.addi %343, %343 : i32
      %1048 = arith.maxui %c8_i32_146, %1047 : i32
      %1049 = arith.extui %1048 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1050 = arith.muli %1049, %c32_i64 : i64
      %1051 = llvm.bitcast %341 : !llvm.ptr<i252> to !llvm.ptr
      %1052 = func.call @realloc(%1051, %1050) : (!llvm.ptr, i64) -> !llvm.ptr
      %1053 = llvm.bitcast %1052 : !llvm.ptr to !llvm.ptr<i252>
      %1054 = llvm.insertvalue %1053, %339[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1055 = llvm.insertvalue %1048, %1054[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1055, %1053 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %339, %341 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %346 = llvm.getelementptr %345#1[%342] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %340, %346 : i252, !llvm.ptr
    %c1_i32_77 = arith.constant 1 : i32
    %347 = arith.addi %342, %c1_i32_77 : i32
    %348 = llvm.insertvalue %347, %345#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb191(%316 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb191(%349: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb190
    %350 = llvm.extractvalue %349[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %351 = llvm.extractvalue %349[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb192(%351 : !llvm.struct<()>)
  ^bb192(%352: !llvm.struct<()>):  // pred: ^bb191
    cf.br ^bb193(%350 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb193(%353: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb192
    cf.br ^bb194(%353 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb194(%354: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb193
    cf.br ^bb195(%312#0 : i64)
  ^bb195(%355: i64):  // pred: ^bb194
    cf.br ^bb196(%57 : i128)
  ^bb196(%356: i128):  // pred: ^bb195
    cf.br ^bb197(%353 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb197(%357: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb196
    cf.br ^bb198(%348 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb198(%358: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb197
    cf.br ^bb199(%355, %356, %357, %358 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb199(%359: i64, %360: i128, %361: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %362: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb198
    %363:3 = call @"core::byte_array::ByteArraySerde::serialize(f0)"(%359, %360, %361, %362) : (i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %363#2, %42 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb200(%42 : !llvm.ptr)
  ^bb200(%364: !llvm.ptr):  // pred: ^bb199
    %365 = llvm.load %364 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %365 : i1, [
      default: ^bb201,
      0: ^bb202,
      1: ^bb203
    ]
  ^bb201:  // pred: ^bb200
    %false_78 = arith.constant false
    cf.assert %false_78, "Invalid enum tag."
    llvm.unreachable
  ^bb202:  // pred: ^bb200
    %366 = llvm.load %364 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %367 = llvm.extractvalue %366[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    cf.br ^bb204
  ^bb203:  // pred: ^bb200
    %368 = llvm.load %364 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %369 = llvm.extractvalue %368[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb214
  ^bb204:  // pred: ^bb202
    cf.br ^bb205(%367 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)
  ^bb205(%370: !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>):  // pred: ^bb204
    %371 = llvm.extractvalue %370[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %372 = llvm.extractvalue %370[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    cf.br ^bb206(%372 : !llvm.struct<()>)
  ^bb206(%373: !llvm.struct<()>):  // pred: ^bb205
    cf.br ^bb207
  ^bb207:  // pred: ^bb206
    %374 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb208(%374, %371 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb208(%375: !llvm.struct<()>, %376: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb207
    %377 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %378 = llvm.insertvalue %375, %377[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %379 = llvm.insertvalue %376, %378[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb209(%379 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb209(%380: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb208
    %true_79 = arith.constant true
    %381 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %382 = llvm.insertvalue %true_79, %381[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %383 = llvm.insertvalue %380, %382[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %383, %44 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb210(%363#0 : i64)
  ^bb210(%384: i64):  // pred: ^bb209
    cf.br ^bb211(%363#1 : i128)
  ^bb211(%385: i128):  // pred: ^bb210
    cf.br ^bb212(%44 : !llvm.ptr)
  ^bb212(%386: !llvm.ptr):  // pred: ^bb211
    cf.br ^bb213(%384, %385, %386 : i64, i128, !llvm.ptr)
  ^bb213(%387: i64, %388: i128, %389: !llvm.ptr):  // pred: ^bb212
    %390 = llvm.load %386 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %384, %385, %390 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb214:  // pred: ^bb203
    cf.br ^bb215(%369 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb215(%391: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb214
    %true_80 = arith.constant true
    %392 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %393 = llvm.insertvalue %true_80, %392[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %394 = llvm.insertvalue %391, %393[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %394, %43 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb216(%363#0 : i64)
  ^bb216(%395: i64):  // pred: ^bb215
    cf.br ^bb217(%363#1 : i128)
  ^bb217(%396: i128):  // pred: ^bb216
    cf.br ^bb218(%43 : !llvm.ptr)
  ^bb218(%397: !llvm.ptr):  // pred: ^bb217
    cf.br ^bb219(%395, %396, %397 : i64, i128, !llvm.ptr)
  ^bb219(%398: i64, %399: i128, %400: !llvm.ptr):  // pred: ^bb218
    %401 = llvm.load %397 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %395, %396, %401 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb220:  // pred: ^bb184
    cf.br ^bb221(%316 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb221(%402: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb220
    cf.br ^bb222(%331 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb222(%403: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb221
    %true_81 = arith.constant true
    %404 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %405 = llvm.insertvalue %true_81, %404[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %406 = llvm.insertvalue %403, %405[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %406, %41 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb223(%312#0 : i64)
  ^bb223(%407: i64):  // pred: ^bb222
    cf.br ^bb224(%57 : i128)
  ^bb224(%408: i128):  // pred: ^bb223
    cf.br ^bb225(%41 : !llvm.ptr)
  ^bb225(%409: !llvm.ptr):  // pred: ^bb224
    cf.br ^bb226(%407, %408, %409 : i64, i128, !llvm.ptr)
  ^bb226(%410: i64, %411: i128, %412: !llvm.ptr):  // pred: ^bb225
    %413 = llvm.load %409 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %407, %408, %413 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb227:  // pred: ^bb175
    cf.br ^bb228(%318 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb228(%414: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb227
    %true_82 = arith.constant true
    %415 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %416 = llvm.insertvalue %true_82, %415[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %417 = llvm.insertvalue %414, %416[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %417, %39 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb229(%312#0 : i64)
  ^bb229(%418: i64):  // pred: ^bb228
    cf.br ^bb230(%57 : i128)
  ^bb230(%419: i128):  // pred: ^bb229
    cf.br ^bb231(%39 : !llvm.ptr)
  ^bb231(%420: !llvm.ptr):  // pred: ^bb230
    cf.br ^bb232(%418, %419, %420 : i64, i128, !llvm.ptr)
  ^bb232(%421: i64, %422: i128, %423: !llvm.ptr):  // pred: ^bb231
    %424 = llvm.load %420 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %418, %419, %424 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb233:  // pred: ^bb161
    cf.br ^bb234(%299 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb234(%425: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb233
    %true_83 = arith.constant true
    %426 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %427 = llvm.insertvalue %true_83, %426[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %428 = llvm.insertvalue %425, %427[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %428, %37 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb235(%293#0 : i64)
  ^bb235(%429: i64):  // pred: ^bb234
    cf.br ^bb236(%57 : i128)
  ^bb236(%430: i128):  // pred: ^bb235
    cf.br ^bb237(%37 : !llvm.ptr)
  ^bb237(%431: !llvm.ptr):  // pred: ^bb236
    cf.br ^bb238(%429, %430, %431 : i64, i128, !llvm.ptr)
  ^bb238(%432: i64, %433: i128, %434: !llvm.ptr):  // pred: ^bb237
    %435 = llvm.load %431 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %429, %430, %435 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb239:  // pred: ^bb147
    cf.br ^bb240(%280 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb240(%436: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb239
    %true_84 = arith.constant true
    %437 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %438 = llvm.insertvalue %true_84, %437[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %439 = llvm.insertvalue %436, %438[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %439, %35 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb241(%274#0 : i64)
  ^bb241(%440: i64):  // pred: ^bb240
    cf.br ^bb242(%57 : i128)
  ^bb242(%441: i128):  // pred: ^bb241
    cf.br ^bb243(%35 : !llvm.ptr)
  ^bb243(%442: !llvm.ptr):  // pred: ^bb242
    cf.br ^bb244(%440, %441, %442 : i64, i128, !llvm.ptr)
  ^bb244(%443: i64, %444: i128, %445: !llvm.ptr):  // pred: ^bb243
    %446 = llvm.load %442 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %440, %441, %446 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb245:  // pred: ^bb132
    cf.br ^bb246(%261 : !llvm.struct<()>)
  ^bb246(%447: !llvm.struct<()>):  // pred: ^bb245
    cf.br ^bb247
  ^bb247:  // pred: ^bb246
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582016_i252 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582016 : i252
    cf.br ^bb248(%257#0 : i64)
  ^bb248(%448: i64):  // pred: ^bb247
    cf.br ^bb249(%c-3618502788666131000275863779947924135206266826270938552493006944358698582016_i252 : i252)
  ^bb249(%449: i252):  // pred: ^bb248
    cf.br ^bb250(%448, %449 : i64, i252)
  ^bb250(%450: i64, %451: i252):  // pred: ^bb249
    %452:2 = call @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<0, 0>>(f17)"(%450, %451) : (i64, i252) -> (i64, !llvm.struct<(i1, array<0 x i8>)>)
    cf.br ^bb251(%452#1 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb251(%453: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb250
    %454 = llvm.extractvalue %453[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.switch %454 : i1, [
      default: ^bb252,
      0: ^bb253,
      1: ^bb254
    ]
  ^bb252:  // pred: ^bb251
    %false_85 = arith.constant false
    cf.assert %false_85, "Invalid enum tag."
    llvm.unreachable
  ^bb253:  // pred: ^bb251
    %455 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb255
  ^bb254:  // pred: ^bb251
    %456 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb367
  ^bb255:  // pred: ^bb253
    cf.br ^bb256(%455 : !llvm.struct<()>)
  ^bb256(%457: !llvm.struct<()>):  // pred: ^bb255
    cf.br ^bb257
  ^bb257:  // pred: ^bb256
    %458 = call @"core::fmt::FormatterDefault::default(f12)"() : () -> !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
    cf.br ^bb258
  ^bb258:  // pred: ^bb257
    %c172180977190876322177717838039515195832848434332511767082422530228238249590_i252_86 = arith.constant 172180977190876322177717838039515195832848434332511767082422530228238249590 : i252
    cf.br ^bb259
  ^bb259:  // pred: ^bb258
    %c31_i32_87 = arith.constant 31 : i32
    cf.br ^bb260(%458 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>)
  ^bb260(%459: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>):  // pred: ^bb259
    %460 = llvm.extractvalue %459[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)> 
    cf.br ^bb261(%452#0 : i64)
  ^bb261(%461: i64):  // pred: ^bb260
    cf.br ^bb262(%460 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb262(%462: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb261
    cf.br ^bb263(%c172180977190876322177717838039515195832848434332511767082422530228238249590_i252_86 : i252)
  ^bb263(%463: i252):  // pred: ^bb262
    cf.br ^bb264(%c31_i32_87 : i32)
  ^bb264(%464: i32):  // pred: ^bb263
    cf.br ^bb265(%461, %462, %463, %464 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb265(%465: i64, %466: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %467: i252, %468: i32):  // pred: ^bb264
    %469:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%465, %466, %467, %468) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %469#1, %23 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb266(%23 : !llvm.ptr)
  ^bb266(%470: !llvm.ptr):  // pred: ^bb265
    %471 = llvm.load %470 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %471 : i1, [
      default: ^bb267,
      0: ^bb268,
      1: ^bb269
    ]
  ^bb267:  // pred: ^bb266
    %false_88 = arith.constant false
    cf.assert %false_88, "Invalid enum tag."
    llvm.unreachable
  ^bb268:  // pred: ^bb266
    %472 = llvm.load %470 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %473 = llvm.extractvalue %472[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb270
  ^bb269:  // pred: ^bb266
    %474 = llvm.load %470 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %475 = llvm.extractvalue %474[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb361
  ^bb270:  // pred: ^bb268
    cf.br ^bb271(%473 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb271(%476: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb270
    %477 = llvm.extractvalue %476[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %478 = llvm.extractvalue %476[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb272(%478 : !llvm.struct<()>)
  ^bb272(%479: !llvm.struct<()>):  // pred: ^bb271
    cf.br ^bb273
  ^bb273:  // pred: ^bb272
    %c172132395238539276156095731494591481422342884673560379810407770160236339248_i252_89 = arith.constant 172132395238539276156095731494591481422342884673560379810407770160236339248 : i252
    cf.br ^bb274
  ^bb274:  // pred: ^bb273
    %c31_i32_90 = arith.constant 31 : i32
    cf.br ^bb275(%469#0 : i64)
  ^bb275(%480: i64):  // pred: ^bb274
    cf.br ^bb276(%477 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb276(%481: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb275
    cf.br ^bb277(%c172132395238539276156095731494591481422342884673560379810407770160236339248_i252_89 : i252)
  ^bb277(%482: i252):  // pred: ^bb276
    cf.br ^bb278(%c31_i32_90 : i32)
  ^bb278(%483: i32):  // pred: ^bb277
    cf.br ^bb279(%480, %481, %482, %483 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb279(%484: i64, %485: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %486: i252, %487: i32):  // pred: ^bb278
    %488:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%484, %485, %486, %487) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %488#1, %25 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb280(%25 : !llvm.ptr)
  ^bb280(%489: !llvm.ptr):  // pred: ^bb279
    %490 = llvm.load %489 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %490 : i1, [
      default: ^bb281,
      0: ^bb282,
      1: ^bb283
    ]
  ^bb281:  // pred: ^bb280
    %false_91 = arith.constant false
    cf.assert %false_91, "Invalid enum tag."
    llvm.unreachable
  ^bb282:  // pred: ^bb280
    %491 = llvm.load %489 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %492 = llvm.extractvalue %491[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb284
  ^bb283:  // pred: ^bb280
    %493 = llvm.load %489 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %494 = llvm.extractvalue %493[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb355
  ^bb284:  // pred: ^bb282
    cf.br ^bb285(%492 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb285(%495: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb284
    %496 = llvm.extractvalue %495[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %497 = llvm.extractvalue %495[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb286(%497 : !llvm.struct<()>)
  ^bb286(%498: !llvm.struct<()>):  // pred: ^bb285
    cf.br ^bb287
  ^bb287:  // pred: ^bb286
    %c4485066453471027246_i252 = arith.constant 4485066453471027246 : i252
    cf.br ^bb288
  ^bb288:  // pred: ^bb287
    %c8_i32 = arith.constant 8 : i32
    cf.br ^bb289(%488#0 : i64)
  ^bb289(%499: i64):  // pred: ^bb288
    cf.br ^bb290(%496 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb290(%500: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb289
    cf.br ^bb291(%c4485066453471027246_i252 : i252)
  ^bb291(%501: i252):  // pred: ^bb290
    cf.br ^bb292(%c8_i32 : i32)
  ^bb292(%502: i32):  // pred: ^bb291
    cf.br ^bb293(%499, %500, %501, %502 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb293(%503: i64, %504: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %505: i252, %506: i32):  // pred: ^bb292
    %507:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%503, %504, %505, %506) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %507#1, %27 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb294(%27 : !llvm.ptr)
  ^bb294(%508: !llvm.ptr):  // pred: ^bb293
    %509 = llvm.load %508 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %509 : i1, [
      default: ^bb295,
      0: ^bb296,
      1: ^bb297
    ]
  ^bb295:  // pred: ^bb294
    %false_92 = arith.constant false
    cf.assert %false_92, "Invalid enum tag."
    llvm.unreachable
  ^bb296:  // pred: ^bb294
    %510 = llvm.load %508 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %511 = llvm.extractvalue %510[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb298
  ^bb297:  // pred: ^bb294
    %512 = llvm.load %508 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %513 = llvm.extractvalue %512[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb349
  ^bb298:  // pred: ^bb296
    cf.br ^bb299
  ^bb299:  // pred: ^bb298
    %514 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb300(%514 : !llvm.struct<()>)
  ^bb300(%515: !llvm.struct<()>):  // pred: ^bb299
    %false_93 = arith.constant false
    %516 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %517 = llvm.insertvalue %false_93, %516[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb301(%517 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb301(%518: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb300
    cf.br ^bb302(%518 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb302(%519: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb301
    call @"core::result::ResultTraitImpl::<(), core::fmt::Error>::unwrap::<core::fmt::ErrorDrop>(f1)"(%29, %519) : (!llvm.ptr, !llvm.struct<(i1, array<0 x i8>)>) -> ()
    %520 = llvm.getelementptr %29[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb303(%520 : !llvm.ptr)
  ^bb303(%521: !llvm.ptr):  // pred: ^bb302
    %522 = llvm.load %521 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %522 : i1, [
      default: ^bb304,
      0: ^bb305,
      1: ^bb306
    ]
  ^bb304:  // pred: ^bb303
    %false_94 = arith.constant false
    cf.assert %false_94, "Invalid enum tag."
    llvm.unreachable
  ^bb305:  // pred: ^bb303
    %523 = llvm.load %521 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>)>)>
    %524 = llvm.extractvalue %523[1] : !llvm.struct<(i1, struct<(struct<()>)>)> 
    cf.br ^bb307
  ^bb306:  // pred: ^bb303
    %525 = llvm.load %521 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %526 = llvm.extractvalue %525[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb342
  ^bb307:  // pred: ^bb305
    cf.br ^bb308(%524 : !llvm.struct<(struct<()>)>)
  ^bb308(%527: !llvm.struct<(struct<()>)>):  // pred: ^bb307
    cf.br ^bb309
  ^bb309:  // pred: ^bb308
    %528 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_95 = arith.constant 0 : i32
    %529 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %530 = llvm.insertvalue %528, %529[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %531 = llvm.insertvalue %c0_i32_95, %530[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %532 = llvm.insertvalue %c0_i32_95, %531[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb310
  ^bb310:  // pred: ^bb309
    %c1997209042069643135709344952807065910992472029923670688473712229447419591075_i252_96 = arith.constant 1997209042069643135709344952807065910992472029923670688473712229447419591075 : i252
    cf.br ^bb311(%c1997209042069643135709344952807065910992472029923670688473712229447419591075_i252_96 : i252)
  ^bb311(%533: i252):  // pred: ^bb310
    cf.br ^bb312(%532, %533 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb312(%534: !llvm.struct<(ptr<i252>, i32, i32)>, %535: i252):  // pred: ^bb311
    %536 = llvm.extractvalue %534[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %537 = llvm.extractvalue %534[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %538 = llvm.extractvalue %534[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %539 = arith.cmpi uge, %537, %538 : i32
    %540:2 = scf.if %539 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32_146 = arith.constant 8 : i32
      %1047 = arith.addi %538, %538 : i32
      %1048 = arith.maxui %c8_i32_146, %1047 : i32
      %1049 = arith.extui %1048 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1050 = arith.muli %1049, %c32_i64 : i64
      %1051 = llvm.bitcast %536 : !llvm.ptr<i252> to !llvm.ptr
      %1052 = func.call @realloc(%1051, %1050) : (!llvm.ptr, i64) -> !llvm.ptr
      %1053 = llvm.bitcast %1052 : !llvm.ptr to !llvm.ptr<i252>
      %1054 = llvm.insertvalue %1053, %534[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1055 = llvm.insertvalue %1048, %1054[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1055, %1053 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %534, %536 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %541 = llvm.getelementptr %540#1[%537] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %535, %541 : i252, !llvm.ptr
    %c1_i32_97 = arith.constant 1 : i32
    %542 = arith.addi %537, %c1_i32_97 : i32
    %543 = llvm.insertvalue %542, %540#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb313(%511 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb313(%544: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb312
    %545 = llvm.extractvalue %544[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %546 = llvm.extractvalue %544[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb314(%546 : !llvm.struct<()>)
  ^bb314(%547: !llvm.struct<()>):  // pred: ^bb313
    cf.br ^bb315(%545 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb315(%548: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb314
    cf.br ^bb316(%548 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb316(%549: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb315
    cf.br ^bb317(%507#0 : i64)
  ^bb317(%550: i64):  // pred: ^bb316
    cf.br ^bb318(%57 : i128)
  ^bb318(%551: i128):  // pred: ^bb317
    cf.br ^bb319(%548 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb319(%552: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb318
    cf.br ^bb320(%543 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb320(%553: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb319
    cf.br ^bb321(%550, %551, %552, %553 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb321(%554: i64, %555: i128, %556: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %557: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb320
    %558:3 = call @"core::byte_array::ByteArraySerde::serialize(f0)"(%554, %555, %556, %557) : (i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %558#2, %31 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb322(%31 : !llvm.ptr)
  ^bb322(%559: !llvm.ptr):  // pred: ^bb321
    %560 = llvm.load %559 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %560 : i1, [
      default: ^bb323,
      0: ^bb324,
      1: ^bb325
    ]
  ^bb323:  // pred: ^bb322
    %false_98 = arith.constant false
    cf.assert %false_98, "Invalid enum tag."
    llvm.unreachable
  ^bb324:  // pred: ^bb322
    %561 = llvm.load %559 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %562 = llvm.extractvalue %561[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    cf.br ^bb326
  ^bb325:  // pred: ^bb322
    %563 = llvm.load %559 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %564 = llvm.extractvalue %563[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb336
  ^bb326:  // pred: ^bb324
    cf.br ^bb327(%562 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)
  ^bb327(%565: !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>):  // pred: ^bb326
    %566 = llvm.extractvalue %565[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %567 = llvm.extractvalue %565[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    cf.br ^bb328(%567 : !llvm.struct<()>)
  ^bb328(%568: !llvm.struct<()>):  // pred: ^bb327
    cf.br ^bb329
  ^bb329:  // pred: ^bb328
    %569 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb330(%569, %566 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb330(%570: !llvm.struct<()>, %571: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb329
    %572 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %573 = llvm.insertvalue %570, %572[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %574 = llvm.insertvalue %571, %573[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb331(%574 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb331(%575: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb330
    %true_99 = arith.constant true
    %576 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %577 = llvm.insertvalue %true_99, %576[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %578 = llvm.insertvalue %575, %577[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %578, %33 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb332(%558#0 : i64)
  ^bb332(%579: i64):  // pred: ^bb331
    cf.br ^bb333(%558#1 : i128)
  ^bb333(%580: i128):  // pred: ^bb332
    cf.br ^bb334(%33 : !llvm.ptr)
  ^bb334(%581: !llvm.ptr):  // pred: ^bb333
    cf.br ^bb335(%579, %580, %581 : i64, i128, !llvm.ptr)
  ^bb335(%582: i64, %583: i128, %584: !llvm.ptr):  // pred: ^bb334
    %585 = llvm.load %581 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %579, %580, %585 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb336:  // pred: ^bb325
    cf.br ^bb337(%564 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb337(%586: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb336
    %true_100 = arith.constant true
    %587 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %588 = llvm.insertvalue %true_100, %587[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %589 = llvm.insertvalue %586, %588[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %589, %32 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb338(%558#0 : i64)
  ^bb338(%590: i64):  // pred: ^bb337
    cf.br ^bb339(%558#1 : i128)
  ^bb339(%591: i128):  // pred: ^bb338
    cf.br ^bb340(%32 : !llvm.ptr)
  ^bb340(%592: !llvm.ptr):  // pred: ^bb339
    cf.br ^bb341(%590, %591, %592 : i64, i128, !llvm.ptr)
  ^bb341(%593: i64, %594: i128, %595: !llvm.ptr):  // pred: ^bb340
    %596 = llvm.load %592 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %590, %591, %596 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb342:  // pred: ^bb306
    cf.br ^bb343(%511 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb343(%597: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb342
    cf.br ^bb344(%526 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb344(%598: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb343
    %true_101 = arith.constant true
    %599 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %600 = llvm.insertvalue %true_101, %599[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %601 = llvm.insertvalue %598, %600[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %601, %30 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb345(%507#0 : i64)
  ^bb345(%602: i64):  // pred: ^bb344
    cf.br ^bb346(%57 : i128)
  ^bb346(%603: i128):  // pred: ^bb345
    cf.br ^bb347(%30 : !llvm.ptr)
  ^bb347(%604: !llvm.ptr):  // pred: ^bb346
    cf.br ^bb348(%602, %603, %604 : i64, i128, !llvm.ptr)
  ^bb348(%605: i64, %606: i128, %607: !llvm.ptr):  // pred: ^bb347
    %608 = llvm.load %604 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %602, %603, %608 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb349:  // pred: ^bb297
    cf.br ^bb350(%513 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb350(%609: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb349
    %true_102 = arith.constant true
    %610 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %611 = llvm.insertvalue %true_102, %610[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %612 = llvm.insertvalue %609, %611[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %612, %28 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb351(%507#0 : i64)
  ^bb351(%613: i64):  // pred: ^bb350
    cf.br ^bb352(%57 : i128)
  ^bb352(%614: i128):  // pred: ^bb351
    cf.br ^bb353(%28 : !llvm.ptr)
  ^bb353(%615: !llvm.ptr):  // pred: ^bb352
    cf.br ^bb354(%613, %614, %615 : i64, i128, !llvm.ptr)
  ^bb354(%616: i64, %617: i128, %618: !llvm.ptr):  // pred: ^bb353
    %619 = llvm.load %615 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %613, %614, %619 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb355:  // pred: ^bb283
    cf.br ^bb356(%494 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb356(%620: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb355
    %true_103 = arith.constant true
    %621 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %622 = llvm.insertvalue %true_103, %621[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %623 = llvm.insertvalue %620, %622[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %623, %26 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb357(%488#0 : i64)
  ^bb357(%624: i64):  // pred: ^bb356
    cf.br ^bb358(%57 : i128)
  ^bb358(%625: i128):  // pred: ^bb357
    cf.br ^bb359(%26 : !llvm.ptr)
  ^bb359(%626: !llvm.ptr):  // pred: ^bb358
    cf.br ^bb360(%624, %625, %626 : i64, i128, !llvm.ptr)
  ^bb360(%627: i64, %628: i128, %629: !llvm.ptr):  // pred: ^bb359
    %630 = llvm.load %626 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %624, %625, %630 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb361:  // pred: ^bb269
    cf.br ^bb362(%475 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb362(%631: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb361
    %true_104 = arith.constant true
    %632 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %633 = llvm.insertvalue %true_104, %632[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %634 = llvm.insertvalue %631, %633[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %634, %24 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb363(%469#0 : i64)
  ^bb363(%635: i64):  // pred: ^bb362
    cf.br ^bb364(%57 : i128)
  ^bb364(%636: i128):  // pred: ^bb363
    cf.br ^bb365(%24 : !llvm.ptr)
  ^bb365(%637: !llvm.ptr):  // pred: ^bb364
    cf.br ^bb366(%635, %636, %637 : i64, i128, !llvm.ptr)
  ^bb366(%638: i64, %639: i128, %640: !llvm.ptr):  // pred: ^bb365
    %641 = llvm.load %637 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %635, %636, %641 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb367:  // pred: ^bb254
    cf.br ^bb368(%456 : !llvm.struct<()>)
  ^bb368(%642: !llvm.struct<()>):  // pred: ^bb367
    cf.br ^bb369
  ^bb369:  // pred: ^bb368
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582017_i252 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582017 : i252
    cf.br ^bb370(%452#0 : i64)
  ^bb370(%643: i64):  // pred: ^bb369
    cf.br ^bb371(%c-3618502788666131000275863779947924135206266826270938552493006944358698582017_i252 : i252)
  ^bb371(%644: i252):  // pred: ^bb370
    cf.br ^bb372(%643, %644 : i64, i252)
  ^bb372(%645: i64, %646: i252):  // pred: ^bb371
    %647:2 = call @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<-1, -1>>(f16)"(%645, %646) : (i64, i252) -> (i64, !llvm.struct<(i1, array<0 x i8>)>)
    cf.br ^bb373(%647#1 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb373(%648: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb372
    %649 = llvm.extractvalue %648[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.switch %649 : i1, [
      default: ^bb374,
      0: ^bb375,
      1: ^bb376
    ]
  ^bb374:  // pred: ^bb373
    %false_105 = arith.constant false
    cf.assert %false_105, "Invalid enum tag."
    llvm.unreachable
  ^bb375:  // pred: ^bb373
    %650 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb377
  ^bb376:  // pred: ^bb373
    %651 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb489
  ^bb377:  // pred: ^bb375
    cf.br ^bb378(%650 : !llvm.struct<()>)
  ^bb378(%652: !llvm.struct<()>):  // pred: ^bb377
    cf.br ^bb379
  ^bb379:  // pred: ^bb378
    %653 = call @"core::fmt::FormatterDefault::default(f12)"() : () -> !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
    cf.br ^bb380
  ^bb380:  // pred: ^bb379
    %c172180977190876322177717838039515195832848434332511767082422530228238249590_i252_106 = arith.constant 172180977190876322177717838039515195832848434332511767082422530228238249590 : i252
    cf.br ^bb381
  ^bb381:  // pred: ^bb380
    %c31_i32_107 = arith.constant 31 : i32
    cf.br ^bb382(%653 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>)
  ^bb382(%654: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>):  // pred: ^bb381
    %655 = llvm.extractvalue %654[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)> 
    cf.br ^bb383(%647#0 : i64)
  ^bb383(%656: i64):  // pred: ^bb382
    cf.br ^bb384(%655 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb384(%657: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb383
    cf.br ^bb385(%c172180977190876322177717838039515195832848434332511767082422530228238249590_i252_106 : i252)
  ^bb385(%658: i252):  // pred: ^bb384
    cf.br ^bb386(%c31_i32_107 : i32)
  ^bb386(%659: i32):  // pred: ^bb385
    cf.br ^bb387(%656, %657, %658, %659 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb387(%660: i64, %661: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %662: i252, %663: i32):  // pred: ^bb386
    %664:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%660, %661, %662, %663) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %664#1, %12 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb388(%12 : !llvm.ptr)
  ^bb388(%665: !llvm.ptr):  // pred: ^bb387
    %666 = llvm.load %665 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %666 : i1, [
      default: ^bb389,
      0: ^bb390,
      1: ^bb391
    ]
  ^bb389:  // pred: ^bb388
    %false_108 = arith.constant false
    cf.assert %false_108, "Invalid enum tag."
    llvm.unreachable
  ^bb390:  // pred: ^bb388
    %667 = llvm.load %665 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %668 = llvm.extractvalue %667[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb392
  ^bb391:  // pred: ^bb388
    %669 = llvm.load %665 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %670 = llvm.extractvalue %669[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb483
  ^bb392:  // pred: ^bb390
    cf.br ^bb393(%668 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb393(%671: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb392
    %672 = llvm.extractvalue %671[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %673 = llvm.extractvalue %671[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb394(%673 : !llvm.struct<()>)
  ^bb394(%674: !llvm.struct<()>):  // pred: ^bb393
    cf.br ^bb395
  ^bb395:  // pred: ^bb394
    %c172132395238539276156095731494591481422342884673560379810407770160186338336_i252 = arith.constant 172132395238539276156095731494591481422342884673560379810407770160186338336 : i252
    cf.br ^bb396
  ^bb396:  // pred: ^bb395
    %c31_i32_109 = arith.constant 31 : i32
    cf.br ^bb397(%664#0 : i64)
  ^bb397(%675: i64):  // pred: ^bb396
    cf.br ^bb398(%672 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb398(%676: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb397
    cf.br ^bb399(%c172132395238539276156095731494591481422342884673560379810407770160186338336_i252 : i252)
  ^bb399(%677: i252):  // pred: ^bb398
    cf.br ^bb400(%c31_i32_109 : i32)
  ^bb400(%678: i32):  // pred: ^bb399
    cf.br ^bb401(%675, %676, %677, %678 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb401(%679: i64, %680: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %681: i252, %682: i32):  // pred: ^bb400
    %683:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%679, %680, %681, %682) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %683#1, %14 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb402(%14 : !llvm.ptr)
  ^bb402(%684: !llvm.ptr):  // pred: ^bb401
    %685 = llvm.load %684 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %685 : i1, [
      default: ^bb403,
      0: ^bb404,
      1: ^bb405
    ]
  ^bb403:  // pred: ^bb402
    %false_110 = arith.constant false
    cf.assert %false_110, "Invalid enum tag."
    llvm.unreachable
  ^bb404:  // pred: ^bb402
    %686 = llvm.load %684 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %687 = llvm.extractvalue %686[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb406
  ^bb405:  // pred: ^bb402
    %688 = llvm.load %684 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %689 = llvm.extractvalue %688[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb477
  ^bb406:  // pred: ^bb404
    cf.br ^bb407(%687 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb407(%690: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb406
    %691 = llvm.extractvalue %690[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %692 = llvm.extractvalue %690[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb408(%692 : !llvm.struct<()>)
  ^bb408(%693: !llvm.struct<()>):  // pred: ^bb407
    cf.br ^bb409
  ^bb409:  // pred: ^bb408
    %c213414867255199290449966_i252 = arith.constant 213414867255199290449966 : i252
    cf.br ^bb410
  ^bb410:  // pred: ^bb409
    %c10_i32 = arith.constant 10 : i32
    cf.br ^bb411(%683#0 : i64)
  ^bb411(%694: i64):  // pred: ^bb410
    cf.br ^bb412(%691 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb412(%695: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb411
    cf.br ^bb413(%c213414867255199290449966_i252 : i252)
  ^bb413(%696: i252):  // pred: ^bb412
    cf.br ^bb414(%c10_i32 : i32)
  ^bb414(%697: i32):  // pred: ^bb413
    cf.br ^bb415(%694, %695, %696, %697 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb415(%698: i64, %699: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %700: i252, %701: i32):  // pred: ^bb414
    %702:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%698, %699, %700, %701) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %702#1, %16 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb416(%16 : !llvm.ptr)
  ^bb416(%703: !llvm.ptr):  // pred: ^bb415
    %704 = llvm.load %703 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %704 : i1, [
      default: ^bb417,
      0: ^bb418,
      1: ^bb419
    ]
  ^bb417:  // pred: ^bb416
    %false_111 = arith.constant false
    cf.assert %false_111, "Invalid enum tag."
    llvm.unreachable
  ^bb418:  // pred: ^bb416
    %705 = llvm.load %703 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %706 = llvm.extractvalue %705[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb420
  ^bb419:  // pred: ^bb416
    %707 = llvm.load %703 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %708 = llvm.extractvalue %707[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb471
  ^bb420:  // pred: ^bb418
    cf.br ^bb421
  ^bb421:  // pred: ^bb420
    %709 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb422(%709 : !llvm.struct<()>)
  ^bb422(%710: !llvm.struct<()>):  // pred: ^bb421
    %false_112 = arith.constant false
    %711 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %712 = llvm.insertvalue %false_112, %711[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb423(%712 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb423(%713: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb422
    cf.br ^bb424(%713 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb424(%714: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb423
    call @"core::result::ResultTraitImpl::<(), core::fmt::Error>::unwrap::<core::fmt::ErrorDrop>(f1)"(%18, %714) : (!llvm.ptr, !llvm.struct<(i1, array<0 x i8>)>) -> ()
    %715 = llvm.getelementptr %18[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb425(%715 : !llvm.ptr)
  ^bb425(%716: !llvm.ptr):  // pred: ^bb424
    %717 = llvm.load %716 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %717 : i1, [
      default: ^bb426,
      0: ^bb427,
      1: ^bb428
    ]
  ^bb426:  // pred: ^bb425
    %false_113 = arith.constant false
    cf.assert %false_113, "Invalid enum tag."
    llvm.unreachable
  ^bb427:  // pred: ^bb425
    %718 = llvm.load %716 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>)>)>
    %719 = llvm.extractvalue %718[1] : !llvm.struct<(i1, struct<(struct<()>)>)> 
    cf.br ^bb429
  ^bb428:  // pred: ^bb425
    %720 = llvm.load %716 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %721 = llvm.extractvalue %720[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb464
  ^bb429:  // pred: ^bb427
    cf.br ^bb430(%719 : !llvm.struct<(struct<()>)>)
  ^bb430(%722: !llvm.struct<(struct<()>)>):  // pred: ^bb429
    cf.br ^bb431
  ^bb431:  // pred: ^bb430
    %723 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_114 = arith.constant 0 : i32
    %724 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %725 = llvm.insertvalue %723, %724[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %726 = llvm.insertvalue %c0_i32_114, %725[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %727 = llvm.insertvalue %c0_i32_114, %726[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb432
  ^bb432:  // pred: ^bb431
    %c1997209042069643135709344952807065910992472029923670688473712229447419591075_i252_115 = arith.constant 1997209042069643135709344952807065910992472029923670688473712229447419591075 : i252
    cf.br ^bb433(%c1997209042069643135709344952807065910992472029923670688473712229447419591075_i252_115 : i252)
  ^bb433(%728: i252):  // pred: ^bb432
    cf.br ^bb434(%727, %728 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb434(%729: !llvm.struct<(ptr<i252>, i32, i32)>, %730: i252):  // pred: ^bb433
    %731 = llvm.extractvalue %729[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %732 = llvm.extractvalue %729[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %733 = llvm.extractvalue %729[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %734 = arith.cmpi uge, %732, %733 : i32
    %735:2 = scf.if %734 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32_146 = arith.constant 8 : i32
      %1047 = arith.addi %733, %733 : i32
      %1048 = arith.maxui %c8_i32_146, %1047 : i32
      %1049 = arith.extui %1048 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1050 = arith.muli %1049, %c32_i64 : i64
      %1051 = llvm.bitcast %731 : !llvm.ptr<i252> to !llvm.ptr
      %1052 = func.call @realloc(%1051, %1050) : (!llvm.ptr, i64) -> !llvm.ptr
      %1053 = llvm.bitcast %1052 : !llvm.ptr to !llvm.ptr<i252>
      %1054 = llvm.insertvalue %1053, %729[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1055 = llvm.insertvalue %1048, %1054[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1055, %1053 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %729, %731 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %736 = llvm.getelementptr %735#1[%732] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %730, %736 : i252, !llvm.ptr
    %c1_i32_116 = arith.constant 1 : i32
    %737 = arith.addi %732, %c1_i32_116 : i32
    %738 = llvm.insertvalue %737, %735#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb435(%706 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb435(%739: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb434
    %740 = llvm.extractvalue %739[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %741 = llvm.extractvalue %739[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb436(%741 : !llvm.struct<()>)
  ^bb436(%742: !llvm.struct<()>):  // pred: ^bb435
    cf.br ^bb437(%740 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb437(%743: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb436
    cf.br ^bb438(%743 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb438(%744: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb437
    cf.br ^bb439(%702#0 : i64)
  ^bb439(%745: i64):  // pred: ^bb438
    cf.br ^bb440(%57 : i128)
  ^bb440(%746: i128):  // pred: ^bb439
    cf.br ^bb441(%743 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb441(%747: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb440
    cf.br ^bb442(%738 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb442(%748: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb441
    cf.br ^bb443(%745, %746, %747, %748 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb443(%749: i64, %750: i128, %751: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %752: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb442
    %753:3 = call @"core::byte_array::ByteArraySerde::serialize(f0)"(%749, %750, %751, %752) : (i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %753#2, %20 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb444(%20 : !llvm.ptr)
  ^bb444(%754: !llvm.ptr):  // pred: ^bb443
    %755 = llvm.load %754 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %755 : i1, [
      default: ^bb445,
      0: ^bb446,
      1: ^bb447
    ]
  ^bb445:  // pred: ^bb444
    %false_117 = arith.constant false
    cf.assert %false_117, "Invalid enum tag."
    llvm.unreachable
  ^bb446:  // pred: ^bb444
    %756 = llvm.load %754 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %757 = llvm.extractvalue %756[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    cf.br ^bb448
  ^bb447:  // pred: ^bb444
    %758 = llvm.load %754 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %759 = llvm.extractvalue %758[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb458
  ^bb448:  // pred: ^bb446
    cf.br ^bb449(%757 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)
  ^bb449(%760: !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>):  // pred: ^bb448
    %761 = llvm.extractvalue %760[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %762 = llvm.extractvalue %760[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    cf.br ^bb450(%762 : !llvm.struct<()>)
  ^bb450(%763: !llvm.struct<()>):  // pred: ^bb449
    cf.br ^bb451
  ^bb451:  // pred: ^bb450
    %764 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb452(%764, %761 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb452(%765: !llvm.struct<()>, %766: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb451
    %767 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %768 = llvm.insertvalue %765, %767[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %769 = llvm.insertvalue %766, %768[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb453(%769 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb453(%770: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb452
    %true_118 = arith.constant true
    %771 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %772 = llvm.insertvalue %true_118, %771[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %773 = llvm.insertvalue %770, %772[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %773, %22 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb454(%753#0 : i64)
  ^bb454(%774: i64):  // pred: ^bb453
    cf.br ^bb455(%753#1 : i128)
  ^bb455(%775: i128):  // pred: ^bb454
    cf.br ^bb456(%22 : !llvm.ptr)
  ^bb456(%776: !llvm.ptr):  // pred: ^bb455
    cf.br ^bb457(%774, %775, %776 : i64, i128, !llvm.ptr)
  ^bb457(%777: i64, %778: i128, %779: !llvm.ptr):  // pred: ^bb456
    %780 = llvm.load %776 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %774, %775, %780 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb458:  // pred: ^bb447
    cf.br ^bb459(%759 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb459(%781: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb458
    %true_119 = arith.constant true
    %782 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %783 = llvm.insertvalue %true_119, %782[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %784 = llvm.insertvalue %781, %783[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %784, %21 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb460(%753#0 : i64)
  ^bb460(%785: i64):  // pred: ^bb459
    cf.br ^bb461(%753#1 : i128)
  ^bb461(%786: i128):  // pred: ^bb460
    cf.br ^bb462(%21 : !llvm.ptr)
  ^bb462(%787: !llvm.ptr):  // pred: ^bb461
    cf.br ^bb463(%785, %786, %787 : i64, i128, !llvm.ptr)
  ^bb463(%788: i64, %789: i128, %790: !llvm.ptr):  // pred: ^bb462
    %791 = llvm.load %787 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %785, %786, %791 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb464:  // pred: ^bb428
    cf.br ^bb465(%706 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb465(%792: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb464
    cf.br ^bb466(%721 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb466(%793: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb465
    %true_120 = arith.constant true
    %794 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %795 = llvm.insertvalue %true_120, %794[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %796 = llvm.insertvalue %793, %795[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %796, %19 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb467(%702#0 : i64)
  ^bb467(%797: i64):  // pred: ^bb466
    cf.br ^bb468(%57 : i128)
  ^bb468(%798: i128):  // pred: ^bb467
    cf.br ^bb469(%19 : !llvm.ptr)
  ^bb469(%799: !llvm.ptr):  // pred: ^bb468
    cf.br ^bb470(%797, %798, %799 : i64, i128, !llvm.ptr)
  ^bb470(%800: i64, %801: i128, %802: !llvm.ptr):  // pred: ^bb469
    %803 = llvm.load %799 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %797, %798, %803 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb471:  // pred: ^bb419
    cf.br ^bb472(%708 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb472(%804: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb471
    %true_121 = arith.constant true
    %805 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %806 = llvm.insertvalue %true_121, %805[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %807 = llvm.insertvalue %804, %806[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %807, %17 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb473(%702#0 : i64)
  ^bb473(%808: i64):  // pred: ^bb472
    cf.br ^bb474(%57 : i128)
  ^bb474(%809: i128):  // pred: ^bb473
    cf.br ^bb475(%17 : !llvm.ptr)
  ^bb475(%810: !llvm.ptr):  // pred: ^bb474
    cf.br ^bb476(%808, %809, %810 : i64, i128, !llvm.ptr)
  ^bb476(%811: i64, %812: i128, %813: !llvm.ptr):  // pred: ^bb475
    %814 = llvm.load %810 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %808, %809, %814 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb477:  // pred: ^bb405
    cf.br ^bb478(%689 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb478(%815: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb477
    %true_122 = arith.constant true
    %816 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %817 = llvm.insertvalue %true_122, %816[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %818 = llvm.insertvalue %815, %817[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %818, %15 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb479(%683#0 : i64)
  ^bb479(%819: i64):  // pred: ^bb478
    cf.br ^bb480(%57 : i128)
  ^bb480(%820: i128):  // pred: ^bb479
    cf.br ^bb481(%15 : !llvm.ptr)
  ^bb481(%821: !llvm.ptr):  // pred: ^bb480
    cf.br ^bb482(%819, %820, %821 : i64, i128, !llvm.ptr)
  ^bb482(%822: i64, %823: i128, %824: !llvm.ptr):  // pred: ^bb481
    %825 = llvm.load %821 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %819, %820, %825 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb483:  // pred: ^bb391
    cf.br ^bb484(%670 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb484(%826: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb483
    %true_123 = arith.constant true
    %827 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %828 = llvm.insertvalue %true_123, %827[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %829 = llvm.insertvalue %826, %828[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %829, %13 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb485(%664#0 : i64)
  ^bb485(%830: i64):  // pred: ^bb484
    cf.br ^bb486(%57 : i128)
  ^bb486(%831: i128):  // pred: ^bb485
    cf.br ^bb487(%13 : !llvm.ptr)
  ^bb487(%832: !llvm.ptr):  // pred: ^bb486
    cf.br ^bb488(%830, %831, %832 : i64, i128, !llvm.ptr)
  ^bb488(%833: i64, %834: i128, %835: !llvm.ptr):  // pred: ^bb487
    %836 = llvm.load %832 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %830, %831, %836 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb489:  // pred: ^bb376
    cf.br ^bb490(%651 : !llvm.struct<()>)
  ^bb490(%837: !llvm.struct<()>):  // pred: ^bb489
    cf.br ^bb491
  ^bb491:  // pred: ^bb490
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582016_i252_124 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582016 : i252
    cf.br ^bb492(%647#0 : i64)
  ^bb492(%838: i64):  // pred: ^bb491
    cf.br ^bb493(%c-3618502788666131000275863779947924135206266826270938552493006944358698582016_i252_124 : i252)
  ^bb493(%839: i252):  // pred: ^bb492
    cf.br ^bb494(%838, %839 : i64, i252)
  ^bb494(%840: i64, %841: i252):  // pred: ^bb493
    %842:2 = call @"program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<-1, -1>>(f14)"(%840, %841) : (i64, i252) -> (i64, !llvm.struct<(i1, array<0 x i8>)>)
    cf.br ^bb495(%842#1 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb495(%843: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb494
    %844 = llvm.extractvalue %843[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.switch %844 : i1, [
      default: ^bb496,
      0: ^bb497,
      1: ^bb498
    ]
  ^bb496:  // pred: ^bb495
    %false_125 = arith.constant false
    cf.assert %false_125, "Invalid enum tag."
    llvm.unreachable
  ^bb497:  // pred: ^bb495
    %845 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb499
  ^bb498:  // pred: ^bb495
    %846 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb611
  ^bb499:  // pred: ^bb497
    cf.br ^bb500(%845 : !llvm.struct<()>)
  ^bb500(%847: !llvm.struct<()>):  // pred: ^bb499
    cf.br ^bb501
  ^bb501:  // pred: ^bb500
    %848 = call @"core::fmt::FormatterDefault::default(f12)"() : () -> !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
    cf.br ^bb502
  ^bb502:  // pred: ^bb501
    %c172180977190876322177717838039515195832848434333118596004974948384901134190_i252_126 = arith.constant 172180977190876322177717838039515195832848434333118596004974948384901134190 : i252
    cf.br ^bb503
  ^bb503:  // pred: ^bb502
    %c31_i32_127 = arith.constant 31 : i32
    cf.br ^bb504(%848 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>)
  ^bb504(%849: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>):  // pred: ^bb503
    %850 = llvm.extractvalue %849[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)> 
    cf.br ^bb505(%842#0 : i64)
  ^bb505(%851: i64):  // pred: ^bb504
    cf.br ^bb506(%850 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb506(%852: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb505
    cf.br ^bb507(%c172180977190876322177717838039515195832848434333118596004974948384901134190_i252_126 : i252)
  ^bb507(%853: i252):  // pred: ^bb506
    cf.br ^bb508(%c31_i32_127 : i32)
  ^bb508(%854: i32):  // pred: ^bb507
    cf.br ^bb509(%851, %852, %853, %854 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb509(%855: i64, %856: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %857: i252, %858: i32):  // pred: ^bb508
    %859:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%855, %856, %857, %858) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %859#1, %1 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb510(%1 : !llvm.ptr)
  ^bb510(%860: !llvm.ptr):  // pred: ^bb509
    %861 = llvm.load %860 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %861 : i1, [
      default: ^bb511,
      0: ^bb512,
      1: ^bb513
    ]
  ^bb511:  // pred: ^bb510
    %false_128 = arith.constant false
    cf.assert %false_128, "Invalid enum tag."
    llvm.unreachable
  ^bb512:  // pred: ^bb510
    %862 = llvm.load %860 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %863 = llvm.extractvalue %862[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb514
  ^bb513:  // pred: ^bb510
    %864 = llvm.load %860 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %865 = llvm.extractvalue %864[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb605
  ^bb514:  // pred: ^bb512
    cf.br ^bb515(%863 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb515(%866: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb514
    %867 = llvm.extractvalue %866[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %868 = llvm.extractvalue %866[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb516(%868 : !llvm.struct<()>)
  ^bb516(%869: !llvm.struct<()>):  // pred: ^bb515
    cf.br ^bb517
  ^bb517:  // pred: ^bb516
    %c175590441458062252655054000563808860724590867505131080439014142883694195006_i252 = arith.constant 175590441458062252655054000563808860724590867505131080439014142883694195006 : i252
    cf.br ^bb518
  ^bb518:  // pred: ^bb517
    %c31_i32_129 = arith.constant 31 : i32
    cf.br ^bb519(%859#0 : i64)
  ^bb519(%870: i64):  // pred: ^bb518
    cf.br ^bb520(%867 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb520(%871: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb519
    cf.br ^bb521(%c175590441458062252655054000563808860724590867505131080439014142883694195006_i252 : i252)
  ^bb521(%872: i252):  // pred: ^bb520
    cf.br ^bb522(%c31_i32_129 : i32)
  ^bb522(%873: i32):  // pred: ^bb521
    cf.br ^bb523(%870, %871, %872, %873 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb523(%874: i64, %875: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %876: i252, %877: i32):  // pred: ^bb522
    %878:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%874, %875, %876, %877) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %878#1, %3 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb524(%3 : !llvm.ptr)
  ^bb524(%879: !llvm.ptr):  // pred: ^bb523
    %880 = llvm.load %879 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %880 : i1, [
      default: ^bb525,
      0: ^bb526,
      1: ^bb527
    ]
  ^bb525:  // pred: ^bb524
    %false_130 = arith.constant false
    cf.assert %false_130, "Invalid enum tag."
    llvm.unreachable
  ^bb526:  // pred: ^bb524
    %881 = llvm.load %879 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %882 = llvm.extractvalue %881[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb528
  ^bb527:  // pred: ^bb524
    %883 = llvm.load %879 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %884 = llvm.extractvalue %883[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb599
  ^bb528:  // pred: ^bb526
    cf.br ^bb529(%882 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb529(%885: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb528
    %886 = llvm.extractvalue %885[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %887 = llvm.extractvalue %885[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb530(%887 : !llvm.struct<()>)
  ^bb530(%888: !llvm.struct<()>):  // pred: ^bb529
    cf.br ^bb531
  ^bb531:  // pred: ^bb530
    %c17495623119495214_i252 = arith.constant 17495623119495214 : i252
    cf.br ^bb532
  ^bb532:  // pred: ^bb531
    %c7_i32_131 = arith.constant 7 : i32
    cf.br ^bb533(%878#0 : i64)
  ^bb533(%889: i64):  // pred: ^bb532
    cf.br ^bb534(%886 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb534(%890: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb533
    cf.br ^bb535(%c17495623119495214_i252 : i252)
  ^bb535(%891: i252):  // pred: ^bb534
    cf.br ^bb536(%c7_i32_131 : i32)
  ^bb536(%892: i32):  // pred: ^bb535
    cf.br ^bb537(%889, %890, %891, %892 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb537(%893: i64, %894: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %895: i252, %896: i32):  // pred: ^bb536
    %897:2 = call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%893, %894, %895, %896) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>)
    llvm.store %897#1, %5 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    cf.br ^bb538(%5 : !llvm.ptr)
  ^bb538(%898: !llvm.ptr):  // pred: ^bb537
    %899 = llvm.load %898 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %899 : i1, [
      default: ^bb539,
      0: ^bb540,
      1: ^bb541
    ]
  ^bb539:  // pred: ^bb538
    %false_132 = arith.constant false
    cf.assert %false_132, "Invalid enum tag."
    llvm.unreachable
  ^bb540:  // pred: ^bb538
    %900 = llvm.load %898 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %901 = llvm.extractvalue %900[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    cf.br ^bb542
  ^bb541:  // pred: ^bb538
    %902 = llvm.load %898 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %903 = llvm.extractvalue %902[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb593
  ^bb542:  // pred: ^bb540
    cf.br ^bb543
  ^bb543:  // pred: ^bb542
    %904 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb544(%904 : !llvm.struct<()>)
  ^bb544(%905: !llvm.struct<()>):  // pred: ^bb543
    %false_133 = arith.constant false
    %906 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %907 = llvm.insertvalue %false_133, %906[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb545(%907 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb545(%908: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb544
    cf.br ^bb546(%908 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb546(%909: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb545
    call @"core::result::ResultTraitImpl::<(), core::fmt::Error>::unwrap::<core::fmt::ErrorDrop>(f1)"(%7, %909) : (!llvm.ptr, !llvm.struct<(i1, array<0 x i8>)>) -> ()
    %910 = llvm.getelementptr %7[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb547(%910 : !llvm.ptr)
  ^bb547(%911: !llvm.ptr):  // pred: ^bb546
    %912 = llvm.load %911 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %912 : i1, [
      default: ^bb548,
      0: ^bb549,
      1: ^bb550
    ]
  ^bb548:  // pred: ^bb547
    %false_134 = arith.constant false
    cf.assert %false_134, "Invalid enum tag."
    llvm.unreachable
  ^bb549:  // pred: ^bb547
    %913 = llvm.load %911 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>)>)>
    %914 = llvm.extractvalue %913[1] : !llvm.struct<(i1, struct<(struct<()>)>)> 
    cf.br ^bb551
  ^bb550:  // pred: ^bb547
    %915 = llvm.load %911 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %916 = llvm.extractvalue %915[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb586
  ^bb551:  // pred: ^bb549
    cf.br ^bb552(%914 : !llvm.struct<(struct<()>)>)
  ^bb552(%917: !llvm.struct<(struct<()>)>):  // pred: ^bb551
    cf.br ^bb553
  ^bb553:  // pred: ^bb552
    %918 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_135 = arith.constant 0 : i32
    %919 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %920 = llvm.insertvalue %918, %919[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %921 = llvm.insertvalue %c0_i32_135, %920[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %922 = llvm.insertvalue %c0_i32_135, %921[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb554
  ^bb554:  // pred: ^bb553
    %c1997209042069643135709344952807065910992472029923670688473712229447419591075_i252_136 = arith.constant 1997209042069643135709344952807065910992472029923670688473712229447419591075 : i252
    cf.br ^bb555(%c1997209042069643135709344952807065910992472029923670688473712229447419591075_i252_136 : i252)
  ^bb555(%923: i252):  // pred: ^bb554
    cf.br ^bb556(%922, %923 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb556(%924: !llvm.struct<(ptr<i252>, i32, i32)>, %925: i252):  // pred: ^bb555
    %926 = llvm.extractvalue %924[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %927 = llvm.extractvalue %924[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %928 = llvm.extractvalue %924[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %929 = arith.cmpi uge, %927, %928 : i32
    %930:2 = scf.if %929 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32_146 = arith.constant 8 : i32
      %1047 = arith.addi %928, %928 : i32
      %1048 = arith.maxui %c8_i32_146, %1047 : i32
      %1049 = arith.extui %1048 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1050 = arith.muli %1049, %c32_i64 : i64
      %1051 = llvm.bitcast %926 : !llvm.ptr<i252> to !llvm.ptr
      %1052 = func.call @realloc(%1051, %1050) : (!llvm.ptr, i64) -> !llvm.ptr
      %1053 = llvm.bitcast %1052 : !llvm.ptr to !llvm.ptr<i252>
      %1054 = llvm.insertvalue %1053, %924[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1055 = llvm.insertvalue %1048, %1054[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1055, %1053 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %924, %926 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %931 = llvm.getelementptr %930#1[%927] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %925, %931 : i252, !llvm.ptr
    %c1_i32_137 = arith.constant 1 : i32
    %932 = arith.addi %927, %c1_i32_137 : i32
    %933 = llvm.insertvalue %932, %930#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb557(%901 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb557(%934: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb556
    %935 = llvm.extractvalue %934[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %936 = llvm.extractvalue %934[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb558(%936 : !llvm.struct<()>)
  ^bb558(%937: !llvm.struct<()>):  // pred: ^bb557
    cf.br ^bb559(%935 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb559(%938: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb558
    cf.br ^bb560(%938 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb560(%939: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb559
    cf.br ^bb561(%897#0 : i64)
  ^bb561(%940: i64):  // pred: ^bb560
    cf.br ^bb562(%57 : i128)
  ^bb562(%941: i128):  // pred: ^bb561
    cf.br ^bb563(%938 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb563(%942: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb562
    cf.br ^bb564(%933 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb564(%943: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb563
    cf.br ^bb565(%940, %941, %942, %943 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb565(%944: i64, %945: i128, %946: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %947: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb564
    %948:3 = call @"core::byte_array::ByteArraySerde::serialize(f0)"(%944, %945, %946, %947) : (i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %948#2, %9 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb566(%9 : !llvm.ptr)
  ^bb566(%949: !llvm.ptr):  // pred: ^bb565
    %950 = llvm.load %949 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %950 : i1, [
      default: ^bb567,
      0: ^bb568,
      1: ^bb569
    ]
  ^bb567:  // pred: ^bb566
    %false_138 = arith.constant false
    cf.assert %false_138, "Invalid enum tag."
    llvm.unreachable
  ^bb568:  // pred: ^bb566
    %951 = llvm.load %949 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %952 = llvm.extractvalue %951[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    cf.br ^bb570
  ^bb569:  // pred: ^bb566
    %953 = llvm.load %949 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %954 = llvm.extractvalue %953[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb580
  ^bb570:  // pred: ^bb568
    cf.br ^bb571(%952 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)
  ^bb571(%955: !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>):  // pred: ^bb570
    %956 = llvm.extractvalue %955[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %957 = llvm.extractvalue %955[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    cf.br ^bb572(%957 : !llvm.struct<()>)
  ^bb572(%958: !llvm.struct<()>):  // pred: ^bb571
    cf.br ^bb573
  ^bb573:  // pred: ^bb572
    %959 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb574(%959, %956 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb574(%960: !llvm.struct<()>, %961: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb573
    %962 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %963 = llvm.insertvalue %960, %962[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %964 = llvm.insertvalue %961, %963[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb575(%964 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb575(%965: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb574
    %true_139 = arith.constant true
    %966 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %967 = llvm.insertvalue %true_139, %966[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %968 = llvm.insertvalue %965, %967[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %968, %11 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb576(%948#0 : i64)
  ^bb576(%969: i64):  // pred: ^bb575
    cf.br ^bb577(%948#1 : i128)
  ^bb577(%970: i128):  // pred: ^bb576
    cf.br ^bb578(%11 : !llvm.ptr)
  ^bb578(%971: !llvm.ptr):  // pred: ^bb577
    cf.br ^bb579(%969, %970, %971 : i64, i128, !llvm.ptr)
  ^bb579(%972: i64, %973: i128, %974: !llvm.ptr):  // pred: ^bb578
    %975 = llvm.load %971 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %969, %970, %975 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb580:  // pred: ^bb569
    cf.br ^bb581(%954 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb581(%976: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb580
    %true_140 = arith.constant true
    %977 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %978 = llvm.insertvalue %true_140, %977[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %979 = llvm.insertvalue %976, %978[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %979, %10 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb582(%948#0 : i64)
  ^bb582(%980: i64):  // pred: ^bb581
    cf.br ^bb583(%948#1 : i128)
  ^bb583(%981: i128):  // pred: ^bb582
    cf.br ^bb584(%10 : !llvm.ptr)
  ^bb584(%982: !llvm.ptr):  // pred: ^bb583
    cf.br ^bb585(%980, %981, %982 : i64, i128, !llvm.ptr)
  ^bb585(%983: i64, %984: i128, %985: !llvm.ptr):  // pred: ^bb584
    %986 = llvm.load %982 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %980, %981, %986 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb586:  // pred: ^bb550
    cf.br ^bb587(%901 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb587(%987: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb586
    cf.br ^bb588(%916 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb588(%988: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb587
    %true_141 = arith.constant true
    %989 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %990 = llvm.insertvalue %true_141, %989[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %991 = llvm.insertvalue %988, %990[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %991, %8 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb589(%897#0 : i64)
  ^bb589(%992: i64):  // pred: ^bb588
    cf.br ^bb590(%57 : i128)
  ^bb590(%993: i128):  // pred: ^bb589
    cf.br ^bb591(%8 : !llvm.ptr)
  ^bb591(%994: !llvm.ptr):  // pred: ^bb590
    cf.br ^bb592(%992, %993, %994 : i64, i128, !llvm.ptr)
  ^bb592(%995: i64, %996: i128, %997: !llvm.ptr):  // pred: ^bb591
    %998 = llvm.load %994 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %992, %993, %998 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb593:  // pred: ^bb541
    cf.br ^bb594(%903 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb594(%999: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb593
    %true_142 = arith.constant true
    %1000 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %1001 = llvm.insertvalue %true_142, %1000[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %1002 = llvm.insertvalue %999, %1001[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %1002, %6 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb595(%897#0 : i64)
  ^bb595(%1003: i64):  // pred: ^bb594
    cf.br ^bb596(%57 : i128)
  ^bb596(%1004: i128):  // pred: ^bb595
    cf.br ^bb597(%6 : !llvm.ptr)
  ^bb597(%1005: !llvm.ptr):  // pred: ^bb596
    cf.br ^bb598(%1003, %1004, %1005 : i64, i128, !llvm.ptr)
  ^bb598(%1006: i64, %1007: i128, %1008: !llvm.ptr):  // pred: ^bb597
    %1009 = llvm.load %1005 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %1003, %1004, %1009 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb599:  // pred: ^bb527
    cf.br ^bb600(%884 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb600(%1010: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb599
    %true_143 = arith.constant true
    %1011 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %1012 = llvm.insertvalue %true_143, %1011[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %1013 = llvm.insertvalue %1010, %1012[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %1013, %4 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb601(%878#0 : i64)
  ^bb601(%1014: i64):  // pred: ^bb600
    cf.br ^bb602(%57 : i128)
  ^bb602(%1015: i128):  // pred: ^bb601
    cf.br ^bb603(%4 : !llvm.ptr)
  ^bb603(%1016: !llvm.ptr):  // pred: ^bb602
    cf.br ^bb604(%1014, %1015, %1016 : i64, i128, !llvm.ptr)
  ^bb604(%1017: i64, %1018: i128, %1019: !llvm.ptr):  // pred: ^bb603
    %1020 = llvm.load %1016 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %1014, %1015, %1020 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb605:  // pred: ^bb513
    cf.br ^bb606(%865 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb606(%1021: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb605
    %true_144 = arith.constant true
    %1022 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %1023 = llvm.insertvalue %true_144, %1022[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %1024 = llvm.insertvalue %1021, %1023[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %1024, %2 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb607(%859#0 : i64)
  ^bb607(%1025: i64):  // pred: ^bb606
    cf.br ^bb608(%57 : i128)
  ^bb608(%1026: i128):  // pred: ^bb607
    cf.br ^bb609(%2 : !llvm.ptr)
  ^bb609(%1027: !llvm.ptr):  // pred: ^bb608
    cf.br ^bb610(%1025, %1026, %1027 : i64, i128, !llvm.ptr)
  ^bb610(%1028: i64, %1029: i128, %1030: !llvm.ptr):  // pred: ^bb609
    %1031 = llvm.load %1027 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %1025, %1026, %1031 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb611:  // pred: ^bb498
    cf.br ^bb612(%846 : !llvm.struct<()>)
  ^bb612(%1032: !llvm.struct<()>):  // pred: ^bb611
    cf.br ^bb613
  ^bb613:  // pred: ^bb612
    %1033 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb614(%1033 : !llvm.struct<()>)
  ^bb614(%1034: !llvm.struct<()>):  // pred: ^bb613
    %1035 = llvm.mlir.undef : !llvm.struct<(struct<()>)>
    %1036 = llvm.insertvalue %1034, %1035[0] : !llvm.struct<(struct<()>)> 
    cf.br ^bb615(%1036 : !llvm.struct<(struct<()>)>)
  ^bb615(%1037: !llvm.struct<(struct<()>)>):  // pred: ^bb614
    %false_145 = arith.constant false
    %1038 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %1039 = llvm.insertvalue %false_145, %1038[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %1039, %0 {alignment = 8 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    cf.br ^bb616(%842#0 : i64)
  ^bb616(%1040: i64):  // pred: ^bb615
    cf.br ^bb617(%57 : i128)
  ^bb617(%1041: i128):  // pred: ^bb616
    cf.br ^bb618(%0 : !llvm.ptr)
  ^bb618(%1042: !llvm.ptr):  // pred: ^bb617
    cf.br ^bb619(%1040, %1041, %1042 : i64, i128, !llvm.ptr)
  ^bb619(%1043: i64, %1044: i128, %1045: !llvm.ptr):  // pred: ^bb618
    %1046 = llvm.load %1042 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %1040, %1041, %1046 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  }
  func.func private @__debug__print_felt252(i64, i64, i64, i64)
  func.func private @__debug__print_i1(i1)
  func.func public @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<0, 0>>(f17)"(%arg0: i64, %arg1: i252) -> (i64, !llvm.struct<(i1, array<0 x i8>)>) attributes {llvm.emit_c_interface} {
    cf.br ^bb1(%arg0, %arg1 : i64, i252)
  ^bb1(%0: i64, %1: i252):  // pred: ^bb0
    cf.br ^bb2(%0, %1 : i64, i252)
  ^bb2(%2: i64, %3: i252):  // pred: ^bb1
    %c1_i64 = arith.constant 1 : i64
    %4 = arith.addi %2, %c1_i64 : i64
    %c1809251394333065606848661391547535052811553607665798349986546028067936010240_i252 = arith.constant 1809251394333065606848661391547535052811553607665798349986546028067936010240 : i252
    %5 = arith.cmpi ugt, %3, %c1809251394333065606848661391547535052811553607665798349986546028067936010240_i252 : i252
    cf.cond_br %5, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252
    %6 = arith.subi %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252, %3 : i252
    %c-1_i252 = arith.constant -1 : i252
    %7 = arith.muli %6, %c-1_i252 : i252
    cf.br ^bb5(%7 : i252)
  ^bb4:  // pred: ^bb2
    cf.br ^bb5(%3 : i252)
  ^bb5(%8: i252):  // 2 preds: ^bb3, ^bb4
    %9 = arith.trunci %8 : i252 to i2
    %c64_i252 = arith.constant 64 : i252
    %10 = arith.trunci %8 : i252 to i64
    %11 = arith.shrui %8, %c64_i252 : i252
    %12 = arith.trunci %11 : i252 to i64
    %13 = arith.shrui %11, %c64_i252 : i252
    %14 = arith.trunci %13 : i252 to i64
    %15 = arith.shrui %13, %c64_i252 : i252
    %16 = arith.trunci %15 : i252 to i64
    call @__debug__print_felt252(%10, %12, %14, %16) : (i64, i64, i64, i64) -> ()
    %c1_i252 = arith.constant 1 : i252
    %c0_i252 = arith.constant 0 : i252
    %17 = arith.cmpi sle, %8, %c1_i252 : i252
    %18 = arith.cmpi sge, %8, %c0_i252 : i252
    %19 = arith.andi %17, %18 : i1
    call @__debug__print_i1(%19) : (i1) -> ()
    cf.cond_br %19, ^bb6, ^bb17
  ^bb6:  // pred: ^bb5
    cf.br ^bb7(%9 : i2)
  ^bb7(%20: i2):  // pred: ^bb6
    %21 = arith.extsi %20 : i2 to i252
    %c0_i252_0 = arith.constant 0 : i252
    %22 = arith.cmpi slt, %21, %c0_i252_0 : i252
    cf.cond_br %22, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %23 = arith.extsi %20 : i2 to i252
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252_1 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252
    %24 = arith.addi %23, %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252_1 : i252
    cf.br ^bb10(%24 : i252)
  ^bb9:  // pred: ^bb7
    %25 = arith.extui %20 : i2 to i252
    cf.br ^bb10(%25 : i252)
  ^bb10(%26: i252):  // 2 preds: ^bb8, ^bb9
    cf.br ^bb11(%26 : i252)
  ^bb11(%27: i252):  // pred: ^bb10
    cf.br ^bb12
  ^bb12:  // pred: ^bb11
    %28 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb13(%28 : !llvm.struct<()>)
  ^bb13(%29: !llvm.struct<()>):  // pred: ^bb12
    %false = arith.constant false
    %30 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %31 = llvm.insertvalue %false, %30[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb14(%4 : i64)
  ^bb14(%32: i64):  // pred: ^bb13
    cf.br ^bb15(%31 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb15(%33: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb14
    cf.br ^bb16(%32, %33 : i64, !llvm.struct<(i1, array<0 x i8>)>)
  ^bb16(%34: i64, %35: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb15
    return %32, %33 : i64, !llvm.struct<(i1, array<0 x i8>)>
  ^bb17:  // pred: ^bb5
    cf.br ^bb18
  ^bb18:  // pred: ^bb17
    %36 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb19(%36 : !llvm.struct<()>)
  ^bb19(%37: !llvm.struct<()>):  // pred: ^bb18
    %true = arith.constant true
    %38 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %39 = llvm.insertvalue %true, %38[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb20(%4 : i64)
  ^bb20(%40: i64):  // pred: ^bb19
    cf.br ^bb21(%39 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb21(%41: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb20
    cf.br ^bb22(%40, %41 : i64, !llvm.struct<(i1, array<0 x i8>)>)
  ^bb22(%42: i64, %43: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb21
    return %40, %41 : i64, !llvm.struct<(i1, array<0 x i8>)>
  }
  func.func public @"core::fmt::FormatterDefault::default(f12)"() -> !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)> attributes {llvm.emit_c_interface} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %0 = call @"core::byte_array::ByteArrayDefault::default(f13)"() : () -> !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    cf.br ^bb3(%0 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb3(%1: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb2
    %2 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
    %3 = llvm.insertvalue %1, %2[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)> 
    cf.br ^bb4(%3 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>)
  ^bb4(%4: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>):  // pred: ^bb3
    cf.br ^bb5(%4 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>)
  ^bb5(%5: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>):  // pred: ^bb4
    return %4 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
  }
  func.func public @"core::byte_array::ByteArrayImpl::append_word(f3)"(%arg0: i64, %arg1: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %arg2: i252, %arg3: i32) -> (i64, !llvm.struct<(i64, array<52 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_1 = arith.constant 1 : i64
    %2 = llvm.alloca %c1_i64_1 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_2 = arith.constant 1 : i64
    %3 = llvm.alloca %c1_i64_2 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_3 = arith.constant 1 : i64
    %4 = llvm.alloca %c1_i64_3 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_4 = arith.constant 1 : i64
    %5 = llvm.alloca %c1_i64_4 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_5 = arith.constant 1 : i64
    %6 = llvm.alloca %c1_i64_5 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_6 = arith.constant 1 : i64
    %7 = llvm.alloca %c1_i64_6 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_7 = arith.constant 1 : i64
    %8 = llvm.alloca %c1_i64_7 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_8 = arith.constant 1 : i64
    %9 = llvm.alloca %c1_i64_8 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_9 = arith.constant 1 : i64
    %10 = llvm.alloca %c1_i64_9 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_10 = arith.constant 1 : i64
    %11 = llvm.alloca %c1_i64_10 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_11 = arith.constant 1 : i64
    %12 = llvm.alloca %c1_i64_11 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_12 = arith.constant 1 : i64
    %13 = llvm.alloca %c1_i64_12 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_13 = arith.constant 1 : i64
    %14 = llvm.alloca %c1_i64_13 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_14 = arith.constant 1 : i64
    %15 = llvm.alloca %c1_i64_14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_15 = arith.constant 1 : i64
    %16 = llvm.alloca %c1_i64_15 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_16 = arith.constant 1 : i64
    %17 = llvm.alloca %c1_i64_16 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_17 = arith.constant 1 : i64
    %18 = llvm.alloca %c1_i64_17 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_18 = arith.constant 1 : i64
    %19 = llvm.alloca %c1_i64_18 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_19 = arith.constant 1 : i64
    %20 = llvm.alloca %c1_i64_19 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_20 = arith.constant 1 : i64
    %21 = llvm.alloca %c1_i64_20 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_21 = arith.constant 1 : i64
    %22 = llvm.alloca %c1_i64_21 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_22 = arith.constant 1 : i64
    %23 = llvm.alloca %c1_i64_22 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_23 = arith.constant 1 : i64
    %24 = llvm.alloca %c1_i64_23 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_24 = arith.constant 1 : i64
    %25 = llvm.alloca %c1_i64_24 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_25 = arith.constant 1 : i64
    %26 = llvm.alloca %c1_i64_25 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_26 = arith.constant 1 : i64
    %27 = llvm.alloca %c1_i64_26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_27 = arith.constant 1 : i64
    %28 = llvm.alloca %c1_i64_27 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_28 = arith.constant 1 : i64
    %29 = llvm.alloca %c1_i64_28 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_29 = arith.constant 1 : i64
    %30 = llvm.alloca %c1_i64_29 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_30 = arith.constant 1 : i64
    %31 = llvm.alloca %c1_i64_30 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_31 = arith.constant 1 : i64
    %32 = llvm.alloca %c1_i64_31 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_32 = arith.constant 1 : i64
    %33 = llvm.alloca %c1_i64_32 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_33 = arith.constant 1 : i64
    %34 = llvm.alloca %c1_i64_33 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_34 = arith.constant 1 : i64
    %35 = llvm.alloca %c1_i64_34 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_35 = arith.constant 1 : i64
    %36 = llvm.alloca %c1_i64_35 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_36 = arith.constant 1 : i64
    %37 = llvm.alloca %c1_i64_36 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_37 = arith.constant 1 : i64
    %38 = llvm.alloca %c1_i64_37 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_38 = arith.constant 1 : i64
    %39 = llvm.alloca %c1_i64_38 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_39 = arith.constant 1 : i64
    %40 = llvm.alloca %c1_i64_39 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_40 = arith.constant 1 : i64
    %41 = llvm.alloca %c1_i64_40 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_41 = arith.constant 1 : i64
    %42 = llvm.alloca %c1_i64_41 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_42 = arith.constant 1 : i64
    %43 = llvm.alloca %c1_i64_42 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_43 = arith.constant 1 : i64
    %44 = llvm.alloca %c1_i64_43 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_44 = arith.constant 1 : i64
    %45 = llvm.alloca %c1_i64_44 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_45 = arith.constant 1 : i64
    %46 = llvm.alloca %c1_i64_45 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_46 = arith.constant 1 : i64
    %47 = llvm.alloca %c1_i64_46 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1, %arg2, %arg3 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32)
  ^bb1(%48: i64, %49: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %50: i252, %51: i32):  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %c0_i32 = arith.constant 0 : i32
    cf.br ^bb3(%51 : i32)
  ^bb3(%52: i32):  // pred: ^bb2
    cf.br ^bb4(%52, %c0_i32 : i32, i32)
  ^bb4(%53: i32, %54: i32):  // pred: ^bb3
    %55 = arith.cmpi eq, %53, %54 : i32
    cf.cond_br %55, ^bb626, ^bb5
  ^bb5:  // pred: ^bb4
    cf.br ^bb6(%49 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb6(%56: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb5
    %57 = llvm.extractvalue %56[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %58 = llvm.extractvalue %56[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %59 = llvm.extractvalue %56[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb7(%48 : i64)
  ^bb7(%60: i64):  // pred: ^bb6
    cf.br ^bb8(%59 : i32)
  ^bb8(%61: i32):  // pred: ^bb7
    cf.br ^bb9(%61 : i32)
  ^bb9(%62: i32):  // pred: ^bb8
    cf.br ^bb10(%52 : i32)
  ^bb10(%63: i32):  // pred: ^bb9
    cf.br ^bb11(%63 : i32)
  ^bb11(%64: i32):  // pred: ^bb10
    cf.br ^bb12(%60, %62, %64 : i64, i32, i32)
  ^bb12(%65: i64, %66: i32, %67: i32):  // pred: ^bb11
    %68:2 = call @"core::integer::U32Add::add(f4)"(%65, %66, %67) : (i64, i32, i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %68#1, %1 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb13(%1 : !llvm.ptr)
  ^bb13(%69: !llvm.ptr):  // pred: ^bb12
    %70 = llvm.load %69 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %70 : i1, [
      default: ^bb14,
      0: ^bb15,
      1: ^bb16
    ]
  ^bb14:  // pred: ^bb13
    %false = arith.constant false
    cf.assert %false, "Invalid enum tag."
    llvm.unreachable
  ^bb15:  // pred: ^bb13
    %71 = llvm.load %69 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %72 = llvm.extractvalue %71[1] : !llvm.struct<(i1, struct<(i32)>)> 
    cf.br ^bb17
  ^bb16:  // pred: ^bb13
    %73 = llvm.load %69 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %74 = llvm.extractvalue %73[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb616
  ^bb17:  // pred: ^bb15
    cf.br ^bb18(%72 : !llvm.struct<(i32)>)
  ^bb18(%75: !llvm.struct<(i32)>):  // pred: ^bb17
    %76 = llvm.extractvalue %75[0] : !llvm.struct<(i32)> 
    cf.br ^bb19
  ^bb19:  // pred: ^bb18
    %c31_i32 = arith.constant 31 : i32
    cf.br ^bb20(%76 : i32)
  ^bb20(%77: i32):  // pred: ^bb19
    cf.br ^bb21(%c31_i32 : i32)
  ^bb21(%78: i32):  // pred: ^bb20
    cf.br ^bb22(%68#0, %77, %78 : i64, i32, i32)
  ^bb22(%79: i64, %80: i32, %81: i32):  // pred: ^bb21
    %c1_i64_47 = arith.constant 1 : i64
    %82 = arith.addi %79, %c1_i64_47 : i64
    %83 = "llvm.intr.usub.with.overflow"(%80, %81) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %84 = llvm.extractvalue %83[0] : !llvm.struct<(i32, i1)> 
    %85 = llvm.extractvalue %83[1] : !llvm.struct<(i32, i1)> 
    cf.cond_br %85, ^bb549, ^bb23
  ^bb23:  // pred: ^bb22
    cf.br ^bb24(%84 : i32)
  ^bb24(%86: i32):  // pred: ^bb23
    cf.br ^bb25
  ^bb25:  // pred: ^bb24
    %c31_i32_48 = arith.constant 31 : i32
    cf.br ^bb26(%77 : i32)
  ^bb26(%87: i32):  // pred: ^bb25
    cf.br ^bb27(%82 : i64)
  ^bb27(%88: i64):  // pred: ^bb26
    cf.br ^bb28(%87, %c31_i32_48 : i32, i32)
  ^bb28(%89: i32, %90: i32):  // pred: ^bb27
    %91 = arith.cmpi eq, %89, %90 : i32
    cf.cond_br %91, ^bb495, ^bb29
  ^bb29:  // pred: ^bb28
    cf.br ^bb30(%63 : i32)
  ^bb30(%92: i32):  // pred: ^bb29
    cf.br ^bb31
  ^bb31:  // pred: ^bb30
    %c31_i32_49 = arith.constant 31 : i32
    cf.br ^bb32(%88 : i64)
  ^bb32(%93: i64):  // pred: ^bb31
    cf.br ^bb33(%87 : i32)
  ^bb33(%94: i32):  // pred: ^bb32
    cf.br ^bb34(%c31_i32_49 : i32)
  ^bb34(%95: i32):  // pred: ^bb33
    cf.br ^bb35(%93, %94, %95 : i64, i32, i32)
  ^bb35(%96: i64, %97: i32, %98: i32):  // pred: ^bb34
    %99:2 = call @"core::integer::U32Sub::sub(f8)"(%96, %97, %98) : (i64, i32, i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %99#1, %13 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb36(%13 : !llvm.ptr)
  ^bb36(%100: !llvm.ptr):  // pred: ^bb35
    %101 = llvm.load %100 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %101 : i1, [
      default: ^bb37,
      0: ^bb38,
      1: ^bb39
    ]
  ^bb37:  // pred: ^bb36
    %false_50 = arith.constant false
    cf.assert %false_50, "Invalid enum tag."
    llvm.unreachable
  ^bb38:  // pred: ^bb36
    %102 = llvm.load %100 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %103 = llvm.extractvalue %102[1] : !llvm.struct<(i1, struct<(i32)>)> 
    cf.br ^bb40
  ^bb39:  // pred: ^bb36
    %104 = llvm.load %100 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %105 = llvm.extractvalue %104[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb486
  ^bb40:  // pred: ^bb38
    cf.br ^bb41(%103 : !llvm.struct<(i32)>)
  ^bb41(%106: !llvm.struct<(i32)>):  // pred: ^bb40
    %107 = llvm.extractvalue %106[0] : !llvm.struct<(i32)> 
    cf.br ^bb42
  ^bb42:  // pred: ^bb41
    %c16_i32 = arith.constant 16 : i32
    cf.br ^bb43(%107 : i32)
  ^bb43(%108: i32):  // pred: ^bb42
    cf.br ^bb44(%108, %c16_i32 : i32, i32)
  ^bb44(%109: i32, %110: i32):  // pred: ^bb43
    %111 = arith.cmpi eq, %109, %110 : i32
    cf.cond_br %111, ^bb395, ^bb45
  ^bb45:  // pred: ^bb44
    cf.br ^bb46
  ^bb46:  // pred: ^bb45
    %c16_i32_51 = arith.constant 16 : i32
    cf.br ^bb47(%108 : i32)
  ^bb47(%112: i32):  // pred: ^bb46
    cf.br ^bb48(%c16_i32_51 : i32)
  ^bb48(%113: i32):  // pred: ^bb47
    cf.br ^bb49(%99#0, %112, %113 : i64, i32, i32)
  ^bb49(%114: i64, %115: i32, %116: i32):  // pred: ^bb48
    %c1_i64_52 = arith.constant 1 : i64
    %117 = arith.addi %114, %c1_i64_52 : i64
    %118 = "llvm.intr.usub.with.overflow"(%115, %116) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %119 = llvm.extractvalue %118[0] : !llvm.struct<(i32, i1)> 
    %120 = llvm.extractvalue %118[1] : !llvm.struct<(i32, i1)> 
    cf.cond_br %120, ^bb209, ^bb50
  ^bb50:  // pred: ^bb49
    cf.br ^bb51(%119 : i32)
  ^bb51(%121: i32):  // pred: ^bb50
    cf.br ^bb52(%117 : i64)
  ^bb52(%122: i64):  // pred: ^bb51
    cf.br ^bb53(%50 : i252)
  ^bb53(%123: i252):  // pred: ^bb52
    cf.br ^bb54(%122, %123 : i64, i252)
  ^bb54(%124: i64, %125: i252):  // pred: ^bb53
    %126:2 = call @"core::integer::u256_from_felt252(f10)"(%124, %125) : (i64, i252) -> (i64, !llvm.struct<(i128, i128)>)
    cf.br ^bb55(%126#1 : !llvm.struct<(i128, i128)>)
  ^bb55(%127: !llvm.struct<(i128, i128)>):  // pred: ^bb54
    %128 = llvm.extractvalue %127[0] : !llvm.struct<(i128, i128)> 
    %129 = llvm.extractvalue %127[1] : !llvm.struct<(i128, i128)> 
    cf.br ^bb56
  ^bb56:  // pred: ^bb55
    %c16_i32_53 = arith.constant 16 : i32
    cf.br ^bb57(%126#0 : i64)
  ^bb57(%130: i64):  // pred: ^bb56
    cf.br ^bb58(%112 : i32)
  ^bb58(%131: i32):  // pred: ^bb57
    cf.br ^bb59(%131 : i32)
  ^bb59(%132: i32):  // pred: ^bb58
    cf.br ^bb60(%c16_i32_53 : i32)
  ^bb60(%133: i32):  // pred: ^bb59
    cf.br ^bb61(%130, %132, %133 : i64, i32, i32)
  ^bb61(%134: i64, %135: i32, %136: i32):  // pred: ^bb60
    %137:2 = call @"core::integer::U32Sub::sub(f8)"(%134, %135, %136) : (i64, i32, i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %137#1, %36 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb62(%36 : !llvm.ptr)
  ^bb62(%138: !llvm.ptr):  // pred: ^bb61
    %139 = llvm.load %138 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %139 : i1, [
      default: ^bb63,
      0: ^bb64,
      1: ^bb65
    ]
  ^bb63:  // pred: ^bb62
    %false_54 = arith.constant false
    cf.assert %false_54, "Invalid enum tag."
    llvm.unreachable
  ^bb64:  // pred: ^bb62
    %140 = llvm.load %138 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %141 = llvm.extractvalue %140[1] : !llvm.struct<(i1, struct<(i32)>)> 
    cf.br ^bb66
  ^bb65:  // pred: ^bb62
    %142 = llvm.load %138 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %143 = llvm.extractvalue %142[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb198
  ^bb66:  // pred: ^bb64
    cf.br ^bb67(%141 : !llvm.struct<(i32)>)
  ^bb67(%144: !llvm.struct<(i32)>):  // pred: ^bb66
    %145 = llvm.extractvalue %144[0] : !llvm.struct<(i32)> 
    cf.br ^bb68(%145 : i32)
  ^bb68(%146: i32):  // pred: ^bb67
    cf.br ^bb69(%146 : i32)
  ^bb69(%147: i32):  // pred: ^bb68
    call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%38, %147) : (!llvm.ptr, i32) -> ()
    %148 = llvm.getelementptr %38[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb70(%148 : !llvm.ptr)
  ^bb70(%149: !llvm.ptr):  // pred: ^bb69
    %150 = llvm.load %149 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %150 : i1, [
      default: ^bb71,
      0: ^bb72,
      1: ^bb73
    ]
  ^bb71:  // pred: ^bb70
    %false_55 = arith.constant false
    cf.assert %false_55, "Invalid enum tag."
    llvm.unreachable
  ^bb72:  // pred: ^bb70
    %151 = llvm.load %149 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i128)>)>
    %152 = llvm.extractvalue %151[1] : !llvm.struct<(i1, struct<(i128)>)> 
    cf.br ^bb74
  ^bb73:  // pred: ^bb70
    %153 = llvm.load %149 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %154 = llvm.extractvalue %153[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb187
  ^bb74:  // pred: ^bb72
    cf.br ^bb75(%152 : !llvm.struct<(i128)>)
  ^bb75(%155: !llvm.struct<(i128)>):  // pred: ^bb74
    %156 = llvm.extractvalue %155[0] : !llvm.struct<(i128)> 
    cf.br ^bb76(%156 : i128)
  ^bb76(%157: i128):  // pred: ^bb75
    cf.br ^bb77(%157 : i128)
  ^bb77(%158: i128):  // pred: ^bb76
    call @"core::integer::u128_try_as_non_zero(f11)"(%40, %158) : (!llvm.ptr, i128) -> ()
    %159 = llvm.getelementptr %40[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb78(%159 : !llvm.ptr)
  ^bb78(%160: !llvm.ptr):  // pred: ^bb77
    %161 = llvm.load %160 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %161 : i1, [
      default: ^bb79,
      0: ^bb80,
      1: ^bb81
    ]
  ^bb79:  // pred: ^bb78
    %false_56 = arith.constant false
    cf.assert %false_56, "Invalid enum tag."
    llvm.unreachable
  ^bb80:  // pred: ^bb78
    %162 = llvm.load %160 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i128)>
    %163 = llvm.extractvalue %162[1] : !llvm.struct<(i1, i128)> 
    cf.br ^bb82
  ^bb81:  // pred: ^bb78
    %164 = llvm.load %160 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<()>)>
    %165 = llvm.extractvalue %164[1] : !llvm.struct<(i1, struct<()>)> 
    cf.br ^bb169
  ^bb82:  // pred: ^bb80
    cf.br ^bb83(%137#0, %129, %163 : i64, i128, i128)
  ^bb83(%166: i64, %167: i128, %168: i128):  // pred: ^bb82
    %c1_i64_57 = arith.constant 1 : i64
    %169 = arith.addi %166, %c1_i64_57 : i64
    %170 = arith.divui %167, %168 : i128
    %171 = arith.remui %167, %168 : i128
    cf.br ^bb84(%171 : i128)
  ^bb84(%172: i128):  // pred: ^bb83
    %173 = arith.extui %172 : i128 to i252
    cf.br ^bb85(%128 : i128)
  ^bb85(%174: i128):  // pred: ^bb84
    %175 = arith.extui %174 : i128 to i252
    cf.br ^bb86(%170 : i128)
  ^bb86(%176: i128):  // pred: ^bb85
    %177 = arith.extui %176 : i128 to i252
    cf.br ^bb87
  ^bb87:  // pred: ^bb86
    %c31_i32_58 = arith.constant 31 : i32
    cf.br ^bb88(%169 : i64)
  ^bb88(%178: i64):  // pred: ^bb87
    cf.br ^bb89(%c31_i32_58 : i32)
  ^bb89(%179: i32):  // pred: ^bb88
    cf.br ^bb90(%61 : i32)
  ^bb90(%180: i32):  // pred: ^bb89
    cf.br ^bb91(%180 : i32)
  ^bb91(%181: i32):  // pred: ^bb90
    cf.br ^bb92(%178, %179, %181 : i64, i32, i32)
  ^bb92(%182: i64, %183: i32, %184: i32):  // pred: ^bb91
    %185:2 = call @"core::integer::U32Sub::sub(f8)"(%182, %183, %184) : (i64, i32, i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %185#1, %42 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb93(%42 : !llvm.ptr)
  ^bb93(%186: !llvm.ptr):  // pred: ^bb92
    %187 = llvm.load %186 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %187 : i1, [
      default: ^bb94,
      0: ^bb95,
      1: ^bb96
    ]
  ^bb94:  // pred: ^bb93
    %false_59 = arith.constant false
    cf.assert %false_59, "Invalid enum tag."
    llvm.unreachable
  ^bb95:  // pred: ^bb93
    %188 = llvm.load %186 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %189 = llvm.extractvalue %188[1] : !llvm.struct<(i1, struct<(i32)>)> 
    cf.br ^bb97
  ^bb96:  // pred: ^bb93
    %190 = llvm.load %186 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %191 = llvm.extractvalue %190[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb157
  ^bb97:  // pred: ^bb95
    cf.br ^bb98(%189 : !llvm.struct<(i32)>)
  ^bb98(%192: !llvm.struct<(i32)>):  // pred: ^bb97
    %193 = llvm.extractvalue %192[0] : !llvm.struct<(i32)> 
    cf.br ^bb99(%185#0 : i64)
  ^bb99(%194: i64):  // pred: ^bb98
    cf.br ^bb100(%193 : i32)
  ^bb100(%195: i32):  // pred: ^bb99
    cf.br ^bb101(%194, %195 : i64, i32)
  ^bb101(%196: i64, %197: i32):  // pred: ^bb100
    %198:2 = call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%196, %197) : (i64, i32) -> (i64, !llvm.struct<(i64, array<32 x i8>)>)
    llvm.store %198#1, %44 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    cf.br ^bb102(%44 : !llvm.ptr)
  ^bb102(%199: !llvm.ptr):  // pred: ^bb101
    %200 = llvm.load %199 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %200 : i1, [
      default: ^bb103,
      0: ^bb104,
      1: ^bb105
    ]
  ^bb103:  // pred: ^bb102
    %false_60 = arith.constant false
    cf.assert %false_60, "Invalid enum tag."
    llvm.unreachable
  ^bb104:  // pred: ^bb102
    %201 = llvm.load %199 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i252)>)>
    %202 = llvm.extractvalue %201[1] : !llvm.struct<(i1, struct<(i252)>)> 
    cf.br ^bb106
  ^bb105:  // pred: ^bb102
    %203 = llvm.load %199 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %204 = llvm.extractvalue %203[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb145
  ^bb106:  // pred: ^bb104
    cf.br ^bb107(%202 : !llvm.struct<(i252)>)
  ^bb107(%205: !llvm.struct<(i252)>):  // pred: ^bb106
    %206 = llvm.extractvalue %205[0] : !llvm.struct<(i252)> 
    cf.br ^bb108(%58, %206 : i252, i252)
  ^bb108(%207: i252, %208: i252):  // pred: ^bb107
    %209 = arith.extui %207 : i252 to i512
    %210 = arith.extui %208 : i252 to i512
    %211 = arith.muli %209, %210 : i512
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512
    %212 = arith.remui %211, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512 : i512
    %213 = arith.cmpi uge, %211, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512 : i512
    %214 = arith.select %213, %212, %211 : i512
    %215 = arith.trunci %214 : i512 to i252
    cf.br ^bb109(%215 : i252)
  ^bb109(%216: i252):  // pred: ^bb108
    cf.br ^bb110(%177, %216 : i252, i252)
  ^bb110(%217: i252, %218: i252):  // pred: ^bb109
    %219 = arith.extui %217 : i252 to i256
    %220 = arith.extui %218 : i252 to i256
    %221 = arith.addi %219, %220 : i256
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256
    %222 = arith.subi %221, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256 : i256
    %223 = arith.cmpi uge, %221, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256 : i256
    %224 = arith.select %223, %222, %221 : i256
    %225 = arith.trunci %224 : i256 to i252
    cf.br ^bb111(%198#0 : i64)
  ^bb111(%226: i64):  // pred: ^bb110
    cf.br ^bb112(%225 : i252)
  ^bb112(%227: i252):  // pred: ^bb111
    cf.br ^bb113(%226, %227 : i64, i252)
  ^bb113(%228: i64, %229: i252):  // pred: ^bb112
    %230:2 = call @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%228, %229) : (i64, i252) -> (i64, !llvm.struct<(i64, array<32 x i8>)>)
    llvm.store %230#1, %46 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    cf.br ^bb114(%46 : !llvm.ptr)
  ^bb114(%231: !llvm.ptr):  // pred: ^bb113
    %232 = llvm.load %231 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %232 : i1, [
      default: ^bb115,
      0: ^bb116,
      1: ^bb117
    ]
  ^bb115:  // pred: ^bb114
    %false_61 = arith.constant false
    cf.assert %false_61, "Invalid enum tag."
    llvm.unreachable
  ^bb116:  // pred: ^bb114
    %233 = llvm.load %231 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i248)>
    %234 = llvm.extractvalue %233[1] : !llvm.struct<(i1, i248)> 
    cf.br ^bb118
  ^bb117:  // pred: ^bb114
    %235 = llvm.load %231 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<()>)>
    %236 = llvm.extractvalue %235[1] : !llvm.struct<(i1, struct<()>)> 
    cf.br ^bb128
  ^bb118:  // pred: ^bb116
    cf.br ^bb119(%57, %234 : !llvm.struct<(ptr<i248>, i32, i32)>, i248)
  ^bb119(%237: !llvm.struct<(ptr<i248>, i32, i32)>, %238: i248):  // pred: ^bb118
    %239 = llvm.extractvalue %237[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %240 = llvm.extractvalue %237[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %241 = llvm.extractvalue %237[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %242 = arith.cmpi uge, %240, %241 : i32
    %243:2 = scf.if %242 -> (!llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>) {
      %c8_i32 = arith.constant 8 : i32
      %1256 = arith.addi %241, %241 : i32
      %1257 = arith.maxui %c8_i32, %1256 : i32
      %1258 = arith.extui %1257 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1259 = arith.muli %1258, %c32_i64 : i64
      %1260 = llvm.bitcast %239 : !llvm.ptr<i248> to !llvm.ptr
      %1261 = func.call @realloc(%1260, %1259) : (!llvm.ptr, i64) -> !llvm.ptr
      %1262 = llvm.bitcast %1261 : !llvm.ptr to !llvm.ptr<i248>
      %1263 = llvm.insertvalue %1262, %237[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
      %1264 = llvm.insertvalue %1257, %1263[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
      scf.yield %1264, %1262 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>
    } else {
      scf.yield %237, %239 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>
    }
    %244 = llvm.getelementptr %243#1[%240] : (!llvm.ptr<i248>, i32) -> !llvm.ptr, i248
    llvm.store %238, %244 : i248, !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %245 = arith.addi %240, %c1_i32 : i32
    %246 = llvm.insertvalue %245, %243#0[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    cf.br ^bb120
  ^bb120:  // pred: ^bb119
    %c340282366920938463463374607431768211456_i252 = arith.constant 340282366920938463463374607431768211456 : i252
    cf.br ^bb121(%173, %c340282366920938463463374607431768211456_i252 : i252, i252)
  ^bb121(%247: i252, %248: i252):  // pred: ^bb120
    %249 = arith.extui %247 : i252 to i512
    %250 = arith.extui %248 : i252 to i512
    %251 = arith.muli %249, %250 : i512
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_62 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512
    %252 = arith.remui %251, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_62 : i512
    %253 = arith.cmpi uge, %251, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_62 : i512
    %254 = arith.select %253, %252, %251 : i512
    %255 = arith.trunci %254 : i512 to i252
    cf.br ^bb122(%255 : i252)
  ^bb122(%256: i252):  // pred: ^bb121
    cf.br ^bb123(%256, %175 : i252, i252)
  ^bb123(%257: i252, %258: i252):  // pred: ^bb122
    %259 = arith.extui %257 : i252 to i256
    %260 = arith.extui %258 : i252 to i256
    %261 = arith.addi %259, %260 : i256
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_63 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256
    %262 = arith.subi %261, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_63 : i256
    %263 = arith.cmpi uge, %261, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_63 : i256
    %264 = arith.select %263, %262, %261 : i256
    %265 = arith.trunci %264 : i256 to i252
    cf.br ^bb124(%246, %265, %180 : !llvm.struct<(ptr<i248>, i32, i32)>, i252, i32)
  ^bb124(%266: !llvm.struct<(ptr<i248>, i32, i32)>, %267: i252, %268: i32):  // pred: ^bb123
    %269 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %270 = llvm.insertvalue %266, %269[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %271 = llvm.insertvalue %267, %270[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %272 = llvm.insertvalue %268, %271[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb125(%230#0 : i64)
  ^bb125(%273: i64):  // pred: ^bb124
    cf.br ^bb126(%272 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb126(%274: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb125
    cf.br ^bb127
  ^bb127:  // pred: ^bb126
    cf.br ^bb296(%131, %273, %274 : i32, i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb128:  // pred: ^bb117
    cf.br ^bb129(%236 : !llvm.struct<()>)
  ^bb129(%275: !llvm.struct<()>):  // pred: ^bb128
    cf.br ^bb130(%131 : i32)
  ^bb130(%276: i32):  // pred: ^bb129
    cf.br ^bb131(%180 : i32)
  ^bb131(%277: i32):  // pred: ^bb130
    cf.br ^bb132(%173 : i252)
  ^bb132(%278: i252):  // pred: ^bb131
    cf.br ^bb133(%175 : i252)
  ^bb133(%279: i252):  // pred: ^bb132
    cf.br ^bb134(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb134(%280: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb133
    %281 = llvm.extractvalue %280[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %282 = llvm.bitcast %281 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%282) : (!llvm.ptr) -> ()
    cf.br ^bb135
  ^bb135:  // pred: ^bb134
    %283 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_64 = arith.constant 0 : i32
    %284 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %285 = llvm.insertvalue %283, %284[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %286 = llvm.insertvalue %c0_i32_64, %285[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %287 = llvm.insertvalue %c0_i32_64, %286[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb136
  ^bb136:  // pred: ^bb135
    %c29721761890975875353235833581453094220424382983267374_i252 = arith.constant 29721761890975875353235833581453094220424382983267374 : i252
    cf.br ^bb137(%c29721761890975875353235833581453094220424382983267374_i252 : i252)
  ^bb137(%288: i252):  // pred: ^bb136
    cf.br ^bb138(%287, %288 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb138(%289: !llvm.struct<(ptr<i252>, i32, i32)>, %290: i252):  // pred: ^bb137
    %291 = llvm.extractvalue %289[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %292 = llvm.extractvalue %289[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %293 = llvm.extractvalue %289[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %294 = arith.cmpi uge, %292, %293 : i32
    %295:2 = scf.if %294 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %1256 = arith.addi %293, %293 : i32
      %1257 = arith.maxui %c8_i32, %1256 : i32
      %1258 = arith.extui %1257 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1259 = arith.muli %1258, %c32_i64 : i64
      %1260 = llvm.bitcast %291 : !llvm.ptr<i252> to !llvm.ptr
      %1261 = func.call @realloc(%1260, %1259) : (!llvm.ptr, i64) -> !llvm.ptr
      %1262 = llvm.bitcast %1261 : !llvm.ptr to !llvm.ptr<i252>
      %1263 = llvm.insertvalue %1262, %289[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1264 = llvm.insertvalue %1257, %1263[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1264, %1262 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %289, %291 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %296 = llvm.getelementptr %295#1[%292] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %290, %296 : i252, !llvm.ptr
    %c1_i32_65 = arith.constant 1 : i32
    %297 = arith.addi %292, %c1_i32_65 : i32
    %298 = llvm.insertvalue %297, %295#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb139
  ^bb139:  // pred: ^bb138
    %299 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb140(%299, %298 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb140(%300: !llvm.struct<()>, %301: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb139
    %302 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %303 = llvm.insertvalue %300, %302[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %304 = llvm.insertvalue %301, %303[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb141(%304 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb141(%305: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb140
    %true = arith.constant true
    %306 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %307 = llvm.insertvalue %true, %306[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %308 = llvm.insertvalue %305, %307[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %308, %47 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb142(%230#0 : i64)
  ^bb142(%309: i64):  // pred: ^bb141
    cf.br ^bb143(%47 : !llvm.ptr)
  ^bb143(%310: !llvm.ptr):  // pred: ^bb142
    cf.br ^bb144(%309, %310 : i64, !llvm.ptr)
  ^bb144(%311: i64, %312: !llvm.ptr):  // pred: ^bb143
    %313 = llvm.load %310 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %309, %313 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb145:  // pred: ^bb105
    cf.br ^bb146(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb146(%314: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb145
    %315 = llvm.extractvalue %314[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %316 = llvm.bitcast %315 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%316) : (!llvm.ptr) -> ()
    cf.br ^bb147(%131 : i32)
  ^bb147(%317: i32):  // pred: ^bb146
    cf.br ^bb148(%180 : i32)
  ^bb148(%318: i32):  // pred: ^bb147
    cf.br ^bb149(%173 : i252)
  ^bb149(%319: i252):  // pred: ^bb148
    cf.br ^bb150(%175 : i252)
  ^bb150(%320: i252):  // pred: ^bb149
    cf.br ^bb151(%177 : i252)
  ^bb151(%321: i252):  // pred: ^bb150
    cf.br ^bb152(%58 : i252)
  ^bb152(%322: i252):  // pred: ^bb151
    cf.br ^bb153(%204 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb153(%323: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb152
    %true_66 = arith.constant true
    %324 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %325 = llvm.insertvalue %true_66, %324[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %326 = llvm.insertvalue %323, %325[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %326, %45 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb154(%198#0 : i64)
  ^bb154(%327: i64):  // pred: ^bb153
    cf.br ^bb155(%45 : !llvm.ptr)
  ^bb155(%328: !llvm.ptr):  // pred: ^bb154
    cf.br ^bb156(%327, %328 : i64, !llvm.ptr)
  ^bb156(%329: i64, %330: !llvm.ptr):  // pred: ^bb155
    %331 = llvm.load %328 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %327, %331 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb157:  // pred: ^bb96
    cf.br ^bb158(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb158(%332: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb157
    %333 = llvm.extractvalue %332[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %334 = llvm.bitcast %333 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%334) : (!llvm.ptr) -> ()
    cf.br ^bb159(%131 : i32)
  ^bb159(%335: i32):  // pred: ^bb158
    cf.br ^bb160(%180 : i32)
  ^bb160(%336: i32):  // pred: ^bb159
    cf.br ^bb161(%173 : i252)
  ^bb161(%337: i252):  // pred: ^bb160
    cf.br ^bb162(%175 : i252)
  ^bb162(%338: i252):  // pred: ^bb161
    cf.br ^bb163(%58 : i252)
  ^bb163(%339: i252):  // pred: ^bb162
    cf.br ^bb164(%177 : i252)
  ^bb164(%340: i252):  // pred: ^bb163
    cf.br ^bb165(%191 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb165(%341: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb164
    %true_67 = arith.constant true
    %342 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %343 = llvm.insertvalue %true_67, %342[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %344 = llvm.insertvalue %341, %343[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %344, %43 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb166(%185#0 : i64)
  ^bb166(%345: i64):  // pred: ^bb165
    cf.br ^bb167(%43 : !llvm.ptr)
  ^bb167(%346: !llvm.ptr):  // pred: ^bb166
    cf.br ^bb168(%345, %346 : i64, !llvm.ptr)
  ^bb168(%347: i64, %348: !llvm.ptr):  // pred: ^bb167
    %349 = llvm.load %346 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %345, %349 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb169:  // pred: ^bb81
    cf.br ^bb170(%165 : !llvm.struct<()>)
  ^bb170(%350: !llvm.struct<()>):  // pred: ^bb169
    cf.br ^bb171(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb171(%351: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb170
    %352 = llvm.extractvalue %351[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %353 = llvm.bitcast %352 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%353) : (!llvm.ptr) -> ()
    cf.br ^bb172(%131 : i32)
  ^bb172(%354: i32):  // pred: ^bb171
    cf.br ^bb173(%61 : i32)
  ^bb173(%355: i32):  // pred: ^bb172
    cf.br ^bb174(%128 : i128)
  ^bb174(%356: i128):  // pred: ^bb173
    cf.br ^bb175(%58 : i252)
  ^bb175(%357: i252):  // pred: ^bb174
    cf.br ^bb176(%129 : i128)
  ^bb176(%358: i128):  // pred: ^bb175
    cf.br ^bb177
  ^bb177:  // pred: ^bb176
    %359 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_68 = arith.constant 0 : i32
    %360 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %361 = llvm.insertvalue %359, %360[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %362 = llvm.insertvalue %c0_i32_68, %361[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %363 = llvm.insertvalue %c0_i32_68, %362[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb178
  ^bb178:  // pred: ^bb177
    %c29721761890975875353235833581453094220424382983267374_i252_69 = arith.constant 29721761890975875353235833581453094220424382983267374 : i252
    cf.br ^bb179(%c29721761890975875353235833581453094220424382983267374_i252_69 : i252)
  ^bb179(%364: i252):  // pred: ^bb178
    cf.br ^bb180(%363, %364 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb180(%365: !llvm.struct<(ptr<i252>, i32, i32)>, %366: i252):  // pred: ^bb179
    %367 = llvm.extractvalue %365[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %368 = llvm.extractvalue %365[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %369 = llvm.extractvalue %365[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %370 = arith.cmpi uge, %368, %369 : i32
    %371:2 = scf.if %370 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %1256 = arith.addi %369, %369 : i32
      %1257 = arith.maxui %c8_i32, %1256 : i32
      %1258 = arith.extui %1257 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1259 = arith.muli %1258, %c32_i64 : i64
      %1260 = llvm.bitcast %367 : !llvm.ptr<i252> to !llvm.ptr
      %1261 = func.call @realloc(%1260, %1259) : (!llvm.ptr, i64) -> !llvm.ptr
      %1262 = llvm.bitcast %1261 : !llvm.ptr to !llvm.ptr<i252>
      %1263 = llvm.insertvalue %1262, %365[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1264 = llvm.insertvalue %1257, %1263[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1264, %1262 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %365, %367 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %372 = llvm.getelementptr %371#1[%368] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %366, %372 : i252, !llvm.ptr
    %c1_i32_70 = arith.constant 1 : i32
    %373 = arith.addi %368, %c1_i32_70 : i32
    %374 = llvm.insertvalue %373, %371#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb181
  ^bb181:  // pred: ^bb180
    %375 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb182(%375, %374 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb182(%376: !llvm.struct<()>, %377: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb181
    %378 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %379 = llvm.insertvalue %376, %378[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %380 = llvm.insertvalue %377, %379[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb183(%380 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb183(%381: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb182
    %true_71 = arith.constant true
    %382 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %383 = llvm.insertvalue %true_71, %382[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %384 = llvm.insertvalue %381, %383[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %384, %41 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb184(%137#0 : i64)
  ^bb184(%385: i64):  // pred: ^bb183
    cf.br ^bb185(%41 : !llvm.ptr)
  ^bb185(%386: !llvm.ptr):  // pred: ^bb184
    cf.br ^bb186(%385, %386 : i64, !llvm.ptr)
  ^bb186(%387: i64, %388: !llvm.ptr):  // pred: ^bb185
    %389 = llvm.load %386 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %385, %389 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb187:  // pred: ^bb73
    cf.br ^bb188(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb188(%390: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb187
    %391 = llvm.extractvalue %390[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %392 = llvm.bitcast %391 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%392) : (!llvm.ptr) -> ()
    cf.br ^bb189(%131 : i32)
  ^bb189(%393: i32):  // pred: ^bb188
    cf.br ^bb190(%61 : i32)
  ^bb190(%394: i32):  // pred: ^bb189
    cf.br ^bb191(%128 : i128)
  ^bb191(%395: i128):  // pred: ^bb190
    cf.br ^bb192(%58 : i252)
  ^bb192(%396: i252):  // pred: ^bb191
    cf.br ^bb193(%129 : i128)
  ^bb193(%397: i128):  // pred: ^bb192
    cf.br ^bb194(%154 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb194(%398: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb193
    %true_72 = arith.constant true
    %399 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %400 = llvm.insertvalue %true_72, %399[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %401 = llvm.insertvalue %398, %400[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %401, %39 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb195(%137#0 : i64)
  ^bb195(%402: i64):  // pred: ^bb194
    cf.br ^bb196(%39 : !llvm.ptr)
  ^bb196(%403: !llvm.ptr):  // pred: ^bb195
    cf.br ^bb197(%402, %403 : i64, !llvm.ptr)
  ^bb197(%404: i64, %405: !llvm.ptr):  // pred: ^bb196
    %406 = llvm.load %403 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %402, %406 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb198:  // pred: ^bb65
    cf.br ^bb199(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb199(%407: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb198
    %408 = llvm.extractvalue %407[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %409 = llvm.bitcast %408 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%409) : (!llvm.ptr) -> ()
    cf.br ^bb200(%131 : i32)
  ^bb200(%410: i32):  // pred: ^bb199
    cf.br ^bb201(%61 : i32)
  ^bb201(%411: i32):  // pred: ^bb200
    cf.br ^bb202(%128 : i128)
  ^bb202(%412: i128):  // pred: ^bb201
    cf.br ^bb203(%58 : i252)
  ^bb203(%413: i252):  // pred: ^bb202
    cf.br ^bb204(%129 : i128)
  ^bb204(%414: i128):  // pred: ^bb203
    cf.br ^bb205(%143 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb205(%415: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb204
    %true_73 = arith.constant true
    %416 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %417 = llvm.insertvalue %true_73, %416[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %418 = llvm.insertvalue %415, %417[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %418, %37 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb206(%137#0 : i64)
  ^bb206(%419: i64):  // pred: ^bb205
    cf.br ^bb207(%37 : !llvm.ptr)
  ^bb207(%420: !llvm.ptr):  // pred: ^bb206
    cf.br ^bb208(%419, %420 : i64, !llvm.ptr)
  ^bb208(%421: i64, %422: !llvm.ptr):  // pred: ^bb207
    %423 = llvm.load %420 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %419, %423 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb209:  // pred: ^bb49
    cf.br ^bb210(%119 : i32)
  ^bb210(%424: i32):  // pred: ^bb209
    cf.br ^bb211(%117 : i64)
  ^bb211(%425: i64):  // pred: ^bb210
    cf.br ^bb212(%50 : i252)
  ^bb212(%426: i252):  // pred: ^bb211
    cf.br ^bb213(%425, %426 : i64, i252)
  ^bb213(%427: i64, %428: i252):  // pred: ^bb212
    %429:2 = call @"core::integer::u256_from_felt252(f10)"(%427, %428) : (i64, i252) -> (i64, !llvm.struct<(i128, i128)>)
    cf.br ^bb214(%429#1 : !llvm.struct<(i128, i128)>)
  ^bb214(%430: !llvm.struct<(i128, i128)>):  // pred: ^bb213
    %431 = llvm.extractvalue %430[0] : !llvm.struct<(i128, i128)> 
    %432 = llvm.extractvalue %430[1] : !llvm.struct<(i128, i128)> 
    cf.br ^bb215(%112 : i32)
  ^bb215(%433: i32):  // pred: ^bb214
    cf.br ^bb216(%433 : i32)
  ^bb216(%434: i32):  // pred: ^bb215
    cf.br ^bb217(%434 : i32)
  ^bb217(%435: i32):  // pred: ^bb216
    call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%22, %435) : (!llvm.ptr, i32) -> ()
    %436 = llvm.getelementptr %22[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb218(%436 : !llvm.ptr)
  ^bb218(%437: !llvm.ptr):  // pred: ^bb217
    %438 = llvm.load %437 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %438 : i1, [
      default: ^bb219,
      0: ^bb220,
      1: ^bb221
    ]
  ^bb219:  // pred: ^bb218
    %false_74 = arith.constant false
    cf.assert %false_74, "Invalid enum tag."
    llvm.unreachable
  ^bb220:  // pred: ^bb218
    %439 = llvm.load %437 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i128)>)>
    %440 = llvm.extractvalue %439[1] : !llvm.struct<(i1, struct<(i128)>)> 
    cf.br ^bb222
  ^bb221:  // pred: ^bb218
    %441 = llvm.load %437 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %442 = llvm.extractvalue %441[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb384
  ^bb222:  // pred: ^bb220
    cf.br ^bb223(%440 : !llvm.struct<(i128)>)
  ^bb223(%443: !llvm.struct<(i128)>):  // pred: ^bb222
    %444 = llvm.extractvalue %443[0] : !llvm.struct<(i128)> 
    cf.br ^bb224(%444 : i128)
  ^bb224(%445: i128):  // pred: ^bb223
    cf.br ^bb225(%445 : i128)
  ^bb225(%446: i128):  // pred: ^bb224
    call @"core::integer::u128_try_as_non_zero(f11)"(%24, %446) : (!llvm.ptr, i128) -> ()
    %447 = llvm.getelementptr %24[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb226(%447 : !llvm.ptr)
  ^bb226(%448: !llvm.ptr):  // pred: ^bb225
    %449 = llvm.load %448 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %449 : i1, [
      default: ^bb227,
      0: ^bb228,
      1: ^bb229
    ]
  ^bb227:  // pred: ^bb226
    %false_75 = arith.constant false
    cf.assert %false_75, "Invalid enum tag."
    llvm.unreachable
  ^bb228:  // pred: ^bb226
    %450 = llvm.load %448 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i128)>
    %451 = llvm.extractvalue %450[1] : !llvm.struct<(i1, i128)> 
    cf.br ^bb230
  ^bb229:  // pred: ^bb226
    %452 = llvm.load %448 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<()>)>
    %453 = llvm.extractvalue %452[1] : !llvm.struct<(i1, struct<()>)> 
    cf.br ^bb366
  ^bb230:  // pred: ^bb228
    cf.br ^bb231(%429#0, %431, %451 : i64, i128, i128)
  ^bb231(%454: i64, %455: i128, %456: i128):  // pred: ^bb230
    %c1_i64_76 = arith.constant 1 : i64
    %457 = arith.addi %454, %c1_i64_76 : i64
    %458 = arith.divui %455, %456 : i128
    %459 = arith.remui %455, %456 : i128
    cf.br ^bb232(%432 : i128)
  ^bb232(%460: i128):  // pred: ^bb231
    %461 = arith.extui %460 : i128 to i252
    cf.br ^bb233
  ^bb233:  // pred: ^bb232
    %c16_i32_77 = arith.constant 16 : i32
    cf.br ^bb234(%457 : i64)
  ^bb234(%462: i64):  // pred: ^bb233
    cf.br ^bb235(%c16_i32_77 : i32)
  ^bb235(%463: i32):  // pred: ^bb234
    cf.br ^bb236(%433 : i32)
  ^bb236(%464: i32):  // pred: ^bb235
    cf.br ^bb237(%464 : i32)
  ^bb237(%465: i32):  // pred: ^bb236
    cf.br ^bb238(%462, %463, %465 : i64, i32, i32)
  ^bb238(%466: i64, %467: i32, %468: i32):  // pred: ^bb237
    %469:2 = call @"core::integer::U32Sub::sub(f8)"(%466, %467, %468) : (i64, i32, i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %469#1, %26 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb239(%26 : !llvm.ptr)
  ^bb239(%470: !llvm.ptr):  // pred: ^bb238
    %471 = llvm.load %470 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %471 : i1, [
      default: ^bb240,
      0: ^bb241,
      1: ^bb242
    ]
  ^bb240:  // pred: ^bb239
    %false_78 = arith.constant false
    cf.assert %false_78, "Invalid enum tag."
    llvm.unreachable
  ^bb241:  // pred: ^bb239
    %472 = llvm.load %470 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %473 = llvm.extractvalue %472[1] : !llvm.struct<(i1, struct<(i32)>)> 
    cf.br ^bb243
  ^bb242:  // pred: ^bb239
    %474 = llvm.load %470 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %475 = llvm.extractvalue %474[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb354
  ^bb243:  // pred: ^bb241
    cf.br ^bb244(%473 : !llvm.struct<(i32)>)
  ^bb244(%476: !llvm.struct<(i32)>):  // pred: ^bb243
    %477 = llvm.extractvalue %476[0] : !llvm.struct<(i32)> 
    cf.br ^bb245(%477 : i32)
  ^bb245(%478: i32):  // pred: ^bb244
    cf.br ^bb246(%478 : i32)
  ^bb246(%479: i32):  // pred: ^bb245
    call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%28, %479) : (!llvm.ptr, i32) -> ()
    %480 = llvm.getelementptr %28[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb247(%480 : !llvm.ptr)
  ^bb247(%481: !llvm.ptr):  // pred: ^bb246
    %482 = llvm.load %481 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %482 : i1, [
      default: ^bb248,
      0: ^bb249,
      1: ^bb250
    ]
  ^bb248:  // pred: ^bb247
    %false_79 = arith.constant false
    cf.assert %false_79, "Invalid enum tag."
    llvm.unreachable
  ^bb249:  // pred: ^bb247
    %483 = llvm.load %481 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i128)>)>
    %484 = llvm.extractvalue %483[1] : !llvm.struct<(i1, struct<(i128)>)> 
    cf.br ^bb251
  ^bb250:  // pred: ^bb247
    %485 = llvm.load %481 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %486 = llvm.extractvalue %485[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb342
  ^bb251:  // pred: ^bb249
    cf.br ^bb252(%484 : !llvm.struct<(i128)>)
  ^bb252(%487: !llvm.struct<(i128)>):  // pred: ^bb251
    %488 = llvm.extractvalue %487[0] : !llvm.struct<(i128)> 
    cf.br ^bb253(%488 : i128)
  ^bb253(%489: i128):  // pred: ^bb252
    %490 = arith.extui %489 : i128 to i252
    cf.br ^bb254(%458 : i128)
  ^bb254(%491: i128):  // pred: ^bb253
    %492 = arith.extui %491 : i128 to i252
    cf.br ^bb255(%459 : i128)
  ^bb255(%493: i128):  // pred: ^bb254
    %494 = arith.extui %493 : i128 to i252
    cf.br ^bb256
  ^bb256:  // pred: ^bb255
    %c31_i32_80 = arith.constant 31 : i32
    cf.br ^bb257(%469#0 : i64)
  ^bb257(%495: i64):  // pred: ^bb256
    cf.br ^bb258(%c31_i32_80 : i32)
  ^bb258(%496: i32):  // pred: ^bb257
    cf.br ^bb259(%61 : i32)
  ^bb259(%497: i32):  // pred: ^bb258
    cf.br ^bb260(%497 : i32)
  ^bb260(%498: i32):  // pred: ^bb259
    cf.br ^bb261(%495, %496, %498 : i64, i32, i32)
  ^bb261(%499: i64, %500: i32, %501: i32):  // pred: ^bb260
    %502:2 = call @"core::integer::U32Sub::sub(f8)"(%499, %500, %501) : (i64, i32, i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %502#1, %30 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb262(%30 : !llvm.ptr)
  ^bb262(%503: !llvm.ptr):  // pred: ^bb261
    %504 = llvm.load %503 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %504 : i1, [
      default: ^bb263,
      0: ^bb264,
      1: ^bb265
    ]
  ^bb263:  // pred: ^bb262
    %false_81 = arith.constant false
    cf.assert %false_81, "Invalid enum tag."
    llvm.unreachable
  ^bb264:  // pred: ^bb262
    %505 = llvm.load %503 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %506 = llvm.extractvalue %505[1] : !llvm.struct<(i1, struct<(i32)>)> 
    cf.br ^bb266
  ^bb265:  // pred: ^bb262
    %507 = llvm.load %503 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %508 = llvm.extractvalue %507[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb329
  ^bb266:  // pred: ^bb264
    cf.br ^bb267(%506 : !llvm.struct<(i32)>)
  ^bb267(%509: !llvm.struct<(i32)>):  // pred: ^bb266
    %510 = llvm.extractvalue %509[0] : !llvm.struct<(i32)> 
    cf.br ^bb268(%502#0 : i64)
  ^bb268(%511: i64):  // pred: ^bb267
    cf.br ^bb269(%510 : i32)
  ^bb269(%512: i32):  // pred: ^bb268
    cf.br ^bb270(%511, %512 : i64, i32)
  ^bb270(%513: i64, %514: i32):  // pred: ^bb269
    %515:2 = call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%513, %514) : (i64, i32) -> (i64, !llvm.struct<(i64, array<32 x i8>)>)
    llvm.store %515#1, %32 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    cf.br ^bb271(%32 : !llvm.ptr)
  ^bb271(%516: !llvm.ptr):  // pred: ^bb270
    %517 = llvm.load %516 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %517 : i1, [
      default: ^bb272,
      0: ^bb273,
      1: ^bb274
    ]
  ^bb272:  // pred: ^bb271
    %false_82 = arith.constant false
    cf.assert %false_82, "Invalid enum tag."
    llvm.unreachable
  ^bb273:  // pred: ^bb271
    %518 = llvm.load %516 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i252)>)>
    %519 = llvm.extractvalue %518[1] : !llvm.struct<(i1, struct<(i252)>)> 
    cf.br ^bb275
  ^bb274:  // pred: ^bb271
    %520 = llvm.load %516 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %521 = llvm.extractvalue %520[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb316
  ^bb275:  // pred: ^bb273
    cf.br ^bb276(%461, %490 : i252, i252)
  ^bb276(%522: i252, %523: i252):  // pred: ^bb275
    %524 = arith.extui %522 : i252 to i512
    %525 = arith.extui %523 : i252 to i512
    %526 = arith.muli %524, %525 : i512
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_83 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512
    %527 = arith.remui %526, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_83 : i512
    %528 = arith.cmpi uge, %526, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_83 : i512
    %529 = arith.select %528, %527, %526 : i512
    %530 = arith.trunci %529 : i512 to i252
    cf.br ^bb277(%530 : i252)
  ^bb277(%531: i252):  // pred: ^bb276
    cf.br ^bb278(%531, %492 : i252, i252)
  ^bb278(%532: i252, %533: i252):  // pred: ^bb277
    %534 = arith.extui %532 : i252 to i256
    %535 = arith.extui %533 : i252 to i256
    %536 = arith.addi %534, %535 : i256
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_84 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256
    %537 = arith.subi %536, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_84 : i256
    %538 = arith.cmpi uge, %536, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_84 : i256
    %539 = arith.select %538, %537, %536 : i256
    %540 = arith.trunci %539 : i256 to i252
    cf.br ^bb279(%519 : !llvm.struct<(i252)>)
  ^bb279(%541: !llvm.struct<(i252)>):  // pred: ^bb278
    %542 = llvm.extractvalue %541[0] : !llvm.struct<(i252)> 
    cf.br ^bb280(%58, %542 : i252, i252)
  ^bb280(%543: i252, %544: i252):  // pred: ^bb279
    %545 = arith.extui %543 : i252 to i512
    %546 = arith.extui %544 : i252 to i512
    %547 = arith.muli %545, %546 : i512
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_85 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512
    %548 = arith.remui %547, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_85 : i512
    %549 = arith.cmpi uge, %547, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_85 : i512
    %550 = arith.select %549, %548, %547 : i512
    %551 = arith.trunci %550 : i512 to i252
    cf.br ^bb281(%540 : i252)
  ^bb281(%552: i252):  // pred: ^bb280
    cf.br ^bb282(%551 : i252)
  ^bb282(%553: i252):  // pred: ^bb281
    cf.br ^bb283(%552, %553 : i252, i252)
  ^bb283(%554: i252, %555: i252):  // pred: ^bb282
    %556 = arith.extui %554 : i252 to i256
    %557 = arith.extui %555 : i252 to i256
    %558 = arith.addi %556, %557 : i256
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_86 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256
    %559 = arith.subi %558, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_86 : i256
    %560 = arith.cmpi uge, %558, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_86 : i256
    %561 = arith.select %560, %559, %558 : i256
    %562 = arith.trunci %561 : i256 to i252
    cf.br ^bb284(%515#0 : i64)
  ^bb284(%563: i64):  // pred: ^bb283
    cf.br ^bb285(%562 : i252)
  ^bb285(%564: i252):  // pred: ^bb284
    cf.br ^bb286(%563, %564 : i64, i252)
  ^bb286(%565: i64, %566: i252):  // pred: ^bb285
    %567:2 = call @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%565, %566) : (i64, i252) -> (i64, !llvm.struct<(i64, array<32 x i8>)>)
    llvm.store %567#1, %34 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    cf.br ^bb287(%34 : !llvm.ptr)
  ^bb287(%568: !llvm.ptr):  // pred: ^bb286
    %569 = llvm.load %568 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %569 : i1, [
      default: ^bb288,
      0: ^bb289,
      1: ^bb290
    ]
  ^bb288:  // pred: ^bb287
    %false_87 = arith.constant false
    cf.assert %false_87, "Invalid enum tag."
    llvm.unreachable
  ^bb289:  // pred: ^bb287
    %570 = llvm.load %568 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i248)>
    %571 = llvm.extractvalue %570[1] : !llvm.struct<(i1, i248)> 
    cf.br ^bb291
  ^bb290:  // pred: ^bb287
    %572 = llvm.load %568 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<()>)>
    %573 = llvm.extractvalue %572[1] : !llvm.struct<(i1, struct<()>)> 
    cf.br ^bb300
  ^bb291:  // pred: ^bb289
    cf.br ^bb292(%57, %571 : !llvm.struct<(ptr<i248>, i32, i32)>, i248)
  ^bb292(%574: !llvm.struct<(ptr<i248>, i32, i32)>, %575: i248):  // pred: ^bb291
    %576 = llvm.extractvalue %574[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %577 = llvm.extractvalue %574[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %578 = llvm.extractvalue %574[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %579 = arith.cmpi uge, %577, %578 : i32
    %580:2 = scf.if %579 -> (!llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>) {
      %c8_i32 = arith.constant 8 : i32
      %1256 = arith.addi %578, %578 : i32
      %1257 = arith.maxui %c8_i32, %1256 : i32
      %1258 = arith.extui %1257 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1259 = arith.muli %1258, %c32_i64 : i64
      %1260 = llvm.bitcast %576 : !llvm.ptr<i248> to !llvm.ptr
      %1261 = func.call @realloc(%1260, %1259) : (!llvm.ptr, i64) -> !llvm.ptr
      %1262 = llvm.bitcast %1261 : !llvm.ptr to !llvm.ptr<i248>
      %1263 = llvm.insertvalue %1262, %574[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
      %1264 = llvm.insertvalue %1257, %1263[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
      scf.yield %1264, %1262 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>
    } else {
      scf.yield %574, %576 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>
    }
    %581 = llvm.getelementptr %580#1[%577] : (!llvm.ptr<i248>, i32) -> !llvm.ptr, i248
    llvm.store %575, %581 : i248, !llvm.ptr
    %c1_i32_88 = arith.constant 1 : i32
    %582 = arith.addi %577, %c1_i32_88 : i32
    %583 = llvm.insertvalue %582, %580#0[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    cf.br ^bb293(%583, %494, %497 : !llvm.struct<(ptr<i248>, i32, i32)>, i252, i32)
  ^bb293(%584: !llvm.struct<(ptr<i248>, i32, i32)>, %585: i252, %586: i32):  // pred: ^bb292
    %587 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %588 = llvm.insertvalue %584, %587[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %589 = llvm.insertvalue %585, %588[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %590 = llvm.insertvalue %586, %589[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb294(%567#0 : i64)
  ^bb294(%591: i64):  // pred: ^bb293
    cf.br ^bb295(%590 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb295(%592: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb294
    cf.br ^bb296(%464, %591, %592 : i32, i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb296(%593: i32, %594: i64, %595: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // 2 preds: ^bb127, ^bb295
    cf.br ^bb297(%594 : i64)
  ^bb297(%596: i64):  // pred: ^bb296
    cf.br ^bb298(%595 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb298(%597: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb297
    cf.br ^bb299
  ^bb299:  // pred: ^bb298
    cf.br ^bb438(%593, %596, %597 : i32, i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb300:  // pred: ^bb290
    cf.br ^bb301(%573 : !llvm.struct<()>)
  ^bb301(%598: !llvm.struct<()>):  // pred: ^bb300
    cf.br ^bb302(%464 : i32)
  ^bb302(%599: i32):  // pred: ^bb301
    cf.br ^bb303(%497 : i32)
  ^bb303(%600: i32):  // pred: ^bb302
    cf.br ^bb304(%494 : i252)
  ^bb304(%601: i252):  // pred: ^bb303
    cf.br ^bb305(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb305(%602: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb304
    %603 = llvm.extractvalue %602[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %604 = llvm.bitcast %603 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%604) : (!llvm.ptr) -> ()
    cf.br ^bb306
  ^bb306:  // pred: ^bb305
    %605 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_89 = arith.constant 0 : i32
    %606 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %607 = llvm.insertvalue %605, %606[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %608 = llvm.insertvalue %c0_i32_89, %607[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %609 = llvm.insertvalue %c0_i32_89, %608[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb307
  ^bb307:  // pred: ^bb306
    %c29721761890975875353235833581453094220424382983267374_i252_90 = arith.constant 29721761890975875353235833581453094220424382983267374 : i252
    cf.br ^bb308(%c29721761890975875353235833581453094220424382983267374_i252_90 : i252)
  ^bb308(%610: i252):  // pred: ^bb307
    cf.br ^bb309(%609, %610 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb309(%611: !llvm.struct<(ptr<i252>, i32, i32)>, %612: i252):  // pred: ^bb308
    %613 = llvm.extractvalue %611[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %614 = llvm.extractvalue %611[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %615 = llvm.extractvalue %611[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %616 = arith.cmpi uge, %614, %615 : i32
    %617:2 = scf.if %616 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %1256 = arith.addi %615, %615 : i32
      %1257 = arith.maxui %c8_i32, %1256 : i32
      %1258 = arith.extui %1257 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1259 = arith.muli %1258, %c32_i64 : i64
      %1260 = llvm.bitcast %613 : !llvm.ptr<i252> to !llvm.ptr
      %1261 = func.call @realloc(%1260, %1259) : (!llvm.ptr, i64) -> !llvm.ptr
      %1262 = llvm.bitcast %1261 : !llvm.ptr to !llvm.ptr<i252>
      %1263 = llvm.insertvalue %1262, %611[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1264 = llvm.insertvalue %1257, %1263[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1264, %1262 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %611, %613 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %618 = llvm.getelementptr %617#1[%614] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %612, %618 : i252, !llvm.ptr
    %c1_i32_91 = arith.constant 1 : i32
    %619 = arith.addi %614, %c1_i32_91 : i32
    %620 = llvm.insertvalue %619, %617#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb310
  ^bb310:  // pred: ^bb309
    %621 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb311(%621, %620 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb311(%622: !llvm.struct<()>, %623: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb310
    %624 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %625 = llvm.insertvalue %622, %624[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %626 = llvm.insertvalue %623, %625[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb312(%626 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb312(%627: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb311
    %true_92 = arith.constant true
    %628 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %629 = llvm.insertvalue %true_92, %628[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %630 = llvm.insertvalue %627, %629[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %630, %35 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb313(%567#0 : i64)
  ^bb313(%631: i64):  // pred: ^bb312
    cf.br ^bb314(%35 : !llvm.ptr)
  ^bb314(%632: !llvm.ptr):  // pred: ^bb313
    cf.br ^bb315(%631, %632 : i64, !llvm.ptr)
  ^bb315(%633: i64, %634: !llvm.ptr):  // pred: ^bb314
    %635 = llvm.load %632 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %631, %635 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb316:  // pred: ^bb274
    cf.br ^bb317(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb317(%636: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb316
    %637 = llvm.extractvalue %636[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %638 = llvm.bitcast %637 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%638) : (!llvm.ptr) -> ()
    cf.br ^bb318(%464 : i32)
  ^bb318(%639: i32):  // pred: ^bb317
    cf.br ^bb319(%497 : i32)
  ^bb319(%640: i32):  // pred: ^bb318
    cf.br ^bb320(%494 : i252)
  ^bb320(%641: i252):  // pred: ^bb319
    cf.br ^bb321(%461 : i252)
  ^bb321(%642: i252):  // pred: ^bb320
    cf.br ^bb322(%58 : i252)
  ^bb322(%643: i252):  // pred: ^bb321
    cf.br ^bb323(%492 : i252)
  ^bb323(%644: i252):  // pred: ^bb322
    cf.br ^bb324(%490 : i252)
  ^bb324(%645: i252):  // pred: ^bb323
    cf.br ^bb325(%521 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb325(%646: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb324
    %true_93 = arith.constant true
    %647 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %648 = llvm.insertvalue %true_93, %647[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %649 = llvm.insertvalue %646, %648[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %649, %33 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb326(%515#0 : i64)
  ^bb326(%650: i64):  // pred: ^bb325
    cf.br ^bb327(%33 : !llvm.ptr)
  ^bb327(%651: !llvm.ptr):  // pred: ^bb326
    cf.br ^bb328(%650, %651 : i64, !llvm.ptr)
  ^bb328(%652: i64, %653: !llvm.ptr):  // pred: ^bb327
    %654 = llvm.load %651 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %650, %654 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb329:  // pred: ^bb265
    cf.br ^bb330(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb330(%655: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb329
    %656 = llvm.extractvalue %655[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %657 = llvm.bitcast %656 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%657) : (!llvm.ptr) -> ()
    cf.br ^bb331(%464 : i32)
  ^bb331(%658: i32):  // pred: ^bb330
    cf.br ^bb332(%497 : i32)
  ^bb332(%659: i32):  // pred: ^bb331
    cf.br ^bb333(%494 : i252)
  ^bb333(%660: i252):  // pred: ^bb332
    cf.br ^bb334(%490 : i252)
  ^bb334(%661: i252):  // pred: ^bb333
    cf.br ^bb335(%461 : i252)
  ^bb335(%662: i252):  // pred: ^bb334
    cf.br ^bb336(%58 : i252)
  ^bb336(%663: i252):  // pred: ^bb335
    cf.br ^bb337(%492 : i252)
  ^bb337(%664: i252):  // pred: ^bb336
    cf.br ^bb338(%508 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb338(%665: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb337
    %true_94 = arith.constant true
    %666 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %667 = llvm.insertvalue %true_94, %666[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %668 = llvm.insertvalue %665, %667[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %668, %31 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb339(%502#0 : i64)
  ^bb339(%669: i64):  // pred: ^bb338
    cf.br ^bb340(%31 : !llvm.ptr)
  ^bb340(%670: !llvm.ptr):  // pred: ^bb339
    cf.br ^bb341(%669, %670 : i64, !llvm.ptr)
  ^bb341(%671: i64, %672: !llvm.ptr):  // pred: ^bb340
    %673 = llvm.load %670 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %669, %673 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb342:  // pred: ^bb250
    cf.br ^bb343(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb343(%674: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb342
    %675 = llvm.extractvalue %674[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %676 = llvm.bitcast %675 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%676) : (!llvm.ptr) -> ()
    cf.br ^bb344(%464 : i32)
  ^bb344(%677: i32):  // pred: ^bb343
    cf.br ^bb345(%61 : i32)
  ^bb345(%678: i32):  // pred: ^bb344
    cf.br ^bb346(%458 : i128)
  ^bb346(%679: i128):  // pred: ^bb345
    cf.br ^bb347(%461 : i252)
  ^bb347(%680: i252):  // pred: ^bb346
    cf.br ^bb348(%58 : i252)
  ^bb348(%681: i252):  // pred: ^bb347
    cf.br ^bb349(%459 : i128)
  ^bb349(%682: i128):  // pred: ^bb348
    cf.br ^bb350(%486 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb350(%683: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb349
    %true_95 = arith.constant true
    %684 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %685 = llvm.insertvalue %true_95, %684[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %686 = llvm.insertvalue %683, %685[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %686, %29 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb351(%469#0 : i64)
  ^bb351(%687: i64):  // pred: ^bb350
    cf.br ^bb352(%29 : !llvm.ptr)
  ^bb352(%688: !llvm.ptr):  // pred: ^bb351
    cf.br ^bb353(%687, %688 : i64, !llvm.ptr)
  ^bb353(%689: i64, %690: !llvm.ptr):  // pred: ^bb352
    %691 = llvm.load %688 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %687, %691 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb354:  // pred: ^bb242
    cf.br ^bb355(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb355(%692: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb354
    %693 = llvm.extractvalue %692[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %694 = llvm.bitcast %693 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%694) : (!llvm.ptr) -> ()
    cf.br ^bb356(%464 : i32)
  ^bb356(%695: i32):  // pred: ^bb355
    cf.br ^bb357(%61 : i32)
  ^bb357(%696: i32):  // pred: ^bb356
    cf.br ^bb358(%458 : i128)
  ^bb358(%697: i128):  // pred: ^bb357
    cf.br ^bb359(%461 : i252)
  ^bb359(%698: i252):  // pred: ^bb358
    cf.br ^bb360(%58 : i252)
  ^bb360(%699: i252):  // pred: ^bb359
    cf.br ^bb361(%459 : i128)
  ^bb361(%700: i128):  // pred: ^bb360
    cf.br ^bb362(%475 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb362(%701: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb361
    %true_96 = arith.constant true
    %702 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %703 = llvm.insertvalue %true_96, %702[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %704 = llvm.insertvalue %701, %703[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %704, %27 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb363(%469#0 : i64)
  ^bb363(%705: i64):  // pred: ^bb362
    cf.br ^bb364(%27 : !llvm.ptr)
  ^bb364(%706: !llvm.ptr):  // pred: ^bb363
    cf.br ^bb365(%705, %706 : i64, !llvm.ptr)
  ^bb365(%707: i64, %708: !llvm.ptr):  // pred: ^bb364
    %709 = llvm.load %706 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %705, %709 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb366:  // pred: ^bb229
    cf.br ^bb367(%453 : !llvm.struct<()>)
  ^bb367(%710: !llvm.struct<()>):  // pred: ^bb366
    cf.br ^bb368(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb368(%711: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb367
    %712 = llvm.extractvalue %711[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %713 = llvm.bitcast %712 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%713) : (!llvm.ptr) -> ()
    cf.br ^bb369(%433 : i32)
  ^bb369(%714: i32):  // pred: ^bb368
    cf.br ^bb370(%61 : i32)
  ^bb370(%715: i32):  // pred: ^bb369
    cf.br ^bb371(%432 : i128)
  ^bb371(%716: i128):  // pred: ^bb370
    cf.br ^bb372(%58 : i252)
  ^bb372(%717: i252):  // pred: ^bb371
    cf.br ^bb373(%431 : i128)
  ^bb373(%718: i128):  // pred: ^bb372
    cf.br ^bb374
  ^bb374:  // pred: ^bb373
    %719 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_97 = arith.constant 0 : i32
    %720 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %721 = llvm.insertvalue %719, %720[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %722 = llvm.insertvalue %c0_i32_97, %721[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %723 = llvm.insertvalue %c0_i32_97, %722[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb375
  ^bb375:  // pred: ^bb374
    %c29721761890975875353235833581453094220424382983267374_i252_98 = arith.constant 29721761890975875353235833581453094220424382983267374 : i252
    cf.br ^bb376(%c29721761890975875353235833581453094220424382983267374_i252_98 : i252)
  ^bb376(%724: i252):  // pred: ^bb375
    cf.br ^bb377(%723, %724 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb377(%725: !llvm.struct<(ptr<i252>, i32, i32)>, %726: i252):  // pred: ^bb376
    %727 = llvm.extractvalue %725[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %728 = llvm.extractvalue %725[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %729 = llvm.extractvalue %725[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %730 = arith.cmpi uge, %728, %729 : i32
    %731:2 = scf.if %730 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %1256 = arith.addi %729, %729 : i32
      %1257 = arith.maxui %c8_i32, %1256 : i32
      %1258 = arith.extui %1257 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1259 = arith.muli %1258, %c32_i64 : i64
      %1260 = llvm.bitcast %727 : !llvm.ptr<i252> to !llvm.ptr
      %1261 = func.call @realloc(%1260, %1259) : (!llvm.ptr, i64) -> !llvm.ptr
      %1262 = llvm.bitcast %1261 : !llvm.ptr to !llvm.ptr<i252>
      %1263 = llvm.insertvalue %1262, %725[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1264 = llvm.insertvalue %1257, %1263[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1264, %1262 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %725, %727 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %732 = llvm.getelementptr %731#1[%728] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %726, %732 : i252, !llvm.ptr
    %c1_i32_99 = arith.constant 1 : i32
    %733 = arith.addi %728, %c1_i32_99 : i32
    %734 = llvm.insertvalue %733, %731#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb378
  ^bb378:  // pred: ^bb377
    %735 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb379(%735, %734 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb379(%736: !llvm.struct<()>, %737: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb378
    %738 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %739 = llvm.insertvalue %736, %738[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %740 = llvm.insertvalue %737, %739[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb380(%740 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb380(%741: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb379
    %true_100 = arith.constant true
    %742 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %743 = llvm.insertvalue %true_100, %742[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %744 = llvm.insertvalue %741, %743[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %744, %25 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb381(%429#0 : i64)
  ^bb381(%745: i64):  // pred: ^bb380
    cf.br ^bb382(%25 : !llvm.ptr)
  ^bb382(%746: !llvm.ptr):  // pred: ^bb381
    cf.br ^bb383(%745, %746 : i64, !llvm.ptr)
  ^bb383(%747: i64, %748: !llvm.ptr):  // pred: ^bb382
    %749 = llvm.load %746 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %745, %749 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb384:  // pred: ^bb221
    cf.br ^bb385(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb385(%750: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb384
    %751 = llvm.extractvalue %750[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %752 = llvm.bitcast %751 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%752) : (!llvm.ptr) -> ()
    cf.br ^bb386(%433 : i32)
  ^bb386(%753: i32):  // pred: ^bb385
    cf.br ^bb387(%61 : i32)
  ^bb387(%754: i32):  // pred: ^bb386
    cf.br ^bb388(%432 : i128)
  ^bb388(%755: i128):  // pred: ^bb387
    cf.br ^bb389(%58 : i252)
  ^bb389(%756: i252):  // pred: ^bb388
    cf.br ^bb390(%431 : i128)
  ^bb390(%757: i128):  // pred: ^bb389
    cf.br ^bb391(%442 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb391(%758: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb390
    %true_101 = arith.constant true
    %759 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %760 = llvm.insertvalue %true_101, %759[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %761 = llvm.insertvalue %758, %760[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %761, %23 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb392(%429#0 : i64)
  ^bb392(%762: i64):  // pred: ^bb391
    cf.br ^bb393(%23 : !llvm.ptr)
  ^bb393(%763: !llvm.ptr):  // pred: ^bb392
    cf.br ^bb394(%762, %763 : i64, !llvm.ptr)
  ^bb394(%764: i64, %765: !llvm.ptr):  // pred: ^bb393
    %766 = llvm.load %763 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %762, %766 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb395:  // pred: ^bb44
    cf.br ^bb396(%99#0 : i64)
  ^bb396(%767: i64):  // pred: ^bb395
    cf.br ^bb397(%50 : i252)
  ^bb397(%768: i252):  // pred: ^bb396
    cf.br ^bb398(%767, %768 : i64, i252)
  ^bb398(%769: i64, %770: i252):  // pred: ^bb397
    %771:2 = call @"core::integer::u256_from_felt252(f10)"(%769, %770) : (i64, i252) -> (i64, !llvm.struct<(i128, i128)>)
    cf.br ^bb399(%771#1 : !llvm.struct<(i128, i128)>)
  ^bb399(%772: !llvm.struct<(i128, i128)>):  // pred: ^bb398
    %773 = llvm.extractvalue %772[0] : !llvm.struct<(i128, i128)> 
    %774 = llvm.extractvalue %772[1] : !llvm.struct<(i128, i128)> 
    cf.br ^bb400(%774 : i128)
  ^bb400(%775: i128):  // pred: ^bb399
    %776 = arith.extui %775 : i128 to i252
    cf.br ^bb401(%773 : i128)
  ^bb401(%777: i128):  // pred: ^bb400
    %778 = arith.extui %777 : i128 to i252
    cf.br ^bb402
  ^bb402:  // pred: ^bb401
    %c31_i32_102 = arith.constant 31 : i32
    cf.br ^bb403(%771#0 : i64)
  ^bb403(%779: i64):  // pred: ^bb402
    cf.br ^bb404(%c31_i32_102 : i32)
  ^bb404(%780: i32):  // pred: ^bb403
    cf.br ^bb405(%61 : i32)
  ^bb405(%781: i32):  // pred: ^bb404
    cf.br ^bb406(%781 : i32)
  ^bb406(%782: i32):  // pred: ^bb405
    cf.br ^bb407(%779, %780, %782 : i64, i32, i32)
  ^bb407(%783: i64, %784: i32, %785: i32):  // pred: ^bb406
    %786:2 = call @"core::integer::U32Sub::sub(f8)"(%783, %784, %785) : (i64, i32, i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %786#1, %15 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb408(%15 : !llvm.ptr)
  ^bb408(%787: !llvm.ptr):  // pred: ^bb407
    %788 = llvm.load %787 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %788 : i1, [
      default: ^bb409,
      0: ^bb410,
      1: ^bb411
    ]
  ^bb409:  // pred: ^bb408
    %false_103 = arith.constant false
    cf.assert %false_103, "Invalid enum tag."
    llvm.unreachable
  ^bb410:  // pred: ^bb408
    %789 = llvm.load %787 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %790 = llvm.extractvalue %789[1] : !llvm.struct<(i1, struct<(i32)>)> 
    cf.br ^bb412
  ^bb411:  // pred: ^bb408
    %791 = llvm.load %787 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %792 = llvm.extractvalue %791[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb475
  ^bb412:  // pred: ^bb410
    cf.br ^bb413(%790 : !llvm.struct<(i32)>)
  ^bb413(%793: !llvm.struct<(i32)>):  // pred: ^bb412
    %794 = llvm.extractvalue %793[0] : !llvm.struct<(i32)> 
    cf.br ^bb414(%786#0 : i64)
  ^bb414(%795: i64):  // pred: ^bb413
    cf.br ^bb415(%794 : i32)
  ^bb415(%796: i32):  // pred: ^bb414
    cf.br ^bb416(%795, %796 : i64, i32)
  ^bb416(%797: i64, %798: i32):  // pred: ^bb415
    %799:2 = call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%797, %798) : (i64, i32) -> (i64, !llvm.struct<(i64, array<32 x i8>)>)
    llvm.store %799#1, %17 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    cf.br ^bb417(%17 : !llvm.ptr)
  ^bb417(%800: !llvm.ptr):  // pred: ^bb416
    %801 = llvm.load %800 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %801 : i1, [
      default: ^bb418,
      0: ^bb419,
      1: ^bb420
    ]
  ^bb418:  // pred: ^bb417
    %false_104 = arith.constant false
    cf.assert %false_104, "Invalid enum tag."
    llvm.unreachable
  ^bb419:  // pred: ^bb417
    %802 = llvm.load %800 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i252)>)>
    %803 = llvm.extractvalue %802[1] : !llvm.struct<(i1, struct<(i252)>)> 
    cf.br ^bb421
  ^bb420:  // pred: ^bb417
    %804 = llvm.load %800 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %805 = llvm.extractvalue %804[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb464
  ^bb421:  // pred: ^bb419
    cf.br ^bb422(%803 : !llvm.struct<(i252)>)
  ^bb422(%806: !llvm.struct<(i252)>):  // pred: ^bb421
    %807 = llvm.extractvalue %806[0] : !llvm.struct<(i252)> 
    cf.br ^bb423(%58, %807 : i252, i252)
  ^bb423(%808: i252, %809: i252):  // pred: ^bb422
    %810 = arith.extui %808 : i252 to i512
    %811 = arith.extui %809 : i252 to i512
    %812 = arith.muli %810, %811 : i512
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_105 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512
    %813 = arith.remui %812, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_105 : i512
    %814 = arith.cmpi uge, %812, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_105 : i512
    %815 = arith.select %814, %813, %812 : i512
    %816 = arith.trunci %815 : i512 to i252
    cf.br ^bb424(%816 : i252)
  ^bb424(%817: i252):  // pred: ^bb423
    cf.br ^bb425(%776, %817 : i252, i252)
  ^bb425(%818: i252, %819: i252):  // pred: ^bb424
    %820 = arith.extui %818 : i252 to i256
    %821 = arith.extui %819 : i252 to i256
    %822 = arith.addi %820, %821 : i256
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_106 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256
    %823 = arith.subi %822, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_106 : i256
    %824 = arith.cmpi uge, %822, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_106 : i256
    %825 = arith.select %824, %823, %822 : i256
    %826 = arith.trunci %825 : i256 to i252
    cf.br ^bb426(%799#0 : i64)
  ^bb426(%827: i64):  // pred: ^bb425
    cf.br ^bb427(%826 : i252)
  ^bb427(%828: i252):  // pred: ^bb426
    cf.br ^bb428(%827, %828 : i64, i252)
  ^bb428(%829: i64, %830: i252):  // pred: ^bb427
    %831:2 = call @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%829, %830) : (i64, i252) -> (i64, !llvm.struct<(i64, array<32 x i8>)>)
    llvm.store %831#1, %19 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    cf.br ^bb429(%19 : !llvm.ptr)
  ^bb429(%832: !llvm.ptr):  // pred: ^bb428
    %833 = llvm.load %832 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %833 : i1, [
      default: ^bb430,
      0: ^bb431,
      1: ^bb432
    ]
  ^bb430:  // pred: ^bb429
    %false_107 = arith.constant false
    cf.assert %false_107, "Invalid enum tag."
    llvm.unreachable
  ^bb431:  // pred: ^bb429
    %834 = llvm.load %832 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i248)>
    %835 = llvm.extractvalue %834[1] : !llvm.struct<(i1, i248)> 
    cf.br ^bb433
  ^bb432:  // pred: ^bb429
    %836 = llvm.load %832 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<()>)>
    %837 = llvm.extractvalue %836[1] : !llvm.struct<(i1, struct<()>)> 
    cf.br ^bb448
  ^bb433:  // pred: ^bb431
    cf.br ^bb434(%57, %835 : !llvm.struct<(ptr<i248>, i32, i32)>, i248)
  ^bb434(%838: !llvm.struct<(ptr<i248>, i32, i32)>, %839: i248):  // pred: ^bb433
    %840 = llvm.extractvalue %838[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %841 = llvm.extractvalue %838[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %842 = llvm.extractvalue %838[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %843 = arith.cmpi uge, %841, %842 : i32
    %844:2 = scf.if %843 -> (!llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>) {
      %c8_i32 = arith.constant 8 : i32
      %1256 = arith.addi %842, %842 : i32
      %1257 = arith.maxui %c8_i32, %1256 : i32
      %1258 = arith.extui %1257 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1259 = arith.muli %1258, %c32_i64 : i64
      %1260 = llvm.bitcast %840 : !llvm.ptr<i248> to !llvm.ptr
      %1261 = func.call @realloc(%1260, %1259) : (!llvm.ptr, i64) -> !llvm.ptr
      %1262 = llvm.bitcast %1261 : !llvm.ptr to !llvm.ptr<i248>
      %1263 = llvm.insertvalue %1262, %838[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
      %1264 = llvm.insertvalue %1257, %1263[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
      scf.yield %1264, %1262 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>
    } else {
      scf.yield %838, %840 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>
    }
    %845 = llvm.getelementptr %844#1[%841] : (!llvm.ptr<i248>, i32) -> !llvm.ptr, i248
    llvm.store %839, %845 : i248, !llvm.ptr
    %c1_i32_108 = arith.constant 1 : i32
    %846 = arith.addi %841, %c1_i32_108 : i32
    %847 = llvm.insertvalue %846, %844#0[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    cf.br ^bb435(%847, %778, %781 : !llvm.struct<(ptr<i248>, i32, i32)>, i252, i32)
  ^bb435(%848: !llvm.struct<(ptr<i248>, i32, i32)>, %849: i252, %850: i32):  // pred: ^bb434
    %851 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %852 = llvm.insertvalue %848, %851[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %853 = llvm.insertvalue %849, %852[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %854 = llvm.insertvalue %850, %853[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb436(%831#0 : i64)
  ^bb436(%855: i64):  // pred: ^bb435
    cf.br ^bb437(%854 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb437(%856: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb436
    cf.br ^bb438(%108, %855, %856 : i32, i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb438(%857: i32, %858: i64, %859: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // 2 preds: ^bb299, ^bb437
    cf.br ^bb439(%859 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb439(%860: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb438
    %861 = llvm.extractvalue %860[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %862 = llvm.extractvalue %860[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %863 = llvm.extractvalue %860[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb440(%863 : i32)
  ^bb440(%864: i32):  // pred: ^bb439
    cf.br ^bb441
  ^bb441:  // pred: ^bb440
    %865 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb442(%861, %862, %857 : !llvm.struct<(ptr<i248>, i32, i32)>, i252, i32)
  ^bb442(%866: !llvm.struct<(ptr<i248>, i32, i32)>, %867: i252, %868: i32):  // pred: ^bb441
    %869 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %870 = llvm.insertvalue %866, %869[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %871 = llvm.insertvalue %867, %870[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %872 = llvm.insertvalue %868, %871[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb443(%872, %865 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<()>)
  ^bb443(%873: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %874: !llvm.struct<()>):  // pred: ^bb442
    %875 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>
    %876 = llvm.insertvalue %873, %875[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %877 = llvm.insertvalue %874, %876[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb444(%877 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb444(%878: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb443
    %false_109 = arith.constant false
    %879 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %880 = llvm.insertvalue %false_109, %879[0] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %881 = llvm.insertvalue %878, %880[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    llvm.store %881, %21 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>, !llvm.ptr
    cf.br ^bb445(%858 : i64)
  ^bb445(%882: i64):  // pred: ^bb444
    cf.br ^bb446(%21 : !llvm.ptr)
  ^bb446(%883: !llvm.ptr):  // pred: ^bb445
    cf.br ^bb447(%882, %883 : i64, !llvm.ptr)
  ^bb447(%884: i64, %885: !llvm.ptr):  // pred: ^bb446
    %886 = llvm.load %883 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %882, %886 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb448:  // pred: ^bb432
    cf.br ^bb449(%837 : !llvm.struct<()>)
  ^bb449(%887: !llvm.struct<()>):  // pred: ^bb448
    cf.br ^bb450(%108 : i32)
  ^bb450(%888: i32):  // pred: ^bb449
    cf.br ^bb451(%781 : i32)
  ^bb451(%889: i32):  // pred: ^bb450
    cf.br ^bb452(%778 : i252)
  ^bb452(%890: i252):  // pred: ^bb451
    cf.br ^bb453(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb453(%891: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb452
    %892 = llvm.extractvalue %891[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %893 = llvm.bitcast %892 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%893) : (!llvm.ptr) -> ()
    cf.br ^bb454
  ^bb454:  // pred: ^bb453
    %894 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_110 = arith.constant 0 : i32
    %895 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %896 = llvm.insertvalue %894, %895[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %897 = llvm.insertvalue %c0_i32_110, %896[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %898 = llvm.insertvalue %c0_i32_110, %897[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb455
  ^bb455:  // pred: ^bb454
    %c29721761890975875353235833581453094220424382983267374_i252_111 = arith.constant 29721761890975875353235833581453094220424382983267374 : i252
    cf.br ^bb456(%c29721761890975875353235833581453094220424382983267374_i252_111 : i252)
  ^bb456(%899: i252):  // pred: ^bb455
    cf.br ^bb457(%898, %899 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb457(%900: !llvm.struct<(ptr<i252>, i32, i32)>, %901: i252):  // pred: ^bb456
    %902 = llvm.extractvalue %900[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %903 = llvm.extractvalue %900[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %904 = llvm.extractvalue %900[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %905 = arith.cmpi uge, %903, %904 : i32
    %906:2 = scf.if %905 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %1256 = arith.addi %904, %904 : i32
      %1257 = arith.maxui %c8_i32, %1256 : i32
      %1258 = arith.extui %1257 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1259 = arith.muli %1258, %c32_i64 : i64
      %1260 = llvm.bitcast %902 : !llvm.ptr<i252> to !llvm.ptr
      %1261 = func.call @realloc(%1260, %1259) : (!llvm.ptr, i64) -> !llvm.ptr
      %1262 = llvm.bitcast %1261 : !llvm.ptr to !llvm.ptr<i252>
      %1263 = llvm.insertvalue %1262, %900[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1264 = llvm.insertvalue %1257, %1263[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1264, %1262 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %900, %902 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %907 = llvm.getelementptr %906#1[%903] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %901, %907 : i252, !llvm.ptr
    %c1_i32_112 = arith.constant 1 : i32
    %908 = arith.addi %903, %c1_i32_112 : i32
    %909 = llvm.insertvalue %908, %906#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb458
  ^bb458:  // pred: ^bb457
    %910 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb459(%910, %909 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb459(%911: !llvm.struct<()>, %912: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb458
    %913 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %914 = llvm.insertvalue %911, %913[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %915 = llvm.insertvalue %912, %914[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb460(%915 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb460(%916: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb459
    %true_113 = arith.constant true
    %917 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %918 = llvm.insertvalue %true_113, %917[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %919 = llvm.insertvalue %916, %918[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %919, %20 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb461(%831#0 : i64)
  ^bb461(%920: i64):  // pred: ^bb460
    cf.br ^bb462(%20 : !llvm.ptr)
  ^bb462(%921: !llvm.ptr):  // pred: ^bb461
    cf.br ^bb463(%920, %921 : i64, !llvm.ptr)
  ^bb463(%922: i64, %923: !llvm.ptr):  // pred: ^bb462
    %924 = llvm.load %921 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %920, %924 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb464:  // pred: ^bb420
    cf.br ^bb465(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb465(%925: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb464
    %926 = llvm.extractvalue %925[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %927 = llvm.bitcast %926 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%927) : (!llvm.ptr) -> ()
    cf.br ^bb466(%108 : i32)
  ^bb466(%928: i32):  // pred: ^bb465
    cf.br ^bb467(%781 : i32)
  ^bb467(%929: i32):  // pred: ^bb466
    cf.br ^bb468(%778 : i252)
  ^bb468(%930: i252):  // pred: ^bb467
    cf.br ^bb469(%776 : i252)
  ^bb469(%931: i252):  // pred: ^bb468
    cf.br ^bb470(%58 : i252)
  ^bb470(%932: i252):  // pred: ^bb469
    cf.br ^bb471(%805 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb471(%933: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb470
    %true_114 = arith.constant true
    %934 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %935 = llvm.insertvalue %true_114, %934[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %936 = llvm.insertvalue %933, %935[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %936, %18 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb472(%799#0 : i64)
  ^bb472(%937: i64):  // pred: ^bb471
    cf.br ^bb473(%18 : !llvm.ptr)
  ^bb473(%938: !llvm.ptr):  // pred: ^bb472
    cf.br ^bb474(%937, %938 : i64, !llvm.ptr)
  ^bb474(%939: i64, %940: !llvm.ptr):  // pred: ^bb473
    %941 = llvm.load %938 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %937, %941 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb475:  // pred: ^bb411
    cf.br ^bb476(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb476(%942: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb475
    %943 = llvm.extractvalue %942[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %944 = llvm.bitcast %943 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%944) : (!llvm.ptr) -> ()
    cf.br ^bb477(%108 : i32)
  ^bb477(%945: i32):  // pred: ^bb476
    cf.br ^bb478(%781 : i32)
  ^bb478(%946: i32):  // pred: ^bb477
    cf.br ^bb479(%778 : i252)
  ^bb479(%947: i252):  // pred: ^bb478
    cf.br ^bb480(%58 : i252)
  ^bb480(%948: i252):  // pred: ^bb479
    cf.br ^bb481(%776 : i252)
  ^bb481(%949: i252):  // pred: ^bb480
    cf.br ^bb482(%792 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb482(%950: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb481
    %true_115 = arith.constant true
    %951 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %952 = llvm.insertvalue %true_115, %951[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %953 = llvm.insertvalue %950, %952[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %953, %16 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb483(%786#0 : i64)
  ^bb483(%954: i64):  // pred: ^bb482
    cf.br ^bb484(%16 : !llvm.ptr)
  ^bb484(%955: !llvm.ptr):  // pred: ^bb483
    cf.br ^bb485(%954, %955 : i64, !llvm.ptr)
  ^bb485(%956: i64, %957: !llvm.ptr):  // pred: ^bb484
    %958 = llvm.load %955 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %954, %958 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb486:  // pred: ^bb39
    cf.br ^bb487(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb487(%959: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb486
    %960 = llvm.extractvalue %959[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %961 = llvm.bitcast %960 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%961) : (!llvm.ptr) -> ()
    cf.br ^bb488(%61 : i32)
  ^bb488(%962: i32):  // pred: ^bb487
    cf.br ^bb489(%58 : i252)
  ^bb489(%963: i252):  // pred: ^bb488
    cf.br ^bb490(%50 : i252)
  ^bb490(%964: i252):  // pred: ^bb489
    cf.br ^bb491(%105 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb491(%965: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb490
    %true_116 = arith.constant true
    %966 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %967 = llvm.insertvalue %true_116, %966[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %968 = llvm.insertvalue %965, %967[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %968, %14 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb492(%99#0 : i64)
  ^bb492(%969: i64):  // pred: ^bb491
    cf.br ^bb493(%14 : !llvm.ptr)
  ^bb493(%970: !llvm.ptr):  // pred: ^bb492
    cf.br ^bb494(%969, %970 : i64, !llvm.ptr)
  ^bb494(%971: i64, %972: !llvm.ptr):  // pred: ^bb493
    %973 = llvm.load %970 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %969, %973 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb495:  // pred: ^bb28
    cf.br ^bb496(%61 : i32)
  ^bb496(%974: i32):  // pred: ^bb495
    cf.br ^bb497(%87 : i32)
  ^bb497(%975: i32):  // pred: ^bb496
    cf.br ^bb498(%88 : i64)
  ^bb498(%976: i64):  // pred: ^bb497
    cf.br ^bb499(%63 : i32)
  ^bb499(%977: i32):  // pred: ^bb498
    cf.br ^bb500(%976, %977 : i64, i32)
  ^bb500(%978: i64, %979: i32):  // pred: ^bb499
    %980:2 = call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%978, %979) : (i64, i32) -> (i64, !llvm.struct<(i64, array<32 x i8>)>)
    llvm.store %980#1, %8 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    cf.br ^bb501(%8 : !llvm.ptr)
  ^bb501(%981: !llvm.ptr):  // pred: ^bb500
    %982 = llvm.load %981 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %982 : i1, [
      default: ^bb502,
      0: ^bb503,
      1: ^bb504
    ]
  ^bb502:  // pred: ^bb501
    %false_117 = arith.constant false
    cf.assert %false_117, "Invalid enum tag."
    llvm.unreachable
  ^bb503:  // pred: ^bb501
    %983 = llvm.load %981 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i252)>)>
    %984 = llvm.extractvalue %983[1] : !llvm.struct<(i1, struct<(i252)>)> 
    cf.br ^bb505
  ^bb504:  // pred: ^bb501
    %985 = llvm.load %981 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %986 = llvm.extractvalue %985[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb541
  ^bb505:  // pred: ^bb503
    cf.br ^bb506(%984 : !llvm.struct<(i252)>)
  ^bb506(%987: !llvm.struct<(i252)>):  // pred: ^bb505
    %988 = llvm.extractvalue %987[0] : !llvm.struct<(i252)> 
    cf.br ^bb507(%58, %988 : i252, i252)
  ^bb507(%989: i252, %990: i252):  // pred: ^bb506
    %991 = arith.extui %989 : i252 to i512
    %992 = arith.extui %990 : i252 to i512
    %993 = arith.muli %991, %992 : i512
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_118 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512
    %994 = arith.remui %993, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_118 : i512
    %995 = arith.cmpi uge, %993, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_118 : i512
    %996 = arith.select %995, %994, %993 : i512
    %997 = arith.trunci %996 : i512 to i252
    cf.br ^bb508(%997 : i252)
  ^bb508(%998: i252):  // pred: ^bb507
    cf.br ^bb509(%50, %998 : i252, i252)
  ^bb509(%999: i252, %1000: i252):  // pred: ^bb508
    %1001 = arith.extui %999 : i252 to i256
    %1002 = arith.extui %1000 : i252 to i256
    %1003 = arith.addi %1001, %1002 : i256
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_119 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256
    %1004 = arith.subi %1003, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_119 : i256
    %1005 = arith.cmpi uge, %1003, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_119 : i256
    %1006 = arith.select %1005, %1004, %1003 : i256
    %1007 = arith.trunci %1006 : i256 to i252
    cf.br ^bb510(%980#0 : i64)
  ^bb510(%1008: i64):  // pred: ^bb509
    cf.br ^bb511(%1007 : i252)
  ^bb511(%1009: i252):  // pred: ^bb510
    cf.br ^bb512(%1008, %1009 : i64, i252)
  ^bb512(%1010: i64, %1011: i252):  // pred: ^bb511
    %1012:2 = call @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%1010, %1011) : (i64, i252) -> (i64, !llvm.struct<(i64, array<32 x i8>)>)
    llvm.store %1012#1, %10 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    cf.br ^bb513(%10 : !llvm.ptr)
  ^bb513(%1013: !llvm.ptr):  // pred: ^bb512
    %1014 = llvm.load %1013 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %1014 : i1, [
      default: ^bb514,
      0: ^bb515,
      1: ^bb516
    ]
  ^bb514:  // pred: ^bb513
    %false_120 = arith.constant false
    cf.assert %false_120, "Invalid enum tag."
    llvm.unreachable
  ^bb515:  // pred: ^bb513
    %1015 = llvm.load %1013 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i248)>
    %1016 = llvm.extractvalue %1015[1] : !llvm.struct<(i1, i248)> 
    cf.br ^bb517
  ^bb516:  // pred: ^bb513
    %1017 = llvm.load %1013 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<()>)>
    %1018 = llvm.extractvalue %1017[1] : !llvm.struct<(i1, struct<()>)> 
    cf.br ^bb528
  ^bb517:  // pred: ^bb515
    cf.br ^bb518(%57, %1016 : !llvm.struct<(ptr<i248>, i32, i32)>, i248)
  ^bb518(%1019: !llvm.struct<(ptr<i248>, i32, i32)>, %1020: i248):  // pred: ^bb517
    %1021 = llvm.extractvalue %1019[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %1022 = llvm.extractvalue %1019[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %1023 = llvm.extractvalue %1019[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %1024 = arith.cmpi uge, %1022, %1023 : i32
    %1025:2 = scf.if %1024 -> (!llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>) {
      %c8_i32 = arith.constant 8 : i32
      %1256 = arith.addi %1023, %1023 : i32
      %1257 = arith.maxui %c8_i32, %1256 : i32
      %1258 = arith.extui %1257 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1259 = arith.muli %1258, %c32_i64 : i64
      %1260 = llvm.bitcast %1021 : !llvm.ptr<i248> to !llvm.ptr
      %1261 = func.call @realloc(%1260, %1259) : (!llvm.ptr, i64) -> !llvm.ptr
      %1262 = llvm.bitcast %1261 : !llvm.ptr to !llvm.ptr<i248>
      %1263 = llvm.insertvalue %1262, %1019[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
      %1264 = llvm.insertvalue %1257, %1263[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
      scf.yield %1264, %1262 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>
    } else {
      scf.yield %1019, %1021 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>
    }
    %1026 = llvm.getelementptr %1025#1[%1022] : (!llvm.ptr<i248>, i32) -> !llvm.ptr, i248
    llvm.store %1020, %1026 : i248, !llvm.ptr
    %c1_i32_121 = arith.constant 1 : i32
    %1027 = arith.addi %1022, %c1_i32_121 : i32
    %1028 = llvm.insertvalue %1027, %1025#0[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    cf.br ^bb519
  ^bb519:  // pred: ^bb518
    %c0_i252 = arith.constant 0 : i252
    cf.br ^bb520
  ^bb520:  // pred: ^bb519
    %c0_i32_122 = arith.constant 0 : i32
    cf.br ^bb521
  ^bb521:  // pred: ^bb520
    %1029 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb522(%1028, %c0_i252, %c0_i32_122 : !llvm.struct<(ptr<i248>, i32, i32)>, i252, i32)
  ^bb522(%1030: !llvm.struct<(ptr<i248>, i32, i32)>, %1031: i252, %1032: i32):  // pred: ^bb521
    %1033 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %1034 = llvm.insertvalue %1030, %1033[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %1035 = llvm.insertvalue %1031, %1034[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %1036 = llvm.insertvalue %1032, %1035[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb523(%1036, %1029 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<()>)
  ^bb523(%1037: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %1038: !llvm.struct<()>):  // pred: ^bb522
    %1039 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>
    %1040 = llvm.insertvalue %1037, %1039[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %1041 = llvm.insertvalue %1038, %1040[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb524(%1041 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb524(%1042: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb523
    %false_123 = arith.constant false
    %1043 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %1044 = llvm.insertvalue %false_123, %1043[0] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %1045 = llvm.insertvalue %1042, %1044[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    llvm.store %1045, %12 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>, !llvm.ptr
    cf.br ^bb525(%1012#0 : i64)
  ^bb525(%1046: i64):  // pred: ^bb524
    cf.br ^bb526(%12 : !llvm.ptr)
  ^bb526(%1047: !llvm.ptr):  // pred: ^bb525
    cf.br ^bb527(%1046, %1047 : i64, !llvm.ptr)
  ^bb527(%1048: i64, %1049: !llvm.ptr):  // pred: ^bb526
    %1050 = llvm.load %1047 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %1046, %1050 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb528:  // pred: ^bb516
    cf.br ^bb529(%1018 : !llvm.struct<()>)
  ^bb529(%1051: !llvm.struct<()>):  // pred: ^bb528
    cf.br ^bb530(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb530(%1052: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb529
    %1053 = llvm.extractvalue %1052[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %1054 = llvm.bitcast %1053 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%1054) : (!llvm.ptr) -> ()
    cf.br ^bb531
  ^bb531:  // pred: ^bb530
    %1055 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_124 = arith.constant 0 : i32
    %1056 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %1057 = llvm.insertvalue %1055, %1056[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %1058 = llvm.insertvalue %c0_i32_124, %1057[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %1059 = llvm.insertvalue %c0_i32_124, %1058[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb532
  ^bb532:  // pred: ^bb531
    %c29721761890975875353235833581453094220424382983267374_i252_125 = arith.constant 29721761890975875353235833581453094220424382983267374 : i252
    cf.br ^bb533(%c29721761890975875353235833581453094220424382983267374_i252_125 : i252)
  ^bb533(%1060: i252):  // pred: ^bb532
    cf.br ^bb534(%1059, %1060 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb534(%1061: !llvm.struct<(ptr<i252>, i32, i32)>, %1062: i252):  // pred: ^bb533
    %1063 = llvm.extractvalue %1061[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %1064 = llvm.extractvalue %1061[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %1065 = llvm.extractvalue %1061[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %1066 = arith.cmpi uge, %1064, %1065 : i32
    %1067:2 = scf.if %1066 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %1256 = arith.addi %1065, %1065 : i32
      %1257 = arith.maxui %c8_i32, %1256 : i32
      %1258 = arith.extui %1257 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %1259 = arith.muli %1258, %c32_i64 : i64
      %1260 = llvm.bitcast %1063 : !llvm.ptr<i252> to !llvm.ptr
      %1261 = func.call @realloc(%1260, %1259) : (!llvm.ptr, i64) -> !llvm.ptr
      %1262 = llvm.bitcast %1261 : !llvm.ptr to !llvm.ptr<i252>
      %1263 = llvm.insertvalue %1262, %1061[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %1264 = llvm.insertvalue %1257, %1263[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %1264, %1262 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %1061, %1063 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %1068 = llvm.getelementptr %1067#1[%1064] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %1062, %1068 : i252, !llvm.ptr
    %c1_i32_126 = arith.constant 1 : i32
    %1069 = arith.addi %1064, %c1_i32_126 : i32
    %1070 = llvm.insertvalue %1069, %1067#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb535
  ^bb535:  // pred: ^bb534
    %1071 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb536(%1071, %1070 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb536(%1072: !llvm.struct<()>, %1073: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb535
    %1074 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %1075 = llvm.insertvalue %1072, %1074[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %1076 = llvm.insertvalue %1073, %1075[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb537(%1076 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb537(%1077: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb536
    %true_127 = arith.constant true
    %1078 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %1079 = llvm.insertvalue %true_127, %1078[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %1080 = llvm.insertvalue %1077, %1079[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %1080, %11 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb538(%1012#0 : i64)
  ^bb538(%1081: i64):  // pred: ^bb537
    cf.br ^bb539(%11 : !llvm.ptr)
  ^bb539(%1082: !llvm.ptr):  // pred: ^bb538
    cf.br ^bb540(%1081, %1082 : i64, !llvm.ptr)
  ^bb540(%1083: i64, %1084: !llvm.ptr):  // pred: ^bb539
    %1085 = llvm.load %1082 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %1081, %1085 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb541:  // pred: ^bb504
    cf.br ^bb542(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb542(%1086: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb541
    %1087 = llvm.extractvalue %1086[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %1088 = llvm.bitcast %1087 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%1088) : (!llvm.ptr) -> ()
    cf.br ^bb543(%50 : i252)
  ^bb543(%1089: i252):  // pred: ^bb542
    cf.br ^bb544(%58 : i252)
  ^bb544(%1090: i252):  // pred: ^bb543
    cf.br ^bb545(%986 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb545(%1091: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb544
    %true_128 = arith.constant true
    %1092 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %1093 = llvm.insertvalue %true_128, %1092[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %1094 = llvm.insertvalue %1091, %1093[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %1094, %9 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb546(%980#0 : i64)
  ^bb546(%1095: i64):  // pred: ^bb545
    cf.br ^bb547(%9 : !llvm.ptr)
  ^bb547(%1096: !llvm.ptr):  // pred: ^bb546
    cf.br ^bb548(%1095, %1096 : i64, !llvm.ptr)
  ^bb548(%1097: i64, %1098: !llvm.ptr):  // pred: ^bb547
    %1099 = llvm.load %1096 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %1095, %1099 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb549:  // pred: ^bb22
    cf.br ^bb550(%84 : i32)
  ^bb550(%1100: i32):  // pred: ^bb549
    cf.br ^bb551(%77 : i32)
  ^bb551(%1101: i32):  // pred: ^bb550
    cf.br ^bb552
  ^bb552:  // pred: ^bb551
    %c0_i32_129 = arith.constant 0 : i32
    cf.br ^bb553(%61 : i32)
  ^bb553(%1102: i32):  // pred: ^bb552
    cf.br ^bb554(%82 : i64)
  ^bb554(%1103: i64):  // pred: ^bb553
    cf.br ^bb555(%1102, %c0_i32_129 : i32, i32)
  ^bb555(%1104: i32, %1105: i32):  // pred: ^bb554
    %1106 = arith.cmpi eq, %1104, %1105 : i32
    cf.cond_br %1106, ^bb603, ^bb556
  ^bb556:  // pred: ^bb555
    cf.br ^bb557(%1103 : i64)
  ^bb557(%1107: i64):  // pred: ^bb556
    cf.br ^bb558(%63 : i32)
  ^bb558(%1108: i32):  // pred: ^bb557
    cf.br ^bb559(%1108 : i32)
  ^bb559(%1109: i32):  // pred: ^bb558
    cf.br ^bb560(%1107, %1109 : i64, i32)
  ^bb560(%1110: i64, %1111: i32):  // pred: ^bb559
    %1112:2 = call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%1110, %1111) : (i64, i32) -> (i64, !llvm.struct<(i64, array<32 x i8>)>)
    llvm.store %1112#1, %4 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    cf.br ^bb561(%4 : !llvm.ptr)
  ^bb561(%1113: !llvm.ptr):  // pred: ^bb560
    %1114 = llvm.load %1113 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %1114 : i1, [
      default: ^bb562,
      0: ^bb563,
      1: ^bb564
    ]
  ^bb562:  // pred: ^bb561
    %false_130 = arith.constant false
    cf.assert %false_130, "Invalid enum tag."
    llvm.unreachable
  ^bb563:  // pred: ^bb561
    %1115 = llvm.load %1113 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i252)>)>
    %1116 = llvm.extractvalue %1115[1] : !llvm.struct<(i1, struct<(i252)>)> 
    cf.br ^bb565
  ^bb564:  // pred: ^bb561
    %1117 = llvm.load %1113 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %1118 = llvm.extractvalue %1117[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb593
  ^bb565:  // pred: ^bb563
    cf.br ^bb566(%1112#0 : i64)
  ^bb566(%1119: i64):  // pred: ^bb565
    cf.br ^bb567(%1102 : i32)
  ^bb567(%1120: i32):  // pred: ^bb566
    cf.br ^bb568(%1108 : i32)
  ^bb568(%1121: i32):  // pred: ^bb567
    cf.br ^bb569(%1119, %1120, %1121 : i64, i32, i32)
  ^bb569(%1122: i64, %1123: i32, %1124: i32):  // pred: ^bb568
    %1125:2 = call @"core::integer::U32Add::add(f4)"(%1122, %1123, %1124) : (i64, i32, i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %1125#1, %6 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb570(%6 : !llvm.ptr)
  ^bb570(%1126: !llvm.ptr):  // pred: ^bb569
    %1127 = llvm.load %1126 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %1127 : i1, [
      default: ^bb571,
      0: ^bb572,
      1: ^bb573
    ]
  ^bb571:  // pred: ^bb570
    %false_131 = arith.constant false
    cf.assert %false_131, "Invalid enum tag."
    llvm.unreachable
  ^bb572:  // pred: ^bb570
    %1128 = llvm.load %1126 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %1129 = llvm.extractvalue %1128[1] : !llvm.struct<(i1, struct<(i32)>)> 
    cf.br ^bb574
  ^bb573:  // pred: ^bb570
    %1130 = llvm.load %1126 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %1131 = llvm.extractvalue %1130[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb584
  ^bb574:  // pred: ^bb572
    cf.br ^bb575(%1116 : !llvm.struct<(i252)>)
  ^bb575(%1132: !llvm.struct<(i252)>):  // pred: ^bb574
    %1133 = llvm.extractvalue %1132[0] : !llvm.struct<(i252)> 
    cf.br ^bb576(%58, %1133 : i252, i252)
  ^bb576(%1134: i252, %1135: i252):  // pred: ^bb575
    %1136 = arith.extui %1134 : i252 to i512
    %1137 = arith.extui %1135 : i252 to i512
    %1138 = arith.muli %1136, %1137 : i512
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_132 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512
    %1139 = arith.remui %1138, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_132 : i512
    %1140 = arith.cmpi uge, %1138, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512_132 : i512
    %1141 = arith.select %1140, %1139, %1138 : i512
    %1142 = arith.trunci %1141 : i512 to i252
    cf.br ^bb577(%1142 : i252)
  ^bb577(%1143: i252):  // pred: ^bb576
    cf.br ^bb578(%50, %1143 : i252, i252)
  ^bb578(%1144: i252, %1145: i252):  // pred: ^bb577
    %1146 = arith.extui %1144 : i252 to i256
    %1147 = arith.extui %1145 : i252 to i256
    %1148 = arith.addi %1146, %1147 : i256
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_133 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256
    %1149 = arith.subi %1148, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_133 : i256
    %1150 = arith.cmpi uge, %1148, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256_133 : i256
    %1151 = arith.select %1150, %1149, %1148 : i256
    %1152 = arith.trunci %1151 : i256 to i252
    cf.br ^bb579(%1129 : !llvm.struct<(i32)>)
  ^bb579(%1153: !llvm.struct<(i32)>):  // pred: ^bb578
    %1154 = llvm.extractvalue %1153[0] : !llvm.struct<(i32)> 
    cf.br ^bb580(%57, %1152, %1154 : !llvm.struct<(ptr<i248>, i32, i32)>, i252, i32)
  ^bb580(%1155: !llvm.struct<(ptr<i248>, i32, i32)>, %1156: i252, %1157: i32):  // pred: ^bb579
    %1158 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %1159 = llvm.insertvalue %1155, %1158[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %1160 = llvm.insertvalue %1156, %1159[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %1161 = llvm.insertvalue %1157, %1160[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb581(%1125#0 : i64)
  ^bb581(%1162: i64):  // pred: ^bb580
    cf.br ^bb582(%1161 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb582(%1163: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb581
    cf.br ^bb583
  ^bb583:  // pred: ^bb582
    cf.br ^bb609(%1162, %1163 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb584:  // pred: ^bb573
    cf.br ^bb585(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb585(%1164: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb584
    %1165 = llvm.extractvalue %1164[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %1166 = llvm.bitcast %1165 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%1166) : (!llvm.ptr) -> ()
    cf.br ^bb586(%1116 : !llvm.struct<(i252)>)
  ^bb586(%1167: !llvm.struct<(i252)>):  // pred: ^bb585
    cf.br ^bb587(%50 : i252)
  ^bb587(%1168: i252):  // pred: ^bb586
    cf.br ^bb588(%58 : i252)
  ^bb588(%1169: i252):  // pred: ^bb587
    cf.br ^bb589(%1131 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb589(%1170: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb588
    %true_134 = arith.constant true
    %1171 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %1172 = llvm.insertvalue %true_134, %1171[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %1173 = llvm.insertvalue %1170, %1172[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %1173, %7 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb590(%1125#0 : i64)
  ^bb590(%1174: i64):  // pred: ^bb589
    cf.br ^bb591(%7 : !llvm.ptr)
  ^bb591(%1175: !llvm.ptr):  // pred: ^bb590
    cf.br ^bb592(%1174, %1175 : i64, !llvm.ptr)
  ^bb592(%1176: i64, %1177: !llvm.ptr):  // pred: ^bb591
    %1178 = llvm.load %1175 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %1174, %1178 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb593:  // pred: ^bb564
    cf.br ^bb594(%58 : i252)
  ^bb594(%1179: i252):  // pred: ^bb593
    cf.br ^bb595(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb595(%1180: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb594
    %1181 = llvm.extractvalue %1180[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %1182 = llvm.bitcast %1181 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%1182) : (!llvm.ptr) -> ()
    cf.br ^bb596(%50 : i252)
  ^bb596(%1183: i252):  // pred: ^bb595
    cf.br ^bb597(%1108 : i32)
  ^bb597(%1184: i32):  // pred: ^bb596
    cf.br ^bb598(%1102 : i32)
  ^bb598(%1185: i32):  // pred: ^bb597
    cf.br ^bb599(%1118 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb599(%1186: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb598
    %true_135 = arith.constant true
    %1187 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %1188 = llvm.insertvalue %true_135, %1187[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %1189 = llvm.insertvalue %1186, %1188[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %1189, %5 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb600(%1112#0 : i64)
  ^bb600(%1190: i64):  // pred: ^bb599
    cf.br ^bb601(%5 : !llvm.ptr)
  ^bb601(%1191: !llvm.ptr):  // pred: ^bb600
    cf.br ^bb602(%1190, %1191 : i64, !llvm.ptr)
  ^bb602(%1192: i64, %1193: !llvm.ptr):  // pred: ^bb601
    %1194 = llvm.load %1191 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %1190, %1194 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb603:  // pred: ^bb555
    cf.br ^bb604(%58 : i252)
  ^bb604(%1195: i252):  // pred: ^bb603
    cf.br ^bb605(%1102 : i32)
  ^bb605(%1196: i32):  // pred: ^bb604
    cf.br ^bb606(%57, %50, %63 : !llvm.struct<(ptr<i248>, i32, i32)>, i252, i32)
  ^bb606(%1197: !llvm.struct<(ptr<i248>, i32, i32)>, %1198: i252, %1199: i32):  // pred: ^bb605
    %1200 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %1201 = llvm.insertvalue %1197, %1200[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %1202 = llvm.insertvalue %1198, %1201[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %1203 = llvm.insertvalue %1199, %1202[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb607(%1103 : i64)
  ^bb607(%1204: i64):  // pred: ^bb606
    cf.br ^bb608(%1203 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb608(%1205: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb607
    cf.br ^bb609(%1204, %1205 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb609(%1206: i64, %1207: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // 2 preds: ^bb583, ^bb608
    cf.br ^bb610
  ^bb610:  // pred: ^bb609
    %1208 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb611(%1207, %1208 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<()>)
  ^bb611(%1209: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %1210: !llvm.struct<()>):  // pred: ^bb610
    %1211 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>
    %1212 = llvm.insertvalue %1209, %1211[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %1213 = llvm.insertvalue %1210, %1212[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb612(%1213 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb612(%1214: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb611
    %false_136 = arith.constant false
    %1215 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %1216 = llvm.insertvalue %false_136, %1215[0] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %1217 = llvm.insertvalue %1214, %1216[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    llvm.store %1217, %3 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>, !llvm.ptr
    cf.br ^bb613(%1206 : i64)
  ^bb613(%1218: i64):  // pred: ^bb612
    cf.br ^bb614(%3 : !llvm.ptr)
  ^bb614(%1219: !llvm.ptr):  // pred: ^bb613
    cf.br ^bb615(%1218, %1219 : i64, !llvm.ptr)
  ^bb615(%1220: i64, %1221: !llvm.ptr):  // pred: ^bb614
    %1222 = llvm.load %1219 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %1218, %1222 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb616:  // pred: ^bb16
    cf.br ^bb617(%57 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb617(%1223: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb616
    %1224 = llvm.extractvalue %1223[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %1225 = llvm.bitcast %1224 : !llvm.ptr<i248> to !llvm.ptr
    call @free(%1225) : (!llvm.ptr) -> ()
    cf.br ^bb618(%50 : i252)
  ^bb618(%1226: i252):  // pred: ^bb617
    cf.br ^bb619(%61 : i32)
  ^bb619(%1227: i32):  // pred: ^bb618
    cf.br ^bb620(%58 : i252)
  ^bb620(%1228: i252):  // pred: ^bb619
    cf.br ^bb621(%63 : i32)
  ^bb621(%1229: i32):  // pred: ^bb620
    cf.br ^bb622(%74 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb622(%1230: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb621
    %true_137 = arith.constant true
    %1231 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %1232 = llvm.insertvalue %true_137, %1231[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %1233 = llvm.insertvalue %1230, %1232[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %1233, %2 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb623(%68#0 : i64)
  ^bb623(%1234: i64):  // pred: ^bb622
    cf.br ^bb624(%2 : !llvm.ptr)
  ^bb624(%1235: !llvm.ptr):  // pred: ^bb623
    cf.br ^bb625(%1234, %1235 : i64, !llvm.ptr)
  ^bb625(%1236: i64, %1237: !llvm.ptr):  // pred: ^bb624
    %1238 = llvm.load %1235 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %1234, %1238 : i64, !llvm.struct<(i64, array<52 x i8>)>
  ^bb626:  // pred: ^bb4
    cf.br ^bb627(%50 : i252)
  ^bb627(%1239: i252):  // pred: ^bb626
    cf.br ^bb628(%52 : i32)
  ^bb628(%1240: i32):  // pred: ^bb627
    cf.br ^bb629
  ^bb629:  // pred: ^bb628
    %1241 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb630(%49, %1241 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<()>)
  ^bb630(%1242: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %1243: !llvm.struct<()>):  // pred: ^bb629
    %1244 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>
    %1245 = llvm.insertvalue %1242, %1244[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %1246 = llvm.insertvalue %1243, %1245[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    cf.br ^bb631(%1246 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)
  ^bb631(%1247: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>):  // pred: ^bb630
    %false_138 = arith.constant false
    %1248 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %1249 = llvm.insertvalue %false_138, %1248[0] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %1250 = llvm.insertvalue %1247, %1249[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    llvm.store %1250, %0 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>, !llvm.ptr
    cf.br ^bb632(%48 : i64)
  ^bb632(%1251: i64):  // pred: ^bb631
    cf.br ^bb633(%0 : !llvm.ptr)
  ^bb633(%1252: !llvm.ptr):  // pred: ^bb632
    cf.br ^bb634(%1251, %1252 : i64, !llvm.ptr)
  ^bb634(%1253: i64, %1254: !llvm.ptr):  // pred: ^bb633
    %1255 = llvm.load %1252 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    return %1251, %1255 : i64, !llvm.struct<(i64, array<52 x i8>)>
  }
  func.func public @"core::result::ResultTraitImpl::<(), core::fmt::Error>::unwrap::<core::fmt::ErrorDrop>(f1)"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(i1, array<0 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg1 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb1(%1: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %c30828113188794245257250221355944970489240709081949230_i252 = arith.constant 30828113188794245257250221355944970489240709081949230 : i252
    cf.br ^bb3(%1 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb3(%2: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb2
    cf.br ^bb4(%c30828113188794245257250221355944970489240709081949230_i252 : i252)
  ^bb4(%3: i252):  // pred: ^bb3
    cf.br ^bb5(%2, %3 : !llvm.struct<(i1, array<0 x i8>)>, i252)
  ^bb5(%4: !llvm.struct<(i1, array<0 x i8>)>, %5: i252):  // pred: ^bb4
    call @"core::result::ResultTraitImpl::<(), core::fmt::Error>::expect::<core::fmt::ErrorDrop>(f2)"(%0, %4, %5) : (!llvm.ptr, !llvm.struct<(i1, array<0 x i8>)>, i252) -> ()
    %6 = llvm.getelementptr %0[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb6(%6 : !llvm.ptr)
  ^bb6(%7: !llvm.ptr):  // pred: ^bb5
    %c24_i64 = arith.constant 24 : i64
    %false = arith.constant false
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %6, %c24_i64, %false) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    return
  }
  func.func public @"core::byte_array::ByteArraySerde::serialize(f0)"(%arg0: i64, %arg1: i128, %arg2: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %arg3: !llvm.struct<(ptr<i252>, i32, i32)>) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_1 = arith.constant 1 : i64
    %2 = llvm.alloca %c1_i64_1 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1, %arg2, %arg3 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb1(%3: i64, %4: i128, %5: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %6: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%5 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb3(%7: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb2
    cf.br ^bb4(%7 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb4(%8: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb3
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb5(%10 : i252)
  ^bb5(%12: i252):  // pred: ^bb4
    cf.br ^bb6(%11 : i32)
  ^bb6(%13: i32):  // pred: ^bb5
    cf.br ^bb7(%3 : i64)
  ^bb7(%14: i64):  // pred: ^bb6
    cf.br ^bb8(%4 : i128)
  ^bb8(%15: i128):  // pred: ^bb7
    cf.br ^bb9(%9 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb9(%16: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb8
    cf.br ^bb10(%6 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb10(%17: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb9
    cf.br ^bb11(%14, %15, %16, %17 : i64, i128, !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb11(%18: i64, %19: i128, %20: !llvm.struct<(ptr<i248>, i32, i32)>, %21: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb10
    %22:3 = call @"core::array::ArraySerde::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>::serialize(f22)"(%18, %19, %20, %21) : (i64, i128, !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %22#2, %0 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb12(%0 : !llvm.ptr)
  ^bb12(%23: !llvm.ptr):  // pred: ^bb11
    %24 = llvm.load %23 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %24 : i1, [
      default: ^bb13,
      0: ^bb14,
      1: ^bb15
    ]
  ^bb13:  // pred: ^bb12
    %false = arith.constant false
    cf.assert %false, "Invalid enum tag."
    llvm.unreachable
  ^bb14:  // pred: ^bb12
    %25 = llvm.load %23 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %26 = llvm.extractvalue %25[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    cf.br ^bb16
  ^bb15:  // pred: ^bb12
    %27 = llvm.load %23 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %28 = llvm.extractvalue %27[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb40
  ^bb16:  // pred: ^bb14
    cf.br ^bb17(%26 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)
  ^bb17(%29: !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>):  // pred: ^bb16
    %30 = llvm.extractvalue %29[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %31 = llvm.extractvalue %29[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    cf.br ^bb18(%31 : !llvm.struct<()>)
  ^bb18(%32: !llvm.struct<()>):  // pred: ^bb17
    cf.br ^bb19(%7 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb19(%33: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb18
    cf.br ^bb20(%33 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb20(%34: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb19
    %35 = llvm.extractvalue %34[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %36 = llvm.extractvalue %34[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %37 = llvm.extractvalue %34[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb21(%35 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb21(%38: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb20
    cf.br ^bb22(%37 : i32)
  ^bb22(%39: i32):  // pred: ^bb21
    cf.br ^bb23(%36 : i252)
  ^bb23(%40: i252):  // pred: ^bb22
    cf.br ^bb24(%30 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb24(%41: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb23
    cf.br ^bb25(%40, %41 : i252, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb25(%42: i252, %43: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb24
    %44:2 = call @"core::Felt252Serde::serialize(f21)"(%42, %43) : (i252, !llvm.struct<(ptr<i252>, i32, i32)>) -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.struct<()>)
    cf.br ^bb26(%44#1 : !llvm.struct<()>)
  ^bb26(%45: !llvm.struct<()>):  // pred: ^bb25
    cf.br ^bb27(%33 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb27(%46: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb26
    %47 = llvm.extractvalue %46[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %48 = llvm.extractvalue %46[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %49 = llvm.extractvalue %46[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb28(%47 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb28(%50: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb27
    cf.br ^bb29(%48 : i252)
  ^bb29(%51: i252):  // pred: ^bb28
    cf.br ^bb30(%49 : i32)
  ^bb30(%52: i32):  // pred: ^bb29
    cf.br ^bb31(%52 : i32)
  ^bb31(%53: i32):  // pred: ^bb30
    %54 = arith.extui %53 : i32 to i252
    cf.br ^bb32(%44#0, %54 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb32(%55: !llvm.struct<(ptr<i252>, i32, i32)>, %56: i252):  // pred: ^bb31
    %57 = llvm.extractvalue %55[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %58 = llvm.extractvalue %55[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %59 = llvm.extractvalue %55[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %60 = arith.cmpi uge, %58, %59 : i32
    %61:2 = scf.if %60 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %94 = arith.addi %59, %59 : i32
      %95 = arith.maxui %c8_i32, %94 : i32
      %96 = arith.extui %95 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %97 = arith.muli %96, %c32_i64 : i64
      %98 = llvm.bitcast %57 : !llvm.ptr<i252> to !llvm.ptr
      %99 = func.call @realloc(%98, %97) : (!llvm.ptr, i64) -> !llvm.ptr
      %100 = llvm.bitcast %99 : !llvm.ptr to !llvm.ptr<i252>
      %101 = llvm.insertvalue %100, %55[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %102 = llvm.insertvalue %95, %101[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %102, %100 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %55, %57 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %62 = llvm.getelementptr %61#1[%58] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %56, %62 : i252, !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %63 = arith.addi %58, %c1_i32 : i32
    %64 = llvm.insertvalue %63, %61#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb33
  ^bb33:  // pred: ^bb32
    %65 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb34(%64, %65 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.struct<()>)
  ^bb34(%66: !llvm.struct<(ptr<i252>, i32, i32)>, %67: !llvm.struct<()>):  // pred: ^bb33
    %68 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>
    %69 = llvm.insertvalue %66, %68[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %70 = llvm.insertvalue %67, %69[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    cf.br ^bb35(%70 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)
  ^bb35(%71: !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>):  // pred: ^bb34
    %false_2 = arith.constant false
    %72 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %73 = llvm.insertvalue %false_2, %72[0] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    %74 = llvm.insertvalue %71, %73[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    llvm.store %74, %2 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>, !llvm.ptr
    cf.br ^bb36(%22#0 : i64)
  ^bb36(%75: i64):  // pred: ^bb35
    cf.br ^bb37(%22#1 : i128)
  ^bb37(%76: i128):  // pred: ^bb36
    cf.br ^bb38(%2 : !llvm.ptr)
  ^bb38(%77: !llvm.ptr):  // pred: ^bb37
    cf.br ^bb39(%75, %76, %77 : i64, i128, !llvm.ptr)
  ^bb39(%78: i64, %79: i128, %80: !llvm.ptr):  // pred: ^bb38
    %81 = llvm.load %77 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %75, %76, %81 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb40:  // pred: ^bb15
    cf.br ^bb41(%7 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb41(%82: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb40
    cf.br ^bb42(%28 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb42(%83: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb41
    %true = arith.constant true
    %84 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %85 = llvm.insertvalue %true, %84[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %86 = llvm.insertvalue %83, %85[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %86, %1 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb43(%22#0 : i64)
  ^bb43(%87: i64):  // pred: ^bb42
    cf.br ^bb44(%22#1 : i128)
  ^bb44(%88: i128):  // pred: ^bb43
    cf.br ^bb45(%1 : !llvm.ptr)
  ^bb45(%89: !llvm.ptr):  // pred: ^bb44
    cf.br ^bb46(%87, %88, %89 : i64, i128, !llvm.ptr)
  ^bb46(%90: i64, %91: i128, %92: !llvm.ptr):  // pred: ^bb45
    %93 = llvm.load %89 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %87, %88, %93 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  }
  func.func public @"program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<0, 0>>(f18)"(%arg0: i64, %arg1: i252) -> (i64, !llvm.struct<(i1, array<0 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i8, array<1 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i8, array<1 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1 : i64, i252)
  ^bb1(%2: i64, %3: i252):  // pred: ^bb0
    cf.br ^bb2(%3 : i252)
  ^bb2(%4: i252):  // pred: ^bb1
    cf.br ^bb3(%2, %4 : i64, i252)
  ^bb3(%5: i64, %6: i252):  // pred: ^bb2
    %c1_i64_1 = arith.constant 1 : i64
    %7 = arith.addi %5, %c1_i64_1 : i64
    %c1809251394333065606848661391547535052811553607665798349986546028067936010240_i252 = arith.constant 1809251394333065606848661391547535052811553607665798349986546028067936010240 : i252
    %8 = arith.cmpi ugt, %6, %c1809251394333065606848661391547535052811553607665798349986546028067936010240_i252 : i252
    cf.cond_br %8, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252
    %9 = arith.subi %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252, %6 : i252
    %c-1_i252 = arith.constant -1 : i252
    %10 = arith.muli %9, %c-1_i252 : i252
    cf.br ^bb6(%10 : i252)
  ^bb5:  // pred: ^bb3
    cf.br ^bb6(%6 : i252)
  ^bb6(%11: i252):  // 2 preds: ^bb4, ^bb5
    %12 = arith.trunci %11 : i252 to i2
    %c64_i252 = arith.constant 64 : i252
    %13 = arith.trunci %11 : i252 to i64
    %14 = arith.shrui %11, %c64_i252 : i252
    %15 = arith.trunci %14 : i252 to i64
    %16 = arith.shrui %14, %c64_i252 : i252
    %17 = arith.trunci %16 : i252 to i64
    %18 = arith.shrui %16, %c64_i252 : i252
    %19 = arith.trunci %18 : i252 to i64
    call @__debug__print_felt252(%13, %15, %17, %19) : (i64, i64, i64, i64) -> ()
    %c1_i252 = arith.constant 1 : i252
    %c0_i252 = arith.constant 0 : i252
    %20 = arith.cmpi sle, %11, %c1_i252 : i252
    %21 = arith.cmpi sge, %11, %c0_i252 : i252
    %22 = arith.andi %20, %21 : i1
    call @__debug__print_i1(%22) : (i1) -> ()
    cf.cond_br %22, ^bb7, ^bb12
  ^bb7:  // pred: ^bb6
    cf.br ^bb8(%12 : i2)
  ^bb8(%23: i2):  // pred: ^bb7
    %false = arith.constant false
    %24 = llvm.mlir.undef : !llvm.struct<(i1, i2)>
    %25 = llvm.insertvalue %false, %24[0] : !llvm.struct<(i1, i2)> 
    %26 = llvm.insertvalue %23, %25[1] : !llvm.struct<(i1, i2)> 
    llvm.store %26, %1 {alignment = 1 : i64} : !llvm.struct<(i1, i2)>, !llvm.ptr
    cf.br ^bb9(%7 : i64)
  ^bb9(%27: i64):  // pred: ^bb8
    cf.br ^bb10(%1 : !llvm.ptr)
  ^bb10(%28: !llvm.ptr):  // pred: ^bb9
    cf.br ^bb11
  ^bb11:  // pred: ^bb10
    cf.br ^bb17(%4, %27, %28 : i252, i64, !llvm.ptr)
  ^bb12:  // pred: ^bb6
    cf.br ^bb13
  ^bb13:  // pred: ^bb12
    %29 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb14(%29 : !llvm.struct<()>)
  ^bb14(%30: !llvm.struct<()>):  // pred: ^bb13
    %true = arith.constant true
    %31 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %32 = llvm.insertvalue %true, %31[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %32, %0 {alignment = 1 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    cf.br ^bb15(%7 : i64)
  ^bb15(%33: i64):  // pred: ^bb14
    cf.br ^bb16(%0 : !llvm.ptr)
  ^bb16(%34: !llvm.ptr):  // pred: ^bb15
    cf.br ^bb17(%4, %33, %34 : i252, i64, !llvm.ptr)
  ^bb17(%35: i252, %36: i64, %37: !llvm.ptr):  // 2 preds: ^bb11, ^bb16
    cf.br ^bb18(%35 : i252)
  ^bb18(%38: i252):  // pred: ^bb17
    cf.br ^bb19(%37, %38 : !llvm.ptr, i252)
  ^bb19(%39: !llvm.ptr, %40: i252):  // pred: ^bb18
    %41 = call @"program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<0, 0>>(f19)"(%39, %40) : (!llvm.ptr, i252) -> !llvm.struct<(i1, array<0 x i8>)>
    cf.br ^bb20(%36 : i64)
  ^bb20(%42: i64):  // pred: ^bb19
    cf.br ^bb21(%41 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb21(%43: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb20
    cf.br ^bb22(%42, %43 : i64, !llvm.struct<(i1, array<0 x i8>)>)
  ^bb22(%44: i64, %45: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb21
    return %42, %43 : i64, !llvm.struct<(i1, array<0 x i8>)>
  }
  func.func public @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<-1, -1>>(f16)"(%arg0: i64, %arg1: i252) -> (i64, !llvm.struct<(i1, array<0 x i8>)>) attributes {llvm.emit_c_interface} {
    cf.br ^bb1(%arg0, %arg1 : i64, i252)
  ^bb1(%0: i64, %1: i252):  // pred: ^bb0
    cf.br ^bb2(%0, %1 : i64, i252)
  ^bb2(%2: i64, %3: i252):  // pred: ^bb1
    %c1_i64 = arith.constant 1 : i64
    %4 = arith.addi %2, %c1_i64 : i64
    %c1809251394333065606848661391547535052811553607665798349986546028067936010240_i252 = arith.constant 1809251394333065606848661391547535052811553607665798349986546028067936010240 : i252
    %5 = arith.cmpi ugt, %3, %c1809251394333065606848661391547535052811553607665798349986546028067936010240_i252 : i252
    cf.cond_br %5, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252
    %6 = arith.subi %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252, %3 : i252
    %c-1_i252 = arith.constant -1 : i252
    %7 = arith.muli %6, %c-1_i252 : i252
    cf.br ^bb5(%7 : i252)
  ^bb4:  // pred: ^bb2
    cf.br ^bb5(%3 : i252)
  ^bb5(%8: i252):  // 2 preds: ^bb3, ^bb4
    %9 = arith.trunci %8 : i252 to i2
    %c64_i252 = arith.constant 64 : i252
    %10 = arith.trunci %8 : i252 to i64
    %11 = arith.shrui %8, %c64_i252 : i252
    %12 = arith.trunci %11 : i252 to i64
    %13 = arith.shrui %11, %c64_i252 : i252
    %14 = arith.trunci %13 : i252 to i64
    %15 = arith.shrui %13, %c64_i252 : i252
    %16 = arith.trunci %15 : i252 to i64
    call @__debug__print_felt252(%10, %12, %14, %16) : (i64, i64, i64, i64) -> ()
    %c0_i252 = arith.constant 0 : i252
    %c-1_i252_0 = arith.constant -1 : i252
    %17 = arith.cmpi sle, %8, %c0_i252 : i252
    %18 = arith.cmpi sge, %8, %c-1_i252_0 : i252
    %19 = arith.andi %17, %18 : i1
    call @__debug__print_i1(%19) : (i1) -> ()
    cf.cond_br %19, ^bb6, ^bb17
  ^bb6:  // pred: ^bb5
    cf.br ^bb7(%9 : i2)
  ^bb7(%20: i2):  // pred: ^bb6
    %21 = arith.extsi %20 : i2 to i252
    %c0_i252_1 = arith.constant 0 : i252
    %22 = arith.cmpi slt, %21, %c0_i252_1 : i252
    cf.cond_br %22, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %23 = arith.extsi %20 : i2 to i252
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252_2 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252
    %24 = arith.addi %23, %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252_2 : i252
    cf.br ^bb10(%24 : i252)
  ^bb9:  // pred: ^bb7
    %25 = arith.extui %20 : i2 to i252
    cf.br ^bb10(%25 : i252)
  ^bb10(%26: i252):  // 2 preds: ^bb8, ^bb9
    cf.br ^bb11(%26 : i252)
  ^bb11(%27: i252):  // pred: ^bb10
    cf.br ^bb12
  ^bb12:  // pred: ^bb11
    %28 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb13(%28 : !llvm.struct<()>)
  ^bb13(%29: !llvm.struct<()>):  // pred: ^bb12
    %false = arith.constant false
    %30 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %31 = llvm.insertvalue %false, %30[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb14(%4 : i64)
  ^bb14(%32: i64):  // pred: ^bb13
    cf.br ^bb15(%31 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb15(%33: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb14
    cf.br ^bb16(%32, %33 : i64, !llvm.struct<(i1, array<0 x i8>)>)
  ^bb16(%34: i64, %35: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb15
    return %32, %33 : i64, !llvm.struct<(i1, array<0 x i8>)>
  ^bb17:  // pred: ^bb5
    cf.br ^bb18
  ^bb18:  // pred: ^bb17
    %36 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb19(%36 : !llvm.struct<()>)
  ^bb19(%37: !llvm.struct<()>):  // pred: ^bb18
    %true = arith.constant true
    %38 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %39 = llvm.insertvalue %true, %38[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb20(%4 : i64)
  ^bb20(%40: i64):  // pred: ^bb19
    cf.br ^bb21(%39 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb21(%41: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb20
    cf.br ^bb22(%40, %41 : i64, !llvm.struct<(i1, array<0 x i8>)>)
  ^bb22(%42: i64, %43: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb21
    return %40, %41 : i64, !llvm.struct<(i1, array<0 x i8>)>
  }
  func.func public @"program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<-1, -1>>(f14)"(%arg0: i64, %arg1: i252) -> (i64, !llvm.struct<(i1, array<0 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i8, array<1 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i8, array<1 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1 : i64, i252)
  ^bb1(%2: i64, %3: i252):  // pred: ^bb0
    cf.br ^bb2(%3 : i252)
  ^bb2(%4: i252):  // pred: ^bb1
    cf.br ^bb3(%2, %4 : i64, i252)
  ^bb3(%5: i64, %6: i252):  // pred: ^bb2
    %c1_i64_1 = arith.constant 1 : i64
    %7 = arith.addi %5, %c1_i64_1 : i64
    %c1809251394333065606848661391547535052811553607665798349986546028067936010240_i252 = arith.constant 1809251394333065606848661391547535052811553607665798349986546028067936010240 : i252
    %8 = arith.cmpi ugt, %6, %c1809251394333065606848661391547535052811553607665798349986546028067936010240_i252 : i252
    cf.cond_br %8, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252
    %9 = arith.subi %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252, %6 : i252
    %c-1_i252 = arith.constant -1 : i252
    %10 = arith.muli %9, %c-1_i252 : i252
    cf.br ^bb6(%10 : i252)
  ^bb5:  // pred: ^bb3
    cf.br ^bb6(%6 : i252)
  ^bb6(%11: i252):  // 2 preds: ^bb4, ^bb5
    %12 = arith.trunci %11 : i252 to i2
    %c64_i252 = arith.constant 64 : i252
    %13 = arith.trunci %11 : i252 to i64
    %14 = arith.shrui %11, %c64_i252 : i252
    %15 = arith.trunci %14 : i252 to i64
    %16 = arith.shrui %14, %c64_i252 : i252
    %17 = arith.trunci %16 : i252 to i64
    %18 = arith.shrui %16, %c64_i252 : i252
    %19 = arith.trunci %18 : i252 to i64
    call @__debug__print_felt252(%13, %15, %17, %19) : (i64, i64, i64, i64) -> ()
    %c0_i252 = arith.constant 0 : i252
    %c-1_i252_2 = arith.constant -1 : i252
    %20 = arith.cmpi sle, %11, %c0_i252 : i252
    %21 = arith.cmpi sge, %11, %c-1_i252_2 : i252
    %22 = arith.andi %20, %21 : i1
    call @__debug__print_i1(%22) : (i1) -> ()
    cf.cond_br %22, ^bb7, ^bb12
  ^bb7:  // pred: ^bb6
    cf.br ^bb8(%12 : i2)
  ^bb8(%23: i2):  // pred: ^bb7
    %false = arith.constant false
    %24 = llvm.mlir.undef : !llvm.struct<(i1, i2)>
    %25 = llvm.insertvalue %false, %24[0] : !llvm.struct<(i1, i2)> 
    %26 = llvm.insertvalue %23, %25[1] : !llvm.struct<(i1, i2)> 
    llvm.store %26, %1 {alignment = 1 : i64} : !llvm.struct<(i1, i2)>, !llvm.ptr
    cf.br ^bb9(%7 : i64)
  ^bb9(%27: i64):  // pred: ^bb8
    cf.br ^bb10(%1 : !llvm.ptr)
  ^bb10(%28: !llvm.ptr):  // pred: ^bb9
    cf.br ^bb11
  ^bb11:  // pred: ^bb10
    cf.br ^bb17(%4, %27, %28 : i252, i64, !llvm.ptr)
  ^bb12:  // pred: ^bb6
    cf.br ^bb13
  ^bb13:  // pred: ^bb12
    %29 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb14(%29 : !llvm.struct<()>)
  ^bb14(%30: !llvm.struct<()>):  // pred: ^bb13
    %true = arith.constant true
    %31 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %32 = llvm.insertvalue %true, %31[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %32, %0 {alignment = 1 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    cf.br ^bb15(%7 : i64)
  ^bb15(%33: i64):  // pred: ^bb14
    cf.br ^bb16(%0 : !llvm.ptr)
  ^bb16(%34: !llvm.ptr):  // pred: ^bb15
    cf.br ^bb17(%4, %33, %34 : i252, i64, !llvm.ptr)
  ^bb17(%35: i252, %36: i64, %37: !llvm.ptr):  // 2 preds: ^bb11, ^bb16
    cf.br ^bb18(%35 : i252)
  ^bb18(%38: i252):  // pred: ^bb17
    cf.br ^bb19(%37, %38 : !llvm.ptr, i252)
  ^bb19(%39: !llvm.ptr, %40: i252):  // pred: ^bb18
    %41 = call @"program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<-1, -1>>(f15)"(%39, %40) : (!llvm.ptr, i252) -> !llvm.struct<(i1, array<0 x i8>)>
    cf.br ^bb20(%36 : i64)
  ^bb20(%42: i64):  // pred: ^bb19
    cf.br ^bb21(%41 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb21(%43: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb20
    cf.br ^bb22(%42, %43 : i64, !llvm.struct<(i1, array<0 x i8>)>)
  ^bb22(%44: i64, %45: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb21
    return %42, %43 : i64, !llvm.struct<(i1, array<0 x i8>)>
  }
  func.func public @"core::byte_array::ByteArrayDefault::default(f13)"() -> !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> attributes {llvm.emit_c_interface} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %0 = llvm.mlir.null : !llvm.ptr<i248>
    %c0_i32 = arith.constant 0 : i32
    %1 = llvm.mlir.undef : !llvm.struct<(ptr<i248>, i32, i32)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %3 = llvm.insertvalue %c0_i32, %2[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %4 = llvm.insertvalue %c0_i32, %3[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    cf.br ^bb3
  ^bb3:  // pred: ^bb2
    %c0_i252 = arith.constant 0 : i252
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    %c0_i32_0 = arith.constant 0 : i32
    cf.br ^bb5(%4, %c0_i252, %c0_i32_0 : !llvm.struct<(ptr<i248>, i32, i32)>, i252, i32)
  ^bb5(%5: !llvm.struct<(ptr<i248>, i32, i32)>, %6: i252, %7: i32):  // pred: ^bb4
    %8 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %9 = llvm.insertvalue %5, %8[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %10 = llvm.insertvalue %6, %9[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %11 = llvm.insertvalue %7, %10[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    cf.br ^bb6(%11 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb6(%12: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb5
    cf.br ^bb7(%12 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb7(%13: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb6
    return %12 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
  }
  func.func public @"core::integer::U32Add::add(f4)"(%arg0: i64, %arg1: i32, %arg2: i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i32, array<4 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_1 = arith.constant 1 : i64
    %2 = llvm.alloca %c1_i64_1 x !llvm.struct<(i32, array<4 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1, %arg2 : i64, i32, i32)
  ^bb1(%3: i64, %4: i32, %5: i32):  // pred: ^bb0
    cf.br ^bb2(%3, %4, %5 : i64, i32, i32)
  ^bb2(%6: i64, %7: i32, %8: i32):  // pred: ^bb1
    %c1_i64_2 = arith.constant 1 : i64
    %9 = arith.addi %6, %c1_i64_2 : i64
    %10 = "llvm.intr.uadd.with.overflow"(%7, %8) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(i32, i1)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(i32, i1)> 
    cf.cond_br %12, ^bb8, ^bb3
  ^bb3:  // pred: ^bb2
    cf.br ^bb4(%11 : i32)
  ^bb4(%13: i32):  // pred: ^bb3
    %false = arith.constant false
    %14 = llvm.mlir.undef : !llvm.struct<(i1, i32)>
    %15 = llvm.insertvalue %false, %14[0] : !llvm.struct<(i1, i32)> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(i1, i32)> 
    llvm.store %16, %2 {alignment = 4 : i64} : !llvm.struct<(i1, i32)>, !llvm.ptr
    cf.br ^bb5(%9 : i64)
  ^bb5(%17: i64):  // pred: ^bb4
    cf.br ^bb6(%2 : !llvm.ptr)
  ^bb6(%18: !llvm.ptr):  // pred: ^bb5
    cf.br ^bb7
  ^bb7:  // pred: ^bb6
    cf.br ^bb12(%17, %18 : i64, !llvm.ptr)
  ^bb8:  // pred: ^bb2
    cf.br ^bb9(%11 : i32)
  ^bb9(%19: i32):  // pred: ^bb8
    %true = arith.constant true
    %20 = llvm.mlir.undef : !llvm.struct<(i1, i32)>
    %21 = llvm.insertvalue %true, %20[0] : !llvm.struct<(i1, i32)> 
    %22 = llvm.insertvalue %19, %21[1] : !llvm.struct<(i1, i32)> 
    llvm.store %22, %0 {alignment = 4 : i64} : !llvm.struct<(i1, i32)>, !llvm.ptr
    cf.br ^bb10(%9 : i64)
  ^bb10(%23: i64):  // pred: ^bb9
    cf.br ^bb11(%0 : !llvm.ptr)
  ^bb11(%24: !llvm.ptr):  // pred: ^bb10
    cf.br ^bb12(%23, %24 : i64, !llvm.ptr)
  ^bb12(%25: i64, %26: !llvm.ptr):  // 2 preds: ^bb7, ^bb11
    cf.br ^bb13
  ^bb13:  // pred: ^bb12
    %c155785504323917466144735657540098748279_i252 = arith.constant 155785504323917466144735657540098748279 : i252
    cf.br ^bb14(%c155785504323917466144735657540098748279_i252 : i252)
  ^bb14(%27: i252):  // pred: ^bb13
    cf.br ^bb15(%26, %27 : !llvm.ptr, i252)
  ^bb15(%28: !llvm.ptr, %29: i252):  // pred: ^bb14
    call @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>(f5)"(%1, %28, %29) : (!llvm.ptr, !llvm.ptr, i252) -> ()
    %30 = llvm.getelementptr %1[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb16(%25 : i64)
  ^bb16(%31: i64):  // pred: ^bb15
    cf.br ^bb17(%30 : !llvm.ptr)
  ^bb17(%32: !llvm.ptr):  // pred: ^bb16
    cf.br ^bb18(%31, %32 : i64, !llvm.ptr)
  ^bb18(%33: i64, %34: !llvm.ptr):  // pred: ^bb17
    %35 = llvm.load %32 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %31, %35 : i64, !llvm.struct<(i64, array<16 x i8>)>
  }
  func.func public @"core::integer::U32Sub::sub(f8)"(%arg0: i64, %arg1: i32, %arg2: i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i32, array<4 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_1 = arith.constant 1 : i64
    %2 = llvm.alloca %c1_i64_1 x !llvm.struct<(i32, array<4 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1, %arg2 : i64, i32, i32)
  ^bb1(%3: i64, %4: i32, %5: i32):  // pred: ^bb0
    cf.br ^bb2(%3, %4, %5 : i64, i32, i32)
  ^bb2(%6: i64, %7: i32, %8: i32):  // pred: ^bb1
    %c1_i64_2 = arith.constant 1 : i64
    %9 = arith.addi %6, %c1_i64_2 : i64
    %10 = "llvm.intr.usub.with.overflow"(%7, %8) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(i32, i1)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(i32, i1)> 
    cf.cond_br %12, ^bb8, ^bb3
  ^bb3:  // pred: ^bb2
    cf.br ^bb4(%11 : i32)
  ^bb4(%13: i32):  // pred: ^bb3
    %false = arith.constant false
    %14 = llvm.mlir.undef : !llvm.struct<(i1, i32)>
    %15 = llvm.insertvalue %false, %14[0] : !llvm.struct<(i1, i32)> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(i1, i32)> 
    llvm.store %16, %2 {alignment = 4 : i64} : !llvm.struct<(i1, i32)>, !llvm.ptr
    cf.br ^bb5(%9 : i64)
  ^bb5(%17: i64):  // pred: ^bb4
    cf.br ^bb6(%2 : !llvm.ptr)
  ^bb6(%18: !llvm.ptr):  // pred: ^bb5
    cf.br ^bb7
  ^bb7:  // pred: ^bb6
    cf.br ^bb12(%17, %18 : i64, !llvm.ptr)
  ^bb8:  // pred: ^bb2
    cf.br ^bb9(%11 : i32)
  ^bb9(%19: i32):  // pred: ^bb8
    %true = arith.constant true
    %20 = llvm.mlir.undef : !llvm.struct<(i1, i32)>
    %21 = llvm.insertvalue %true, %20[0] : !llvm.struct<(i1, i32)> 
    %22 = llvm.insertvalue %19, %21[1] : !llvm.struct<(i1, i32)> 
    llvm.store %22, %0 {alignment = 4 : i64} : !llvm.struct<(i1, i32)>, !llvm.ptr
    cf.br ^bb10(%9 : i64)
  ^bb10(%23: i64):  // pred: ^bb9
    cf.br ^bb11(%0 : !llvm.ptr)
  ^bb11(%24: !llvm.ptr):  // pred: ^bb10
    cf.br ^bb12(%23, %24 : i64, !llvm.ptr)
  ^bb12(%25: i64, %26: !llvm.ptr):  // 2 preds: ^bb7, ^bb11
    cf.br ^bb13
  ^bb13:  // pred: ^bb12
    %c155785504329508738615720351733824384887_i252 = arith.constant 155785504329508738615720351733824384887 : i252
    cf.br ^bb14(%c155785504329508738615720351733824384887_i252 : i252)
  ^bb14(%27: i252):  // pred: ^bb13
    cf.br ^bb15(%26, %27 : !llvm.ptr, i252)
  ^bb15(%28: !llvm.ptr, %29: i252):  // pred: ^bb14
    call @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>(f5)"(%1, %28, %29) : (!llvm.ptr, !llvm.ptr, i252) -> ()
    %30 = llvm.getelementptr %1[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb16(%25 : i64)
  ^bb16(%31: i64):  // pred: ^bb15
    cf.br ^bb17(%30 : !llvm.ptr)
  ^bb17(%32: !llvm.ptr):  // pred: ^bb16
    cf.br ^bb18(%31, %32 : i64, !llvm.ptr)
  ^bb18(%33: i64, %34: !llvm.ptr):  // pred: ^bb17
    %35 = llvm.load %32 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %31, %35 : i64, !llvm.struct<(i64, array<16 x i8>)>
  }
  func.func public @"core::integer::u256_from_felt252(f10)"(%arg0: i64, %arg1: i252) -> (i64, !llvm.struct<(i128, i128)>) attributes {llvm.emit_c_interface} {
    cf.br ^bb1(%arg0, %arg1 : i64, i252)
  ^bb1(%0: i64, %1: i252):  // pred: ^bb0
    cf.br ^bb2(%0, %1 : i64, i252)
  ^bb2(%2: i64, %3: i252):  // pred: ^bb1
    %c1_i64 = arith.constant 1 : i64
    %4 = arith.addi %2, %c1_i64 : i64
    %c1_i252 = arith.constant 1 : i252
    %c128_i252 = arith.constant 128 : i252
    %5 = arith.shli %c1_i252, %c128_i252 : i252
    %6 = arith.cmpi uge, %3, %5 : i252
    %7 = arith.trunci %3 : i252 to i128
    %8 = arith.shrui %3, %c128_i252 : i252
    %9 = arith.trunci %8 : i252 to i128
    cf.cond_br %6, ^bb9, ^bb3
  ^bb3:  // pred: ^bb2
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    %c0_i128 = arith.constant 0 : i128
    cf.br ^bb5(%7, %c0_i128 : i128, i128)
  ^bb5(%10: i128, %11: i128):  // pred: ^bb4
    %12 = llvm.mlir.undef : !llvm.struct<(i128, i128)>
    %13 = llvm.insertvalue %10, %12[0] : !llvm.struct<(i128, i128)> 
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<(i128, i128)> 
    cf.br ^bb6(%4 : i64)
  ^bb6(%15: i64):  // pred: ^bb5
    cf.br ^bb7(%14 : !llvm.struct<(i128, i128)>)
  ^bb7(%16: !llvm.struct<(i128, i128)>):  // pred: ^bb6
    cf.br ^bb8(%15, %16 : i64, !llvm.struct<(i128, i128)>)
  ^bb8(%17: i64, %18: !llvm.struct<(i128, i128)>):  // pred: ^bb7
    return %15, %16 : i64, !llvm.struct<(i128, i128)>
  ^bb9:  // pred: ^bb2
    cf.br ^bb10(%7, %9 : i128, i128)
  ^bb10(%19: i128, %20: i128):  // pred: ^bb9
    %21 = llvm.mlir.undef : !llvm.struct<(i128, i128)>
    %22 = llvm.insertvalue %19, %21[0] : !llvm.struct<(i128, i128)> 
    %23 = llvm.insertvalue %20, %22[1] : !llvm.struct<(i128, i128)> 
    cf.br ^bb11(%4 : i64)
  ^bb11(%24: i64):  // pred: ^bb10
    cf.br ^bb12(%23 : !llvm.struct<(i128, i128)>)
  ^bb12(%25: !llvm.struct<(i128, i128)>):  // pred: ^bb11
    cf.br ^bb13(%24, %25 : i64, !llvm.struct<(i128, i128)>)
  ^bb13(%26: i64, %27: !llvm.struct<(i128, i128)>):  // pred: ^bb12
    return %24, %25 : i64, !llvm.struct<(i128, i128)>
  }
  func.func public @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%arg0: !llvm.ptr, %arg1: i32) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg1 : i32)
  ^bb1(%2: i32):  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %c0_i32 = arith.constant 0 : i32
    cf.br ^bb3(%2 : i32)
  ^bb3(%3: i32):  // pred: ^bb2
    cf.br ^bb4(%3, %c0_i32 : i32, i32)
  ^bb4(%4: i32, %5: i32):  // pred: ^bb3
    %6 = arith.cmpi eq, %4, %5 : i32
    cf.cond_br %6, ^bb176, ^bb5
  ^bb5:  // pred: ^bb4
    cf.br ^bb6
  ^bb6:  // pred: ^bb5
    %c1_i32 = arith.constant 1 : i32
    cf.br ^bb7(%3 : i32)
  ^bb7(%7: i32):  // pred: ^bb6
    cf.br ^bb8(%7, %c1_i32 : i32, i32)
  ^bb8(%8: i32, %9: i32):  // pred: ^bb7
    %10 = arith.cmpi eq, %8, %9 : i32
    cf.cond_br %10, ^bb169, ^bb9
  ^bb9:  // pred: ^bb8
    cf.br ^bb10
  ^bb10:  // pred: ^bb9
    %c2_i32 = arith.constant 2 : i32
    cf.br ^bb11(%7 : i32)
  ^bb11(%11: i32):  // pred: ^bb10
    cf.br ^bb12(%11, %c2_i32 : i32, i32)
  ^bb12(%12: i32, %13: i32):  // pred: ^bb11
    %14 = arith.cmpi eq, %12, %13 : i32
    cf.cond_br %14, ^bb162, ^bb13
  ^bb13:  // pred: ^bb12
    cf.br ^bb14
  ^bb14:  // pred: ^bb13
    %c3_i32 = arith.constant 3 : i32
    cf.br ^bb15(%11 : i32)
  ^bb15(%15: i32):  // pred: ^bb14
    cf.br ^bb16(%15, %c3_i32 : i32, i32)
  ^bb16(%16: i32, %17: i32):  // pred: ^bb15
    %18 = arith.cmpi eq, %16, %17 : i32
    cf.cond_br %18, ^bb155, ^bb17
  ^bb17:  // pred: ^bb16
    cf.br ^bb18
  ^bb18:  // pred: ^bb17
    %c4_i32 = arith.constant 4 : i32
    cf.br ^bb19(%15 : i32)
  ^bb19(%19: i32):  // pred: ^bb18
    cf.br ^bb20(%19, %c4_i32 : i32, i32)
  ^bb20(%20: i32, %21: i32):  // pred: ^bb19
    %22 = arith.cmpi eq, %20, %21 : i32
    cf.cond_br %22, ^bb148, ^bb21
  ^bb21:  // pred: ^bb20
    cf.br ^bb22
  ^bb22:  // pred: ^bb21
    %c5_i32 = arith.constant 5 : i32
    cf.br ^bb23(%19 : i32)
  ^bb23(%23: i32):  // pred: ^bb22
    cf.br ^bb24(%23, %c5_i32 : i32, i32)
  ^bb24(%24: i32, %25: i32):  // pred: ^bb23
    %26 = arith.cmpi eq, %24, %25 : i32
    cf.cond_br %26, ^bb141, ^bb25
  ^bb25:  // pred: ^bb24
    cf.br ^bb26
  ^bb26:  // pred: ^bb25
    %c6_i32 = arith.constant 6 : i32
    cf.br ^bb27(%23 : i32)
  ^bb27(%27: i32):  // pred: ^bb26
    cf.br ^bb28(%27, %c6_i32 : i32, i32)
  ^bb28(%28: i32, %29: i32):  // pred: ^bb27
    %30 = arith.cmpi eq, %28, %29 : i32
    cf.cond_br %30, ^bb134, ^bb29
  ^bb29:  // pred: ^bb28
    cf.br ^bb30
  ^bb30:  // pred: ^bb29
    %c7_i32 = arith.constant 7 : i32
    cf.br ^bb31(%27 : i32)
  ^bb31(%31: i32):  // pred: ^bb30
    cf.br ^bb32(%31, %c7_i32 : i32, i32)
  ^bb32(%32: i32, %33: i32):  // pred: ^bb31
    %34 = arith.cmpi eq, %32, %33 : i32
    cf.cond_br %34, ^bb127, ^bb33
  ^bb33:  // pred: ^bb32
    cf.br ^bb34
  ^bb34:  // pred: ^bb33
    %c8_i32 = arith.constant 8 : i32
    cf.br ^bb35(%31 : i32)
  ^bb35(%35: i32):  // pred: ^bb34
    cf.br ^bb36(%35, %c8_i32 : i32, i32)
  ^bb36(%36: i32, %37: i32):  // pred: ^bb35
    %38 = arith.cmpi eq, %36, %37 : i32
    cf.cond_br %38, ^bb120, ^bb37
  ^bb37:  // pred: ^bb36
    cf.br ^bb38
  ^bb38:  // pred: ^bb37
    %c9_i32 = arith.constant 9 : i32
    cf.br ^bb39(%35 : i32)
  ^bb39(%39: i32):  // pred: ^bb38
    cf.br ^bb40(%39, %c9_i32 : i32, i32)
  ^bb40(%40: i32, %41: i32):  // pred: ^bb39
    %42 = arith.cmpi eq, %40, %41 : i32
    cf.cond_br %42, ^bb113, ^bb41
  ^bb41:  // pred: ^bb40
    cf.br ^bb42
  ^bb42:  // pred: ^bb41
    %c10_i32 = arith.constant 10 : i32
    cf.br ^bb43(%39 : i32)
  ^bb43(%43: i32):  // pred: ^bb42
    cf.br ^bb44(%43, %c10_i32 : i32, i32)
  ^bb44(%44: i32, %45: i32):  // pred: ^bb43
    %46 = arith.cmpi eq, %44, %45 : i32
    cf.cond_br %46, ^bb106, ^bb45
  ^bb45:  // pred: ^bb44
    cf.br ^bb46
  ^bb46:  // pred: ^bb45
    %c11_i32 = arith.constant 11 : i32
    cf.br ^bb47(%43 : i32)
  ^bb47(%47: i32):  // pred: ^bb46
    cf.br ^bb48(%47, %c11_i32 : i32, i32)
  ^bb48(%48: i32, %49: i32):  // pred: ^bb47
    %50 = arith.cmpi eq, %48, %49 : i32
    cf.cond_br %50, ^bb99, ^bb49
  ^bb49:  // pred: ^bb48
    cf.br ^bb50
  ^bb50:  // pred: ^bb49
    %c12_i32 = arith.constant 12 : i32
    cf.br ^bb51(%47 : i32)
  ^bb51(%51: i32):  // pred: ^bb50
    cf.br ^bb52(%51, %c12_i32 : i32, i32)
  ^bb52(%52: i32, %53: i32):  // pred: ^bb51
    %54 = arith.cmpi eq, %52, %53 : i32
    cf.cond_br %54, ^bb92, ^bb53
  ^bb53:  // pred: ^bb52
    cf.br ^bb54
  ^bb54:  // pred: ^bb53
    %c13_i32 = arith.constant 13 : i32
    cf.br ^bb55(%51 : i32)
  ^bb55(%55: i32):  // pred: ^bb54
    cf.br ^bb56(%55, %c13_i32 : i32, i32)
  ^bb56(%56: i32, %57: i32):  // pred: ^bb55
    %58 = arith.cmpi eq, %56, %57 : i32
    cf.cond_br %58, ^bb85, ^bb57
  ^bb57:  // pred: ^bb56
    cf.br ^bb58
  ^bb58:  // pred: ^bb57
    %c14_i32 = arith.constant 14 : i32
    cf.br ^bb59(%55 : i32)
  ^bb59(%59: i32):  // pred: ^bb58
    cf.br ^bb60(%59, %c14_i32 : i32, i32)
  ^bb60(%60: i32, %61: i32):  // pred: ^bb59
    %62 = arith.cmpi eq, %60, %61 : i32
    cf.cond_br %62, ^bb78, ^bb61
  ^bb61:  // pred: ^bb60
    cf.br ^bb62
  ^bb62:  // pred: ^bb61
    %c15_i32 = arith.constant 15 : i32
    cf.br ^bb63(%59, %c15_i32 : i32, i32)
  ^bb63(%63: i32, %64: i32):  // pred: ^bb62
    %65 = arith.cmpi eq, %63, %64 : i32
    cf.cond_br %65, ^bb74, ^bb64
  ^bb64:  // pred: ^bb63
    cf.br ^bb65
  ^bb65:  // pred: ^bb64
    %66 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_1 = arith.constant 0 : i32
    %67 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %68 = llvm.insertvalue %66, %67[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %69 = llvm.insertvalue %c0_i32_1, %68[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %70 = llvm.insertvalue %c0_i32_1, %69[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb66
  ^bb66:  // pred: ^bb65
    %c573087285299505011920718992710461799_i252 = arith.constant 573087285299505011920718992710461799 : i252
    cf.br ^bb67(%c573087285299505011920718992710461799_i252 : i252)
  ^bb67(%71: i252):  // pred: ^bb66
    cf.br ^bb68(%70, %71 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb68(%72: !llvm.struct<(ptr<i252>, i32, i32)>, %73: i252):  // pred: ^bb67
    %74 = llvm.extractvalue %72[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %75 = llvm.extractvalue %72[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %76 = llvm.extractvalue %72[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %77 = arith.cmpi uge, %75, %76 : i32
    %78:2 = scf.if %77 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32_6 = arith.constant 8 : i32
      %163 = arith.addi %76, %76 : i32
      %164 = arith.maxui %c8_i32_6, %163 : i32
      %165 = arith.extui %164 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %166 = arith.muli %165, %c32_i64 : i64
      %167 = llvm.bitcast %74 : !llvm.ptr<i252> to !llvm.ptr
      %168 = func.call @realloc(%167, %166) : (!llvm.ptr, i64) -> !llvm.ptr
      %169 = llvm.bitcast %168 : !llvm.ptr to !llvm.ptr<i252>
      %170 = llvm.insertvalue %169, %72[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %171 = llvm.insertvalue %164, %170[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %171, %169 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %72, %74 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %79 = llvm.getelementptr %78#1[%75] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %73, %79 : i252, !llvm.ptr
    %c1_i32_2 = arith.constant 1 : i32
    %80 = arith.addi %75, %c1_i32_2 : i32
    %81 = llvm.insertvalue %80, %78#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb69
  ^bb69:  // pred: ^bb68
    %82 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb70(%82, %81 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb70(%83: !llvm.struct<()>, %84: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb69
    %85 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %86 = llvm.insertvalue %83, %85[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %87 = llvm.insertvalue %84, %86[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb71(%87 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb71(%88: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb70
    %true = arith.constant true
    %89 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %90 = llvm.insertvalue %true, %89[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %91 = llvm.insertvalue %88, %90[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %91, %1 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb72(%1 : !llvm.ptr)
  ^bb72(%92: !llvm.ptr):  // pred: ^bb71
    cf.br ^bb73(%92 : !llvm.ptr)
  ^bb73(%93: !llvm.ptr):  // pred: ^bb72
    %c24_i64 = arith.constant 24 : i64
    %false = arith.constant false
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %92, %c24_i64, %false) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    return
  ^bb74:  // pred: ^bb63
    cf.br ^bb75
  ^bb75:  // pred: ^bb74
    %c1329227995784915872903807060280344576_i128 = arith.constant 1329227995784915872903807060280344576 : i128
    cf.br ^bb76(%c1329227995784915872903807060280344576_i128 : i128)
  ^bb76(%94: i128):  // pred: ^bb75
    cf.br ^bb77
  ^bb77:  // pred: ^bb76
    cf.br ^bb82(%94 : i128)
  ^bb78:  // pred: ^bb60
    cf.br ^bb79(%59 : i32)
  ^bb79(%95: i32):  // pred: ^bb78
    cf.br ^bb80
  ^bb80:  // pred: ^bb79
    %c5192296858534827628530496329220096_i128 = arith.constant 5192296858534827628530496329220096 : i128
    cf.br ^bb81(%c5192296858534827628530496329220096_i128 : i128)
  ^bb81(%96: i128):  // pred: ^bb80
    cf.br ^bb82(%96 : i128)
  ^bb82(%97: i128):  // 2 preds: ^bb77, ^bb81
    cf.br ^bb83(%97 : i128)
  ^bb83(%98: i128):  // pred: ^bb82
    cf.br ^bb84
  ^bb84:  // pred: ^bb83
    cf.br ^bb89(%98 : i128)
  ^bb85:  // pred: ^bb56
    cf.br ^bb86(%55 : i32)
  ^bb86(%99: i32):  // pred: ^bb85
    cf.br ^bb87
  ^bb87:  // pred: ^bb86
    %c20282409603651670423947251286016_i128 = arith.constant 20282409603651670423947251286016 : i128
    cf.br ^bb88(%c20282409603651670423947251286016_i128 : i128)
  ^bb88(%100: i128):  // pred: ^bb87
    cf.br ^bb89(%100 : i128)
  ^bb89(%101: i128):  // 2 preds: ^bb84, ^bb88
    cf.br ^bb90(%101 : i128)
  ^bb90(%102: i128):  // pred: ^bb89
    cf.br ^bb91
  ^bb91:  // pred: ^bb90
    cf.br ^bb96(%102 : i128)
  ^bb92:  // pred: ^bb52
    cf.br ^bb93(%51 : i32)
  ^bb93(%103: i32):  // pred: ^bb92
    cf.br ^bb94
  ^bb94:  // pred: ^bb93
    %c79228162514264337593543950336_i128 = arith.constant 79228162514264337593543950336 : i128
    cf.br ^bb95(%c79228162514264337593543950336_i128 : i128)
  ^bb95(%104: i128):  // pred: ^bb94
    cf.br ^bb96(%104 : i128)
  ^bb96(%105: i128):  // 2 preds: ^bb91, ^bb95
    cf.br ^bb97(%105 : i128)
  ^bb97(%106: i128):  // pred: ^bb96
    cf.br ^bb98
  ^bb98:  // pred: ^bb97
    cf.br ^bb103(%106 : i128)
  ^bb99:  // pred: ^bb48
    cf.br ^bb100(%47 : i32)
  ^bb100(%107: i32):  // pred: ^bb99
    cf.br ^bb101
  ^bb101:  // pred: ^bb100
    %c309485009821345068724781056_i128 = arith.constant 309485009821345068724781056 : i128
    cf.br ^bb102(%c309485009821345068724781056_i128 : i128)
  ^bb102(%108: i128):  // pred: ^bb101
    cf.br ^bb103(%108 : i128)
  ^bb103(%109: i128):  // 2 preds: ^bb98, ^bb102
    cf.br ^bb104(%109 : i128)
  ^bb104(%110: i128):  // pred: ^bb103
    cf.br ^bb105
  ^bb105:  // pred: ^bb104
    cf.br ^bb110(%110 : i128)
  ^bb106:  // pred: ^bb44
    cf.br ^bb107(%43 : i32)
  ^bb107(%111: i32):  // pred: ^bb106
    cf.br ^bb108
  ^bb108:  // pred: ^bb107
    %c1208925819614629174706176_i128 = arith.constant 1208925819614629174706176 : i128
    cf.br ^bb109(%c1208925819614629174706176_i128 : i128)
  ^bb109(%112: i128):  // pred: ^bb108
    cf.br ^bb110(%112 : i128)
  ^bb110(%113: i128):  // 2 preds: ^bb105, ^bb109
    cf.br ^bb111(%113 : i128)
  ^bb111(%114: i128):  // pred: ^bb110
    cf.br ^bb112
  ^bb112:  // pred: ^bb111
    cf.br ^bb117(%114 : i128)
  ^bb113:  // pred: ^bb40
    cf.br ^bb114(%39 : i32)
  ^bb114(%115: i32):  // pred: ^bb113
    cf.br ^bb115
  ^bb115:  // pred: ^bb114
    %c4722366482869645213696_i128 = arith.constant 4722366482869645213696 : i128
    cf.br ^bb116(%c4722366482869645213696_i128 : i128)
  ^bb116(%116: i128):  // pred: ^bb115
    cf.br ^bb117(%116 : i128)
  ^bb117(%117: i128):  // 2 preds: ^bb112, ^bb116
    cf.br ^bb118(%117 : i128)
  ^bb118(%118: i128):  // pred: ^bb117
    cf.br ^bb119
  ^bb119:  // pred: ^bb118
    cf.br ^bb124(%118 : i128)
  ^bb120:  // pred: ^bb36
    cf.br ^bb121(%35 : i32)
  ^bb121(%119: i32):  // pred: ^bb120
    cf.br ^bb122
  ^bb122:  // pred: ^bb121
    %c18446744073709551616_i128 = arith.constant 18446744073709551616 : i128
    cf.br ^bb123(%c18446744073709551616_i128 : i128)
  ^bb123(%120: i128):  // pred: ^bb122
    cf.br ^bb124(%120 : i128)
  ^bb124(%121: i128):  // 2 preds: ^bb119, ^bb123
    cf.br ^bb125(%121 : i128)
  ^bb125(%122: i128):  // pred: ^bb124
    cf.br ^bb126
  ^bb126:  // pred: ^bb125
    cf.br ^bb131(%122 : i128)
  ^bb127:  // pred: ^bb32
    cf.br ^bb128(%31 : i32)
  ^bb128(%123: i32):  // pred: ^bb127
    cf.br ^bb129
  ^bb129:  // pred: ^bb128
    %c72057594037927936_i128 = arith.constant 72057594037927936 : i128
    cf.br ^bb130(%c72057594037927936_i128 : i128)
  ^bb130(%124: i128):  // pred: ^bb129
    cf.br ^bb131(%124 : i128)
  ^bb131(%125: i128):  // 2 preds: ^bb126, ^bb130
    cf.br ^bb132(%125 : i128)
  ^bb132(%126: i128):  // pred: ^bb131
    cf.br ^bb133
  ^bb133:  // pred: ^bb132
    cf.br ^bb138(%126 : i128)
  ^bb134:  // pred: ^bb28
    cf.br ^bb135(%27 : i32)
  ^bb135(%127: i32):  // pred: ^bb134
    cf.br ^bb136
  ^bb136:  // pred: ^bb135
    %c281474976710656_i128 = arith.constant 281474976710656 : i128
    cf.br ^bb137(%c281474976710656_i128 : i128)
  ^bb137(%128: i128):  // pred: ^bb136
    cf.br ^bb138(%128 : i128)
  ^bb138(%129: i128):  // 2 preds: ^bb133, ^bb137
    cf.br ^bb139(%129 : i128)
  ^bb139(%130: i128):  // pred: ^bb138
    cf.br ^bb140
  ^bb140:  // pred: ^bb139
    cf.br ^bb145(%130 : i128)
  ^bb141:  // pred: ^bb24
    cf.br ^bb142(%23 : i32)
  ^bb142(%131: i32):  // pred: ^bb141
    cf.br ^bb143
  ^bb143:  // pred: ^bb142
    %c1099511627776_i128 = arith.constant 1099511627776 : i128
    cf.br ^bb144(%c1099511627776_i128 : i128)
  ^bb144(%132: i128):  // pred: ^bb143
    cf.br ^bb145(%132 : i128)
  ^bb145(%133: i128):  // 2 preds: ^bb140, ^bb144
    cf.br ^bb146(%133 : i128)
  ^bb146(%134: i128):  // pred: ^bb145
    cf.br ^bb147
  ^bb147:  // pred: ^bb146
    cf.br ^bb152(%134 : i128)
  ^bb148:  // pred: ^bb20
    cf.br ^bb149(%19 : i32)
  ^bb149(%135: i32):  // pred: ^bb148
    cf.br ^bb150
  ^bb150:  // pred: ^bb149
    %c4294967296_i128 = arith.constant 4294967296 : i128
    cf.br ^bb151(%c4294967296_i128 : i128)
  ^bb151(%136: i128):  // pred: ^bb150
    cf.br ^bb152(%136 : i128)
  ^bb152(%137: i128):  // 2 preds: ^bb147, ^bb151
    cf.br ^bb153(%137 : i128)
  ^bb153(%138: i128):  // pred: ^bb152
    cf.br ^bb154
  ^bb154:  // pred: ^bb153
    cf.br ^bb159(%138 : i128)
  ^bb155:  // pred: ^bb16
    cf.br ^bb156(%15 : i32)
  ^bb156(%139: i32):  // pred: ^bb155
    cf.br ^bb157
  ^bb157:  // pred: ^bb156
    %c16777216_i128 = arith.constant 16777216 : i128
    cf.br ^bb158(%c16777216_i128 : i128)
  ^bb158(%140: i128):  // pred: ^bb157
    cf.br ^bb159(%140 : i128)
  ^bb159(%141: i128):  // 2 preds: ^bb154, ^bb158
    cf.br ^bb160(%141 : i128)
  ^bb160(%142: i128):  // pred: ^bb159
    cf.br ^bb161
  ^bb161:  // pred: ^bb160
    cf.br ^bb166(%142 : i128)
  ^bb162:  // pred: ^bb12
    cf.br ^bb163(%11 : i32)
  ^bb163(%143: i32):  // pred: ^bb162
    cf.br ^bb164
  ^bb164:  // pred: ^bb163
    %c65536_i128 = arith.constant 65536 : i128
    cf.br ^bb165(%c65536_i128 : i128)
  ^bb165(%144: i128):  // pred: ^bb164
    cf.br ^bb166(%144 : i128)
  ^bb166(%145: i128):  // 2 preds: ^bb161, ^bb165
    cf.br ^bb167(%145 : i128)
  ^bb167(%146: i128):  // pred: ^bb166
    cf.br ^bb168
  ^bb168:  // pred: ^bb167
    cf.br ^bb173(%146 : i128)
  ^bb169:  // pred: ^bb8
    cf.br ^bb170(%7 : i32)
  ^bb170(%147: i32):  // pred: ^bb169
    cf.br ^bb171
  ^bb171:  // pred: ^bb170
    %c256_i128 = arith.constant 256 : i128
    cf.br ^bb172(%c256_i128 : i128)
  ^bb172(%148: i128):  // pred: ^bb171
    cf.br ^bb173(%148 : i128)
  ^bb173(%149: i128):  // 2 preds: ^bb168, ^bb172
    cf.br ^bb174(%149 : i128)
  ^bb174(%150: i128):  // pred: ^bb173
    cf.br ^bb175
  ^bb175:  // pred: ^bb174
    cf.br ^bb180(%150 : i128)
  ^bb176:  // pred: ^bb4
    cf.br ^bb177(%3 : i32)
  ^bb177(%151: i32):  // pred: ^bb176
    cf.br ^bb178
  ^bb178:  // pred: ^bb177
    %c1_i128 = arith.constant 1 : i128
    cf.br ^bb179(%c1_i128 : i128)
  ^bb179(%152: i128):  // pred: ^bb178
    cf.br ^bb180(%152 : i128)
  ^bb180(%153: i128):  // 2 preds: ^bb175, ^bb179
    cf.br ^bb181(%153 : i128)
  ^bb181(%154: i128):  // pred: ^bb180
    %155 = llvm.mlir.undef : !llvm.struct<(i128)>
    %156 = llvm.insertvalue %154, %155[0] : !llvm.struct<(i128)> 
    cf.br ^bb182(%156 : !llvm.struct<(i128)>)
  ^bb182(%157: !llvm.struct<(i128)>):  // pred: ^bb181
    %false_3 = arith.constant false
    %158 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i128)>)>
    %159 = llvm.insertvalue %false_3, %158[0] : !llvm.struct<(i1, struct<(i128)>)> 
    %160 = llvm.insertvalue %157, %159[1] : !llvm.struct<(i1, struct<(i128)>)> 
    llvm.store %160, %0 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(i128)>)>, !llvm.ptr
    cf.br ^bb183(%0 : !llvm.ptr)
  ^bb183(%161: !llvm.ptr):  // pred: ^bb182
    cf.br ^bb184(%161 : !llvm.ptr)
  ^bb184(%162: !llvm.ptr):  // pred: ^bb183
    %c24_i64_4 = arith.constant 24 : i64
    %false_5 = arith.constant false
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %161, %c24_i64_4, %false_5) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    return
  }
  func.func public @"core::integer::u128_try_as_non_zero(f11)"(%arg0: !llvm.ptr, %arg1: i128) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg1 : i128)
  ^bb1(%2: i128):  // pred: ^bb0
    cf.br ^bb2(%2 : i128)
  ^bb2(%3: i128):  // pred: ^bb1
    %c0_i128 = arith.constant 0 : i128
    %4 = arith.cmpi eq, %3, %c0_i128 : i128
    cf.cond_br %4, ^bb3, ^bb8
  ^bb3:  // pred: ^bb2
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    %5 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb5(%5 : !llvm.struct<()>)
  ^bb5(%6: !llvm.struct<()>):  // pred: ^bb4
    %true = arith.constant true
    %7 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %8 = llvm.insertvalue %true, %7[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %8, %1 {alignment = 8 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    cf.br ^bb6(%1 : !llvm.ptr)
  ^bb6(%9: !llvm.ptr):  // pred: ^bb5
    cf.br ^bb7(%9 : !llvm.ptr)
  ^bb7(%10: !llvm.ptr):  // pred: ^bb6
    %c24_i64 = arith.constant 24 : i64
    %false = arith.constant false
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %9, %c24_i64, %false) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    return
  ^bb8:  // pred: ^bb2
    cf.br ^bb9(%3 : i128)
  ^bb9(%11: i128):  // pred: ^bb8
    %false_1 = arith.constant false
    %12 = llvm.mlir.undef : !llvm.struct<(i1, i128)>
    %13 = llvm.insertvalue %false_1, %12[0] : !llvm.struct<(i1, i128)> 
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<(i1, i128)> 
    llvm.store %14, %0 {alignment = 8 : i64} : !llvm.struct<(i1, i128)>, !llvm.ptr
    cf.br ^bb10(%0 : !llvm.ptr)
  ^bb10(%15: !llvm.ptr):  // pred: ^bb9
    cf.br ^bb11(%15 : !llvm.ptr)
  ^bb11(%16: !llvm.ptr):  // pred: ^bb10
    %c24_i64_2 = arith.constant 24 : i64
    %false_3 = arith.constant false
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %15, %c24_i64_2, %false_3) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    return
  }
  func.func public @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%arg0: i64, %arg1: i32) -> (i64, !llvm.struct<(i64, array<32 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_1 = arith.constant 1 : i64
    %2 = llvm.alloca %c1_i64_1 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_2 = arith.constant 1 : i64
    %3 = llvm.alloca %c1_i64_2 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_3 = arith.constant 1 : i64
    %4 = llvm.alloca %c1_i64_3 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_4 = arith.constant 1 : i64
    %5 = llvm.alloca %c1_i64_4 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_5 = arith.constant 1 : i64
    %6 = llvm.alloca %c1_i64_5 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1 : i64, i32)
  ^bb1(%7: i64, %8: i32):  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %c16_i32 = arith.constant 16 : i32
    cf.br ^bb3(%8 : i32)
  ^bb3(%9: i32):  // pred: ^bb2
    cf.br ^bb4(%c16_i32 : i32)
  ^bb4(%10: i32):  // pred: ^bb3
    cf.br ^bb5(%7, %9, %10 : i64, i32, i32)
  ^bb5(%11: i64, %12: i32, %13: i32):  // pred: ^bb4
    %c1_i64_6 = arith.constant 1 : i64
    %14 = arith.addi %11, %c1_i64_6 : i64
    %15 = "llvm.intr.usub.with.overflow"(%12, %13) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %16 = llvm.extractvalue %15[0] : !llvm.struct<(i32, i1)> 
    %17 = llvm.extractvalue %15[1] : !llvm.struct<(i32, i1)> 
    cf.cond_br %17, ^bb43, ^bb6
  ^bb6:  // pred: ^bb5
    cf.br ^bb7(%16 : i32)
  ^bb7(%18: i32):  // pred: ^bb6
    cf.br ^bb8
  ^bb8:  // pred: ^bb7
    %c16_i32_7 = arith.constant 16 : i32
    cf.br ^bb9(%14 : i64)
  ^bb9(%19: i64):  // pred: ^bb8
    cf.br ^bb10(%9 : i32)
  ^bb10(%20: i32):  // pred: ^bb9
    cf.br ^bb11(%c16_i32_7 : i32)
  ^bb11(%21: i32):  // pred: ^bb10
    cf.br ^bb12(%19, %20, %21 : i64, i32, i32)
  ^bb12(%22: i64, %23: i32, %24: i32):  // pred: ^bb11
    %25:2 = call @"core::integer::U32Sub::sub(f8)"(%22, %23, %24) : (i64, i32, i32) -> (i64, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %25#1, %3 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb13(%3 : !llvm.ptr)
  ^bb13(%26: !llvm.ptr):  // pred: ^bb12
    %27 = llvm.load %26 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %27 : i1, [
      default: ^bb14,
      0: ^bb15,
      1: ^bb16
    ]
  ^bb14:  // pred: ^bb13
    %false = arith.constant false
    cf.assert %false, "Invalid enum tag."
    llvm.unreachable
  ^bb15:  // pred: ^bb13
    %28 = llvm.load %26 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %29 = llvm.extractvalue %28[1] : !llvm.struct<(i1, struct<(i32)>)> 
    cf.br ^bb17
  ^bb16:  // pred: ^bb13
    %30 = llvm.load %26 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %31 = llvm.extractvalue %30[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb38
  ^bb17:  // pred: ^bb15
    cf.br ^bb18(%29 : !llvm.struct<(i32)>)
  ^bb18(%32: !llvm.struct<(i32)>):  // pred: ^bb17
    %33 = llvm.extractvalue %32[0] : !llvm.struct<(i32)> 
    cf.br ^bb19(%33 : i32)
  ^bb19(%34: i32):  // pred: ^bb18
    cf.br ^bb20(%34 : i32)
  ^bb20(%35: i32):  // pred: ^bb19
    call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%5, %35) : (!llvm.ptr, i32) -> ()
    %36 = llvm.getelementptr %5[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb21(%36 : !llvm.ptr)
  ^bb21(%37: !llvm.ptr):  // pred: ^bb20
    %38 = llvm.load %37 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %38 : i1, [
      default: ^bb22,
      0: ^bb23,
      1: ^bb24
    ]
  ^bb22:  // pred: ^bb21
    %false_8 = arith.constant false
    cf.assert %false_8, "Invalid enum tag."
    llvm.unreachable
  ^bb23:  // pred: ^bb21
    %39 = llvm.load %37 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i128)>)>
    %40 = llvm.extractvalue %39[1] : !llvm.struct<(i1, struct<(i128)>)> 
    cf.br ^bb25
  ^bb24:  // pred: ^bb21
    %41 = llvm.load %37 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %42 = llvm.extractvalue %41[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb33
  ^bb25:  // pred: ^bb23
    cf.br ^bb26(%40 : !llvm.struct<(i128)>)
  ^bb26(%43: !llvm.struct<(i128)>):  // pred: ^bb25
    %44 = llvm.extractvalue %43[0] : !llvm.struct<(i128)> 
    cf.br ^bb27(%44 : i128)
  ^bb27(%45: i128):  // pred: ^bb26
    %46 = arith.extui %45 : i128 to i252
    cf.br ^bb28
  ^bb28:  // pred: ^bb27
    %c340282366920938463463374607431768211456_i252 = arith.constant 340282366920938463463374607431768211456 : i252
    cf.br ^bb29(%46, %c340282366920938463463374607431768211456_i252 : i252, i252)
  ^bb29(%47: i252, %48: i252):  // pred: ^bb28
    %49 = arith.extui %47 : i252 to i512
    %50 = arith.extui %48 : i252 to i512
    %51 = arith.muli %49, %50 : i512
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512
    %52 = arith.remui %51, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512 : i512
    %53 = arith.cmpi uge, %51, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i512 : i512
    %54 = arith.select %53, %52, %51 : i512
    %55 = arith.trunci %54 : i512 to i252
    cf.br ^bb30(%25#0 : i64)
  ^bb30(%56: i64):  // pred: ^bb29
    cf.br ^bb31(%55 : i252)
  ^bb31(%57: i252):  // pred: ^bb30
    cf.br ^bb32
  ^bb32:  // pred: ^bb31
    cf.br ^bb57(%56, %57 : i64, i252)
  ^bb33:  // pred: ^bb24
    cf.br ^bb34(%42 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb34(%58: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb33
    %true = arith.constant true
    %59 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %60 = llvm.insertvalue %true, %59[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %61, %6 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb35(%25#0 : i64)
  ^bb35(%62: i64):  // pred: ^bb34
    cf.br ^bb36(%6 : !llvm.ptr)
  ^bb36(%63: !llvm.ptr):  // pred: ^bb35
    cf.br ^bb37(%62, %63 : i64, !llvm.ptr)
  ^bb37(%64: i64, %65: !llvm.ptr):  // pred: ^bb36
    %66 = llvm.load %63 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<32 x i8>)>
    return %62, %66 : i64, !llvm.struct<(i64, array<32 x i8>)>
  ^bb38:  // pred: ^bb16
    cf.br ^bb39(%31 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb39(%67: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb38
    %true_9 = arith.constant true
    %68 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %69 = llvm.insertvalue %true_9, %68[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %70 = llvm.insertvalue %67, %69[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %70, %4 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb40(%25#0 : i64)
  ^bb40(%71: i64):  // pred: ^bb39
    cf.br ^bb41(%4 : !llvm.ptr)
  ^bb41(%72: !llvm.ptr):  // pred: ^bb40
    cf.br ^bb42(%71, %72 : i64, !llvm.ptr)
  ^bb42(%73: i64, %74: !llvm.ptr):  // pred: ^bb41
    %75 = llvm.load %72 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<32 x i8>)>
    return %71, %75 : i64, !llvm.struct<(i64, array<32 x i8>)>
  ^bb43:  // pred: ^bb5
    cf.br ^bb44(%16 : i32)
  ^bb44(%76: i32):  // pred: ^bb43
    cf.br ^bb45(%9 : i32)
  ^bb45(%77: i32):  // pred: ^bb44
    cf.br ^bb46(%77 : i32)
  ^bb46(%78: i32):  // pred: ^bb45
    call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%0, %78) : (!llvm.ptr, i32) -> ()
    %79 = llvm.getelementptr %0[0] : (!llvm.ptr) -> !llvm.ptr, i8
    cf.br ^bb47(%14 : i64)
  ^bb47(%80: i64):  // pred: ^bb46
    cf.br ^bb48(%79 : !llvm.ptr)
  ^bb48(%81: !llvm.ptr):  // pred: ^bb47
    %82 = llvm.load %81 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %82 : i1, [
      default: ^bb49,
      0: ^bb50,
      1: ^bb51
    ]
  ^bb49:  // pred: ^bb48
    %false_10 = arith.constant false
    cf.assert %false_10, "Invalid enum tag."
    llvm.unreachable
  ^bb50:  // pred: ^bb48
    %83 = llvm.load %81 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i128)>)>
    %84 = llvm.extractvalue %83[1] : !llvm.struct<(i1, struct<(i128)>)> 
    cf.br ^bb52
  ^bb51:  // pred: ^bb48
    %85 = llvm.load %81 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %86 = llvm.extractvalue %85[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb63
  ^bb52:  // pred: ^bb50
    cf.br ^bb53(%84 : !llvm.struct<(i128)>)
  ^bb53(%87: !llvm.struct<(i128)>):  // pred: ^bb52
    %88 = llvm.extractvalue %87[0] : !llvm.struct<(i128)> 
    cf.br ^bb54(%88 : i128)
  ^bb54(%89: i128):  // pred: ^bb53
    %90 = arith.extui %89 : i128 to i252
    cf.br ^bb55(%80 : i64)
  ^bb55(%91: i64):  // pred: ^bb54
    cf.br ^bb56(%90 : i252)
  ^bb56(%92: i252):  // pred: ^bb55
    cf.br ^bb57(%91, %92 : i64, i252)
  ^bb57(%93: i64, %94: i252):  // 2 preds: ^bb32, ^bb56
    cf.br ^bb58(%94 : i252)
  ^bb58(%95: i252):  // pred: ^bb57
    %96 = llvm.mlir.undef : !llvm.struct<(i252)>
    %97 = llvm.insertvalue %95, %96[0] : !llvm.struct<(i252)> 
    cf.br ^bb59(%97 : !llvm.struct<(i252)>)
  ^bb59(%98: !llvm.struct<(i252)>):  // pred: ^bb58
    %false_11 = arith.constant false
    %99 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i252)>)>
    %100 = llvm.insertvalue %false_11, %99[0] : !llvm.struct<(i1, struct<(i252)>)> 
    %101 = llvm.insertvalue %98, %100[1] : !llvm.struct<(i1, struct<(i252)>)> 
    llvm.store %101, %2 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(i252)>)>, !llvm.ptr
    cf.br ^bb60(%93 : i64)
  ^bb60(%102: i64):  // pred: ^bb59
    cf.br ^bb61(%2 : !llvm.ptr)
  ^bb61(%103: !llvm.ptr):  // pred: ^bb60
    cf.br ^bb62(%102, %103 : i64, !llvm.ptr)
  ^bb62(%104: i64, %105: !llvm.ptr):  // pred: ^bb61
    %106 = llvm.load %103 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<32 x i8>)>
    return %102, %106 : i64, !llvm.struct<(i64, array<32 x i8>)>
  ^bb63:  // pred: ^bb51
    cf.br ^bb64(%86 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb64(%107: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb63
    %true_12 = arith.constant true
    %108 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %109 = llvm.insertvalue %true_12, %108[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %110 = llvm.insertvalue %107, %109[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %110, %1 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb65(%80 : i64)
  ^bb65(%111: i64):  // pred: ^bb64
    cf.br ^bb66(%1 : !llvm.ptr)
  ^bb66(%112: !llvm.ptr):  // pred: ^bb65
    cf.br ^bb67(%111, %112 : i64, !llvm.ptr)
  ^bb67(%113: i64, %114: !llvm.ptr):  // pred: ^bb66
    %115 = llvm.load %112 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<32 x i8>)>
    return %111, %115 : i64, !llvm.struct<(i64, array<32 x i8>)>
  }
  func.func public @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%arg0: i64, %arg1: i252) -> (i64, !llvm.struct<(i64, array<32 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1 : i64, i252)
  ^bb1(%2: i64, %3: i252):  // pred: ^bb0
    cf.br ^bb2(%2, %3 : i64, i252)
  ^bb2(%4: i64, %5: i252):  // pred: ^bb1
    %c1_i64_1 = arith.constant 1 : i64
    %6 = arith.addi %4, %c1_i64_1 : i64
    %c452312848583266388373324160190187140051835877600158453279131187530910662655_i252 = arith.constant 452312848583266388373324160190187140051835877600158453279131187530910662655 : i252
    %7 = arith.cmpi ule, %5, %c452312848583266388373324160190187140051835877600158453279131187530910662655_i252 : i252
    cf.cond_br %7, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %8 = arith.trunci %5 : i252 to i248
    cf.br ^bb5
  ^bb4:  // pred: ^bb2
    cf.br ^bb10
  ^bb5:  // pred: ^bb3
    cf.br ^bb6(%8 : i248)
  ^bb6(%9: i248):  // pred: ^bb5
    %false = arith.constant false
    %10 = llvm.mlir.undef : !llvm.struct<(i1, i248)>
    %11 = llvm.insertvalue %false, %10[0] : !llvm.struct<(i1, i248)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<(i1, i248)> 
    llvm.store %12, %1 {alignment = 8 : i64} : !llvm.struct<(i1, i248)>, !llvm.ptr
    cf.br ^bb7(%6 : i64)
  ^bb7(%13: i64):  // pred: ^bb6
    cf.br ^bb8(%1 : !llvm.ptr)
  ^bb8(%14: !llvm.ptr):  // pred: ^bb7
    cf.br ^bb9(%13, %14 : i64, !llvm.ptr)
  ^bb9(%15: i64, %16: !llvm.ptr):  // pred: ^bb8
    %17 = llvm.load %14 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<32 x i8>)>
    return %13, %17 : i64, !llvm.struct<(i64, array<32 x i8>)>
  ^bb10:  // pred: ^bb4
    cf.br ^bb11
  ^bb11:  // pred: ^bb10
    %18 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb12(%18 : !llvm.struct<()>)
  ^bb12(%19: !llvm.struct<()>):  // pred: ^bb11
    %true = arith.constant true
    %20 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %21 = llvm.insertvalue %true, %20[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %21, %0 {alignment = 8 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    cf.br ^bb13(%6 : i64)
  ^bb13(%22: i64):  // pred: ^bb12
    cf.br ^bb14(%0 : !llvm.ptr)
  ^bb14(%23: !llvm.ptr):  // pred: ^bb13
    cf.br ^bb15(%22, %23 : i64, !llvm.ptr)
  ^bb15(%24: i64, %25: !llvm.ptr):  // pred: ^bb14
    %26 = llvm.load %23 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<32 x i8>)>
    return %22, %26 : i64, !llvm.struct<(i64, array<32 x i8>)>
  }
  func.func public @"core::result::ResultTraitImpl::<(), core::fmt::Error>::expect::<core::fmt::ErrorDrop>(f2)"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(i1, array<0 x i8>)>, %arg2: i252) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg1, %arg2 : !llvm.struct<(i1, array<0 x i8>)>, i252)
  ^bb1(%2: !llvm.struct<(i1, array<0 x i8>)>, %3: i252):  // pred: ^bb0
    cf.br ^bb2(%2 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb2(%4: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb1
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.switch %5 : i1, [
      default: ^bb3,
      0: ^bb4,
      1: ^bb5
    ]
  ^bb3:  // pred: ^bb2
    %false = arith.constant false
    cf.assert %false, "Invalid enum tag."
    llvm.unreachable
  ^bb4:  // pred: ^bb2
    %6 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb6
  ^bb5:  // pred: ^bb2
    %7 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb12
  ^bb6:  // pred: ^bb4
    cf.br ^bb7(%3 : i252)
  ^bb7(%8: i252):  // pred: ^bb6
    cf.br ^bb8(%6 : !llvm.struct<()>)
  ^bb8(%9: !llvm.struct<()>):  // pred: ^bb7
    %10 = llvm.mlir.undef : !llvm.struct<(struct<()>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(struct<()>)> 
    cf.br ^bb9(%11 : !llvm.struct<(struct<()>)>)
  ^bb9(%12: !llvm.struct<(struct<()>)>):  // pred: ^bb8
    %false_1 = arith.constant false
    %13 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %14 = llvm.insertvalue %false_1, %13[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %14, %1 {alignment = 8 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    cf.br ^bb10(%1 : !llvm.ptr)
  ^bb10(%15: !llvm.ptr):  // pred: ^bb9
    cf.br ^bb11(%15 : !llvm.ptr)
  ^bb11(%16: !llvm.ptr):  // pred: ^bb10
    %c24_i64 = arith.constant 24 : i64
    %false_2 = arith.constant false
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %15, %c24_i64, %false_2) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    return
  ^bb12:  // pred: ^bb5
    cf.br ^bb13(%7 : !llvm.struct<()>)
  ^bb13(%17: !llvm.struct<()>):  // pred: ^bb12
    cf.br ^bb14
  ^bb14:  // pred: ^bb13
    %18 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32 = arith.constant 0 : i32
    %19 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %21 = llvm.insertvalue %c0_i32, %20[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %22 = llvm.insertvalue %c0_i32, %21[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb15(%22, %3 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb15(%23: !llvm.struct<(ptr<i252>, i32, i32)>, %24: i252):  // pred: ^bb14
    %25 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %26 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %27 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %28 = arith.cmpi uge, %26, %27 : i32
    %29:2 = scf.if %28 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %45 = arith.addi %27, %27 : i32
      %46 = arith.maxui %c8_i32, %45 : i32
      %47 = arith.extui %46 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %48 = arith.muli %47, %c32_i64 : i64
      %49 = llvm.bitcast %25 : !llvm.ptr<i252> to !llvm.ptr
      %50 = func.call @realloc(%49, %48) : (!llvm.ptr, i64) -> !llvm.ptr
      %51 = llvm.bitcast %50 : !llvm.ptr to !llvm.ptr<i252>
      %52 = llvm.insertvalue %51, %23[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %53 = llvm.insertvalue %46, %52[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %53, %51 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %23, %25 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %30 = llvm.getelementptr %29#1[%26] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %24, %30 : i252, !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %31 = arith.addi %26, %c1_i32 : i32
    %32 = llvm.insertvalue %31, %29#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb16
  ^bb16:  // pred: ^bb15
    %33 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb17(%33, %32 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb17(%34: !llvm.struct<()>, %35: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb16
    %36 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %37 = llvm.insertvalue %34, %36[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb18(%38 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb18(%39: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb17
    %true = arith.constant true
    %40 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %41 = llvm.insertvalue %true, %40[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %42 = llvm.insertvalue %39, %41[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %42, %0 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb19(%0 : !llvm.ptr)
  ^bb19(%43: !llvm.ptr):  // pred: ^bb18
    cf.br ^bb20(%43 : !llvm.ptr)
  ^bb20(%44: !llvm.ptr):  // pred: ^bb19
    %c24_i64_3 = arith.constant 24 : i64
    %false_4 = arith.constant false
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %43, %c24_i64_3, %false_4) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    return
  }
  func.func public @"core::array::ArraySerde::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>::serialize(f22)"(%arg0: i64, %arg1: i128, %arg2: !llvm.struct<(ptr<i248>, i32, i32)>, %arg3: !llvm.struct<(ptr<i252>, i32, i32)>) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1, %arg2, %arg3 : i64, i128, !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb1(%1: i64, %2: i128, %3: !llvm.struct<(ptr<i248>, i32, i32)>, %4: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%3 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb3(%5: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb2
    cf.br ^bb4(%5 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb4(%6: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb3
    %7 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    cf.br ^bb5(%7 : i32)
  ^bb5(%8: i32):  // pred: ^bb4
    %9 = arith.extui %8 : i32 to i252
    cf.br ^bb6(%9 : i252)
  ^bb6(%10: i252):  // pred: ^bb5
    cf.br ^bb7(%4, %10 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb7(%11: !llvm.struct<(ptr<i252>, i32, i32)>, %12: i252):  // pred: ^bb6
    %13 = llvm.extractvalue %11[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %14 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %15 = llvm.extractvalue %11[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %16 = arith.cmpi uge, %14, %15 : i32
    %17:2 = scf.if %16 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %37 = arith.addi %15, %15 : i32
      %38 = arith.maxui %c8_i32, %37 : i32
      %39 = arith.extui %38 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %40 = arith.muli %39, %c32_i64 : i64
      %41 = llvm.bitcast %13 : !llvm.ptr<i252> to !llvm.ptr
      %42 = func.call @realloc(%41, %40) : (!llvm.ptr, i64) -> !llvm.ptr
      %43 = llvm.bitcast %42 : !llvm.ptr to !llvm.ptr<i252>
      %44 = llvm.insertvalue %43, %11[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %45 = llvm.insertvalue %38, %44[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %45, %43 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %11, %13 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %18 = llvm.getelementptr %17#1[%14] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %12, %18 : i252, !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %19 = arith.addi %14, %c1_i32 : i32
    %20 = llvm.insertvalue %19, %17#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb8(%5 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb8(%21: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb7
    %22 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>
    %23 = llvm.insertvalue %21, %22[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)> 
    cf.br ^bb9(%1 : i64)
  ^bb9(%24: i64):  // pred: ^bb8
    cf.br ^bb10(%2 : i128)
  ^bb10(%25: i128):  // pred: ^bb9
    cf.br ^bb11(%23 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>)
  ^bb11(%26: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>):  // pred: ^bb10
    cf.br ^bb12(%20 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb12(%27: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb11
    cf.br ^bb13(%24, %25, %26, %27 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb13(%28: i64, %29: i128, %30: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, %31: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb12
    %32:3 = call @"core::array::serialize_array_helper::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>(f23)"(%28, %29, %30, %31) : (i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>)
    llvm.store %32#2, %0 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb14(%32#0, %32#1, %0 : i64, i128, !llvm.ptr)
  ^bb14(%33: i64, %34: i128, %35: !llvm.ptr):  // pred: ^bb13
    %36 = llvm.load %0 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %32#0, %32#1, %36 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  }
  func.func public @"core::Felt252Serde::serialize(f21)"(%arg0: i252, %arg1: !llvm.struct<(ptr<i252>, i32, i32)>) -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.struct<()>) attributes {llvm.emit_c_interface} {
    cf.br ^bb1(%arg0, %arg1 : i252, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb1(%0: i252, %1: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb0
    cf.br ^bb2(%0 : i252)
  ^bb2(%2: i252):  // pred: ^bb1
    cf.br ^bb3(%1, %2 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb3(%3: !llvm.struct<(ptr<i252>, i32, i32)>, %4: i252):  // pred: ^bb2
    %5 = llvm.extractvalue %3[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %6 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %7 = llvm.extractvalue %3[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %8 = arith.cmpi uge, %6, %7 : i32
    %9:2 = scf.if %8 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %17 = arith.addi %7, %7 : i32
      %18 = arith.maxui %c8_i32, %17 : i32
      %19 = arith.extui %18 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %20 = arith.muli %19, %c32_i64 : i64
      %21 = llvm.bitcast %5 : !llvm.ptr<i252> to !llvm.ptr
      %22 = func.call @realloc(%21, %20) : (!llvm.ptr, i64) -> !llvm.ptr
      %23 = llvm.bitcast %22 : !llvm.ptr to !llvm.ptr<i252>
      %24 = llvm.insertvalue %23, %3[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %25 = llvm.insertvalue %18, %24[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %25, %23 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %3, %5 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %10 = llvm.getelementptr %9#1[%6] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %4, %10 : i252, !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %11 = arith.addi %6, %c1_i32 : i32
    %12 = llvm.insertvalue %11, %9#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    %13 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb5(%12 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb5(%14: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb4
    cf.br ^bb6(%14, %13 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.struct<()>)
  ^bb6(%15: !llvm.struct<(ptr<i252>, i32, i32)>, %16: !llvm.struct<()>):  // pred: ^bb5
    return %14, %13 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.struct<()>
  }
  func.func public @"program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<0, 0>>(f19)"(%arg0: !llvm.ptr, %arg1: i252) -> !llvm.struct<(i1, array<0 x i8>)> attributes {llvm.emit_c_interface} {
    cf.br ^bb1(%arg0, %arg1 : !llvm.ptr, i252)
  ^bb1(%0: !llvm.ptr, %1: i252):  // pred: ^bb0
    cf.br ^bb2(%0 : !llvm.ptr)
  ^bb2(%2: !llvm.ptr):  // pred: ^bb1
    %3 = llvm.load %2 {alignment = 1 : i64} : !llvm.ptr -> i1
    cf.switch %3 : i1, [
      default: ^bb3,
      0: ^bb4,
      1: ^bb5
    ]
  ^bb3:  // pred: ^bb2
    %false = arith.constant false
    cf.assert %false, "Invalid enum tag."
    llvm.unreachable
  ^bb4:  // pred: ^bb2
    %4 = llvm.load %2 {alignment = 1 : i64} : !llvm.ptr -> !llvm.struct<(i1, i2)>
    %5 = llvm.extractvalue %4[1] : !llvm.struct<(i1, i2)> 
    cf.br ^bb6
  ^bb5:  // pred: ^bb2
    %6 = llvm.load %2 {alignment = 1 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<()>)>
    %7 = llvm.extractvalue %6[1] : !llvm.struct<(i1, struct<()>)> 
    cf.br ^bb25
  ^bb6:  // pred: ^bb4
    cf.br ^bb7(%5 : i2)
  ^bb7(%8: i2):  // pred: ^bb6
    %9 = arith.extsi %8 : i2 to i252
    %c0_i252 = arith.constant 0 : i252
    %10 = arith.cmpi slt, %9, %c0_i252 : i252
    cf.cond_br %10, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %11 = arith.extsi %8 : i2 to i252
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252
    %12 = arith.addi %11, %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252 : i252
    cf.br ^bb10(%12 : i252)
  ^bb9:  // pred: ^bb7
    %13 = arith.extui %8 : i2 to i252
    cf.br ^bb10(%13 : i252)
  ^bb10(%14: i252):  // 2 preds: ^bb8, ^bb9
    cf.br ^bb11(%14, %1 : i252, i252)
  ^bb11(%15: i252, %16: i252):  // pred: ^bb10
    %17 = arith.extui %15 : i252 to i256
    %18 = arith.extui %16 : i252 to i256
    %19 = arith.subi %17, %18 : i256
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256
    %20 = arith.addi %19, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256 : i256
    %21 = arith.cmpi ult, %17, %18 : i256
    %22 = arith.select %21, %20, %19 : i256
    %23 = arith.trunci %22 : i256 to i252
    cf.br ^bb12(%23 : i252)
  ^bb12(%24: i252):  // pred: ^bb11
    cf.br ^bb13(%24 : i252)
  ^bb13(%25: i252):  // pred: ^bb12
    %c0_i252_0 = arith.constant 0 : i252
    %26 = arith.cmpi eq, %25, %c0_i252_0 : i252
    cf.cond_br %26, ^bb14, ^bb19
  ^bb14:  // pred: ^bb13
    cf.br ^bb15
  ^bb15:  // pred: ^bb14
    %27 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb16(%27 : !llvm.struct<()>)
  ^bb16(%28: !llvm.struct<()>):  // pred: ^bb15
    %true = arith.constant true
    %29 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %30 = llvm.insertvalue %true, %29[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb17(%30 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb17(%31: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb16
    cf.br ^bb18(%31 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb18(%32: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb17
    return %31 : !llvm.struct<(i1, array<0 x i8>)>
  ^bb19:  // pred: ^bb13
    cf.br ^bb20(%25 : i252)
  ^bb20(%33: i252):  // pred: ^bb19
    cf.br ^bb21
  ^bb21:  // pred: ^bb20
    %34 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb22(%34 : !llvm.struct<()>)
  ^bb22(%35: !llvm.struct<()>):  // pred: ^bb21
    %false_1 = arith.constant false
    %36 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %37 = llvm.insertvalue %false_1, %36[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb23(%37 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb23(%38: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb22
    cf.br ^bb24(%38 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb24(%39: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb23
    return %38 : !llvm.struct<(i1, array<0 x i8>)>
  ^bb25:  // pred: ^bb5
    cf.br ^bb26(%7 : !llvm.struct<()>)
  ^bb26(%40: !llvm.struct<()>):  // pred: ^bb25
    cf.br ^bb27(%1 : i252)
  ^bb27(%41: i252):  // pred: ^bb26
    cf.br ^bb28
  ^bb28:  // pred: ^bb27
    %42 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb29(%42 : !llvm.struct<()>)
  ^bb29(%43: !llvm.struct<()>):  // pred: ^bb28
    %false_2 = arith.constant false
    %44 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %45 = llvm.insertvalue %false_2, %44[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb30(%45 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb30(%46: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb29
    cf.br ^bb31(%46 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb31(%47: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb30
    return %46 : !llvm.struct<(i1, array<0 x i8>)>
  }
  func.func public @"program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<-1, -1>>(f15)"(%arg0: !llvm.ptr, %arg1: i252) -> !llvm.struct<(i1, array<0 x i8>)> attributes {llvm.emit_c_interface} {
    cf.br ^bb1(%arg0, %arg1 : !llvm.ptr, i252)
  ^bb1(%0: !llvm.ptr, %1: i252):  // pred: ^bb0
    cf.br ^bb2(%0 : !llvm.ptr)
  ^bb2(%2: !llvm.ptr):  // pred: ^bb1
    %3 = llvm.load %2 {alignment = 1 : i64} : !llvm.ptr -> i1
    cf.switch %3 : i1, [
      default: ^bb3,
      0: ^bb4,
      1: ^bb5
    ]
  ^bb3:  // pred: ^bb2
    %false = arith.constant false
    cf.assert %false, "Invalid enum tag."
    llvm.unreachable
  ^bb4:  // pred: ^bb2
    %4 = llvm.load %2 {alignment = 1 : i64} : !llvm.ptr -> !llvm.struct<(i1, i2)>
    %5 = llvm.extractvalue %4[1] : !llvm.struct<(i1, i2)> 
    cf.br ^bb6
  ^bb5:  // pred: ^bb2
    %6 = llvm.load %2 {alignment = 1 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<()>)>
    %7 = llvm.extractvalue %6[1] : !llvm.struct<(i1, struct<()>)> 
    cf.br ^bb25
  ^bb6:  // pred: ^bb4
    cf.br ^bb7(%5 : i2)
  ^bb7(%8: i2):  // pred: ^bb6
    %9 = arith.extsi %8 : i2 to i252
    %c0_i252 = arith.constant 0 : i252
    %10 = arith.cmpi slt, %9, %c0_i252 : i252
    cf.cond_br %10, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %11 = arith.extsi %8 : i2 to i252
    %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252
    %12 = arith.addi %11, %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252 : i252
    cf.br ^bb10(%12 : i252)
  ^bb9:  // pred: ^bb7
    %13 = arith.extui %8 : i2 to i252
    cf.br ^bb10(%13 : i252)
  ^bb10(%14: i252):  // 2 preds: ^bb8, ^bb9
    cf.br ^bb11(%14, %1 : i252, i252)
  ^bb11(%15: i252, %16: i252):  // pred: ^bb10
    %17 = arith.extui %15 : i252 to i256
    %18 = arith.extui %16 : i252 to i256
    %19 = arith.subi %17, %18 : i256
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256
    %20 = arith.addi %19, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256 : i256
    %21 = arith.cmpi ult, %17, %18 : i256
    %22 = arith.select %21, %20, %19 : i256
    %23 = arith.trunci %22 : i256 to i252
    cf.br ^bb12(%23 : i252)
  ^bb12(%24: i252):  // pred: ^bb11
    cf.br ^bb13(%24 : i252)
  ^bb13(%25: i252):  // pred: ^bb12
    %c0_i252_0 = arith.constant 0 : i252
    %26 = arith.cmpi eq, %25, %c0_i252_0 : i252
    cf.cond_br %26, ^bb14, ^bb19
  ^bb14:  // pred: ^bb13
    cf.br ^bb15
  ^bb15:  // pred: ^bb14
    %27 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb16(%27 : !llvm.struct<()>)
  ^bb16(%28: !llvm.struct<()>):  // pred: ^bb15
    %true = arith.constant true
    %29 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %30 = llvm.insertvalue %true, %29[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb17(%30 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb17(%31: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb16
    cf.br ^bb18(%31 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb18(%32: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb17
    return %31 : !llvm.struct<(i1, array<0 x i8>)>
  ^bb19:  // pred: ^bb13
    cf.br ^bb20(%25 : i252)
  ^bb20(%33: i252):  // pred: ^bb19
    cf.br ^bb21
  ^bb21:  // pred: ^bb20
    %34 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb22(%34 : !llvm.struct<()>)
  ^bb22(%35: !llvm.struct<()>):  // pred: ^bb21
    %false_1 = arith.constant false
    %36 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %37 = llvm.insertvalue %false_1, %36[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb23(%37 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb23(%38: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb22
    cf.br ^bb24(%38 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb24(%39: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb23
    return %38 : !llvm.struct<(i1, array<0 x i8>)>
  ^bb25:  // pred: ^bb5
    cf.br ^bb26(%7 : !llvm.struct<()>)
  ^bb26(%40: !llvm.struct<()>):  // pred: ^bb25
    cf.br ^bb27(%1 : i252)
  ^bb27(%41: i252):  // pred: ^bb26
    cf.br ^bb28
  ^bb28:  // pred: ^bb27
    %42 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb29(%42 : !llvm.struct<()>)
  ^bb29(%43: !llvm.struct<()>):  // pred: ^bb28
    %false_2 = arith.constant false
    %44 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %45 = llvm.insertvalue %false_2, %44[0] : !llvm.struct<(i1, array<0 x i8>)> 
    cf.br ^bb30(%45 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb30(%46: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb29
    cf.br ^bb31(%46 : !llvm.struct<(i1, array<0 x i8>)>)
  ^bb31(%47: !llvm.struct<(i1, array<0 x i8>)>):  // pred: ^bb30
    return %46 : !llvm.struct<(i1, array<0 x i8>)>
  }
  func.func public @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>(f5)"(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i252) attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg1, %arg2 : !llvm.ptr, i252)
  ^bb1(%2: !llvm.ptr, %3: i252):  // pred: ^bb0
    cf.br ^bb2(%2 : !llvm.ptr)
  ^bb2(%4: !llvm.ptr):  // pred: ^bb1
    %5 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i1
    cf.switch %5 : i1, [
      default: ^bb3,
      0: ^bb4,
      1: ^bb5
    ]
  ^bb3:  // pred: ^bb2
    %false = arith.constant false
    cf.assert %false, "Invalid enum tag."
    llvm.unreachable
  ^bb4:  // pred: ^bb2
    %6 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> !llvm.struct<(i1, i32)>
    %7 = llvm.extractvalue %6[1] : !llvm.struct<(i1, i32)> 
    cf.br ^bb6
  ^bb5:  // pred: ^bb2
    %8 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> !llvm.struct<(i1, i32)>
    %9 = llvm.extractvalue %8[1] : !llvm.struct<(i1, i32)> 
    cf.br ^bb12
  ^bb6:  // pred: ^bb4
    cf.br ^bb7(%3 : i252)
  ^bb7(%10: i252):  // pred: ^bb6
    cf.br ^bb8(%7 : i32)
  ^bb8(%11: i32):  // pred: ^bb7
    %12 = llvm.mlir.undef : !llvm.struct<(i32)>
    %13 = llvm.insertvalue %11, %12[0] : !llvm.struct<(i32)> 
    cf.br ^bb9(%13 : !llvm.struct<(i32)>)
  ^bb9(%14: !llvm.struct<(i32)>):  // pred: ^bb8
    %false_1 = arith.constant false
    %15 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i32)>)>
    %16 = llvm.insertvalue %false_1, %15[0] : !llvm.struct<(i1, struct<(i32)>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(i1, struct<(i32)>)> 
    llvm.store %17, %1 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(i32)>)>, !llvm.ptr
    cf.br ^bb10(%1 : !llvm.ptr)
  ^bb10(%18: !llvm.ptr):  // pred: ^bb9
    cf.br ^bb11(%18 : !llvm.ptr)
  ^bb11(%19: !llvm.ptr):  // pred: ^bb10
    %c24_i64 = arith.constant 24 : i64
    %false_2 = arith.constant false
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %18, %c24_i64, %false_2) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    return
  ^bb12:  // pred: ^bb5
    cf.br ^bb13(%9 : i32)
  ^bb13(%20: i32):  // pred: ^bb12
    cf.br ^bb14
  ^bb14:  // pred: ^bb13
    %21 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32 = arith.constant 0 : i32
    %22 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %23 = llvm.insertvalue %21, %22[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %24 = llvm.insertvalue %c0_i32, %23[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %25 = llvm.insertvalue %c0_i32, %24[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb15(%25, %3 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb15(%26: !llvm.struct<(ptr<i252>, i32, i32)>, %27: i252):  // pred: ^bb14
    %28 = llvm.extractvalue %26[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %29 = llvm.extractvalue %26[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %30 = llvm.extractvalue %26[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %31 = arith.cmpi uge, %29, %30 : i32
    %32:2 = scf.if %31 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %48 = arith.addi %30, %30 : i32
      %49 = arith.maxui %c8_i32, %48 : i32
      %50 = arith.extui %49 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %51 = arith.muli %50, %c32_i64 : i64
      %52 = llvm.bitcast %28 : !llvm.ptr<i252> to !llvm.ptr
      %53 = func.call @realloc(%52, %51) : (!llvm.ptr, i64) -> !llvm.ptr
      %54 = llvm.bitcast %53 : !llvm.ptr to !llvm.ptr<i252>
      %55 = llvm.insertvalue %54, %26[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %56 = llvm.insertvalue %49, %55[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %56, %54 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %26, %28 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %33 = llvm.getelementptr %32#1[%29] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %27, %33 : i252, !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %34 = arith.addi %29, %c1_i32 : i32
    %35 = llvm.insertvalue %34, %32#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb16
  ^bb16:  // pred: ^bb15
    %36 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb17(%36, %35 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb17(%37: !llvm.struct<()>, %38: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb16
    %39 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %40 = llvm.insertvalue %37, %39[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %41 = llvm.insertvalue %38, %40[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb18(%41 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb18(%42: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb17
    %true = arith.constant true
    %43 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %44 = llvm.insertvalue %true, %43[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %45 = llvm.insertvalue %42, %44[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %45, %0 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb19(%0 : !llvm.ptr)
  ^bb19(%46: !llvm.ptr):  // pred: ^bb18
    cf.br ^bb20(%46 : !llvm.ptr)
  ^bb20(%47: !llvm.ptr):  // pred: ^bb19
    %c24_i64_3 = arith.constant 24 : i64
    %false_4 = arith.constant false
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %46, %c24_i64_3, %false_4) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    return
  }
  func.func public @"core::array::serialize_array_helper::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>(f23)"(%arg0: i64, %arg1: i128, %arg2: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, %arg3: !llvm.struct<(ptr<i252>, i32, i32)>) -> (i64, i128, !llvm.struct<(i64, array<16 x i8>)>) attributes {llvm.emit_c_interface} {
    %alloca = memref.alloca() : memref<index>
    %idx0 = index.constant 0
    memref.store %idx0, %alloca[] : memref<index>
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_0 = arith.constant 1 : i64
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i64, array<8 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_1 = arith.constant 1 : i64
    %2 = llvm.alloca %c1_i64_1 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_2 = arith.constant 1 : i64
    %3 = llvm.alloca %c1_i64_2 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %c1_i64_3 = arith.constant 1 : i64
    %4 = llvm.alloca %c1_i64_3 x !llvm.struct<(i64, array<8 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    cf.br ^bb1(%arg0, %arg1, %arg2, %arg3 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb1(%5: i64, %6: i128, %7: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, %8: !llvm.struct<(ptr<i252>, i32, i32)>):  // 2 preds: ^bb0, ^bb40
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    cf.br ^bb3
  ^bb3:  // pred: ^bb2
    %9 = llvm.mlir.undef : !llvm.array<0 x i8>
    cf.br ^bb4(%9 : !llvm.array<0 x i8>)
  ^bb4(%10: !llvm.array<0 x i8>):  // pred: ^bb3
    cf.br ^bb5(%5, %6, %10 : i64, i128, !llvm.array<0 x i8>)
  ^bb5(%11: i64, %12: i128, %13: !llvm.array<0 x i8>):  // pred: ^bb4
    %c1_i64_4 = arith.constant 1 : i64
    %14 = arith.addi %11, %c1_i64_4 : i64
    %c2670_i128 = arith.constant 2670 : i128
    %15 = arith.cmpi uge, %12, %c2670_i128 : i128
    %16 = llvm.call_intrinsic "llvm.usub.sat"(%12, %c2670_i128) : (i128, i128) -> i128  {intrin = "llvm.usub.sat"}
    cf.cond_br %15, ^bb6, ^bb55
  ^bb6:  // pred: ^bb5
    cf.br ^bb7(%7 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>)
  ^bb7(%17: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>):  // pred: ^bb6
    %18 = llvm.extractvalue %17[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)> 
    cf.br ^bb8
  ^bb8:  // pred: ^bb7
    cf.br ^bb9(%14 : i64)
  ^bb9(%19: i64):  // pred: ^bb8
    cf.br ^bb10(%18 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb10(%20: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb9
    %21 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %c0_i32 = arith.constant 0 : i32
    %22 = arith.cmpi eq, %21, %c0_i32 : i32
    cf.cond_br %22, ^bb12, ^bb11
  ^bb11:  // pred: ^bb10
    %23 = llvm.extractvalue %20[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %24 = llvm.getelementptr %23[%c0_i32] : (!llvm.ptr<i248>, i32) -> !llvm.ptr, i248
    %25 = llvm.mlir.null : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %26 = call @realloc(%25, %c32_i64) : (!llvm.ptr, i64) -> !llvm.ptr
    %27 = llvm.load %24 {alignment = 8 : i64} : !llvm.ptr -> i248
    llvm.store %27, %26 {alignment = 8 : i64} : i248, !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %28 = llvm.getelementptr %23[%c1_i32] : (!llvm.ptr<i248>, i32) -> !llvm.ptr, i248
    %29 = arith.subi %21, %c1_i32 : i32
    %30 = arith.extui %29 : i32 to i64
    %c32_i64_5 = arith.constant 32 : i64
    %31 = arith.muli %30, %c32_i64_5 : i64
    %32 = llvm.bitcast %23 : !llvm.ptr<i248> to !llvm.ptr
    "llvm.intr.memmove"(%32, %28, %31) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %33 = llvm.insertvalue %29, %20[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    cf.br ^bb13
  ^bb12:  // pred: ^bb10
    cf.br ^bb18
  ^bb13:  // pred: ^bb11
    cf.br ^bb14(%26 : !llvm.ptr)
  ^bb14(%34: !llvm.ptr):  // pred: ^bb13
    %false = arith.constant false
    %35 = llvm.mlir.undef : !llvm.struct<(i1, ptr)>
    %36 = llvm.insertvalue %false, %35[0] : !llvm.struct<(i1, ptr)> 
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(i1, ptr)> 
    llvm.store %37, %4 {alignment = 8 : i64} : !llvm.struct<(i1, ptr)>, !llvm.ptr
    cf.br ^bb15(%33 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb15(%38: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb14
    cf.br ^bb16(%4 : !llvm.ptr)
  ^bb16(%39: !llvm.ptr):  // pred: ^bb15
    cf.br ^bb17
  ^bb17:  // pred: ^bb16
    cf.br ^bb23(%8, %19, %16, %38, %39 : !llvm.struct<(ptr<i252>, i32, i32)>, i64, i128, !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr)
  ^bb18:  // pred: ^bb12
    cf.br ^bb19
  ^bb19:  // pred: ^bb18
    %40 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb20(%40 : !llvm.struct<()>)
  ^bb20(%41: !llvm.struct<()>):  // pred: ^bb19
    %true = arith.constant true
    %42 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %43 = llvm.insertvalue %true, %42[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %43, %1 {alignment = 8 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    cf.br ^bb21(%20 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb21(%44: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb20
    cf.br ^bb22(%1 : !llvm.ptr)
  ^bb22(%45: !llvm.ptr):  // pred: ^bb21
    cf.br ^bb23(%8, %19, %16, %44, %45 : !llvm.struct<(ptr<i252>, i32, i32)>, i64, i128, !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr)
  ^bb23(%46: !llvm.struct<(ptr<i252>, i32, i32)>, %47: i64, %48: i128, %49: !llvm.struct<(ptr<i248>, i32, i32)>, %50: !llvm.ptr):  // 2 preds: ^bb17, ^bb22
    cf.br ^bb24(%50 : !llvm.ptr)
  ^bb24(%51: !llvm.ptr):  // pred: ^bb23
    %52 = llvm.load %51 {alignment = 8 : i64} : !llvm.ptr -> i1
    cf.switch %52 : i1, [
      default: ^bb25,
      0: ^bb26,
      1: ^bb27
    ]
  ^bb25:  // pred: ^bb24
    %false_6 = arith.constant false
    cf.assert %false_6, "Invalid enum tag."
    llvm.unreachable
  ^bb26:  // pred: ^bb24
    %53 = llvm.load %51 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, ptr)>
    %54 = llvm.extractvalue %53[1] : !llvm.struct<(i1, ptr)> 
    cf.br ^bb28
  ^bb27:  // pred: ^bb24
    %55 = llvm.load %51 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<()>)>
    %56 = llvm.extractvalue %55[1] : !llvm.struct<(i1, struct<()>)> 
    cf.br ^bb44
  ^bb28:  // pred: ^bb26
    cf.br ^bb29
  ^bb29:  // pred: ^bb28
    cf.br ^bb30(%54 : !llvm.ptr)
  ^bb30(%57: !llvm.ptr):  // pred: ^bb29
    %58 = llvm.load %57 {alignment = 8 : i64} : !llvm.ptr -> i248
    call @free(%57) : (!llvm.ptr) -> ()
    cf.br ^bb31(%58 : i248)
  ^bb31(%59: i248):  // pred: ^bb30
    cf.br ^bb32(%59 : i248)
  ^bb32(%60: i248):  // pred: ^bb31
    %61 = arith.extui %60 : i248 to i252
    cf.br ^bb33(%61 : i252)
  ^bb33(%62: i252):  // pred: ^bb32
    cf.br ^bb34(%46, %62 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb34(%63: !llvm.struct<(ptr<i252>, i32, i32)>, %64: i252):  // pred: ^bb33
    %65 = llvm.extractvalue %63[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %66 = llvm.extractvalue %63[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %67 = llvm.extractvalue %63[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %68 = arith.cmpi uge, %66, %67 : i32
    %69:2 = scf.if %68 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %153 = arith.addi %67, %67 : i32
      %154 = arith.maxui %c8_i32, %153 : i32
      %155 = arith.extui %154 : i32 to i64
      %c32_i64_14 = arith.constant 32 : i64
      %156 = arith.muli %155, %c32_i64_14 : i64
      %157 = llvm.bitcast %65 : !llvm.ptr<i252> to !llvm.ptr
      %158 = func.call @realloc(%157, %156) : (!llvm.ptr, i64) -> !llvm.ptr
      %159 = llvm.bitcast %158 : !llvm.ptr to !llvm.ptr<i252>
      %160 = llvm.insertvalue %159, %63[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %161 = llvm.insertvalue %154, %160[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %161, %159 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %63, %65 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %70 = llvm.getelementptr %69#1[%66] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %64, %70 : i252, !llvm.ptr
    %c1_i32_7 = arith.constant 1 : i32
    %71 = arith.addi %66, %c1_i32_7 : i32
    %72 = llvm.insertvalue %71, %69#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb35(%49 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb35(%73: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb34
    %74 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>
    %75 = llvm.insertvalue %73, %74[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)> 
    cf.br ^bb36(%47 : i64)
  ^bb36(%76: i64):  // pred: ^bb35
    cf.br ^bb37(%48 : i128)
  ^bb37(%77: i128):  // pred: ^bb36
    cf.br ^bb38(%75 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>)
  ^bb38(%78: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>):  // pred: ^bb37
    cf.br ^bb39(%72 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb39(%79: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb38
    cf.br ^bb40(%76, %77, %78, %79 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb40(%80: i64, %81: i128, %82: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, %83: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb39
    %84 = memref.load %alloca[] : memref<index>
    %idx1 = index.constant 1
    %85 = index.add %84, %idx1
    memref.store %85, %alloca[] : memref<index>
    cf.br ^bb1(%80, %81, %82, %83 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb41(%86: i64, %87: i128, %88: !llvm.struct<(i64, array<16 x i8>)>):  // pred: ^bb42
    llvm.store %88, %3 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    cf.br ^bb42(%86, %87, %3 : i64, i128, !llvm.ptr)
  ^bb42(%89: i64, %90: i128, %91: !llvm.ptr):  // pred: ^bb41
    %92 = memref.load %alloca[] : memref<index>
    %idx0_8 = index.constant 0
    %93 = index.cmp eq(%92, %idx0_8)
    %idx1_9 = index.constant 1
    %94 = index.sub %92, %idx1_9
    memref.store %94, %alloca[] : memref<index>
    %95 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    cf.cond_br %93, ^bb43, ^bb41(%86, %87, %95 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>)
  ^bb43:  // pred: ^bb42
    %96 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %86, %87, %96 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb44:  // pred: ^bb27
    cf.br ^bb45
  ^bb45:  // pred: ^bb44
    cf.br ^bb46(%56 : !llvm.struct<()>)
  ^bb46(%97: !llvm.struct<()>):  // pred: ^bb45
    cf.br ^bb47(%49 : !llvm.struct<(ptr<i248>, i32, i32)>)
  ^bb47(%98: !llvm.struct<(ptr<i248>, i32, i32)>):  // pred: ^bb46
    cf.br ^bb48
  ^bb48:  // pred: ^bb47
    %99 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb49(%46, %99 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.struct<()>)
  ^bb49(%100: !llvm.struct<(ptr<i252>, i32, i32)>, %101: !llvm.struct<()>):  // pred: ^bb48
    %102 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>
    %103 = llvm.insertvalue %100, %102[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %104 = llvm.insertvalue %101, %103[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    cf.br ^bb50(%104 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)
  ^bb50(%105: !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>):  // pred: ^bb49
    %false_10 = arith.constant false
    %106 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %107 = llvm.insertvalue %false_10, %106[0] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    %108 = llvm.insertvalue %105, %107[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    llvm.store %108, %2 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>, !llvm.ptr
    cf.br ^bb51(%47 : i64)
  ^bb51(%109: i64):  // pred: ^bb50
    cf.br ^bb52(%48 : i128)
  ^bb52(%110: i128):  // pred: ^bb51
    cf.br ^bb53(%2 : !llvm.ptr)
  ^bb53(%111: !llvm.ptr):  // pred: ^bb52
    cf.br ^bb54(%109, %110, %111 : i64, i128, !llvm.ptr)
  ^bb54(%112: i64, %113: i128, %114: !llvm.ptr):  // pred: ^bb53
    %115 = llvm.load %111 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %109, %110, %115 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  ^bb55:  // pred: ^bb5
    cf.br ^bb56(%8 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb56(%116: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb55
    %117 = llvm.extractvalue %116[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %118 = llvm.bitcast %117 : !llvm.ptr<i252> to !llvm.ptr
    call @free(%118) : (!llvm.ptr) -> ()
    cf.br ^bb57(%7 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>)
  ^bb57(%119: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>):  // pred: ^bb56
    cf.br ^bb58
  ^bb58:  // pred: ^bb57
    %120 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32_11 = arith.constant 0 : i32
    %121 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %122 = llvm.insertvalue %120, %121[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %123 = llvm.insertvalue %c0_i32_11, %122[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %124 = llvm.insertvalue %c0_i32_11, %123[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb59
  ^bb59:  // pred: ^bb58
    %c375233589013918064796019_i252 = arith.constant 375233589013918064796019 : i252
    cf.br ^bb60(%c375233589013918064796019_i252 : i252)
  ^bb60(%125: i252):  // pred: ^bb59
    cf.br ^bb61(%124, %125 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb61(%126: !llvm.struct<(ptr<i252>, i32, i32)>, %127: i252):  // pred: ^bb60
    %128 = llvm.extractvalue %126[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %129 = llvm.extractvalue %126[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %130 = llvm.extractvalue %126[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %131 = arith.cmpi uge, %129, %130 : i32
    %132:2 = scf.if %131 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %153 = arith.addi %130, %130 : i32
      %154 = arith.maxui %c8_i32, %153 : i32
      %155 = arith.extui %154 : i32 to i64
      %c32_i64_14 = arith.constant 32 : i64
      %156 = arith.muli %155, %c32_i64_14 : i64
      %157 = llvm.bitcast %128 : !llvm.ptr<i252> to !llvm.ptr
      %158 = func.call @realloc(%157, %156) : (!llvm.ptr, i64) -> !llvm.ptr
      %159 = llvm.bitcast %158 : !llvm.ptr to !llvm.ptr<i252>
      %160 = llvm.insertvalue %159, %126[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %161 = llvm.insertvalue %154, %160[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %161, %159 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %126, %128 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %133 = llvm.getelementptr %132#1[%129] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %127, %133 : i252, !llvm.ptr
    %c1_i32_12 = arith.constant 1 : i32
    %134 = arith.addi %129, %c1_i32_12 : i32
    %135 = llvm.insertvalue %134, %132#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb62
  ^bb62:  // pred: ^bb61
    %136 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb63(%136, %135 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb63(%137: !llvm.struct<()>, %138: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb62
    %139 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %140 = llvm.insertvalue %137, %139[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %141 = llvm.insertvalue %138, %140[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb64(%141 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb64(%142: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb63
    %true_13 = arith.constant true
    %143 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %144 = llvm.insertvalue %true_13, %143[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %145 = llvm.insertvalue %142, %144[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %145, %0 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    cf.br ^bb65(%14 : i64)
  ^bb65(%146: i64):  // pred: ^bb64
    cf.br ^bb66(%16 : i128)
  ^bb66(%147: i128):  // pred: ^bb65
    cf.br ^bb67(%0 : !llvm.ptr)
  ^bb67(%148: !llvm.ptr):  // pred: ^bb66
    cf.br ^bb68(%146, %147, %148 : i64, i128, !llvm.ptr)
  ^bb68(%149: i64, %150: i128, %151: !llvm.ptr):  // pred: ^bb67
    %152 = llvm.load %148 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    return %146, %147, %152 : i64, i128, !llvm.struct<(i64, array<16 x i8>)>
  }
}
