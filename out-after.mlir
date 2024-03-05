module attributes {llvm.data_layout = ""} {
  llvm.mlir.global private constant @assert_msg_8(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_7(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_6(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_5(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_4(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_3(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_2(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.mlir.global private constant @assert_msg_1(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.func @abort()
  llvm.func @puts(!llvm.ptr)
  llvm.mlir.global private constant @assert_msg_0(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @free(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @"program::program::special_casts::test_felt252_downcasts(f20)"(%arg0: i64, %arg1: i128) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(256 : i64) : i64
    %1 = llvm.mlir.constant(17495623119495214 : i252) : i252
    %2 = llvm.mlir.constant(175590441458062252655054000563808860724590867505131080439014142883694195006 : i252) : i252
    %3 = llvm.mlir.constant(10 : i32) : i32
    %4 = llvm.mlir.constant(213414867255199290449966 : i252) : i252
    %5 = llvm.mlir.constant(172132395238539276156095731494591481422342884673560379810407770160186338336 : i252) : i252
    %6 = llvm.mlir.constant(-3618502788666131000275863779947924135206266826270938552493006944358698582017 : i252) : i252
    %7 = llvm.mlir.constant(4485066453471027246 : i252) : i252
    %8 = llvm.mlir.constant(-3618502788666131000275863779947924135206266826270938552493006944358698582016 : i252) : i252
    %9 = llvm.mlir.constant(4 : i32) : i32
    %10 = llvm.mlir.constant(808017966 : i252) : i252
    %11 = llvm.mlir.constant(175590441458062252655054000563808860724590867505131080439014981759796133416 : i252) : i252
    %12 = llvm.mlir.constant(172180977190876322177717838039515195832848434333118596004974948384901134190 : i252) : i252
    %13 = llvm.mlir.constant(0 : i252) : i252
    %14 = llvm.mlir.constant(true) : i1
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(8 : i32) : i32
    %17 = llvm.mlir.constant(1997209042069643135709344952807065910992472029923670688473712229447419591075 : i252) : i252
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(7 : i32) : i32
    %20 = llvm.mlir.constant(17519790900469806 : i252) : i252
    %21 = llvm.mlir.constant(172132395238539276156095731494591481422342884673560379810407770160236339248 : i252) : i252
    %22 = llvm.mlir.constant(31 : i32) : i32
    %23 = llvm.mlir.constant(172180977190876322177717838039515195832848434332511767082422530228238249590 : i252) : i252
    %24 = llvm.mlir.constant(false) : i1
    %25 = llvm.mlir.constant(1 : i252) : i252
    %26 = llvm.mlir.constant(1 : i64) : i64
    %27 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %28 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %29 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %30 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %31 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %32 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %33 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %34 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %35 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %36 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %37 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %38 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %39 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %40 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %41 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %42 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %43 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %44 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %45 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %46 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %47 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %48 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %49 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %50 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %51 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %52 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %53 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %54 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %55 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %56 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %57 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %58 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %59 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %60 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %61 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %62 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %63 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %64 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %65 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %66 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %67 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %68 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %69 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %70 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %71 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %72 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %73 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %74 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %75 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %76 = llvm.alloca %26 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %77 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %78 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %79 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %80 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %81 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %82 = llvm.alloca %26 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %83 = llvm.call @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<0, 0>>(f17)"(%arg0, %25) : (i64, i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    %84 = llvm.extractvalue %83[0] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %85 = llvm.extractvalue %83[1] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %86 = llvm.extractvalue %85[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.switch %86 : i1, ^bb1 [
      0: ^bb3(%84, %23, %72, %72, %72, %21, %74, %74, %74, %20, %19, %76, %76, %76, %78, %78, %80, %80, %80, %82, %82, %80, %81, %81, %78, %79, %79, %76, %77, %77, %74, %75, %75, %72, %73, %73 : i64, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr),
      1: ^bb4
    ]
  ^bb1:  // 10 preds: ^bb0, ^bb3, ^bb4, ^bb5, ^bb7, ^bb8, ^bb9, ^bb11, ^bb12, ^bb13
    llvm.cond_br %24, ^bb2, ^bb15
  ^bb2:  // pred: ^bb1
    llvm.unreachable
  ^bb3(%87: i64, %88: i252, %89: !llvm.ptr, %90: !llvm.ptr, %91: !llvm.ptr, %92: i252, %93: !llvm.ptr, %94: !llvm.ptr, %95: !llvm.ptr, %96: i252, %97: i32, %98: !llvm.ptr, %99: !llvm.ptr, %100: !llvm.ptr, %101: !llvm.ptr, %102: !llvm.ptr, %103: !llvm.ptr, %104: !llvm.ptr, %105: !llvm.ptr, %106: !llvm.ptr, %107: !llvm.ptr, %108: !llvm.ptr, %109: !llvm.ptr, %110: !llvm.ptr, %111: !llvm.ptr, %112: !llvm.ptr, %113: !llvm.ptr, %114: !llvm.ptr, %115: !llvm.ptr, %116: !llvm.ptr, %117: !llvm.ptr, %118: !llvm.ptr, %119: !llvm.ptr, %120: !llvm.ptr, %121: !llvm.ptr, %122: !llvm.ptr):  // 5 preds: ^bb0, ^bb4, ^bb11, ^bb12, ^bb13
    %123 = llvm.call @"core::fmt::FormatterDefault::default(f12)"() : () -> !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
    %124 = llvm.extractvalue %123[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)> 
    %125 = llvm.call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%87, %124, %88, %22) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
    %126 = llvm.extractvalue %125[0] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    %127 = llvm.extractvalue %125[1] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    llvm.store %127, %89 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    %128 = llvm.load %90 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %128 : i1, ^bb1 [
      0: ^bb5(%91, %126, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119 : !llvm.ptr, i64, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr),
      1: ^bb6(%120, %121, %122, %126, %arg1 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i128)
    ]
  ^bb4:  // pred: ^bb0
    %129 = llvm.call @"program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<0, 0>>(f18)"(%84, %13) : (i64, i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    %130 = llvm.extractvalue %129[0] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %131 = llvm.extractvalue %129[1] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %132 = llvm.extractvalue %131[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.switch %132 : i1, ^bb1 [
      0: ^bb3(%130, %12, %61, %61, %61, %11, %63, %63, %63, %10, %9, %65, %65, %65, %67, %67, %69, %69, %69, %71, %71, %69, %70, %70, %67, %68, %68, %65, %66, %66, %63, %64, %64, %61, %62, %62 : i64, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr),
      1: ^bb11
    ]
  ^bb5(%133: !llvm.ptr, %134: i64, %135: i252, %136: !llvm.ptr, %137: !llvm.ptr, %138: !llvm.ptr, %139: i252, %140: i32, %141: !llvm.ptr, %142: !llvm.ptr, %143: !llvm.ptr, %144: !llvm.ptr, %145: !llvm.ptr, %146: !llvm.ptr, %147: !llvm.ptr, %148: !llvm.ptr, %149: !llvm.ptr, %150: !llvm.ptr, %151: !llvm.ptr, %152: !llvm.ptr, %153: !llvm.ptr, %154: !llvm.ptr, %155: !llvm.ptr, %156: !llvm.ptr, %157: !llvm.ptr, %158: !llvm.ptr, %159: !llvm.ptr, %160: !llvm.ptr, %161: !llvm.ptr, %162: !llvm.ptr):  // pred: ^bb3
    %163 = llvm.load %133 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %164 = llvm.extractvalue %163[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %165 = llvm.extractvalue %164[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %166 = llvm.call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%134, %165, %135, %22) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
    %167 = llvm.extractvalue %166[0] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    %168 = llvm.extractvalue %166[1] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    llvm.store %168, %136 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    %169 = llvm.load %137 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %169 : i1, ^bb1 [
      0: ^bb7(%138, %167, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149, %150, %151, %152, %153, %154, %155, %156, %157, %158, %159 : !llvm.ptr, i64, i252, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr),
      1: ^bb6(%160, %161, %162, %167, %arg1 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i128)
    ]
  ^bb6(%170: !llvm.ptr, %171: !llvm.ptr, %172: !llvm.ptr, %173: i64, %174: i128):  // 5 preds: ^bb3, ^bb5, ^bb7, ^bb8, ^bb9
    %175 = llvm.load %170 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %176 = llvm.extractvalue %175[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %177 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %178 = llvm.insertvalue %14, %177[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %179 = llvm.insertvalue %176, %178[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %179, %171 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    %180 = llvm.load %172 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    %181 = llvm.mlir.undef : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %182 = llvm.insertvalue %173, %181[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %183 = llvm.insertvalue %174, %182[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %184 = llvm.insertvalue %180, %183[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.return %184 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
  ^bb7(%185: !llvm.ptr, %186: i64, %187: i252, %188: i32, %189: !llvm.ptr, %190: !llvm.ptr, %191: !llvm.ptr, %192: !llvm.ptr, %193: !llvm.ptr, %194: !llvm.ptr, %195: !llvm.ptr, %196: !llvm.ptr, %197: !llvm.ptr, %198: !llvm.ptr, %199: !llvm.ptr, %200: !llvm.ptr, %201: !llvm.ptr, %202: !llvm.ptr, %203: !llvm.ptr, %204: !llvm.ptr, %205: !llvm.ptr, %206: !llvm.ptr, %207: !llvm.ptr):  // pred: ^bb5
    %208 = llvm.load %185 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %209 = llvm.extractvalue %208[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %210 = llvm.extractvalue %209[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %211 = llvm.call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%186, %210, %187, %188) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
    %212 = llvm.extractvalue %211[0] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    %213 = llvm.extractvalue %211[1] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    llvm.store %213, %189 {alignment = 8 : i64} : !llvm.struct<(i64, array<52 x i8>)>, !llvm.ptr
    %214 = llvm.load %190 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %214 : i1, ^bb1 [
      0: ^bb8(%191, %192, %193, %212, %194, %195, %196, %197, %198, %199, %200, %201, %202, %203, %204, %212 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64),
      1: ^bb6(%205, %206, %207, %212, %arg1 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i128)
    ]
  ^bb8(%215: !llvm.ptr, %216: !llvm.ptr, %217: !llvm.ptr, %218: i64, %219: !llvm.ptr, %220: !llvm.ptr, %221: !llvm.ptr, %222: !llvm.ptr, %223: !llvm.ptr, %224: !llvm.ptr, %225: !llvm.ptr, %226: !llvm.ptr, %227: !llvm.ptr, %228: !llvm.ptr, %229: !llvm.ptr, %230: i64):  // pred: ^bb7
    %231 = llvm.load %215 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %232 = llvm.extractvalue %231[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %233 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %234 = llvm.insertvalue %24, %233[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.call @"core::result::ResultTraitImpl::<(), core::fmt::Error>::unwrap::<core::fmt::ErrorDrop>(f1)"(%216, %234) : (!llvm.ptr, !llvm.struct<(i1, array<0 x i8>)>) -> ()
    %235 = llvm.load %217 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %235 : i1, ^bb1 [
      0: ^bb9(%232, %218, %219, %220, %221, %222, %223, %224, %225, %226 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr),
      1: ^bb6(%227, %228, %229, %230, %arg1 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i128)
    ]
  ^bb9(%236: !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>, %237: i64, %238: !llvm.ptr, %239: !llvm.ptr, %240: !llvm.ptr, %241: !llvm.ptr, %242: !llvm.ptr, %243: !llvm.ptr, %244: !llvm.ptr, %245: !llvm.ptr):  // pred: ^bb8
    %246 = llvm.mlir.null : !llvm.ptr<i252>
    %247 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %248 = llvm.insertvalue %246, %247[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %249 = llvm.insertvalue %18, %248[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %250 = llvm.insertvalue %18, %249[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %251 = llvm.bitcast %246 : !llvm.ptr<i252> to !llvm.ptr
    %252 = llvm.call @realloc(%251, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %253 = llvm.bitcast %252 : !llvm.ptr to !llvm.ptr<i252>
    %254 = llvm.insertvalue %253, %250[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %255 = llvm.insertvalue %16, %254[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %256 = llvm.getelementptr %253[0] : (!llvm.ptr<i252>) -> !llvm.ptr, i252
    llvm.store %17, %256 : i252, !llvm.ptr
    %257 = llvm.insertvalue %15, %255[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %258 = llvm.extractvalue %236[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %259 = llvm.call @"core::byte_array::ByteArraySerde::serialize(f0)"(%237, %arg1, %258, %257) : (i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %260 = llvm.extractvalue %259[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %261 = llvm.extractvalue %259[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %262 = llvm.extractvalue %259[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.store %262, %238 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %263 = llvm.load %239 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %263 : i1, ^bb1 [
      0: ^bb10(%240, %241, %242, %260, %261 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i128),
      1: ^bb6(%243, %244, %245, %260, %261 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i128)
    ]
  ^bb10(%264: !llvm.ptr, %265: !llvm.ptr, %266: !llvm.ptr, %267: i64, %268: i128):  // pred: ^bb9
    %269 = llvm.load %264 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %270 = llvm.extractvalue %269[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    %271 = llvm.extractvalue %270[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %272 = llvm.mlir.undef : !llvm.struct<()>
    %273 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %274 = llvm.insertvalue %272, %273[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %275 = llvm.insertvalue %271, %274[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %276 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %277 = llvm.insertvalue %14, %276[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %278 = llvm.insertvalue %275, %277[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %278, %265 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    %279 = llvm.load %266 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    %280 = llvm.mlir.undef : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %281 = llvm.insertvalue %267, %280[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %282 = llvm.insertvalue %268, %281[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %283 = llvm.insertvalue %279, %282[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.return %283 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
  ^bb11:  // pred: ^bb4
    %284 = llvm.call @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<0, 0>>(f17)"(%130, %8) : (i64, i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    %285 = llvm.extractvalue %284[0] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %286 = llvm.extractvalue %284[1] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %287 = llvm.extractvalue %286[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.switch %287 : i1, ^bb1 [
      0: ^bb3(%285, %23, %50, %50, %50, %21, %52, %52, %52, %7, %16, %54, %54, %54, %56, %56, %58, %58, %58, %60, %60, %58, %59, %59, %56, %57, %57, %54, %55, %55, %52, %53, %53, %50, %51, %51 : i64, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr),
      1: ^bb12
    ]
  ^bb12:  // pred: ^bb11
    %288 = llvm.call @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<-1, -1>>(f16)"(%285, %6) : (i64, i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    %289 = llvm.extractvalue %288[0] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %290 = llvm.extractvalue %288[1] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %291 = llvm.extractvalue %290[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.switch %291 : i1, ^bb1 [
      0: ^bb3(%289, %23, %39, %39, %39, %5, %41, %41, %41, %4, %3, %43, %43, %43, %45, %45, %47, %47, %47, %49, %49, %47, %48, %48, %45, %46, %46, %43, %44, %44, %41, %42, %42, %39, %40, %40 : i64, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr),
      1: ^bb13
    ]
  ^bb13:  // pred: ^bb12
    %292 = llvm.call @"program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<-1, -1>>(f14)"(%289, %8) : (i64, i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    %293 = llvm.extractvalue %292[0] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %294 = llvm.extractvalue %292[1] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %295 = llvm.extractvalue %294[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.switch %295 : i1, ^bb1 [
      0: ^bb3(%293, %12, %28, %28, %28, %2, %30, %30, %30, %1, %19, %32, %32, %32, %34, %34, %36, %36, %36, %38, %38, %36, %37, %37, %34, %35, %35, %32, %33, %33, %30, %31, %31, %28, %29, %29 : i64, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, !llvm.ptr, !llvm.ptr, !llvm.ptr, i252, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr),
      1: ^bb14
    ]
  ^bb14:  // pred: ^bb13
    %296 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %297 = llvm.insertvalue %24, %296[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %297, %27 {alignment = 8 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    %298 = llvm.load %27 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    %299 = llvm.mlir.undef : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %300 = llvm.insertvalue %293, %299[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %301 = llvm.insertvalue %arg1, %300[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %302 = llvm.insertvalue %298, %301[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.return %302 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
  ^bb15:  // pred: ^bb1
    %303 = llvm.mlir.addressof @assert_msg_0 : !llvm.ptr
    llvm.call @puts(%303) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_program::program::special_casts::test_felt252_downcasts(f20)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i128) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"program::program::special_casts::test_felt252_downcasts(f20)"(%arg1, %arg2) : (i64, i128) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @__debug__print_felt252(i64, i64, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @__debug__print_i1(i1) attributes {sym_visibility = "private"}
  llvm.func @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<0, 0>>(f17)"(%arg0: i64, %arg1: i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(0 : i252) : i252
    %2 = llvm.mlir.constant(1 : i252) : i252
    %3 = llvm.mlir.constant(64 : i252) : i252
    %4 = llvm.mlir.constant(-1 : i252) : i252
    %5 = llvm.mlir.constant(-3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252) : i252
    %6 = llvm.mlir.constant(1809251394333065606848661391547535052811553607665798349986546028067936010240 : i252) : i252
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.add %arg0, %7  : i64
    %9 = llvm.icmp "ugt" %arg1, %6 : i252
    llvm.cond_br %9, ^bb1, ^bb2(%arg1 : i252)
  ^bb1:  // pred: ^bb0
    %10 = llvm.sub %5, %arg1  : i252
    %11 = llvm.mul %10, %4  : i252
    llvm.br ^bb2(%11 : i252)
  ^bb2(%12: i252):  // 2 preds: ^bb0, ^bb1
    %13 = llvm.trunc %12 : i252 to i64
    %14 = llvm.lshr %12, %3  : i252
    %15 = llvm.trunc %14 : i252 to i64
    %16 = llvm.lshr %14, %3  : i252
    %17 = llvm.trunc %16 : i252 to i64
    %18 = llvm.lshr %16, %3  : i252
    %19 = llvm.trunc %18 : i252 to i64
    llvm.call @__debug__print_felt252(%13, %15, %17, %19) : (i64, i64, i64, i64) -> ()
    %20 = llvm.icmp "sle" %12, %2 : i252
    %21 = llvm.icmp "sge" %12, %1 : i252
    %22 = llvm.and %20, %21  : i1
    llvm.call @__debug__print_i1(%22) : (i1) -> ()
    %23 = llvm.xor %22, %0  : i1
    %24 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %25 = llvm.insertvalue %23, %24[0] : !llvm.struct<(i1, array<0 x i8>)> 
    %26 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    %27 = llvm.insertvalue %8, %26[0] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %28 = llvm.insertvalue %25, %27[1] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    llvm.return %28 : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
  }
  llvm.func @"_mlir_ciface_program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<0, 0>>(f17)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<0, 0>>(f17)"(%arg1, %arg2) : (i64, i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::fmt::FormatterDefault::default(f12)"() -> !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::byte_array::ByteArrayDefault::default(f13)"() : () -> !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %1 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)> 
    llvm.return %2 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
  }
  llvm.func @"_mlir_ciface_core::fmt::FormatterDefault::default(f12)"(%arg0: !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::fmt::FormatterDefault::default(f12)"() : () -> !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>
    llvm.store %0, %arg0 : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::byte_array::ByteArrayImpl::append_word(f3)"(%arg0: i64, %arg1: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %arg2: i252, %arg3: i32) -> !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(256 : i64) : i64
    %1 = llvm.mlir.constant(340282366920938463463374607431768211456 : i512) : i512
    %2 = llvm.mlir.constant(0 : i252) : i252
    %3 = llvm.mlir.constant(true) : i1
    %4 = llvm.mlir.constant(29721761890975875353235833581453094220424382983267374 : i252) : i252
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(32 : i64) : i64
    %7 = llvm.mlir.constant(8 : i32) : i32
    %8 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %9 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %10 = llvm.mlir.constant(16 : i32) : i32
    %11 = llvm.mlir.constant(31 : i32) : i32
    %12 = llvm.mlir.constant(false) : i1
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = llvm.mlir.constant(1 : i64) : i64
    %15 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %16 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %17 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %18 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %19 = llvm.alloca %14 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %20 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %21 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %22 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %23 = llvm.alloca %14 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %24 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %25 = llvm.alloca %14 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %26 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %27 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %28 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %29 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %30 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %31 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %32 = llvm.alloca %14 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %33 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %34 = llvm.alloca %14 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %35 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %36 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %37 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %38 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %39 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %40 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %41 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %42 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %43 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %44 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %45 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %46 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %47 = llvm.alloca %14 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %48 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %49 = llvm.alloca %14 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %50 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %51 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %52 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %53 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %54 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %55 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %56 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %57 = llvm.alloca %14 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %58 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %59 = llvm.alloca %14 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %60 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %61 = llvm.alloca %14 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %62 = llvm.alloca %14 x !llvm.struct<(i64, array<52 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %63 = llvm.icmp "eq" %arg3, %13 : i32
    llvm.cond_br %63, ^bb56, ^bb1(%arg1 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb1(%64: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // pred: ^bb0
    %65 = llvm.extractvalue %64[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %66 = llvm.extractvalue %64[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %67 = llvm.extractvalue %64[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %68 = llvm.call @"core::integer::U32Add::add(f4)"(%arg0, %67, %arg3) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %69 = llvm.extractvalue %68[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %70 = llvm.extractvalue %68[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.store %70, %16 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %71 = llvm.load %16 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %71 : i1, ^bb2 [
      0: ^bb4,
      1: ^bb5(%16, %17, %17, %69 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb2:  // 22 preds: ^bb1, ^bb7, ^bb10, ^bb11, ^bb12, ^bb13, ^bb15, ^bb16, ^bb22, ^bb23, ^bb24, ^bb25, ^bb26, ^bb27, ^bb28, ^bb34, ^bb35, ^bb36, ^bb43, ^bb44, ^bb51, ^bb52
    llvm.cond_br %12, ^bb3, ^bb57
  ^bb3:  // pred: ^bb2
    llvm.unreachable
  ^bb4:  // pred: ^bb1
    %72 = llvm.load %16 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %73 = llvm.extractvalue %72[1] : !llvm.struct<(i1, struct<(i32)>)> 
    %74 = llvm.extractvalue %73[0] : !llvm.struct<(i32)> 
    %75 = llvm.add %69, %14  : i64
    %76 = "llvm.intr.usub.with.overflow"(%74, %11) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %77 = llvm.extractvalue %76[1] : !llvm.struct<(i32, i1)> 
    llvm.cond_br %77, ^bb50(%67 : i32), ^bb6(%74 : i32)
  ^bb5(%78: !llvm.ptr, %79: !llvm.ptr, %80: !llvm.ptr, %81: i64):  // 16 preds: ^bb1, ^bb7, ^bb10, ^bb11, ^bb13, ^bb15, ^bb22, ^bb24, ^bb25, ^bb26, ^bb27, ^bb34, ^bb35, ^bb43, ^bb51, ^bb52
    %82 = llvm.load %78 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %83 = llvm.extractvalue %82[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %84 = llvm.extractvalue %65[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %85 = llvm.bitcast %84 : !llvm.ptr<i248> to !llvm.ptr
    llvm.call @free(%85) : (!llvm.ptr) -> ()
    %86 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %87 = llvm.insertvalue %3, %86[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %88 = llvm.insertvalue %83, %87[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %88, %79 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    %89 = llvm.load %80 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    %90 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
    %91 = llvm.insertvalue %81, %90[0] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    %92 = llvm.insertvalue %89, %91[1] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    llvm.return %92 : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
  ^bb6(%93: i32):  // pred: ^bb4
    %94 = llvm.icmp "eq" %93, %11 : i32
    llvm.cond_br %94, ^bb43(%75 : i64), ^bb7(%75 : i64)
  ^bb7(%95: i64):  // pred: ^bb6
    %96 = llvm.call @"core::integer::U32Sub::sub(f8)"(%95, %93, %11) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %97 = llvm.extractvalue %96[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %98 = llvm.extractvalue %96[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.store %98, %28 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %99 = llvm.load %28 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %99 : i1, ^bb2 [
      0: ^bb8,
      1: ^bb5(%28, %29, %29, %97 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb8:  // pred: ^bb7
    %100 = llvm.load %28 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %101 = llvm.extractvalue %100[1] : !llvm.struct<(i1, struct<(i32)>)> 
    %102 = llvm.extractvalue %101[0] : !llvm.struct<(i32)> 
    %103 = llvm.icmp "eq" %102, %10 : i32
    llvm.cond_br %103, ^bb34(%97 : i64), ^bb9(%102 : i32)
  ^bb9(%104: i32):  // pred: ^bb8
    %105 = llvm.add %97, %14  : i64
    %106 = "llvm.intr.usub.with.overflow"(%104, %10) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %107 = llvm.extractvalue %106[1] : !llvm.struct<(i32, i1)> 
    llvm.cond_br %107, ^bb22(%105 : i64), ^bb10(%105 : i64)
  ^bb10(%108: i64):  // pred: ^bb9
    %109 = llvm.call @"core::integer::u256_from_felt252(f10)"(%108, %arg2) : (i64, i252) -> !llvm.struct<(i64, struct<(i128, i128)>)>
    %110 = llvm.extractvalue %109[0] : !llvm.struct<(i64, struct<(i128, i128)>)> 
    %111 = llvm.extractvalue %109[1] : !llvm.struct<(i64, struct<(i128, i128)>)> 
    %112 = llvm.extractvalue %111[0] : !llvm.struct<(i128, i128)> 
    %113 = llvm.extractvalue %111[1] : !llvm.struct<(i128, i128)> 
    %114 = llvm.call @"core::integer::U32Sub::sub(f8)"(%110, %104, %10) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %115 = llvm.extractvalue %114[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %116 = llvm.extractvalue %114[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.store %116, %51 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %117 = llvm.load %51 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %117 : i1, ^bb2 [
      0: ^bb11,
      1: ^bb5(%51, %52, %52, %115 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb11:  // pred: ^bb10
    %118 = llvm.load %51 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %119 = llvm.extractvalue %118[1] : !llvm.struct<(i1, struct<(i32)>)> 
    %120 = llvm.extractvalue %119[0] : !llvm.struct<(i32)> 
    llvm.call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%53, %120) : (!llvm.ptr, i32) -> ()
    %121 = llvm.load %53 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %121 : i1, ^bb2 [
      0: ^bb12,
      1: ^bb5(%53, %54, %54, %115 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb12:  // pred: ^bb11
    %122 = llvm.load %53 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i128)>)>
    %123 = llvm.extractvalue %122[1] : !llvm.struct<(i1, struct<(i128)>)> 
    %124 = llvm.extractvalue %123[0] : !llvm.struct<(i128)> 
    llvm.call @"core::integer::u128_try_as_non_zero(f11)"(%55, %124) : (!llvm.ptr, i128) -> ()
    %125 = llvm.load %55 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %125 : i1, ^bb2 [
      0: ^bb13,
      1: ^bb14(%56, %56, %115 : !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb13:  // pred: ^bb12
    %126 = llvm.load %55 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i128)>
    %127 = llvm.extractvalue %126[1] : !llvm.struct<(i1, i128)> 
    %128 = llvm.add %115, %14  : i64
    %129 = llvm.udiv %113, %127  : i128
    %130 = llvm.urem %113, %127  : i128
    %131 = llvm.call @"core::integer::U32Sub::sub(f8)"(%128, %11, %67) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %132 = llvm.extractvalue %131[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %133 = llvm.extractvalue %131[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.store %133, %57 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %134 = llvm.load %57 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %134 : i1, ^bb2 [
      0: ^bb15,
      1: ^bb5(%57, %58, %58, %132 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb14(%135: !llvm.ptr, %136: !llvm.ptr, %137: i64):  // 6 preds: ^bb12, ^bb16, ^bb23, ^bb28, ^bb36, ^bb44
    %138 = llvm.extractvalue %65[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %139 = llvm.bitcast %138 : !llvm.ptr<i248> to !llvm.ptr
    llvm.call @free(%139) : (!llvm.ptr) -> ()
    %140 = llvm.mlir.null : !llvm.ptr<i252>
    %141 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %142 = llvm.insertvalue %140, %141[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %143 = llvm.insertvalue %13, %142[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %144 = llvm.insertvalue %13, %143[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %145 = llvm.bitcast %140 : !llvm.ptr<i252> to !llvm.ptr
    %146 = llvm.call @realloc(%145, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %147 = llvm.bitcast %146 : !llvm.ptr to !llvm.ptr<i252>
    %148 = llvm.insertvalue %147, %144[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %149 = llvm.insertvalue %7, %148[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %150 = llvm.getelementptr %147[0] : (!llvm.ptr<i252>) -> !llvm.ptr, i252
    llvm.store %4, %150 : i252, !llvm.ptr
    %151 = llvm.insertvalue %5, %149[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %152 = llvm.mlir.undef : !llvm.struct<()>
    %153 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %154 = llvm.insertvalue %152, %153[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %155 = llvm.insertvalue %151, %154[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %156 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %157 = llvm.insertvalue %3, %156[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %158 = llvm.insertvalue %155, %157[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %158, %135 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    %159 = llvm.load %136 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    %160 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
    %161 = llvm.insertvalue %137, %160[0] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    %162 = llvm.insertvalue %159, %161[1] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    llvm.return %162 : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
  ^bb15:  // pred: ^bb13
    %163 = llvm.load %57 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %164 = llvm.extractvalue %163[1] : !llvm.struct<(i1, struct<(i32)>)> 
    %165 = llvm.extractvalue %164[0] : !llvm.struct<(i32)> 
    %166 = llvm.call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%132, %165) : (i64, i32) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %167 = llvm.extractvalue %166[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %168 = llvm.extractvalue %166[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.store %168, %59 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    %169 = llvm.load %59 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %169 : i1, ^bb2 [
      0: ^bb16,
      1: ^bb5(%59, %60, %60, %167 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb16:  // pred: ^bb15
    %170 = llvm.load %59 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i252)>)>
    %171 = llvm.extractvalue %170[1] : !llvm.struct<(i1, struct<(i252)>)> 
    %172 = llvm.extractvalue %171[0] : !llvm.struct<(i252)> 
    %173 = llvm.zext %66 : i252 to i512
    %174 = llvm.zext %172 : i252 to i512
    %175 = llvm.mul %173, %174  : i512
    %176 = llvm.urem %175, %9  : i512
    %177 = llvm.icmp "uge" %175, %9 : i512
    %178 = llvm.select %177, %176, %175 : i1, i512
    %179 = llvm.trunc %178 : i512 to i252
    %180 = llvm.zext %129 : i128 to i256
    %181 = llvm.zext %179 : i252 to i256
    %182 = llvm.add %180, %181  : i256
    %183 = llvm.sub %182, %8  : i256
    %184 = llvm.icmp "uge" %182, %8 : i256
    %185 = llvm.select %184, %183, %182 : i1, i256
    %186 = llvm.trunc %185 : i256 to i252
    %187 = llvm.call @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%167, %186) : (i64, i252) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %188 = llvm.extractvalue %187[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %189 = llvm.extractvalue %187[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.store %189, %61 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    %190 = llvm.load %61 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %190 : i1, ^bb2 [
      0: ^bb17,
      1: ^bb14(%62, %62, %188 : !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb17:  // pred: ^bb16
    %191 = llvm.load %61 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i248)>
    %192 = llvm.extractvalue %191[1] : !llvm.struct<(i1, i248)> 
    %193 = llvm.extractvalue %65[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %194 = llvm.extractvalue %65[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %195 = llvm.extractvalue %65[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %196 = llvm.icmp "uge" %194, %195 : i32
    llvm.cond_br %196, ^bb18, ^bb19
  ^bb18:  // pred: ^bb17
    %197 = llvm.add %195, %195  : i32
    %198 = llvm.intr.umax(%197, %7)  : (i32, i32) -> i32
    %199 = llvm.zext %198 : i32 to i64
    %200 = llvm.mul %199, %6  : i64
    %201 = llvm.bitcast %193 : !llvm.ptr<i248> to !llvm.ptr
    %202 = llvm.call @realloc(%201, %200) : (!llvm.ptr, i64) -> !llvm.ptr
    %203 = llvm.bitcast %202 : !llvm.ptr to !llvm.ptr<i248>
    %204 = llvm.insertvalue %203, %65[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %205 = llvm.insertvalue %198, %204[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    llvm.br ^bb20(%205, %203 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>)
  ^bb19:  // pred: ^bb17
    llvm.br ^bb20(%65, %193 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>)
  ^bb20(%206: !llvm.struct<(ptr<i248>, i32, i32)>, %207: !llvm.ptr<i248>):  // 2 preds: ^bb18, ^bb19
    llvm.br ^bb21
  ^bb21:  // pred: ^bb20
    %208 = llvm.getelementptr %207[%194] : (!llvm.ptr<i248>, i32) -> !llvm.ptr, i248
    llvm.store %192, %208 : i248, !llvm.ptr
    %209 = llvm.add %194, %5  : i32
    %210 = llvm.insertvalue %209, %206[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %211 = llvm.zext %130 : i128 to i512
    %212 = llvm.mul %211, %1  : i512
    %213 = llvm.urem %212, %9  : i512
    %214 = llvm.icmp "uge" %212, %9 : i512
    %215 = llvm.select %214, %213, %212 : i1, i512
    %216 = llvm.trunc %215 : i512 to i252
    %217 = llvm.zext %216 : i252 to i256
    %218 = llvm.zext %112 : i128 to i256
    %219 = llvm.add %217, %218  : i256
    %220 = llvm.sub %219, %8  : i256
    %221 = llvm.icmp "uge" %219, %8 : i256
    %222 = llvm.select %221, %220, %219 : i1, i256
    %223 = llvm.trunc %222 : i256 to i252
    %224 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %225 = llvm.insertvalue %210, %224[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %226 = llvm.insertvalue %223, %225[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %227 = llvm.insertvalue %67, %226[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    llvm.br ^bb42(%104, %188, %227 : i32, i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb22(%228: i64):  // pred: ^bb9
    %229 = llvm.call @"core::integer::u256_from_felt252(f10)"(%228, %arg2) : (i64, i252) -> !llvm.struct<(i64, struct<(i128, i128)>)>
    %230 = llvm.extractvalue %229[0] : !llvm.struct<(i64, struct<(i128, i128)>)> 
    %231 = llvm.extractvalue %229[1] : !llvm.struct<(i64, struct<(i128, i128)>)> 
    %232 = llvm.extractvalue %231[0] : !llvm.struct<(i128, i128)> 
    %233 = llvm.extractvalue %231[1] : !llvm.struct<(i128, i128)> 
    llvm.call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%37, %104) : (!llvm.ptr, i32) -> ()
    %234 = llvm.load %37 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %234 : i1, ^bb2 [
      0: ^bb23,
      1: ^bb5(%37, %38, %38, %230 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb23:  // pred: ^bb22
    %235 = llvm.load %37 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i128)>)>
    %236 = llvm.extractvalue %235[1] : !llvm.struct<(i1, struct<(i128)>)> 
    %237 = llvm.extractvalue %236[0] : !llvm.struct<(i128)> 
    llvm.call @"core::integer::u128_try_as_non_zero(f11)"(%39, %237) : (!llvm.ptr, i128) -> ()
    %238 = llvm.load %39 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %238 : i1, ^bb2 [
      0: ^bb24,
      1: ^bb14(%40, %40, %230 : !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb24:  // pred: ^bb23
    %239 = llvm.load %39 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i128)>
    %240 = llvm.extractvalue %239[1] : !llvm.struct<(i1, i128)> 
    %241 = llvm.add %230, %14  : i64
    %242 = llvm.udiv %232, %240  : i128
    %243 = llvm.urem %232, %240  : i128
    %244 = llvm.call @"core::integer::U32Sub::sub(f8)"(%241, %10, %104) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %245 = llvm.extractvalue %244[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %246 = llvm.extractvalue %244[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.store %246, %41 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %247 = llvm.load %41 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %247 : i1, ^bb2 [
      0: ^bb25,
      1: ^bb5(%41, %42, %42, %245 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb25:  // pred: ^bb24
    %248 = llvm.load %41 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %249 = llvm.extractvalue %248[1] : !llvm.struct<(i1, struct<(i32)>)> 
    %250 = llvm.extractvalue %249[0] : !llvm.struct<(i32)> 
    llvm.call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%43, %250) : (!llvm.ptr, i32) -> ()
    %251 = llvm.load %43 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %251 : i1, ^bb2 [
      0: ^bb26,
      1: ^bb5(%43, %44, %44, %245 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb26:  // pred: ^bb25
    %252 = llvm.load %43 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i128)>)>
    %253 = llvm.extractvalue %252[1] : !llvm.struct<(i1, struct<(i128)>)> 
    %254 = llvm.extractvalue %253[0] : !llvm.struct<(i128)> 
    %255 = llvm.zext %243 : i128 to i252
    %256 = llvm.call @"core::integer::U32Sub::sub(f8)"(%245, %11, %67) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %257 = llvm.extractvalue %256[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %258 = llvm.extractvalue %256[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.store %258, %45 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %259 = llvm.load %45 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %259 : i1, ^bb2 [
      0: ^bb27,
      1: ^bb5(%45, %46, %46, %257 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb27:  // pred: ^bb26
    %260 = llvm.load %45 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %261 = llvm.extractvalue %260[1] : !llvm.struct<(i1, struct<(i32)>)> 
    %262 = llvm.extractvalue %261[0] : !llvm.struct<(i32)> 
    %263 = llvm.call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%257, %262) : (i64, i32) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %264 = llvm.extractvalue %263[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %265 = llvm.extractvalue %263[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.store %265, %47 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    %266 = llvm.load %47 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %266 : i1, ^bb2 [
      0: ^bb28,
      1: ^bb5(%47, %48, %48, %264 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb28:  // pred: ^bb27
    %267 = llvm.load %47 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i252)>)>
    %268 = llvm.extractvalue %267[1] : !llvm.struct<(i1, struct<(i252)>)> 
    %269 = llvm.zext %233 : i128 to i512
    %270 = llvm.zext %254 : i128 to i512
    %271 = llvm.mul %269, %270  : i512
    %272 = llvm.urem %271, %9  : i512
    %273 = llvm.icmp "uge" %271, %9 : i512
    %274 = llvm.select %273, %272, %271 : i1, i512
    %275 = llvm.trunc %274 : i512 to i252
    %276 = llvm.zext %275 : i252 to i256
    %277 = llvm.zext %242 : i128 to i256
    %278 = llvm.add %276, %277  : i256
    %279 = llvm.sub %278, %8  : i256
    %280 = llvm.icmp "uge" %278, %8 : i256
    %281 = llvm.select %280, %279, %278 : i1, i256
    %282 = llvm.trunc %281 : i256 to i252
    %283 = llvm.extractvalue %268[0] : !llvm.struct<(i252)> 
    %284 = llvm.zext %66 : i252 to i512
    %285 = llvm.zext %283 : i252 to i512
    %286 = llvm.mul %284, %285  : i512
    %287 = llvm.urem %286, %9  : i512
    %288 = llvm.icmp "uge" %286, %9 : i512
    %289 = llvm.select %288, %287, %286 : i1, i512
    %290 = llvm.trunc %289 : i512 to i252
    %291 = llvm.zext %282 : i252 to i256
    %292 = llvm.zext %290 : i252 to i256
    %293 = llvm.add %291, %292  : i256
    %294 = llvm.sub %293, %8  : i256
    %295 = llvm.icmp "uge" %293, %8 : i256
    %296 = llvm.select %295, %294, %293 : i1, i256
    %297 = llvm.trunc %296 : i256 to i252
    %298 = llvm.call @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%264, %297) : (i64, i252) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %299 = llvm.extractvalue %298[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %300 = llvm.extractvalue %298[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.store %300, %49 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    %301 = llvm.load %49 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %301 : i1, ^bb2 [
      0: ^bb29,
      1: ^bb14(%50, %50, %299 : !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb29:  // pred: ^bb28
    %302 = llvm.load %49 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i248)>
    %303 = llvm.extractvalue %302[1] : !llvm.struct<(i1, i248)> 
    %304 = llvm.extractvalue %65[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %305 = llvm.extractvalue %65[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %306 = llvm.extractvalue %65[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %307 = llvm.icmp "uge" %305, %306 : i32
    llvm.cond_br %307, ^bb30, ^bb31
  ^bb30:  // pred: ^bb29
    %308 = llvm.add %306, %306  : i32
    %309 = llvm.intr.umax(%308, %7)  : (i32, i32) -> i32
    %310 = llvm.zext %309 : i32 to i64
    %311 = llvm.mul %310, %6  : i64
    %312 = llvm.bitcast %304 : !llvm.ptr<i248> to !llvm.ptr
    %313 = llvm.call @realloc(%312, %311) : (!llvm.ptr, i64) -> !llvm.ptr
    %314 = llvm.bitcast %313 : !llvm.ptr to !llvm.ptr<i248>
    %315 = llvm.insertvalue %314, %65[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %316 = llvm.insertvalue %309, %315[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    llvm.br ^bb32(%316, %314 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>)
  ^bb31:  // pred: ^bb29
    llvm.br ^bb32(%65, %304 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>)
  ^bb32(%317: !llvm.struct<(ptr<i248>, i32, i32)>, %318: !llvm.ptr<i248>):  // 2 preds: ^bb30, ^bb31
    llvm.br ^bb33
  ^bb33:  // pred: ^bb32
    %319 = llvm.getelementptr %318[%305] : (!llvm.ptr<i248>, i32) -> !llvm.ptr, i248
    llvm.store %303, %319 : i248, !llvm.ptr
    %320 = llvm.add %305, %5  : i32
    %321 = llvm.insertvalue %320, %317[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %322 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %323 = llvm.insertvalue %321, %322[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %324 = llvm.insertvalue %255, %323[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %325 = llvm.insertvalue %67, %324[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    llvm.br ^bb42(%104, %299, %325 : i32, i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb34(%326: i64):  // pred: ^bb8
    %327 = llvm.call @"core::integer::u256_from_felt252(f10)"(%326, %arg2) : (i64, i252) -> !llvm.struct<(i64, struct<(i128, i128)>)>
    %328 = llvm.extractvalue %327[0] : !llvm.struct<(i64, struct<(i128, i128)>)> 
    %329 = llvm.extractvalue %327[1] : !llvm.struct<(i64, struct<(i128, i128)>)> 
    %330 = llvm.extractvalue %329[0] : !llvm.struct<(i128, i128)> 
    %331 = llvm.extractvalue %329[1] : !llvm.struct<(i128, i128)> 
    %332 = llvm.zext %330 : i128 to i252
    %333 = llvm.call @"core::integer::U32Sub::sub(f8)"(%328, %11, %67) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %334 = llvm.extractvalue %333[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %335 = llvm.extractvalue %333[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.store %335, %30 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %336 = llvm.load %30 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %336 : i1, ^bb2 [
      0: ^bb35,
      1: ^bb5(%30, %31, %31, %334 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb35:  // pred: ^bb34
    %337 = llvm.load %30 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %338 = llvm.extractvalue %337[1] : !llvm.struct<(i1, struct<(i32)>)> 
    %339 = llvm.extractvalue %338[0] : !llvm.struct<(i32)> 
    %340 = llvm.call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%334, %339) : (i64, i32) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %341 = llvm.extractvalue %340[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %342 = llvm.extractvalue %340[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.store %342, %32 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    %343 = llvm.load %32 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %343 : i1, ^bb2 [
      0: ^bb36,
      1: ^bb5(%32, %33, %33, %341 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb36:  // pred: ^bb35
    %344 = llvm.load %32 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i252)>)>
    %345 = llvm.extractvalue %344[1] : !llvm.struct<(i1, struct<(i252)>)> 
    %346 = llvm.extractvalue %345[0] : !llvm.struct<(i252)> 
    %347 = llvm.zext %66 : i252 to i512
    %348 = llvm.zext %346 : i252 to i512
    %349 = llvm.mul %347, %348  : i512
    %350 = llvm.urem %349, %9  : i512
    %351 = llvm.icmp "uge" %349, %9 : i512
    %352 = llvm.select %351, %350, %349 : i1, i512
    %353 = llvm.trunc %352 : i512 to i252
    %354 = llvm.zext %331 : i128 to i256
    %355 = llvm.zext %353 : i252 to i256
    %356 = llvm.add %354, %355  : i256
    %357 = llvm.sub %356, %8  : i256
    %358 = llvm.icmp "uge" %356, %8 : i256
    %359 = llvm.select %358, %357, %356 : i1, i256
    %360 = llvm.trunc %359 : i256 to i252
    %361 = llvm.call @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%341, %360) : (i64, i252) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %362 = llvm.extractvalue %361[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %363 = llvm.extractvalue %361[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.store %363, %34 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    %364 = llvm.load %34 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %364 : i1, ^bb2 [
      0: ^bb37,
      1: ^bb14(%35, %35, %362 : !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb37:  // pred: ^bb36
    %365 = llvm.load %34 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i248)>
    %366 = llvm.extractvalue %365[1] : !llvm.struct<(i1, i248)> 
    %367 = llvm.extractvalue %65[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %368 = llvm.extractvalue %65[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %369 = llvm.extractvalue %65[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %370 = llvm.icmp "uge" %368, %369 : i32
    llvm.cond_br %370, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    %371 = llvm.add %369, %369  : i32
    %372 = llvm.intr.umax(%371, %7)  : (i32, i32) -> i32
    %373 = llvm.zext %372 : i32 to i64
    %374 = llvm.mul %373, %6  : i64
    %375 = llvm.bitcast %367 : !llvm.ptr<i248> to !llvm.ptr
    %376 = llvm.call @realloc(%375, %374) : (!llvm.ptr, i64) -> !llvm.ptr
    %377 = llvm.bitcast %376 : !llvm.ptr to !llvm.ptr<i248>
    %378 = llvm.insertvalue %377, %65[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %379 = llvm.insertvalue %372, %378[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    llvm.br ^bb40(%379, %377 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>)
  ^bb39:  // pred: ^bb37
    llvm.br ^bb40(%65, %367 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>)
  ^bb40(%380: !llvm.struct<(ptr<i248>, i32, i32)>, %381: !llvm.ptr<i248>):  // 2 preds: ^bb38, ^bb39
    llvm.br ^bb41
  ^bb41:  // pred: ^bb40
    %382 = llvm.getelementptr %381[%368] : (!llvm.ptr<i248>, i32) -> !llvm.ptr, i248
    llvm.store %366, %382 : i248, !llvm.ptr
    %383 = llvm.add %368, %5  : i32
    %384 = llvm.insertvalue %383, %380[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %385 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %386 = llvm.insertvalue %384, %385[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %387 = llvm.insertvalue %332, %386[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %388 = llvm.insertvalue %67, %387[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    llvm.br ^bb42(%102, %362, %388 : i32, i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb42(%389: i32, %390: i64, %391: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // 3 preds: ^bb21, ^bb33, ^bb41
    %392 = llvm.extractvalue %391[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %393 = llvm.extractvalue %391[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %394 = llvm.mlir.undef : !llvm.struct<()>
    %395 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %396 = llvm.insertvalue %392, %395[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %397 = llvm.insertvalue %393, %396[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %398 = llvm.insertvalue %389, %397[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %399 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>
    %400 = llvm.insertvalue %398, %399[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %401 = llvm.insertvalue %394, %400[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %402 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %403 = llvm.insertvalue %12, %402[0] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %404 = llvm.insertvalue %401, %403[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    llvm.store %404, %36 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>, !llvm.ptr
    %405 = llvm.load %36 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    %406 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
    %407 = llvm.insertvalue %390, %406[0] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    %408 = llvm.insertvalue %405, %407[1] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    llvm.return %408 : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
  ^bb43(%409: i64):  // pred: ^bb6
    %410 = llvm.call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%409, %arg3) : (i64, i32) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %411 = llvm.extractvalue %410[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %412 = llvm.extractvalue %410[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.store %412, %23 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    %413 = llvm.load %23 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %413 : i1, ^bb2 [
      0: ^bb44,
      1: ^bb5(%23, %24, %24, %411 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb44:  // pred: ^bb43
    %414 = llvm.load %23 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i252)>)>
    %415 = llvm.extractvalue %414[1] : !llvm.struct<(i1, struct<(i252)>)> 
    %416 = llvm.extractvalue %415[0] : !llvm.struct<(i252)> 
    %417 = llvm.zext %66 : i252 to i512
    %418 = llvm.zext %416 : i252 to i512
    %419 = llvm.mul %417, %418  : i512
    %420 = llvm.urem %419, %9  : i512
    %421 = llvm.icmp "uge" %419, %9 : i512
    %422 = llvm.select %421, %420, %419 : i1, i512
    %423 = llvm.trunc %422 : i512 to i252
    %424 = llvm.zext %arg2 : i252 to i256
    %425 = llvm.zext %423 : i252 to i256
    %426 = llvm.add %424, %425  : i256
    %427 = llvm.sub %426, %8  : i256
    %428 = llvm.icmp "uge" %426, %8 : i256
    %429 = llvm.select %428, %427, %426 : i1, i256
    %430 = llvm.trunc %429 : i256 to i252
    %431 = llvm.call @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%411, %430) : (i64, i252) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %432 = llvm.extractvalue %431[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %433 = llvm.extractvalue %431[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.store %433, %25 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    %434 = llvm.load %25 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %434 : i1, ^bb2 [
      0: ^bb45,
      1: ^bb14(%26, %26, %432 : !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb45:  // pred: ^bb44
    %435 = llvm.load %25 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, i248)>
    %436 = llvm.extractvalue %435[1] : !llvm.struct<(i1, i248)> 
    %437 = llvm.extractvalue %65[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %438 = llvm.extractvalue %65[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %439 = llvm.extractvalue %65[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %440 = llvm.icmp "uge" %438, %439 : i32
    llvm.cond_br %440, ^bb46, ^bb47
  ^bb46:  // pred: ^bb45
    %441 = llvm.add %439, %439  : i32
    %442 = llvm.intr.umax(%441, %7)  : (i32, i32) -> i32
    %443 = llvm.zext %442 : i32 to i64
    %444 = llvm.mul %443, %6  : i64
    %445 = llvm.bitcast %437 : !llvm.ptr<i248> to !llvm.ptr
    %446 = llvm.call @realloc(%445, %444) : (!llvm.ptr, i64) -> !llvm.ptr
    %447 = llvm.bitcast %446 : !llvm.ptr to !llvm.ptr<i248>
    %448 = llvm.insertvalue %447, %65[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %449 = llvm.insertvalue %442, %448[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    llvm.br ^bb48(%449, %447 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>)
  ^bb47:  // pred: ^bb45
    llvm.br ^bb48(%65, %437 : !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr<i248>)
  ^bb48(%450: !llvm.struct<(ptr<i248>, i32, i32)>, %451: !llvm.ptr<i248>):  // 2 preds: ^bb46, ^bb47
    llvm.br ^bb49
  ^bb49:  // pred: ^bb48
    %452 = llvm.getelementptr %451[%438] : (!llvm.ptr<i248>, i32) -> !llvm.ptr, i248
    llvm.store %436, %452 : i248, !llvm.ptr
    %453 = llvm.add %438, %5  : i32
    %454 = llvm.insertvalue %453, %450[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %455 = llvm.mlir.undef : !llvm.struct<()>
    %456 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %457 = llvm.insertvalue %454, %456[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %458 = llvm.insertvalue %2, %457[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %459 = llvm.insertvalue %13, %458[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %460 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>
    %461 = llvm.insertvalue %459, %460[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %462 = llvm.insertvalue %455, %461[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %463 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %464 = llvm.insertvalue %12, %463[0] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %465 = llvm.insertvalue %462, %464[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    llvm.store %465, %27 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>, !llvm.ptr
    %466 = llvm.load %27 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    %467 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
    %468 = llvm.insertvalue %432, %467[0] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    %469 = llvm.insertvalue %466, %468[1] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    llvm.return %469 : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
  ^bb50(%470: i32):  // pred: ^bb4
    %471 = llvm.icmp "eq" %470, %13 : i32
    llvm.cond_br %471, ^bb54(%65, %arg2, %arg3 : !llvm.struct<(ptr<i248>, i32, i32)>, i252, i32), ^bb51(%75 : i64)
  ^bb51(%472: i64):  // pred: ^bb50
    %473 = llvm.call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%472, %arg3) : (i64, i32) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %474 = llvm.extractvalue %473[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %475 = llvm.extractvalue %473[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.store %475, %19 {alignment = 8 : i64} : !llvm.struct<(i64, array<32 x i8>)>, !llvm.ptr
    %476 = llvm.load %19 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %476 : i1, ^bb2 [
      0: ^bb52,
      1: ^bb5(%19, %20, %20, %474 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb52:  // pred: ^bb51
    %477 = llvm.load %19 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i252)>)>
    %478 = llvm.extractvalue %477[1] : !llvm.struct<(i1, struct<(i252)>)> 
    %479 = llvm.call @"core::integer::U32Add::add(f4)"(%474, %470, %arg3) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %480 = llvm.extractvalue %479[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %481 = llvm.extractvalue %479[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.store %481, %21 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %482 = llvm.load %21 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %482 : i1, ^bb2 [
      0: ^bb53,
      1: ^bb5(%21, %22, %22, %480 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb53:  // pred: ^bb52
    %483 = llvm.load %21 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %484 = llvm.extractvalue %483[1] : !llvm.struct<(i1, struct<(i32)>)> 
    %485 = llvm.extractvalue %478[0] : !llvm.struct<(i252)> 
    %486 = llvm.zext %66 : i252 to i512
    %487 = llvm.zext %485 : i252 to i512
    %488 = llvm.mul %486, %487  : i512
    %489 = llvm.urem %488, %9  : i512
    %490 = llvm.icmp "uge" %488, %9 : i512
    %491 = llvm.select %490, %489, %488 : i1, i512
    %492 = llvm.trunc %491 : i512 to i252
    %493 = llvm.zext %arg2 : i252 to i256
    %494 = llvm.zext %492 : i252 to i256
    %495 = llvm.add %493, %494  : i256
    %496 = llvm.sub %495, %8  : i256
    %497 = llvm.icmp "uge" %495, %8 : i256
    %498 = llvm.select %497, %496, %495 : i1, i256
    %499 = llvm.trunc %498 : i256 to i252
    %500 = llvm.extractvalue %484[0] : !llvm.struct<(i32)> 
    %501 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %502 = llvm.insertvalue %65, %501[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %503 = llvm.insertvalue %499, %502[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %504 = llvm.insertvalue %500, %503[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    llvm.br ^bb55(%480, %504 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb54(%505: !llvm.struct<(ptr<i248>, i32, i32)>, %506: i252, %507: i32):  // pred: ^bb50
    %508 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %509 = llvm.insertvalue %505, %508[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %510 = llvm.insertvalue %506, %509[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %511 = llvm.insertvalue %507, %510[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    llvm.br ^bb55(%75, %511 : i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>)
  ^bb55(%512: i64, %513: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>):  // 2 preds: ^bb53, ^bb54
    %514 = llvm.mlir.undef : !llvm.struct<()>
    %515 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>
    %516 = llvm.insertvalue %513, %515[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %517 = llvm.insertvalue %514, %516[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %518 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %519 = llvm.insertvalue %12, %518[0] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %520 = llvm.insertvalue %517, %519[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    llvm.store %520, %18 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>, !llvm.ptr
    %521 = llvm.load %18 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    %522 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
    %523 = llvm.insertvalue %512, %522[0] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    %524 = llvm.insertvalue %521, %523[1] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    llvm.return %524 : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
  ^bb56:  // pred: ^bb0
    %525 = llvm.mlir.undef : !llvm.struct<()>
    %526 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>
    %527 = llvm.insertvalue %arg1, %526[0] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %528 = llvm.insertvalue %525, %527[1] : !llvm.struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)> 
    %529 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>
    %530 = llvm.insertvalue %12, %529[0] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    %531 = llvm.insertvalue %528, %530[1] : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)> 
    llvm.store %531, %15 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, struct<()>)>)>, !llvm.ptr
    %532 = llvm.load %15 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<52 x i8>)>
    %533 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
    %534 = llvm.insertvalue %arg0, %533[0] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    %535 = llvm.insertvalue %532, %534[1] : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)> 
    llvm.return %535 : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
  ^bb57:  // pred: ^bb2
    %536 = llvm.mlir.addressof @assert_msg_1 : !llvm.ptr
    llvm.call @puts(%536) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_core::byte_array::ByteArrayImpl::append_word(f3)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %arg3: i252, %arg4: i32) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::byte_array::ByteArrayImpl::append_word(f3)"(%arg1, %arg2, %arg3, %arg4) : (i64, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, i252, i32) -> !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, struct<(i64, array<52 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<(), core::fmt::Error>::unwrap::<core::fmt::ErrorDrop>(f1)"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(i1, array<0 x i8>)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(false) : i1
    %1 = llvm.mlir.constant(24 : i64) : i64
    %2 = llvm.mlir.constant(30828113188794245257250221355944970489240709081949230 : i252) : i252
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    llvm.call @"core::result::ResultTraitImpl::<(), core::fmt::Error>::expect::<core::fmt::ErrorDrop>(f2)"(%4, %arg1, %2) : (!llvm.ptr, !llvm.struct<(i1, array<0 x i8>)>, i252) -> ()
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %4, %1, %0) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    llvm.return
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<(), core::fmt::Error>::unwrap::<core::fmt::ErrorDrop>(f1)"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(i1, array<0 x i8>)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    llvm.call @"core::result::ResultTraitImpl::<(), core::fmt::Error>::unwrap::<core::fmt::ErrorDrop>(f1)"(%arg0, %arg1) : (!llvm.ptr, !llvm.struct<(i1, array<0 x i8>)>) -> ()
    llvm.return
  }
  llvm.func @"core::byte_array::ByteArraySerde::serialize(f0)"(%arg0: i64, %arg1: i128, %arg2: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %arg3: !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(32 : i64) : i64
    %3 = llvm.mlir.constant(8 : i32) : i32
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %7 = llvm.alloca %5 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %8 = llvm.alloca %5 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %9 = llvm.extractvalue %arg2[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %10 = llvm.call @"core::array::ArraySerde::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>::serialize(f22)"(%arg0, %arg1, %9, %arg3) : (i64, i128, !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.store %13, %6 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %14 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %14 : i1, ^bb1 [
      0: ^bb3,
      1: ^bb8
    ]
  ^bb1:  // pred: ^bb0
    llvm.cond_br %4, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.unreachable
  ^bb3:  // pred: ^bb0
    %15 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %16 = llvm.extractvalue %15[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %18 = llvm.extractvalue %arg2[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %19 = llvm.call @"core::Felt252Serde::serialize(f21)"(%18, %17) : (i252, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>
    %20 = llvm.extractvalue %19[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %21 = llvm.extractvalue %19[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %22 = llvm.extractvalue %arg2[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %23 = llvm.zext %22 : i32 to i252
    %24 = llvm.extractvalue %20[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %25 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %26 = llvm.extractvalue %20[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %27 = llvm.icmp "uge" %25, %26 : i32
    llvm.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.add %26, %26  : i32
    %29 = llvm.intr.umax(%28, %3)  : (i32, i32) -> i32
    %30 = llvm.zext %29 : i32 to i64
    %31 = llvm.mul %30, %2  : i64
    %32 = llvm.bitcast %24 : !llvm.ptr<i252> to !llvm.ptr
    %33 = llvm.call @realloc(%32, %31) : (!llvm.ptr, i64) -> !llvm.ptr
    %34 = llvm.bitcast %33 : !llvm.ptr to !llvm.ptr<i252>
    %35 = llvm.insertvalue %34, %20[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %36 = llvm.insertvalue %29, %35[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    llvm.br ^bb6(%36, %34 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>)
  ^bb5:  // pred: ^bb3
    llvm.br ^bb6(%20, %24 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>)
  ^bb6(%37: !llvm.struct<(ptr<i252>, i32, i32)>, %38: !llvm.ptr<i252>):  // 2 preds: ^bb4, ^bb5
    llvm.br ^bb7
  ^bb7:  // pred: ^bb6
    %39 = llvm.getelementptr %38[%25] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %23, %39 : i252, !llvm.ptr
    %40 = llvm.add %25, %1  : i32
    %41 = llvm.insertvalue %40, %37[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %42 = llvm.mlir.undef : !llvm.struct<()>
    %43 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>
    %44 = llvm.insertvalue %41, %43[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %45 = llvm.insertvalue %42, %44[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %46 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %47 = llvm.insertvalue %4, %46[0] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    %48 = llvm.insertvalue %45, %47[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    llvm.store %48, %8 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>, !llvm.ptr
    %49 = llvm.load %8 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    %50 = llvm.mlir.undef : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %51 = llvm.insertvalue %11, %50[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %52 = llvm.insertvalue %12, %51[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %53 = llvm.insertvalue %49, %52[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.return %53 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
  ^bb8:  // pred: ^bb0
    %54 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %55 = llvm.extractvalue %54[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %56 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %57 = llvm.insertvalue %0, %56[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %58 = llvm.insertvalue %55, %57[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %58, %7 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    %59 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    %60 = llvm.mlir.undef : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %61 = llvm.insertvalue %11, %60[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %62 = llvm.insertvalue %12, %61[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %63 = llvm.insertvalue %59, %62[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.return %63 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
  ^bb9:  // pred: ^bb1
    %64 = llvm.mlir.addressof @assert_msg_2 : !llvm.ptr
    llvm.call @puts(%64) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_core::byte_array::ByteArraySerde::serialize(f0)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i128, %arg3: !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, %arg4: !llvm.struct<(ptr<i252>, i32, i32)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::byte_array::ByteArraySerde::serialize(f0)"(%arg1, %arg2, %arg3, %arg4) : (i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<0, 0>>(f18)"(%arg0: i64, %arg1: i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(0 : i252) : i252
    %3 = llvm.mlir.constant(1 : i252) : i252
    %4 = llvm.mlir.constant(64 : i252) : i252
    %5 = llvm.mlir.constant(-1 : i252) : i252
    %6 = llvm.mlir.constant(-3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252) : i252
    %7 = llvm.mlir.constant(1809251394333065606848661391547535052811553607665798349986546028067936010240 : i252) : i252
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.alloca %8 x !llvm.struct<(i8, array<1 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
    %10 = llvm.alloca %8 x !llvm.struct<(i8, array<1 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
    %11 = llvm.add %arg0, %8  : i64
    %12 = llvm.icmp "ugt" %arg1, %7 : i252
    llvm.cond_br %12, ^bb1, ^bb2(%arg1 : i252)
  ^bb1:  // pred: ^bb0
    %13 = llvm.sub %6, %arg1  : i252
    %14 = llvm.mul %13, %5  : i252
    llvm.br ^bb2(%14 : i252)
  ^bb2(%15: i252):  // 2 preds: ^bb0, ^bb1
    %16 = llvm.trunc %15 : i252 to i2
    %17 = llvm.trunc %15 : i252 to i64
    %18 = llvm.lshr %15, %4  : i252
    %19 = llvm.trunc %18 : i252 to i64
    %20 = llvm.lshr %18, %4  : i252
    %21 = llvm.trunc %20 : i252 to i64
    %22 = llvm.lshr %20, %4  : i252
    %23 = llvm.trunc %22 : i252 to i64
    llvm.call @__debug__print_felt252(%17, %19, %21, %23) : (i64, i64, i64, i64) -> ()
    %24 = llvm.icmp "sle" %15, %3 : i252
    %25 = llvm.icmp "sge" %15, %2 : i252
    %26 = llvm.and %24, %25  : i1
    llvm.call @__debug__print_i1(%26) : (i1) -> ()
    llvm.cond_br %26, ^bb3(%16 : i2), ^bb4
  ^bb3(%27: i2):  // pred: ^bb2
    %28 = llvm.mlir.undef : !llvm.struct<(i1, i2)>
    %29 = llvm.insertvalue %1, %28[0] : !llvm.struct<(i1, i2)> 
    %30 = llvm.insertvalue %27, %29[1] : !llvm.struct<(i1, i2)> 
    llvm.store %30, %10 {alignment = 1 : i64} : !llvm.struct<(i1, i2)>, !llvm.ptr
    llvm.br ^bb5(%arg1, %11, %10 : i252, i64, !llvm.ptr)
  ^bb4:  // pred: ^bb2
    %31 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %32 = llvm.insertvalue %0, %31[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %32, %9 {alignment = 1 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    llvm.br ^bb5(%arg1, %11, %9 : i252, i64, !llvm.ptr)
  ^bb5(%33: i252, %34: i64, %35: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
    %36 = llvm.call @"program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<0, 0>>(f19)"(%35, %33) : (!llvm.ptr, i252) -> !llvm.struct<(i1, array<0 x i8>)>
    %37 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    %38 = llvm.insertvalue %34, %37[0] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %39 = llvm.insertvalue %36, %38[1] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    llvm.return %39 : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
  }
  llvm.func @"_mlir_ciface_program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<0, 0>>(f18)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<0, 0>>(f18)"(%arg1, %arg2) : (i64, i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<-1, -1>>(f16)"(%arg0: i64, %arg1: i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(0 : i252) : i252
    %2 = llvm.mlir.constant(64 : i252) : i252
    %3 = llvm.mlir.constant(-1 : i252) : i252
    %4 = llvm.mlir.constant(-3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252) : i252
    %5 = llvm.mlir.constant(1809251394333065606848661391547535052811553607665798349986546028067936010240 : i252) : i252
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.add %arg0, %6  : i64
    %8 = llvm.icmp "ugt" %arg1, %5 : i252
    llvm.cond_br %8, ^bb1, ^bb2(%arg1 : i252)
  ^bb1:  // pred: ^bb0
    %9 = llvm.sub %4, %arg1  : i252
    %10 = llvm.mul %9, %3  : i252
    llvm.br ^bb2(%10 : i252)
  ^bb2(%11: i252):  // 2 preds: ^bb0, ^bb1
    %12 = llvm.trunc %11 : i252 to i64
    %13 = llvm.lshr %11, %2  : i252
    %14 = llvm.trunc %13 : i252 to i64
    %15 = llvm.lshr %13, %2  : i252
    %16 = llvm.trunc %15 : i252 to i64
    %17 = llvm.lshr %15, %2  : i252
    %18 = llvm.trunc %17 : i252 to i64
    llvm.call @__debug__print_felt252(%12, %14, %16, %18) : (i64, i64, i64, i64) -> ()
    %19 = llvm.icmp "sle" %11, %1 : i252
    %20 = llvm.icmp "sge" %11, %3 : i252
    %21 = llvm.and %19, %20  : i1
    llvm.call @__debug__print_i1(%21) : (i1) -> ()
    %22 = llvm.xor %21, %0  : i1
    %23 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<(i1, array<0 x i8>)> 
    %25 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    %26 = llvm.insertvalue %7, %25[0] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %27 = llvm.insertvalue %24, %26[1] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    llvm.return %27 : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
  }
  llvm.func @"_mlir_ciface_program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<-1, -1>>(f16)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"program::program::special_casts::downcast_invalid::<core::felt252, program::program::special_casts::BoundedInt::<-1, -1>>(f16)"(%arg1, %arg2) : (i64, i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<-1, -1>>(f14)"(%arg0: i64, %arg1: i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(0 : i252) : i252
    %3 = llvm.mlir.constant(64 : i252) : i252
    %4 = llvm.mlir.constant(-1 : i252) : i252
    %5 = llvm.mlir.constant(-3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252) : i252
    %6 = llvm.mlir.constant(1809251394333065606848661391547535052811553607665798349986546028067936010240 : i252) : i252
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x !llvm.struct<(i8, array<1 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
    %9 = llvm.alloca %7 x !llvm.struct<(i8, array<1 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
    %10 = llvm.add %arg0, %7  : i64
    %11 = llvm.icmp "ugt" %arg1, %6 : i252
    llvm.cond_br %11, ^bb1, ^bb2(%arg1 : i252)
  ^bb1:  // pred: ^bb0
    %12 = llvm.sub %5, %arg1  : i252
    %13 = llvm.mul %12, %4  : i252
    llvm.br ^bb2(%13 : i252)
  ^bb2(%14: i252):  // 2 preds: ^bb0, ^bb1
    %15 = llvm.trunc %14 : i252 to i2
    %16 = llvm.trunc %14 : i252 to i64
    %17 = llvm.lshr %14, %3  : i252
    %18 = llvm.trunc %17 : i252 to i64
    %19 = llvm.lshr %17, %3  : i252
    %20 = llvm.trunc %19 : i252 to i64
    %21 = llvm.lshr %19, %3  : i252
    %22 = llvm.trunc %21 : i252 to i64
    llvm.call @__debug__print_felt252(%16, %18, %20, %22) : (i64, i64, i64, i64) -> ()
    %23 = llvm.icmp "sle" %14, %2 : i252
    %24 = llvm.icmp "sge" %14, %4 : i252
    %25 = llvm.and %23, %24  : i1
    llvm.call @__debug__print_i1(%25) : (i1) -> ()
    llvm.cond_br %25, ^bb3(%15 : i2), ^bb4
  ^bb3(%26: i2):  // pred: ^bb2
    %27 = llvm.mlir.undef : !llvm.struct<(i1, i2)>
    %28 = llvm.insertvalue %1, %27[0] : !llvm.struct<(i1, i2)> 
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<(i1, i2)> 
    llvm.store %29, %9 {alignment = 1 : i64} : !llvm.struct<(i1, i2)>, !llvm.ptr
    llvm.br ^bb5(%arg1, %10, %9 : i252, i64, !llvm.ptr)
  ^bb4:  // pred: ^bb2
    %30 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %31 = llvm.insertvalue %0, %30[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %31, %8 {alignment = 1 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    llvm.br ^bb5(%arg1, %10, %8 : i252, i64, !llvm.ptr)
  ^bb5(%32: i252, %33: i64, %34: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
    %35 = llvm.call @"program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<-1, -1>>(f15)"(%34, %32) : (!llvm.ptr, i252) -> !llvm.struct<(i1, array<0 x i8>)>
    %36 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    %37 = llvm.insertvalue %33, %36[0] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)> 
    llvm.return %38 : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
  }
  llvm.func @"_mlir_ciface_program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<-1, -1>>(f14)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"program::program::special_casts::felt252_downcast_valid::<program::program::special_casts::BoundedInt::<-1, -1>>(f14)"(%arg1, %arg2) : (i64, i252) -> !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, struct<(i1, array<0 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::byte_array::ByteArrayDefault::default(f13)"() -> !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(0 : i252) : i252
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.null : !llvm.ptr<i248>
    %3 = llvm.mlir.undef : !llvm.struct<(ptr<i248>, i32, i32)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %6 = llvm.insertvalue %1, %5[2] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %7 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %9 = llvm.insertvalue %0, %8[1] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    %10 = llvm.insertvalue %1, %9[2] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)> 
    llvm.return %10 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
  }
  llvm.func @"_mlir_ciface_core::byte_array::ByteArrayDefault::default(f13)"(%arg0: !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::byte_array::ByteArrayDefault::default(f13)"() : () -> !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>
    llvm.store %0, %arg0 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>, i252, i32)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::integer::U32Add::add(f4)"(%arg0: i64, %arg1: i32, %arg2: i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(155785504323917466144735657540098748279 : i252) : i252
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.struct<(i32, array<4 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %3 = llvm.alloca %1 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %4 = llvm.alloca %1 x !llvm.struct<(i32, array<4 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %5 = llvm.add %arg0, %1  : i64
    %6 = "llvm.intr.uadd.with.overflow"(%arg1, %arg2) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(i32, i1)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(i32, i1)> 
    %9 = llvm.select %8, %2, %4 : i1, !llvm.ptr
    %10 = llvm.select %8, %2, %4 : i1, !llvm.ptr
    %11 = llvm.mlir.undef : !llvm.struct<(i1, i32)>
    %12 = llvm.insertvalue %8, %11[0] : !llvm.struct<(i1, i32)> 
    %13 = llvm.insertvalue %7, %12[1] : !llvm.struct<(i1, i32)> 
    llvm.store %13, %9 {alignment = 4 : i64} : !llvm.struct<(i1, i32)>, !llvm.ptr
    llvm.call @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>(f5)"(%3, %10, %0) : (!llvm.ptr, !llvm.ptr, i252) -> ()
    %14 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    %15 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %16 = llvm.insertvalue %5, %15[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.return %17 : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
  }
  llvm.func @"_mlir_ciface_core::integer::U32Add::add(f4)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i32, %arg3: i32) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::integer::U32Add::add(f4)"(%arg1, %arg2, %arg3) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::integer::U32Sub::sub(f8)"(%arg0: i64, %arg1: i32, %arg2: i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(155785504329508738615720351733824384887 : i252) : i252
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.struct<(i32, array<4 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %3 = llvm.alloca %1 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %4 = llvm.alloca %1 x !llvm.struct<(i32, array<4 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %5 = llvm.add %arg0, %1  : i64
    %6 = "llvm.intr.usub.with.overflow"(%arg1, %arg2) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(i32, i1)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(i32, i1)> 
    %9 = llvm.select %8, %2, %4 : i1, !llvm.ptr
    %10 = llvm.select %8, %2, %4 : i1, !llvm.ptr
    %11 = llvm.mlir.undef : !llvm.struct<(i1, i32)>
    %12 = llvm.insertvalue %8, %11[0] : !llvm.struct<(i1, i32)> 
    %13 = llvm.insertvalue %7, %12[1] : !llvm.struct<(i1, i32)> 
    llvm.store %13, %9 {alignment = 4 : i64} : !llvm.struct<(i1, i32)>, !llvm.ptr
    llvm.call @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>(f5)"(%3, %10, %0) : (!llvm.ptr, !llvm.ptr, i252) -> ()
    %14 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    %15 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %16 = llvm.insertvalue %5, %15[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.return %17 : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
  }
  llvm.func @"_mlir_ciface_core::integer::U32Sub::sub(f8)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i32, %arg3: i32) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::integer::U32Sub::sub(f8)"(%arg1, %arg2, %arg3) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::integer::u256_from_felt252(f10)"(%arg0: i64, %arg1: i252) -> !llvm.struct<(i64, struct<(i128, i128)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(340282366920938463463374607431768211456 : i252) : i252
    %1 = llvm.mlir.constant(0 : i128) : i128
    %2 = llvm.mlir.constant(128 : i252) : i252
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.add %arg0, %3  : i64
    %5 = llvm.icmp "uge" %arg1, %0 : i252
    %6 = llvm.trunc %arg1 : i252 to i128
    %7 = llvm.lshr %arg1, %2  : i252
    %8 = llvm.trunc %7 : i252 to i128
    %9 = llvm.select %5, %8, %1 : i1, i128
    %10 = llvm.mlir.undef : !llvm.struct<(i128, i128)>
    %11 = llvm.insertvalue %6, %10[0] : !llvm.struct<(i128, i128)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<(i128, i128)> 
    %13 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i128, i128)>)>
    %14 = llvm.insertvalue %4, %13[0] : !llvm.struct<(i64, struct<(i128, i128)>)> 
    %15 = llvm.insertvalue %12, %14[1] : !llvm.struct<(i64, struct<(i128, i128)>)> 
    llvm.return %15 : !llvm.struct<(i64, struct<(i128, i128)>)>
  }
  llvm.func @"_mlir_ciface_core::integer::u256_from_felt252(f10)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::integer::u256_from_felt252(f10)"(%arg1, %arg2) : (i64, i252) -> !llvm.struct<(i64, struct<(i128, i128)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, struct<(i128, i128)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%arg0: !llvm.ptr, %arg1: i32) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(256 : i64) : i64
    %1 = llvm.mlir.constant(1 : i128) : i128
    %2 = llvm.mlir.constant(256 : i128) : i128
    %3 = llvm.mlir.constant(65536 : i128) : i128
    %4 = llvm.mlir.constant(16777216 : i128) : i128
    %5 = llvm.mlir.constant(4294967296 : i128) : i128
    %6 = llvm.mlir.constant(1099511627776 : i128) : i128
    %7 = llvm.mlir.constant(281474976710656 : i128) : i128
    %8 = llvm.mlir.constant(72057594037927936 : i128) : i128
    %9 = llvm.mlir.constant(18446744073709551616 : i128) : i128
    %10 = llvm.mlir.constant(4722366482869645213696 : i128) : i128
    %11 = llvm.mlir.constant(1208925819614629174706176 : i128) : i128
    %12 = llvm.mlir.constant(309485009821345068724781056 : i128) : i128
    %13 = llvm.mlir.constant(79228162514264337593543950336 : i128) : i128
    %14 = llvm.mlir.constant(20282409603651670423947251286016 : i128) : i128
    %15 = llvm.mlir.constant(5192296858534827628530496329220096 : i128) : i128
    %16 = llvm.mlir.constant(1329227995784915872903807060280344576 : i128) : i128
    %17 = llvm.mlir.constant(false) : i1
    %18 = llvm.mlir.constant(24 : i64) : i64
    %19 = llvm.mlir.constant(true) : i1
    %20 = llvm.mlir.constant(573087285299505011920718992710461799 : i252) : i252
    %21 = llvm.mlir.constant(15 : i32) : i32
    %22 = llvm.mlir.constant(14 : i32) : i32
    %23 = llvm.mlir.constant(13 : i32) : i32
    %24 = llvm.mlir.constant(12 : i32) : i32
    %25 = llvm.mlir.constant(11 : i32) : i32
    %26 = llvm.mlir.constant(10 : i32) : i32
    %27 = llvm.mlir.constant(9 : i32) : i32
    %28 = llvm.mlir.constant(8 : i32) : i32
    %29 = llvm.mlir.constant(7 : i32) : i32
    %30 = llvm.mlir.constant(6 : i32) : i32
    %31 = llvm.mlir.constant(5 : i32) : i32
    %32 = llvm.mlir.constant(4 : i32) : i32
    %33 = llvm.mlir.constant(3 : i32) : i32
    %34 = llvm.mlir.constant(2 : i32) : i32
    %35 = llvm.mlir.constant(1 : i32) : i32
    %36 = llvm.mlir.constant(0 : i32) : i32
    %37 = llvm.mlir.constant(1 : i64) : i64
    %38 = llvm.alloca %37 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %39 = llvm.alloca %37 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %40 = llvm.icmp "eq" %arg1, %36 : i32
    llvm.cond_br %40, ^bb17(%1 : i128), ^bb1(%arg1 : i32)
  ^bb1(%41: i32):  // pred: ^bb0
    %42 = llvm.icmp "eq" %41, %35 : i32
    llvm.cond_br %42, ^bb17(%2 : i128), ^bb2(%41 : i32)
  ^bb2(%43: i32):  // pred: ^bb1
    %44 = llvm.icmp "eq" %43, %34 : i32
    llvm.cond_br %44, ^bb17(%3 : i128), ^bb3(%43 : i32)
  ^bb3(%45: i32):  // pred: ^bb2
    %46 = llvm.icmp "eq" %45, %33 : i32
    llvm.cond_br %46, ^bb17(%4 : i128), ^bb4(%45 : i32)
  ^bb4(%47: i32):  // pred: ^bb3
    %48 = llvm.icmp "eq" %47, %32 : i32
    llvm.cond_br %48, ^bb17(%5 : i128), ^bb5(%47 : i32)
  ^bb5(%49: i32):  // pred: ^bb4
    %50 = llvm.icmp "eq" %49, %31 : i32
    llvm.cond_br %50, ^bb17(%6 : i128), ^bb6(%49 : i32)
  ^bb6(%51: i32):  // pred: ^bb5
    %52 = llvm.icmp "eq" %51, %30 : i32
    llvm.cond_br %52, ^bb17(%7 : i128), ^bb7(%51 : i32)
  ^bb7(%53: i32):  // pred: ^bb6
    %54 = llvm.icmp "eq" %53, %29 : i32
    llvm.cond_br %54, ^bb17(%8 : i128), ^bb8(%53 : i32)
  ^bb8(%55: i32):  // pred: ^bb7
    %56 = llvm.icmp "eq" %55, %28 : i32
    llvm.cond_br %56, ^bb17(%9 : i128), ^bb9(%55 : i32)
  ^bb9(%57: i32):  // pred: ^bb8
    %58 = llvm.icmp "eq" %57, %27 : i32
    llvm.cond_br %58, ^bb17(%10 : i128), ^bb10(%57 : i32)
  ^bb10(%59: i32):  // pred: ^bb9
    %60 = llvm.icmp "eq" %59, %26 : i32
    llvm.cond_br %60, ^bb17(%11 : i128), ^bb11(%59 : i32)
  ^bb11(%61: i32):  // pred: ^bb10
    %62 = llvm.icmp "eq" %61, %25 : i32
    llvm.cond_br %62, ^bb17(%12 : i128), ^bb12(%61 : i32)
  ^bb12(%63: i32):  // pred: ^bb11
    %64 = llvm.icmp "eq" %63, %24 : i32
    llvm.cond_br %64, ^bb17(%13 : i128), ^bb13(%63 : i32)
  ^bb13(%65: i32):  // pred: ^bb12
    %66 = llvm.icmp "eq" %65, %23 : i32
    llvm.cond_br %66, ^bb17(%14 : i128), ^bb14(%65 : i32)
  ^bb14(%67: i32):  // pred: ^bb13
    %68 = llvm.icmp "eq" %67, %22 : i32
    llvm.cond_br %68, ^bb17(%15 : i128), ^bb15(%67, %21 : i32, i32)
  ^bb15(%69: i32, %70: i32):  // pred: ^bb14
    %71 = llvm.icmp "eq" %69, %70 : i32
    llvm.cond_br %71, ^bb17(%16 : i128), ^bb16
  ^bb16:  // pred: ^bb15
    %72 = llvm.mlir.null : !llvm.ptr<i252>
    %73 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %74 = llvm.insertvalue %72, %73[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %75 = llvm.insertvalue %36, %74[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %76 = llvm.insertvalue %36, %75[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %77 = llvm.bitcast %72 : !llvm.ptr<i252> to !llvm.ptr
    %78 = llvm.call @realloc(%77, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %79 = llvm.bitcast %78 : !llvm.ptr to !llvm.ptr<i252>
    %80 = llvm.insertvalue %79, %76[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %81 = llvm.insertvalue %28, %80[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %82 = llvm.getelementptr %79[0] : (!llvm.ptr<i252>) -> !llvm.ptr, i252
    llvm.store %20, %82 : i252, !llvm.ptr
    %83 = llvm.insertvalue %35, %81[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %84 = llvm.mlir.undef : !llvm.struct<()>
    %85 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %86 = llvm.insertvalue %84, %85[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %87 = llvm.insertvalue %83, %86[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %88 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %89 = llvm.insertvalue %19, %88[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %90 = llvm.insertvalue %87, %89[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %90, %39 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %39, %18, %17) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    llvm.return
  ^bb17(%91: i128):  // 16 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15
    %92 = llvm.mlir.undef : !llvm.struct<(i128)>
    %93 = llvm.insertvalue %91, %92[0] : !llvm.struct<(i128)> 
    %94 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i128)>)>
    %95 = llvm.insertvalue %17, %94[0] : !llvm.struct<(i1, struct<(i128)>)> 
    %96 = llvm.insertvalue %93, %95[1] : !llvm.struct<(i1, struct<(i128)>)> 
    llvm.store %96, %38 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(i128)>)>, !llvm.ptr
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %38, %18, %17) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    llvm.return
  }
  llvm.func @"_mlir_ciface_core::bytes_31::one_shift_left_bytes_u128(f7)"(%arg0: !llvm.ptr, %arg1: i32) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    llvm.call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%arg0, %arg1) : (!llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @"core::integer::u128_try_as_non_zero(f11)"(%arg0: !llvm.ptr, %arg1: i128) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(false) : i1
    %1 = llvm.mlir.constant(24 : i64) : i64
    %2 = llvm.mlir.constant(true) : i1
    %3 = llvm.mlir.constant(0 : i128) : i128
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %6 = llvm.alloca %4 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %7 = llvm.icmp "eq" %arg1, %3 : i128
    llvm.cond_br %7, ^bb1, ^bb2(%arg1 : i128)
  ^bb1:  // pred: ^bb0
    %8 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %9 = llvm.insertvalue %2, %8[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %9, %6 {alignment = 8 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %6, %1, %0) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    llvm.return
  ^bb2(%10: i128):  // pred: ^bb0
    %11 = llvm.mlir.undef : !llvm.struct<(i1, i128)>
    %12 = llvm.insertvalue %0, %11[0] : !llvm.struct<(i1, i128)> 
    %13 = llvm.insertvalue %10, %12[1] : !llvm.struct<(i1, i128)> 
    llvm.store %13, %5 {alignment = 8 : i64} : !llvm.struct<(i1, i128)>, !llvm.ptr
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %5, %1, %0) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    llvm.return
  }
  llvm.func @"_mlir_ciface_core::integer::u128_try_as_non_zero(f11)"(%arg0: !llvm.ptr, %arg1: i128) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    llvm.call @"core::integer::u128_try_as_non_zero(f11)"(%arg0, %arg1) : (!llvm.ptr, i128) -> ()
    llvm.return
  }
  llvm.func @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%arg0: i64, %arg1: i32) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(340282366920938463463374607431768211456 : i512) : i512
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %3 = llvm.mlir.constant(false) : i1
    %4 = llvm.mlir.constant(16 : i32) : i32
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %7 = llvm.alloca %5 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %8 = llvm.alloca %5 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %9 = llvm.alloca %5 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %10 = llvm.alloca %5 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %11 = llvm.alloca %5 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %12 = llvm.alloca %5 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %13 = llvm.add %arg0, %5  : i64
    %14 = "llvm.intr.usub.with.overflow"(%arg1, %4) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %15 = llvm.extractvalue %14[1] : !llvm.struct<(i32, i1)> 
    llvm.cond_br %15, ^bb7(%arg1 : i32), ^bb1(%13 : i64)
  ^bb1(%16: i64):  // pred: ^bb0
    %17 = llvm.call @"core::integer::U32Sub::sub(f8)"(%16, %arg1, %4) : (i64, i32, i32) -> !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)>
    %18 = llvm.extractvalue %17[0] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    %19 = llvm.extractvalue %17[1] : !llvm.struct<(i64, struct<(i64, array<16 x i8>)>)> 
    llvm.store %19, %9 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %20 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %20 : i1, ^bb2 [
      0: ^bb4,
      1: ^bb5(%9, %10, %10, %18 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb2:  // 3 preds: ^bb1, ^bb4, ^bb7
    llvm.cond_br %3, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.unreachable
  ^bb4:  // pred: ^bb1
    %21 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i32)>)>
    %22 = llvm.extractvalue %21[1] : !llvm.struct<(i1, struct<(i32)>)> 
    %23 = llvm.extractvalue %22[0] : !llvm.struct<(i32)> 
    llvm.call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%11, %23) : (!llvm.ptr, i32) -> ()
    %24 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %24 : i1, ^bb2 [
      0: ^bb6,
      1: ^bb5(%11, %12, %12, %18 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb5(%25: !llvm.ptr, %26: !llvm.ptr, %27: !llvm.ptr, %28: i64):  // 3 preds: ^bb1, ^bb4, ^bb7
    %29 = llvm.load %25 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %30 = llvm.extractvalue %29[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %31 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %32 = llvm.insertvalue %1, %31[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %33, %26 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    %34 = llvm.load %27 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<32 x i8>)>
    %35 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %36 = llvm.insertvalue %28, %35[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.return %37 : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
  ^bb6:  // pred: ^bb4
    %38 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i128)>)>
    %39 = llvm.extractvalue %38[1] : !llvm.struct<(i1, struct<(i128)>)> 
    %40 = llvm.extractvalue %39[0] : !llvm.struct<(i128)> 
    %41 = llvm.zext %40 : i128 to i512
    %42 = llvm.mul %41, %0  : i512
    %43 = llvm.urem %42, %2  : i512
    %44 = llvm.icmp "uge" %42, %2 : i512
    %45 = llvm.select %44, %43, %42 : i1, i512
    %46 = llvm.trunc %45 : i512 to i252
    llvm.br ^bb9(%18, %46 : i64, i252)
  ^bb7(%47: i32):  // pred: ^bb0
    llvm.call @"core::bytes_31::one_shift_left_bytes_u128(f7)"(%6, %47) : (!llvm.ptr, i32) -> ()
    %48 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %48 : i1, ^bb2 [
      0: ^bb8,
      1: ^bb5(%6, %7, %7, %13 : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
    ]
  ^bb8:  // pred: ^bb7
    %49 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, struct<(i128)>)>
    %50 = llvm.extractvalue %49[1] : !llvm.struct<(i1, struct<(i128)>)> 
    %51 = llvm.extractvalue %50[0] : !llvm.struct<(i128)> 
    %52 = llvm.zext %51 : i128 to i252
    llvm.br ^bb9(%13, %52 : i64, i252)
  ^bb9(%53: i64, %54: i252):  // 2 preds: ^bb6, ^bb8
    %55 = llvm.mlir.undef : !llvm.struct<(i252)>
    %56 = llvm.insertvalue %54, %55[0] : !llvm.struct<(i252)> 
    %57 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i252)>)>
    %58 = llvm.insertvalue %3, %57[0] : !llvm.struct<(i1, struct<(i252)>)> 
    %59 = llvm.insertvalue %56, %58[1] : !llvm.struct<(i1, struct<(i252)>)> 
    llvm.store %59, %8 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(i252)>)>, !llvm.ptr
    %60 = llvm.load %8 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<32 x i8>)>
    %61 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %62 = llvm.insertvalue %53, %61[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %63 = llvm.insertvalue %60, %62[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.return %63 : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
  ^bb10:  // pred: ^bb2
    %64 = llvm.mlir.addressof @assert_msg_3 : !llvm.ptr
    llvm.call @puts(%64) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_core::bytes_31::one_shift_left_bytes_felt252(f6)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i32) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::bytes_31::one_shift_left_bytes_felt252(f6)"(%arg1, %arg2) : (i64, i32) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%arg0: i64, %arg1: i252) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(452312848583266388373324160190187140051835877600158453279131187530910662655 : i252) : i252
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %5 = llvm.alloca %3 x !llvm.struct<(i64, array<32 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %6 = llvm.add %arg0, %3  : i64
    %7 = llvm.icmp "ule" %arg1, %2 : i252
    llvm.cond_br %7, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %8 = llvm.trunc %arg1 : i252 to i248
    %9 = llvm.mlir.undef : !llvm.struct<(i1, i248)>
    %10 = llvm.insertvalue %1, %9[0] : !llvm.struct<(i1, i248)> 
    %11 = llvm.insertvalue %8, %10[1] : !llvm.struct<(i1, i248)> 
    llvm.store %11, %5 {alignment = 8 : i64} : !llvm.struct<(i1, i248)>, !llvm.ptr
    %12 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<32 x i8>)>
    %13 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %14 = llvm.insertvalue %6, %13[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %15 = llvm.insertvalue %12, %14[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.return %15 : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
  ^bb2:  // pred: ^bb0
    %16 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %17 = llvm.insertvalue %0, %16[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %17, %4 {alignment = 8 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    %18 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<32 x i8>)>
    %19 = llvm.mlir.undef : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    %20 = llvm.insertvalue %6, %19[0] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)> 
    llvm.return %21 : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
  }
  llvm.func @"_mlir_ciface_core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::bytes_31::Felt252TryIntoBytes31::try_into(f9)"(%arg1, %arg2) : (i64, i252) -> !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, struct<(i64, array<32 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<(), core::fmt::Error>::expect::<core::fmt::ErrorDrop>(f2)"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(i1, array<0 x i8>)>, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(256 : i64) : i64
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(8 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(24 : i64) : i64
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %9 = llvm.alloca %7 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %10 = llvm.extractvalue %arg1[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.switch %10 : i1, ^bb1 [
      0: ^bb3,
      1: ^bb4
    ]
  ^bb1:  // pred: ^bb0
    llvm.cond_br %6, ^bb2, ^bb5
  ^bb2:  // pred: ^bb1
    llvm.unreachable
  ^bb3:  // pred: ^bb0
    %11 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %12 = llvm.insertvalue %6, %11[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %12, %9 {alignment = 8 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %9, %5, %6) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    llvm.return
  ^bb4:  // pred: ^bb0
    %13 = llvm.mlir.null : !llvm.ptr<i252>
    %14 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %16 = llvm.insertvalue %4, %15[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %17 = llvm.insertvalue %4, %16[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %18 = llvm.bitcast %13 : !llvm.ptr<i252> to !llvm.ptr
    %19 = llvm.call @realloc(%18, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %20 = llvm.bitcast %19 : !llvm.ptr to !llvm.ptr<i252>
    %21 = llvm.insertvalue %20, %17[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %22 = llvm.insertvalue %3, %21[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %23 = llvm.getelementptr %20[0] : (!llvm.ptr<i252>) -> !llvm.ptr, i252
    llvm.store %arg2, %23 : i252, !llvm.ptr
    %24 = llvm.insertvalue %2, %22[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %25 = llvm.mlir.undef : !llvm.struct<()>
    %26 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %27 = llvm.insertvalue %25, %26[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %28 = llvm.insertvalue %24, %27[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %29 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %30 = llvm.insertvalue %1, %29[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %31, %8 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %8, %5, %6) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    llvm.return
  ^bb5:  // pred: ^bb1
    %32 = llvm.mlir.addressof @assert_msg_4 : !llvm.ptr
    llvm.call @puts(%32) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<(), core::fmt::Error>::expect::<core::fmt::ErrorDrop>(f2)"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(i1, array<0 x i8>)>, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    llvm.call @"core::result::ResultTraitImpl::<(), core::fmt::Error>::expect::<core::fmt::ErrorDrop>(f2)"(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.struct<(i1, array<0 x i8>)>, i252) -> ()
    llvm.return
  }
  llvm.func @"core::array::ArraySerde::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>::serialize(f22)"(%arg0: i64, %arg1: i128, %arg2: !llvm.struct<(ptr<i248>, i32, i32)>, %arg3: !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(32 : i64) : i64
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %5 = llvm.extractvalue %arg2[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %6 = llvm.zext %5 : i32 to i252
    %7 = llvm.extractvalue %arg3[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %8 = llvm.extractvalue %arg3[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %9 = llvm.extractvalue %arg3[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %10 = llvm.icmp "uge" %8, %9 : i32
    llvm.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %11 = llvm.add %9, %9  : i32
    %12 = llvm.intr.umax(%11, %2)  : (i32, i32) -> i32
    %13 = llvm.zext %12 : i32 to i64
    %14 = llvm.mul %13, %1  : i64
    %15 = llvm.bitcast %7 : !llvm.ptr<i252> to !llvm.ptr
    %16 = llvm.call @realloc(%15, %14) : (!llvm.ptr, i64) -> !llvm.ptr
    %17 = llvm.bitcast %16 : !llvm.ptr to !llvm.ptr<i252>
    %18 = llvm.insertvalue %17, %arg3[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %19 = llvm.insertvalue %12, %18[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    llvm.br ^bb3(%19, %17 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%arg3, %7 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>)
  ^bb3(%20: !llvm.struct<(ptr<i252>, i32, i32)>, %21: !llvm.ptr<i252>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %22 = llvm.getelementptr %21[%8] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %6, %22 : i252, !llvm.ptr
    %23 = llvm.add %8, %0  : i32
    %24 = llvm.insertvalue %23, %20[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %25 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>
    %26 = llvm.insertvalue %arg2, %25[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)> 
    %27 = llvm.call @"core::array::serialize_array_helper::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>(f23)"(%arg0, %arg1, %26, %24) : (i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %28 = llvm.extractvalue %27[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %29 = llvm.extractvalue %27[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %30 = llvm.extractvalue %27[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.store %30, %4 {alignment = 8 : i64} : !llvm.struct<(i64, array<16 x i8>)>, !llvm.ptr
    %31 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    %32 = llvm.mlir.undef : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %33 = llvm.insertvalue %28, %32[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %34 = llvm.insertvalue %29, %33[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %35 = llvm.insertvalue %31, %34[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.return %35 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
  }
  llvm.func @"_mlir_ciface_core::array::ArraySerde::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>::serialize(f22)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i128, %arg3: !llvm.struct<(ptr<i248>, i32, i32)>, %arg4: !llvm.struct<(ptr<i252>, i32, i32)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::array::ArraySerde::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>::serialize(f22)"(%arg1, %arg2, %arg3, %arg4) : (i64, i128, !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::Felt252Serde::serialize(f21)"(%arg0: i252, %arg1: !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
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
  llvm.func @"_mlir_ciface_core::Felt252Serde::serialize(f21)"(%arg0: !llvm.ptr, %arg1: i252, %arg2: !llvm.struct<(ptr<i252>, i32, i32)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::Felt252Serde::serialize(f21)"(%arg1, %arg2) : (i252, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>
    llvm.store %0, %arg0 : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<0, 0>>(f19)"(%arg0: !llvm.ptr, %arg1: i252) -> !llvm.struct<(i1, array<0 x i8>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %2 = llvm.mlir.constant(-3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252) : i252
    %3 = llvm.mlir.constant(0 : i252) : i252
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr -> i1
    llvm.switch %5 : i1, ^bb1 [
      0: ^bb3,
      1: ^bb4(%4 : i1)
    ]
  ^bb1:  // pred: ^bb0
    llvm.cond_br %4, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    llvm.unreachable
  ^bb3:  // pred: ^bb0
    %6 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr -> !llvm.struct<(i1, i2)>
    %7 = llvm.extractvalue %6[1] : !llvm.struct<(i1, i2)> 
    %8 = llvm.sext %7 : i2 to i252
    %9 = llvm.icmp "slt" %8, %3 : i252
    llvm.cond_br %9, ^bb5, ^bb6
  ^bb4(%10: i1):  // 3 preds: ^bb0, ^bb7, ^bb7
    %11 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.return %12 : !llvm.struct<(i1, array<0 x i8>)>
  ^bb5:  // pred: ^bb3
    %13 = llvm.sext %7 : i2 to i252
    %14 = llvm.add %13, %2  : i252
    llvm.br ^bb7(%14, %arg1 : i252, i252)
  ^bb6:  // pred: ^bb3
    %15 = llvm.zext %7 : i2 to i252
    llvm.br ^bb7(%15, %arg1 : i252, i252)
  ^bb7(%16: i252, %17: i252):  // 2 preds: ^bb5, ^bb6
    %18 = llvm.zext %16 : i252 to i256
    %19 = llvm.zext %17 : i252 to i256
    %20 = llvm.sub %18, %19  : i256
    %21 = llvm.add %20, %1  : i256
    %22 = llvm.icmp "ult" %18, %19 : i256
    %23 = llvm.select %22, %21, %20 : i1, i256
    %24 = llvm.trunc %23 : i256 to i252
    %25 = llvm.icmp "eq" %24, %3 : i252
    llvm.cond_br %25, ^bb4(%0 : i1), ^bb4(%4 : i1)
  ^bb8:  // pred: ^bb1
    %26 = llvm.mlir.addressof @assert_msg_5 : !llvm.ptr
    llvm.call @puts(%26) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<0, 0>>(f19)"(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<0, 0>>(f19)"(%arg1, %arg2) : (!llvm.ptr, i252) -> !llvm.struct<(i1, array<0 x i8>)>
    llvm.store %0, %arg0 : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<-1, -1>>(f15)"(%arg0: !llvm.ptr, %arg1: i252) -> !llvm.struct<(i1, array<0 x i8>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %2 = llvm.mlir.constant(-3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252) : i252
    %3 = llvm.mlir.constant(0 : i252) : i252
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr -> i1
    llvm.switch %5 : i1, ^bb1 [
      0: ^bb3,
      1: ^bb4(%4 : i1)
    ]
  ^bb1:  // pred: ^bb0
    llvm.cond_br %4, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    llvm.unreachable
  ^bb3:  // pred: ^bb0
    %6 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr -> !llvm.struct<(i1, i2)>
    %7 = llvm.extractvalue %6[1] : !llvm.struct<(i1, i2)> 
    %8 = llvm.sext %7 : i2 to i252
    %9 = llvm.icmp "slt" %8, %3 : i252
    llvm.cond_br %9, ^bb5, ^bb6
  ^bb4(%10: i1):  // 3 preds: ^bb0, ^bb7, ^bb7
    %11 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.return %12 : !llvm.struct<(i1, array<0 x i8>)>
  ^bb5:  // pred: ^bb3
    %13 = llvm.sext %7 : i2 to i252
    %14 = llvm.add %13, %2  : i252
    llvm.br ^bb7(%14, %arg1 : i252, i252)
  ^bb6:  // pred: ^bb3
    %15 = llvm.zext %7 : i2 to i252
    llvm.br ^bb7(%15, %arg1 : i252, i252)
  ^bb7(%16: i252, %17: i252):  // 2 preds: ^bb5, ^bb6
    %18 = llvm.zext %16 : i252 to i256
    %19 = llvm.zext %17 : i252 to i256
    %20 = llvm.sub %18, %19  : i256
    %21 = llvm.add %20, %1  : i256
    %22 = llvm.icmp "ult" %18, %19 : i256
    %23 = llvm.select %22, %21, %20 : i1, i256
    %24 = llvm.trunc %23 : i256 to i252
    %25 = llvm.icmp "eq" %24, %3 : i252
    llvm.cond_br %25, ^bb4(%0 : i1), ^bb4(%4 : i1)
  ^bb8:  // pred: ^bb1
    %26 = llvm.mlir.addressof @assert_msg_6 : !llvm.ptr
    llvm.call @puts(%26) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<-1, -1>>(f15)"(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"program::program::special_casts::is_some_of::<program::program::special_casts::BoundedInt::<-1, -1>>(f15)"(%arg1, %arg2) : (!llvm.ptr, i252) -> !llvm.struct<(i1, array<0 x i8>)>
    llvm.store %0, %arg0 : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>(f5)"(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(256 : i64) : i64
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(8 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(24 : i64) : i64
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %9 = llvm.alloca %7 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %10 = llvm.load %arg1 {alignment = 4 : i64} : !llvm.ptr -> i1
    llvm.switch %10 : i1, ^bb1 [
      0: ^bb3,
      1: ^bb4
    ]
  ^bb1:  // pred: ^bb0
    llvm.cond_br %6, ^bb2, ^bb5
  ^bb2:  // pred: ^bb1
    llvm.unreachable
  ^bb3:  // pred: ^bb0
    %11 = llvm.load %arg1 {alignment = 4 : i64} : !llvm.ptr -> !llvm.struct<(i1, i32)>
    %12 = llvm.extractvalue %11[1] : !llvm.struct<(i1, i32)> 
    %13 = llvm.mlir.undef : !llvm.struct<(i32)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(i32)> 
    %15 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i32)>)>
    %16 = llvm.insertvalue %6, %15[0] : !llvm.struct<(i1, struct<(i32)>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(i1, struct<(i32)>)> 
    llvm.store %17, %9 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(i32)>)>, !llvm.ptr
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %9, %5, %6) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    llvm.return
  ^bb4:  // pred: ^bb0
    %18 = llvm.mlir.null : !llvm.ptr<i252>
    %19 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %21 = llvm.insertvalue %4, %20[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %22 = llvm.insertvalue %4, %21[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %23 = llvm.bitcast %18 : !llvm.ptr<i252> to !llvm.ptr
    %24 = llvm.call @realloc(%23, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %25 = llvm.bitcast %24 : !llvm.ptr to !llvm.ptr<i252>
    %26 = llvm.insertvalue %25, %22[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %27 = llvm.insertvalue %3, %26[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %28 = llvm.getelementptr %25[0] : (!llvm.ptr<i252>) -> !llvm.ptr, i252
    llvm.store %arg2, %28 : i252, !llvm.ptr
    %29 = llvm.insertvalue %2, %27[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %30 = llvm.mlir.undef : !llvm.struct<()>
    %31 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %33 = llvm.insertvalue %29, %32[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %34 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %35 = llvm.insertvalue %1, %34[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %36, %8 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    llvm.call_intrinsic "llvm.memcpy.inline"(%arg0, %8, %5, %6) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()  {intrin = "llvm.memcpy.inline"}
    llvm.return
  ^bb5:  // pred: ^bb1
    %37 = llvm.mlir.addressof @assert_msg_7 : !llvm.ptr
    llvm.call @puts(%37) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>(f5)"(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    llvm.call @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>(f5)"(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i252) -> ()
    llvm.return
  }
  llvm.func @"core::array::serialize_array_helper::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>(f23)"(%arg0: i64, %arg1: i128, %arg2: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, %arg3: !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(256 : i64) : i64
    %1 = llvm.mlir.constant(375233589013918064796019 : i252) : i252
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(8 : i32) : i32
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.constant(false) : i1
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(32 : i64) : i64
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(2670 : i128) : i128
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.mlir.constant(0 : i64) : i64
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.alloca %12 x i64 : (i64) -> !llvm.ptr
    %14 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(ptr, ptr, i64)> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(ptr, ptr, i64)> 
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.insertvalue %17, %16[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %11, %13 : i64, !llvm.ptr
    %19 = llvm.alloca %10 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %20 = llvm.alloca %10 x !llvm.struct<(i64, array<8 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %21 = llvm.alloca %10 x !llvm.struct<(i64, array<16 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %22 = llvm.alloca %10 x !llvm.struct<(i64, array<8 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    llvm.br ^bb1(%arg0, %arg1, %arg2, %arg3 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb1(%23: i64, %24: i128, %25: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, %26: !llvm.struct<(ptr<i252>, i32, i32)>):  // 2 preds: ^bb0, ^bb12
    %27 = llvm.add %23, %10  : i64
    %28 = llvm.icmp "uge" %24, %9 : i128
    %29 = llvm.call_intrinsic "llvm.usub.sat"(%24, %9) : (i128, i128) -> i128  {intrin = "llvm.usub.sat"}
    llvm.cond_br %28, ^bb2(%25 : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>), ^bb14(%26 : !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb2(%30: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>):  // pred: ^bb1
    %31 = llvm.extractvalue %30[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)> 
    %32 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %33 = llvm.icmp "eq" %32, %8 : i32
    llvm.cond_br %33, ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    %34 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %35 = llvm.getelementptr %34[0] : (!llvm.ptr<i248>) -> !llvm.ptr, i248
    %36 = llvm.mlir.null : !llvm.ptr
    %37 = llvm.call @realloc(%36, %7) : (!llvm.ptr, i64) -> !llvm.ptr
    %38 = llvm.load %35 {alignment = 8 : i64} : !llvm.ptr -> i248
    llvm.store %38, %37 {alignment = 8 : i64} : i248, !llvm.ptr
    %39 = llvm.getelementptr %34[1] : (!llvm.ptr<i248>) -> !llvm.ptr, i248
    %40 = llvm.sub %32, %6  : i32
    %41 = llvm.zext %40 : i32 to i64
    %42 = llvm.mul %41, %7  : i64
    %43 = llvm.bitcast %34 : !llvm.ptr<i248> to !llvm.ptr
    "llvm.intr.memmove"(%43, %39, %42) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %44 = llvm.insertvalue %40, %31[1] : !llvm.struct<(ptr<i248>, i32, i32)> 
    %45 = llvm.mlir.undef : !llvm.struct<(i1, ptr)>
    %46 = llvm.insertvalue %5, %45[0] : !llvm.struct<(i1, ptr)> 
    %47 = llvm.insertvalue %37, %46[1] : !llvm.struct<(i1, ptr)> 
    llvm.store %47, %22 {alignment = 8 : i64} : !llvm.struct<(i1, ptr)>, !llvm.ptr
    llvm.br ^bb5(%26, %27, %29, %44, %22 : !llvm.struct<(ptr<i252>, i32, i32)>, i64, i128, !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr)
  ^bb4:  // pred: ^bb2
    %48 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>)>
    %49 = llvm.insertvalue %4, %48[0] : !llvm.struct<(i1, array<0 x i8>)> 
    llvm.store %49, %20 {alignment = 8 : i64} : !llvm.struct<(i1, array<0 x i8>)>, !llvm.ptr
    llvm.br ^bb5(%26, %27, %29, %31, %20 : !llvm.struct<(ptr<i252>, i32, i32)>, i64, i128, !llvm.struct<(ptr<i248>, i32, i32)>, !llvm.ptr)
  ^bb5(%50: !llvm.struct<(ptr<i252>, i32, i32)>, %51: i64, %52: i128, %53: !llvm.struct<(ptr<i248>, i32, i32)>, %54: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
    %55 = llvm.load %54 {alignment = 8 : i64} : !llvm.ptr -> i1
    llvm.switch %55 : i1, ^bb6 [
      0: ^bb8,
      1: ^bb13
    ]
  ^bb6:  // pred: ^bb5
    llvm.cond_br %5, ^bb7, ^bb15
  ^bb7:  // pred: ^bb6
    llvm.unreachable
  ^bb8:  // pred: ^bb5
    %56 = llvm.load %54 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i1, ptr)>
    %57 = llvm.extractvalue %56[1] : !llvm.struct<(i1, ptr)> 
    %58 = llvm.load %57 {alignment = 8 : i64} : !llvm.ptr -> i248
    llvm.call @free(%57) : (!llvm.ptr) -> ()
    %59 = llvm.zext %58 : i248 to i252
    %60 = llvm.extractvalue %50[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %61 = llvm.extractvalue %50[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %62 = llvm.extractvalue %50[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %63 = llvm.icmp "uge" %61, %62 : i32
    llvm.cond_br %63, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %64 = llvm.add %62, %62  : i32
    %65 = llvm.intr.umax(%64, %3)  : (i32, i32) -> i32
    %66 = llvm.zext %65 : i32 to i64
    %67 = llvm.mul %66, %7  : i64
    %68 = llvm.bitcast %60 : !llvm.ptr<i252> to !llvm.ptr
    %69 = llvm.call @realloc(%68, %67) : (!llvm.ptr, i64) -> !llvm.ptr
    %70 = llvm.bitcast %69 : !llvm.ptr to !llvm.ptr<i252>
    %71 = llvm.insertvalue %70, %50[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %72 = llvm.insertvalue %65, %71[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    llvm.br ^bb11(%72, %70 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>)
  ^bb10:  // pred: ^bb8
    llvm.br ^bb11(%50, %60 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>)
  ^bb11(%73: !llvm.struct<(ptr<i252>, i32, i32)>, %74: !llvm.ptr<i252>):  // 2 preds: ^bb9, ^bb10
    llvm.br ^bb12
  ^bb12:  // pred: ^bb11
    %75 = llvm.getelementptr %74[%61] : (!llvm.ptr<i252>, i32) -> !llvm.ptr, i252
    llvm.store %59, %75 : i252, !llvm.ptr
    %76 = llvm.add %61, %6  : i32
    %77 = llvm.insertvalue %76, %73[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %78 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>
    %79 = llvm.insertvalue %53, %78[0] : !llvm.struct<(struct<(ptr<i248>, i32, i32)>)> 
    %80 = llvm.load %13 : !llvm.ptr -> i64
    %81 = llvm.add %80, %2  : i64
    llvm.store %81, %13 : i64, !llvm.ptr
    llvm.br ^bb1(%51, %52, %79, %77 : i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb13:  // pred: ^bb5
    %82 = llvm.mlir.undef : !llvm.struct<()>
    %83 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>
    %84 = llvm.insertvalue %50, %83[0] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %85 = llvm.insertvalue %82, %84[1] : !llvm.struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)> 
    %86 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>
    %87 = llvm.insertvalue %5, %86[0] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    %88 = llvm.insertvalue %85, %87[1] : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)> 
    llvm.store %88, %21 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<(ptr<i252>, i32, i32)>, struct<()>)>)>, !llvm.ptr
    %89 = llvm.load %21 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    %90 = llvm.mlir.undef : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %91 = llvm.insertvalue %51, %90[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %92 = llvm.insertvalue %52, %91[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %93 = llvm.insertvalue %89, %92[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.return %93 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
  ^bb14(%94: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb1
    %95 = llvm.extractvalue %94[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %96 = llvm.bitcast %95 : !llvm.ptr<i252> to !llvm.ptr
    llvm.call @free(%96) : (!llvm.ptr) -> ()
    %97 = llvm.mlir.null : !llvm.ptr<i252>
    %98 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %99 = llvm.insertvalue %97, %98[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %100 = llvm.insertvalue %8, %99[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %101 = llvm.insertvalue %8, %100[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %102 = llvm.bitcast %97 : !llvm.ptr<i252> to !llvm.ptr
    %103 = llvm.call @realloc(%102, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %104 = llvm.bitcast %103 : !llvm.ptr to !llvm.ptr<i252>
    %105 = llvm.insertvalue %104, %101[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %106 = llvm.insertvalue %3, %105[2] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %107 = llvm.getelementptr %104[0] : (!llvm.ptr<i252>) -> !llvm.ptr, i252
    llvm.store %1, %107 : i252, !llvm.ptr
    %108 = llvm.insertvalue %6, %106[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %109 = llvm.mlir.undef : !llvm.struct<()>
    %110 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %111 = llvm.insertvalue %109, %110[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %112 = llvm.insertvalue %108, %111[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %113 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %114 = llvm.insertvalue %4, %113[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %115 = llvm.insertvalue %112, %114[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    llvm.store %115, %19 {alignment = 8 : i64} : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>, !llvm.ptr
    %116 = llvm.load %19 {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(i64, array<16 x i8>)>
    %117 = llvm.mlir.undef : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    %118 = llvm.insertvalue %27, %117[0] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %119 = llvm.insertvalue %29, %118[1] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    %120 = llvm.insertvalue %116, %119[2] : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)> 
    llvm.return %120 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
  ^bb15:  // pred: ^bb6
    %121 = llvm.mlir.addressof @assert_msg_8 : !llvm.ptr
    llvm.call @puts(%121) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @"_mlir_ciface_core::array::serialize_array_helper::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>(f23)"(%arg0: !llvm.ptr, %arg1: i64, %arg2: i128, %arg3: !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, %arg4: !llvm.struct<(ptr<i252>, i32, i32)>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"core::array::serialize_array_helper::<core::bytes_31::bytes31, core::serde::into_felt252_based::SerdeImpl::<core::bytes_31::bytes31, core::bytes_31::bytes31Copy, core::bytes_31::Bytes31IntoFelt252, core::bytes_31::Felt252TryIntoBytes31>, core::bytes_31::bytes31Drop>(f23)"(%arg1, %arg2, %arg3, %arg4) : (i64, i128, !llvm.struct<(struct<(ptr<i248>, i32, i32)>)>, !llvm.struct<(ptr<i252>, i32, i32)>) -> !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(i64, i128, struct<(i64, array<16 x i8>)>)>, !llvm.ptr
    llvm.return
  }
}
