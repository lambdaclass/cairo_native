func.func @print(%0 : !llvm.struct<(i32, i32, !llvm.ptr)>) -> () {
    // Allocate buffer.
    %1 = memref.alloca() : memref<126xi8>

    // Copy "[DEBUG] ".
    %2 = memref.get_global @lit0 : memref<8xi8>
    %3 = memref.subview %1[0][8][1] : memref<126xi8> to memref<8xi8>
    memref.copy %2, %3 : memref<8xi8> to memref<8xi8>

    // Copy " (raw: ".
    %4 = memref.get_global @lit1 : memref<7xi8>
    %5 = memref.subview %1[39][7][1] : memref<126xi8> to memref<7xi8, strided<[1], offset: 39>>
    memref.copy %4, %5 : memref<7xi8> to memref<7xi8, strided<[1], offset: 39>>

    // For each element in the array:
    %6 = index.constant 0
    %7 = llvm.extractvalue %0[0] : !llvm.struct<(i32, i32, !llvm.ptr)>
    %8 = index.castu %7 : i32 to index
    %9 = index.constant 1
    scf.for %10 = %6 to %8 step %9 {
        // Load element to print.
        %11 = llvm.extractvalue %0[2] : !llvm.struct<(i32, i32, !llvm.ptr)>

        %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
        %13 = arith.constant 5 : i64
        %14 = index.castu %10 : index to i64
        %15 = arith.shli %14, %13 : i64
        %16 = arith.addi %12, %15 : i64
        %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
        %18 = llvm.load %17 : !llvm.ptr -> i256

        // Copy string value replacing zeros with spaces.
        %19 = index.constant 32
        %20 = memref.subview %1[8][32][1] : memref<126xi8> to memref<32xi8, strided<[1], offset: 8>>
        scf.for %21 = %6 to %19 step %9 {
            // Compute byte to write.
            %22 = index.constant 3
            %23 = index.shl %21, %22
            %24 = index.castu %23 : index to i256
            %25 = arith.shrui %18, %24 : i256
            %26 = arith.trunci %25 : i256 to i8

            // Map null byte (0) to ' '.
            %27 = arith.constant 0 : i8
            %28 = arith.cmpi eq, %26, %27 : i8
            %29 = arith.constant 32 : i8
            %30 = arith.select %28, %29, %26 : i8

            // Write byte into the buffer.
            %31 = index.constant 30
            %32 = index.sub %31, %21
            memref.store %30, %20[%32] : memref<32xi8, strided<[1], offset: 8>>
        }

    //     // Run algorithm to write decimal value.
    //     %36 = memref.alloca() : memref<77xi8>
    //     // %37 = func.call @felt252_bin2dec(%36, %21) : (memref<77xi8>, i256) -> index
    //     %37 = index.constant 0

    //     // Copy the result.
    //     %38 = index.constant 46
    //     %39 = index.constant 76
    //     %40 = index.sub %39, %37
    //     %41 = memref.subview %36[%37][%40][%4] : memref<77xi8> to memref<?xi8, strided<[1], offset: ?>>
    //     %42 = memref.subview %1[%38][%40][%4] : memref<126xi8> to memref<?xi8, strided<[1], offset: ?>>
    //     %43 = memref.cast %41 : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8>
    //     %44 = memref.cast %42 : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8>
    //     memref.copy %43, %44 : memref<?xi8> to memref<?xi8>

    //     // Copy ")\0".
    //     %45 = index.add %38, %40
    //     %46 = index.constant 2
    //     %47 = memref.subview %1[%45][%46][%4] : memref<126xi8> to memref<?xi8, strided<[1], offset: ?>>
    //     %48 = memref.cast %47 : memref<?xi8, strided<[1], offset: ?>> to memref<2xi8>
    //     %49 = memref.get_global @lit2 : memref<2xi8>
    //     memref.copy %49, %48 : memref<2xi8> to memref<2xi8>

    //     // TODO: Call `puts()`.
    //     %t3 = arith.constant 0 : i8
    //     %t4 = index.constant 125
    //     memref.store %t3, %1[%t4] : memref<126xi8>
    //     %t0 = memref.extract_aligned_pointer_as_index %1 : memref<126xi8> -> index
    //     %t1 = index.castu %t0 : index to i64
    //     %t2 = llvm.inttoptr %t1 : i64 to !llvm.ptr
    //     func.call @puts(%t2) : (!llvm.ptr) -> i32
    }

    func.return
}

func.func @felt252_bin2dec(%0 : memref<77xi8>, %1 : i256) -> index {
    // // Clear the buffer to zero.
    // %2 = memref.extract_aligned_pointer_as_index %0 : memref<77xi8> -> index
    // %3 = index.castu %2 : index to i64
    // %4 = llvm.inttoptr %3 : i64 to !llvm.ptr<i8>
    // %5 = arith.constant 0 : i8
    // %6 = arith.constant 77 : i32
    // %7 = arith.constant 0 : i1
    // "llvm.intr.memset"(%4, %5, %6, %7) : (!llvm.ptr<i8>, i8, i32, i1) -> ()

    // // Count number of bits required.
    // %8 = math.ctlz %1 : i256
    // %9 = arith.trunci %8 : i256 to i8
    // %10 = arith.subi %5, %9 : i8

    // // Handle special case: zero.
    // %11 = arith.constant 0 : i8
    // %12 = arith.cmpi eq, %10, %11 : i8
    // %13 = scf.if %12 -> index {
    //     // Write a zero at the end and return the index.
    //     %14 = arith.constant 48 : i8
    //     %15 = index.constant 75
    //     memref.store %14, %0[%15] : memref<77xi8>
    //     scf.yield %15 : index
    // } else {
    //     // For each (required) bit in MSB to LSB order:
    //     %16 = index.constant 0
    //     %17 = index.castu %10 : i8 to index
    //     %18 = index.constant 1
    //     scf.for %19 = %16 to %17 step %18 {
    //         %20 = index.sub %17, %18
    //         %21 = index.sub %20, %19
    //         %22 = index.castu %21 : index to i256
    //         %23 = arith.shrui %1, %22 : i256
    //         %24 = arith.trunci %23 : i256 to i1

    //         // Shift & add.
    //         %25 = index.constant 76
    //         scf.for %26 = %16 to %25 step %18 iter_args(%27 = %24) -> i1 {
    //             // Load byte.
    //             %28 = index.constant 75
    //             %29 = index.sub %28, %26
    //             %30 = memref.load %0[%29] : memref<77xi8>

    //             // Add 3 if value >= 5.
    //             %31 = arith.constant 5 : i8
    //             %32 = arith.cmpi uge, %30, %31 : i8
    //             %33 = arith.constant 3 : i8
    //             %34 = arith.addi %30, %33 : i8
    //             %35 = arith.select %32, %34, %30 : i8

    //             // Shift 1 bit to the left.
    //             %36 = arith.constant 1 : i8
    //             %37 = arith.shli %35, %36 : i8

    //             // Insert carry-in bit and truncate to 4 bits.
    //             %38 = llvm.zext %27 : i1 to i8
    //             %39 = arith.ori %37, %38 : i8
    //             %40 = arith.constant 15 : i8
    //             %41 = arith.andi %39, %40 : i8

    //             // Store byte.
    //             memref.store %41, %0[%29] : memref<77xi8>

    //             // Extract carry and send it to the next iteration.
    //             %42 = arith.shrui %35, %33 : i8
    //             %43 = arith.trunci %42 : i8 to i1
    //             scf.yield %43 : i1
    //         }
    //     }

    //     // Find first non-zero digit index.
    //     %44 = scf.while (%45 = %16) : (index) -> (index) {
    //         %46 = memref.load %0[%45] : memref<77xi8>
    //         %47 = arith.cmpi eq, %46, %5 : i8
    //         scf.condition(%47) %45 : index
    //     } do {
    //     ^0(%48 : index):
    //         %49 = index.add %48, %18
    //         scf.yield %49 : index
    //     }

    //     // Convert BCD to ascii digits.
    //     %50 = index.constant 76
    //     scf.for %51 = %44 to %50 step %18 {
    //         %52 = memref.load %0[%51] : memref<77xi8>
    //         %53 = arith.constant 48 : i8
    //         %54 = arith.addi %52, %53 : i8
    //         memref.store %54, %0[%51] : memref<77xi8>
    //     }

    //     scf.yield %44 : index
    // }

    // // Return the first digit offset.
    // return %13 : index

    %t = index.constant 0
    return %t : index
}

func.func private @puts(!llvm.ptr) -> i32

memref.global "private" constant @lit0 : memref<8xi8> = dense<[91, 68, 69, 66, 85, 71, 93, 32]>
memref.global "private" constant @lit1 : memref<7xi8> = dense<[32, 40, 114, 97, 119, 58, 32]>
memref.global "private" constant @lit2 : memref<2xi8> = dense<[41, 0]>




// func.func @main() {
//     %0 = memref.get_global @values : memref<2xi256>
//     %1 = memref.extract_aligned_pointer_as_index %0 : memref<2xi256> -> index
//     %2 = index.castu %1 : index to i64
//     %3 = llvm.inttoptr %2 : i64 to !llvm.ptr

//     %4 = llvm.mlir.undef : !llvm.struct<(i32, i32, !llvm.ptr)>
//     %5 = arith.constant 2 : i32
//     %6 = llvm.insertvalue %5, %4[0] : !llvm.struct<(i32, i32, !llvm.ptr)>
//     %7 = llvm.insertvalue %5, %6[1] : !llvm.struct<(i32, i32, !llvm.ptr)>
//     %8 = llvm.insertvalue %3, %7[2] : !llvm.struct<(i32, i32, !llvm.ptr)>

//     func.call @print(%8) : (!llvm.struct<(i32, i32, !llvm.ptr)>) -> ()
//     func.return
// }

// memref.global "private" constant @values : memref<2xi256> = dense<[
//     5735816763073854953388147237921,
//     0
// ]>


// // memref.global "private" constant @template : memref<46xi8> = dense<[
// //     91, 68, 69, 66, 85, 71, 93, 32,
// //     32, 32, 32, 32, 32, 32, 32, 32,
// //     32, 32, 32, 32, 32, 32, 32, 32,
// //     32, 32, 32, 32, 32, 32, 32, 32,
// //     32, 32, 32, 32, 32, 32, 32, 32,
// //     40, 114, 97, 119, 58, 32
// // ]>
