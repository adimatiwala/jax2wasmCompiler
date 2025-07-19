pub fn simd_add_helper() -> String {
    r#"
  (func $simd_add (param $a i32) (param $b i32) (param $out i32) (param $len i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop $simd_loop
      (br_if $exit (i32.ge_s (local.get $i) (local.get $len)))
      ;; Process 4 floats at once using SIMD
      (v128.store 
        (i32.add (local.get $out) (i32.mul (local.get $i) (i32.const 4)))
        (f32x4.add
          (v128.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (v128.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))
        )
      )
      (local.set $i (i32.add (local.get $i) (i32.const 4))) ;; Skip by 4 floats
      (br $simd_loop)
    )
    $exit
  )
    "#.to_string()
}

pub fn simd_mul_helper() -> String {
    r#"
  (func $simd_mul (param $a i32) (param $b i32) (param $out i32) (param $len i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop $simd_loop
      (br_if $exit (i32.ge_s (local.get $i) (local.get $len)))
      (v128.store 
        (i32.add (local.get $out) (i32.mul (local.get $i) (i32.const 4)))
        (f32x4.mul
          (v128.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (v128.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))
        )
      )
      (local.set $i (i32.add (local.get $i) (i32.const 4)))
      (br $simd_loop)
    )
    $exit
  )

  (func $simd_reduce_sum (param $ptr i32) (param $len i32) (result f32)
    (local $i i32) (local $sum v128) (local $scalar_sum f32)
    (local.set $sum (v128.const f32x4 0 0 0 0))
    (local.set $i (i32.const 0))
    
    ;; SIMD reduction loop
    (loop $simd_reduce_loop
      (br_if $simd_done (i32.ge_s (local.get $i) (local.get $len)))
      (local.set $sum
        (f32x4.add (local.get $sum)
          (v128.load (i32.add (local.get $ptr) (i32.mul (local.get $i) (i32.const 4))))
        )
      )
      (local.set $i (i32.add (local.get $i) (i32.const 4)))
      (br $simd_reduce_loop)
    )
    
    ;; Extract and sum the 4 components
    (local.set $scalar_sum
      (f32.add
        (f32.add
          (f32x4.extract_lane 0 (local.get $sum))
          (f32x4.extract_lane 1 (local.get $sum))
        )
        (f32.add
          (f32x4.extract_lane 2 (local.get $sum))
          (f32x4.extract_lane 3 (local.get $sum))
        )
      )
    )
    
    (return (local.get $scalar_sum))
    $simd_done
  )
    "#.to_string()
}
