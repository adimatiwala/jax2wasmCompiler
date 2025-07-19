mod hlo_parser;
mod hlo2wasm;
mod memory;
mod runtime_helpers;
mod simd;

use std::fs;
use std::path::Path;

struct CompilationTarget {
    name: &'static str,
    input_hlo: &'static str,
    output_wat: &'static str,
    output_wasm: &'static str,
}

fn main() {
    let targets = vec![
        CompilationTarget {
            name: "Forward Pass",
            input_hlo: "forward_optimized.hlo",
            output_wat: "target/forward.wat",
            output_wasm: "target/forward.wasm",
        },
        CompilationTarget {
            name: "Gradient Pass", 
            input_hlo: "grad_optimized.hlo",
            output_wat: "target/gradient.wat",
            output_wasm: "target/gradient.wasm",
        },
    ];

    std::fs::create_dir_all("target").expect("Failed to create target directory");

    for target in targets {
        println!("Compiling {} from {}", target.name, target.input_hlo);
        
        if !Path::new(target.input_hlo).exists() {
            println!("{} not found, skipping", target.input_hlo);
            continue;
        }

        let hlo = fs::read_to_string(target.input_hlo)
            .expect(&format!("Failed to read {}", target.input_hlo));
        
        let wat = hlo2wasm::generate_wat(&hlo, target.name);
        
        fs::write(target.output_wat, &wat)
            .expect(&format!("Failed to write {}", target.output_wat));
        
        println!("{} WAT written to {}", target.name, target.output_wat);
        
        compile_to_wasm(target.output_wat, target.output_wasm);
    }

    println!("\nCompilation complete! Both forward and gradient passes ready.");
}

fn compile_to_wasm(wat_path: &str, wasm_path: &str) {
    use std::process::Command;
    
    match Command::new("wat2wasm")
        .arg(wat_path)
        .arg("-o")
        .arg(wasm_path)
        .output()
    {
        Ok(_) => println!("Binary WASM written to {}", wasm_path),
        Err(_) => println!("wat2wasm not found, skipping binary compilation"),
    }
}
