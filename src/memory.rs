use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub offset: u32,
    pub size_bytes: u32,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub is_parameter: bool,
}

pub struct MemoryManager {
    pub base_offset: u32,
    pub current_offset: u32,
    pub tensors: HashMap<String, TensorInfo>,
    pub parameter_count: u32,
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            base_offset: 2048, // Start after WASM stack space
            current_offset: 2048,
            tensors: HashMap::new(),
            parameter_count: 0,
        }
    }

    fn align_to(&mut self, alignment: u32) {
        if self.current_offset % alignment != 0 {
            self.current_offset += alignment - (self.current_offset % alignment);
        }
    }

    fn bytes_per_element(dtype: &str) -> u32 {
        match dtype {
            "f32" => 4,
            "f64" => 8,
            "i32" => 4,
            "i64" => 8,
            "i8" => 1,
            "i16" => 2,
            _ => 4, // Default to f32
        }
    }

    pub fn allocate_tensor(&mut self, name: &str, dtype: &str, shape: &[usize], is_parameter: bool) -> u32 {
        let element_count = if shape.is_empty() { 1 } else { shape.iter().product() };
        let size_bytes = element_count as u32 * Self::bytes_per_element(dtype);
        
        // Align to 16 bytes for SIMD operations
        self.align_to(16);
        
        let offset = self.current_offset;
        
        if is_parameter {
            self.parameter_count += 1;
        }
        
        self.tensors.insert(
            name.to_string(),
            TensorInfo {
                offset,
                size_bytes,
                shape: shape.to_vec(),
                dtype: dtype.to_string(),
                is_parameter,
            },
        );
        
        self.current_offset += size_bytes;
        offset
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn get_offset(&self, name: &str) -> Option<u32> {
        self.tensors.get(name).map(|info| info.offset)
    }

    pub fn get_total_memory_size(&self) -> u32 {
        self.current_offset - self.base_offset
    }

    pub fn get_parameters(&self) -> Vec<(&String, &TensorInfo)> {
        self.tensors
            .iter()
            .filter(|(_, info)| info.is_parameter)
            .collect()
    }

    pub fn reset(&mut self) {
        self.current_offset = self.base_offset;
        self.tensors.clear();
        self.parameter_count = 0;
    }
}
