pub struct EngineConfig {
    pub call_stack_depth: usize,
    pub quantum: u32,
}

impl EngineConfig {
    pub fn default() -> Self {
        Self {
            call_stack_depth: 32,
            quantum: 0,
        }
    }
}
