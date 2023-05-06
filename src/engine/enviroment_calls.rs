use super::Engine;

pub type EnviromentCall = fn(&mut Engine) -> Result<&mut Engine, u64>;
