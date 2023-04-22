pub struct Memory {
    //TODO: hashmap with the start bytes as key and end bytes as offset
}
impl Memory {
    pub fn read_addr8(&mut self, address: u64) -> u8 {
        0
    }
    pub fn read_addr64(&mut self, address: u64) -> u64 {
        0
    }

    pub fn set_addr8(&mut self, address: u64, value: u8) -> u8 {
        0
    }
    pub fn set_addr64(&mut self, address: u64, value: u64) -> u64 {
        0
    }
}
