use hashbrown::HashMap;
use log::trace;

use crate::{engine::Page, RuntimeErrors};

pub struct Memory {
    //TODO: hashmap with the start bytes as key and end bytes as offset
    inner: HashMap<u64, Page>,
}

impl Memory {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn map_vec(&mut self, address: u64, vec: Vec<u8>) {}
}

impl Memory {
    pub fn read_addr8(&mut self, address: u64) -> Result<u8, RuntimeErrors> {
        let (page, offset) = addr_to_page(address);
        println!("page {} offset {}", page, offset);
        match self.inner.get(&page) {
            Some(page) => {
                let val = page.data[offset as usize];
                Ok(val)
            }
            None => Err(RuntimeErrors::PageUnmapped(page)),
        }
        // trace!("Value read {} from page {} offset {}", val, page, offset);
    }
    pub fn read_addr64(&mut self, address: u64) -> u64 {
        unimplemented!()
    }

    pub fn set_addr8(&mut self, address: u64, value: u8) -> Result<(), RuntimeErrors> {
        let (page, offset) = addr_to_page(address);
        let ret: Option<(&u64, &mut Page)> = self.inner.get_key_value_mut(&page);
        match ret {
            Some((_, page)) => {
                page.data[offset as usize] = value;
            }
            None => {
                let mut pg = Page::new();
                pg.data[offset as usize] = value;
                self.inner.insert(page, pg);
                println!("Mapped page {}", page);
            }
        }
        Ok(())
    }
    pub fn set_addr64(&mut self, address: u64, value: u64) -> u64 {
        unimplemented!()
    }
}

fn addr_to_page(addr: u64) -> (u64, u64) {
    (addr / 8192, addr % 8192)
}
