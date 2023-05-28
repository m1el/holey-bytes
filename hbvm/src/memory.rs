use crate::engine::VMPage;

use {
    crate::{engine::Page, RuntimeErrors},
    alloc::vec::Vec,
    hashbrown::HashMap,
    log::trace,
};

pub struct Memory {
    inner: HashMap<u64, Page>,
}

impl Memory {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
        //
    }

    pub fn map_vec(&mut self, address: u64, vec: Vec<u8>) {
        panic!("Mapping vectors into pages is not supported yet");
    }
}

impl Memory {
    pub fn read_addr8(&mut self, address: u64) -> Result<u8, RuntimeErrors> {
        let (page, offset) = addr_to_page(address);
        trace!("page {} offset {}", page, offset);
        match self.inner.get(&page) {
            Some(page) => {
                let val = page.data()[offset as usize];
                trace!("Value {}", val);
                Ok(val)
            }
            None => {
                trace!("page not mapped");
                Err(RuntimeErrors::PageNotMapped(page))
            }
        }
    }
    pub fn read_addr64(&mut self, address: u64) -> u64 {
        unimplemented!()
    }

    pub fn set_addr8(&mut self, address: u64, value: u8) -> Result<(), RuntimeErrors> {
        let (page, offset) = addr_to_page(address);
        let ret: Option<(&u64, &mut Page)> = self.inner.get_key_value_mut(&page);
        match ret {
            Some((_, page)) => {
                page.data()[offset as usize] = value;
            }
            None => {
                let mut pg = VMPage::default();
                pg.data[offset as usize] = value;
                self.inner.insert(page, Page::VMPage(pg));
                trace!("Mapped page {}", page);
            }
        }
        Ok(())
    }
    pub fn set_addr64(&mut self, address: u64, value: u64) -> Result<(), RuntimeErrors> {
        unimplemented!()
    }
}

fn addr_to_page(addr: u64) -> (u64, u64) {
    (addr / 8192, addr % 8192)
}
