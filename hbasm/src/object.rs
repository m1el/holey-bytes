//! Code object

use {rhai::ImmutableString, std::collections::HashMap};

/// Section tabel
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Text,
    Data,
}

/// Symbol entry (in what section, where)
#[derive(Clone, Copy, Debug)]
pub struct SymbolEntry {
    pub location: Section,
    pub offset:   usize,
}

/// Relocation table key
#[derive(Clone, Debug)]
pub enum RelocKey {
    /// Direct reference
    Symbol(usize),
    /// Indirect reference
    Label(ImmutableString),
}

/// Relocation type
#[derive(Clone, Copy, Debug)]
pub enum RelocType {
    Rel32,
    Rel16,
    Abs64,
}

/// Relocation table entry
#[derive(Clone, Debug)]
pub struct RelocEntry {
    pub key: RelocKey,
    pub ty:  RelocType,
}

/// Object code
#[derive(Clone, Debug, Default)]
pub struct Sections {
    pub text: Vec<u8>,
    pub data: Vec<u8>,
}

/// Object
#[derive(Clone, Debug, Default)]
pub struct Object {
    /// Vectors with sections
    pub sections: Sections,
    /// Symbol table
    pub symbols:  Vec<Option<SymbolEntry>>,
    /// Labels to symbols table
    pub labels:   HashMap<ImmutableString, usize>,
    /// Relocation table
    pub relocs:   HashMap<usize, RelocEntry>,
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct SymbolRef(pub usize);

impl Object {
    /// Insert symbol at current location in specified section
    pub fn symbol(&mut self, section: Section) -> SymbolRef {
        let section_buf = match section {
            Section::Text => &mut self.sections.text,
            Section::Data => &mut self.sections.data,
        };

        self.symbols.push(Some(SymbolEntry {
            location: section,
            offset:   section_buf.len(),
        }));

        SymbolRef(self.symbols.len() - 1)
    }

    /// Insert to relocation table and write zeroes to code
    pub fn relocation(&mut self, key: RelocKey, ty: RelocType) {
        self.relocs
            .insert(self.sections.text.len(), RelocEntry { key, ty });

        self.sections.text.extend(match ty {
            RelocType::Rel32 => &[0_u8; 4] as &[u8],
            RelocType::Rel16 => &[0; 2],
            RelocType::Abs64 => &[0; 8],
        });
    }
}
