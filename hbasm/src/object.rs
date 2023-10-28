use {rhai::ImmutableString, std::collections::HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Text,
    Data,
}

#[derive(Clone, Copy, Debug)]
pub struct SymbolEntry {
    pub location: Section,
    pub offset:   usize,
}

#[derive(Clone, Debug)]
pub enum RelocKey {
    Symbol(usize),
    Label(ImmutableString),
}

#[derive(Clone, Copy, Debug)]
pub enum RelocType {
    Rel32,
    Rel16,
    Abs64,
}

#[derive(Clone, Debug)]
pub struct RelocEntry {
    pub key: RelocKey,
    pub ty:  RelocType,
}

#[derive(Clone, Debug, Default)]
pub struct Sections {
    pub text: Vec<u8>,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct Object {
    pub sections: Sections,
    pub symbols:  Vec<Option<SymbolEntry>>,
    pub labels:   HashMap<ImmutableString, usize>,
    pub relocs:   HashMap<usize, RelocEntry>,
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct SymbolRef(pub usize);

impl Object {
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
