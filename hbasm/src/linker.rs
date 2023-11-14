use {
    crate::{
        object::{RelocKey, RelocType, Section},
        SharedObject,
    },
    std::io::Write,
};

pub fn link(object: SharedObject, out: &mut impl Write) -> std::io::Result<()> {
    let obj = &mut *object.borrow_mut();
    for (&loc, entry) in &obj.relocs {
        let value = match &entry.key {
            RelocKey::Symbol(sym) => obj.symbols[*sym],
            RelocKey::Label(label) => obj.symbols[obj.labels[label]],
        }
        .ok_or_else(|| std::io::Error::other("Invalid symbol"))?;

        let offset = match value.location {
            Section::Text => value.offset,
            Section::Data => value.offset + obj.sections.text.len(),
        };

        match entry.ty {
            RelocType::Rel32 => obj.sections.text[loc..loc + 4]
                .copy_from_slice(&((offset as isize - loc as isize) as i32).to_le_bytes()),
            RelocType::Rel16 => obj.sections.text[loc..loc + 2]
                .copy_from_slice(&((offset as isize - loc as isize) as i16).to_le_bytes()),
            RelocType::Abs64 => obj.sections.text[loc..loc + 8]
                .copy_from_slice(&(offset as isize - loc as isize).to_le_bytes()),
        }
    }

    out.write_all(&obj.sections.text)?;
    out.write_all(&obj.sections.data)
}
