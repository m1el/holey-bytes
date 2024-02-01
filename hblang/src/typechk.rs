use crate::lexer::Ty;

#[derive(Clone, Debug)]
pub enum Type {
    Builtin(Ty),
    Struct(StructType),
    Pointer(Box<Type>),
}

#[derive(Clone, Debug)]
pub struct StructType {
    pub name:   String,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub ty:   Type,
}
