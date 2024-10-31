use {crate::reg::Reg, alloc::vec::Vec, core::ops::Range};

type Nid = u16;

pub trait Ctx {
    fn uses_of(&self, nid: Nid) -> impl Iterator<Item = Nid>;
    fn params_of(&self, nid: Nid) -> impl Iterator<Item = Nid>;
    fn args_of(&self, nid: Nid) -> impl Iterator<Item = Nid>;
    fn dom_of(&self, nid: Nid) -> Nid;
}

pub struct Env<'a, C: Ctx> {
    ctx: &'a C,
    func: &'a Func,
    res: &'a mut Res,
}

impl<'a, C: Ctx> Env<'a, C> {
    pub fn new(ctx: &'a C, func: &'a Func, res: &'a mut Res) -> Self {
        Self { ctx, func, res }
    }

    pub fn run(&mut self) {
        self.res.reg_to_node.clear();
        self.res.reg_to_node.resize(self.func.instrs.len(), 0);

        let mut bundle = Bundle::new(self.func.instrs.len());
        for &inst in &self.func.instrs {
            for uinst in self.ctx.uses_of(inst) {
                let mut cursor = self.ctx.dom_of(uinst);
                while cursor != self.ctx.dom_of(inst) {
                    let mut range = self.func.blocks
                        [self.func.id_to_block[cursor as usize] as usize]
                        .range
                        .clone();
                    range.start = range.start.max(inst as usize);
                    range.end = range.end.min(uinst as usize);
                    bundle.add(range);
                    cursor = self.ctx.dom_of(cursor);
                }
            }

            match self.res.bundles.iter_mut().enumerate().find(|(_, b)| !b.overlaps(&bundle)) {
                Some((i, other)) => {
                    other.merge(&bundle);
                    bundle.clear();
                    self.res.reg_to_node[inst as usize] = i as Reg;
                }
                None => {
                    self.res.reg_to_node[inst as usize] = self.res.bundles.len() as Reg;
                    self.res.bundles.push(bundle);
                    bundle = Bundle::new(self.func.instrs.len());
                }
            }
        }
    }
}

pub struct Res {
    bundles: Vec<Bundle>,
    pub reg_to_node: Vec<Reg>,
}

pub struct Bundle {
    //unit_range: Range<usize>,
    //set: BitSet,
    taken: Vec<bool>,
}

impl Bundle {
    fn new(size: usize) -> Self {
        Self { taken: vec![false; size] }
    }

    fn add(&mut self, range: Range<usize>) {
        self.taken[range].fill(true);
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.taken.iter().zip(other.taken.iter()).any(|(a, b)| a & b)
    }

    fn merge(&mut self, other: &Self) {
        debug_assert!(!self.overlaps(other));
        self.taken.iter_mut().zip(other.taken.iter()).for_each(|(a, b)| *a = *b);
    }

    fn clear(&mut self) {
        self.taken.fill(false);
    }

    //fn overlaps(&self, other: &Self) -> bool {
    //    if self.unit_range.start >= other.unit_range.end
    //        || self.unit_range.end <= other.unit_range.start
    //    {
    //        return false;
    //    }

    //    let [mut a, mut b] = [self, other];
    //    if a.unit_range.start > b.unit_range.start {
    //        mem::swap(&mut a, &mut b);
    //    }
    //    let [mut tmp_a, mut tmp_b] = [0; 2];
    //    let [units_a, units_b] = [a.set.units(&mut tmp_a), b.set.units(&mut tmp_b)];
    //    let len = a.unit_range.len().min(b.unit_range.len());
    //    let [units_a, units_b] =
    //        [&units_a[b.unit_range.start - a.unit_range.start..][..len], &units_b[..len]];
    //    units_a.iter().zip(units_b).any(|(&a, &b)| a & b != 0)
    //}

    //fn merge(mut self, mut other: Self) -> Self {
    //    debug_assert!(!self.overlaps(&other));

    //    if self.unit_range.start > other.unit_range.start {
    //        mem::swap(&mut self, &mut other);
    //    }

    //    let final_range = self.unit_range.start..self.unit_range.end.max(other.unit_range.end);

    //    self.set.reserve(final_range.len());

    //    let mut tmp = 0;
    //    let other_units = other.set.units(&mut tmp);

    //    match self.set.units_mut() {
    //        Ok(units) => {
    //            units[other.unit_range.start - self.unit_range.start..]
    //                .iter_mut()
    //                .zip(other_units)
    //                .for_each(|(a, b)| *a |= b);
    //        }
    //        Err(view) => view.add_mask(tmp),
    //    }

    //    self
    //}
}

pub struct Func {
    pub blocks: Vec<Block>,
    pub instrs: Vec<Nid>,
    pub id_to_instr: Vec<Nid>,
    pub id_to_block: Vec<Nid>,
}

pub struct Block {
    pub range: Range<usize>,
    pub start_id: Nid,
    pub eld_id: Nid,
}
