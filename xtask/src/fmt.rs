use {
    crate::{utils::IterExt, ROOT},
    argh::FromArgs,
    std::{
        fs::File,
        io::{self, BufRead, BufReader, BufWriter, Seek, Write},
    },
};

/// Format `instructions.in`
#[derive(Debug, FromArgs, PartialEq)]
#[argh(subcommand, name = "fmt")]
pub struct Command {
    /// renumber instructions in their definition order
    #[argh(switch, short = 'r')]
    renumber: bool,
}

pub fn command(args: Command) -> io::Result<()> {
    let mut file =
        File::options().read(true).write(true).open(ROOT.join("hbbytecode/instructions.in"))?;

    // Extract records
    let reader = BufReader::new(&file);
    let mut recs = vec![];
    let mut lens = [0_usize; 4];

    for rec in reader.split(b';').filter_map(|r| {
        r.map(|ln| {
            let s = String::from_utf8_lossy(&ln);
            let s = s.trim_matches('\n');
            if s.is_empty() {
                return None;
            }

            s.split(',').map(|s| Box::<str>::from(s.trim())).collect_array::<4>().map(Ok::<_, ()>)
        })
        .transpose()
    }) {
        let rec = rec?.expect("Valid record format");
        for (current, next) in lens.iter_mut().zip(rec.iter()) {
            *current = (*current).max(next.len());
        }

        recs.push(rec);
    }

    // Clear file!
    file.set_len(0)?;
    file.seek(std::io::SeekFrom::Start(0))?;

    let mut writer = BufWriter::new(file);

    let ord_opco_len = digit_count(recs.len()) as usize;
    for (n, rec) in recs.iter().enumerate() {
        // Write opcode number
        if args.renumber {
            let n = format!("{n:#04X}");
            write!(writer, "{n}, {}", padding(ord_opco_len, &n))?;
        } else {
            write!(writer, "{}, {}", rec[0], padding(lens[0], &rec[0]))?;
        }

        // Write other fields
        writeln!(
            writer,
            "{}, {}{},{} {}{};",
            rec[1],
            padding(lens[1], &rec[1]),
            rec[2],
            padding(lens[2], &rec[2]),
            rec[3],
            padding(lens[3], &rec[3]),
        )?;
    }

    Ok(())
}

fn padding(req: usize, s: &str) -> Box<str> {
    " ".repeat(req.saturating_sub(s.len())).into()
}

#[inline]
fn digit_count(n: usize) -> u32 {
    n.checked_ilog10().unwrap_or(0) + 1
}
