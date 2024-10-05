#![feature(array_windows)]
#![feature(write_all_vectored)]
use {
    aes_gcm::{
        aead::{self, AeadMutInPlace},
        AeadCore, Aes256Gcm, KeyInit,
    },
    ed25519_dalek::ed25519::signature::Signer,
    rand_core::OsRng,
    std::{
        collections::{HashMap, HashSet},
        fmt, fs,
        io::{self, IoSlice, IoSliceMut, Read, Write},
        mem::{self, MaybeUninit},
        net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream},
        path::PathBuf,
        slice,
        str::FromStr,
        sync::{atomic, OnceLock},
        time,
    },
    x25519_dalek::{EphemeralSecret, SharedSecret},
};

static CONN_COUNT: atomic::AtomicUsize = atomic::AtomicUsize::new(0);
static USER_DATA_DIR: OnceLock<PathBuf> = OnceLock::new();
static SERVER_SECRET: OnceLock<ed25519_dalek::SigningKey> = OnceLock::new();

#[derive(Default)]
struct Cli {
    program: String,
    args: Vec<String>,
    flags: HashSet<String>,
    options: HashMap<String, String>,
}

impl Cli {
    pub fn parse() -> Self {
        let mut s = Self::default();
        let mut args = std::env::args();
        s.program = args.next().unwrap();

        for arg in args {
            if let Some(arg) = arg.strip_prefix("--") {
                match arg.split_once('=') {
                    Some((name, value)) => _ = s.options.insert(name.to_owned(), value.to_owned()),
                    None => _ = s.flags.insert(arg.to_string()),
                }
            } else {
                s.args.push(arg);
            }
        }

        s
    }

    pub fn arg(&self, index: usize) -> &str {
        self.args.get(index).map_or("", String::as_str)
    }

    pub fn expect_option(&self, name: &str) -> &str {
        self.options.get(name).unwrap_or_else(|| panic!("--{name} is mandatory"))
    }

    pub fn expect_poption<T: FromStr<Err: fmt::Display>>(&self, name: &str) -> T {
        self.expect_option(name).parse::<T>().unwrap_or_else(|e| {
            panic!("failed to parse --{name} as {}: {e}", std::any::type_name::<T>())
        })
    }
}

type Subcommand<'a> = (&'a str, &'a str, fn(&Cli) -> io::Result<()>);

fn help<T>(subs: &[(&str, &str, T)]) -> io::Result<()> {
    for (name, desc, _) in subs {
        eprintln!("{name} - {desc}");
    }
    Err(io::ErrorKind::NotFound.into())
}

const SUBCOMMANDS: &[Subcommand] = &[
    ("help", "print command descriptions", |_| help(SUBCOMMANDS)),
    ("serve", "run the server", |cli| {
        let port = cli.expect_poption::<u16>("port");
        let max_conns = cli.expect_poption::<usize>("max-conns");
        USER_DATA_DIR.set(cli.expect_poption("user-data-path")).unwrap();
        SERVER_SECRET.set(cli.expect_poption::<HexSk>("secret").0).unwrap();
        let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, port)).unwrap();
        for incoming in listener.incoming() {
            match incoming {
                Ok(c) => {
                    if CONN_COUNT.fetch_add(1, atomic::Ordering::Relaxed) >= max_conns {
                        CONN_COUNT.fetch_sub(1, atomic::Ordering::Relaxed);
                        continue;
                    }

                    std::thread::spawn(move || {
                        _ = handle_client(c);
                        CONN_COUNT.fetch_sub(1, atomic::Ordering::Relaxed);
                    });
                }
                Err(e) => {
                    eprintln!("accepting conn conn: {e}")
                }
            }
        }
        Ok(())
    }),
    ("make-profile", "create profile file (private key + name)", |cli| {
        let name = cli.expect_option("name");
        let name = str_as_username(name)
            .ok_or(io::ErrorKind::InvalidData)
            .ctx("name is limmited to 32 characters")?;

        let &key = ed25519_dalek::SigningKey::generate(&mut rand_core::OsRng).as_bytes();
        let profile = UserProfile { name, key };

        let out_file = cli.expect_option("out-file");
        _ = fs::write(out_file, as_bytes(&profile)).ctx("while saving profile file");
        Ok(())
    }),
    ("consume", "connect to server and do an action", |cli| {
        let profile_path = cli.expect_option("profile");
        let mut profile_file = fs::File::open(profile_path).ctx("opening profile file")?;
        let profile: UserProfile = read_struct(&mut profile_file).ctx("reading the profile")?;
        let sx = x25519_dalek::EphemeralSecret::random_from_rng(OsRng);
        let auth = UserAuth::sign(profile, &sx);

        let addr = cli.expect_poption::<SocketAddrV4>("addr");
        let mut stream = TcpStream::connect(addr).ctx("creating connection to the server")?;
        write_struct(&mut stream, &auth).ctx("sending initial handshake packet")?;

        let HexPk(server_identity) = cli.expect_poption("server-identity");
        let sauth: ServerAuth = read_struct(&mut stream).ctx("reading server auth")?;
        let secret = sauth
            .verify(auth.pk, server_identity, sx)
            .map_err(|_| io::ErrorKind::PermissionDenied)
            .ctx("authenticating server")?;
        let stream = EncriptedStream::new(stream, secret);

        select_subcommand(1, CONSUME_SUBCOMMAND, cli)(cli, stream)
    }),
];

type ConsumeSubcommand<'a> = (&'a str, &'a str, fn(&Cli, EncriptedStream) -> io::Result<()>);

const CONSUME_SUBCOMMAND: &[ConsumeSubcommand] = &[
    ("help", "this help message", |_, _| help(CONSUME_SUBCOMMAND)),
    ("ping", "ping the server to check the connection", |_, mut stream| {
        write_struct(&mut stream, &Qid::Ping)
    }), //
];

fn hex_to_array<const SIZE: usize>(s: &str) -> Result<[u8; SIZE], &'static str> {
    let mut buf = [0u8; SIZE];

    if s.len() != SIZE * 2 {
        return Err("expected 64 character hex string");
    }

    fn byte_to_hex(val: u8) -> Result<u8, &'static str> {
        Ok(match val {
            b'0'..=b'9' => val - b'0',
            b'a'..=b'f' => val - b'a' + 10,
            b'A'..=b'F' => val - b'A' + 10,
            _ => return Err("invalid hex char"),
        })
    }

    for (dst, &[a, b]) in buf.iter_mut().zip(s.as_bytes().array_windows()) {
        *dst = byte_to_hex(a)? | (byte_to_hex(b)? << 4);
    }

    Ok(buf)
}

struct HexPk(ed25519_dalek::VerifyingKey);

impl std::str::FromStr for HexPk {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ed25519_dalek::VerifyingKey::from_bytes(&hex_to_array(s)?)
            .map_err(|_| "hex code does not represent the valid key")
            .map(Self)
    }
}

struct HexSk(ed25519_dalek::SigningKey);

impl std::str::FromStr for HexSk {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(ed25519_dalek::SigningKey::from_bytes(&hex_to_array(s)?)))
    }
}

fn select_subcommand<'a, T>(depth: usize, list: &'a [(&str, &str, T)], cli: &Cli) -> &'a T {
    &list.iter().find(|&&(name, ..)| name == cli.arg(depth)).unwrap_or(&list[0]).2
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    select_subcommand(0, SUBCOMMANDS, &cli)(&cli)
}

fn handle_client(mut stream: TcpStream) -> io::Result<()> {
    let (user, sec) = {
        let user_auth: UserAuth = read_struct(&mut stream).ctx("reading auth packet")?;
        let sx = x25519_dalek::EphemeralSecret::random_from_rng(OsRng);
        let pk = x25519_dalek::PublicKey::from(&sx);
        let user = UserData::load(&user_auth, sx).ctx("loading user data")?;
        let sauth = ServerAuth::sign(&user_auth, SERVER_SECRET.get().unwrap(), pk);
        write_struct(&mut stream, &sauth).ctx("sending handshare response")?;
        user
    };

    let mut stream = EncriptedStream::new(stream, sec);

    loop {
        match Qid::try_from(read_struct::<u16>(&mut stream)?)? {
            Qid::Ping => write_struct(&mut stream, &Aid::Pong)?,
        }
    }
}

#[repr(u16)]
enum Aid {
    Pong,
}

#[repr(u16)]
enum Qid {
    Ping,
}

impl TryFrom<u16> for Qid {
    type Error = io::ErrorKind;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        if value <= Self::Ping as u16 {
            Ok(unsafe { mem::transmute::<u16, Qid>(value) })
        } else {
            Err(io::ErrorKind::NotFound)
        }
    }
}

trait Ctx {
    fn ctx(self, label: &str) -> Self;
}

impl<O, E: fmt::Display> Ctx for Result<O, E> {
    fn ctx(self, label: &str) -> Self {
        if let Err(e) = &self {
            eprintln!("{label}: {e}")
        }
        self
    }
}

const ASOC_DATA: &[u8] = b"testicle torsion vizard";

struct EncriptedStream {
    inner: TcpStream,
    key: SharedSecret,
    buf: Vec<u8>,
}

impl EncriptedStream {
    fn new(inner: TcpStream, key: SharedSecret) -> Self {
        Self { inner, key, buf: Default::default() }
    }
}

impl Read for EncriptedStream {
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let mut tag = MaybeUninit::<aead::Tag<Aes256Gcm>>::uninit();
        let mut nonce = MaybeUninit::<aead::Nonce<Aes256Gcm>>::uninit();

        let mut bufs = &mut [
            IoSliceMut::new(as_mut_bytes(&mut tag)),
            IoSliceMut::new(as_mut_bytes(&mut nonce)),
            IoSliceMut::new(buf),
        ][..];

        loop {
            let red = self.inner.read_vectored(bufs)?;
            if red == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            IoSliceMut::advance_slices(&mut bufs, red);
            if bufs.is_empty() {
                break;
            }
        }

        unsafe {
            Aes256Gcm::new(self.key.as_bytes().into())
                .decrypt_in_place_detached(&nonce.assume_init(), ASOC_DATA, buf, &tag.assume_init())
                .map_err(|_| io::ErrorKind::PermissionDenied)?;
        }

        Ok(())
    }

    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        unimplemented!()
    }
}

impl Write for EncriptedStream {
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.buf.clear();
        self.buf.extend(buf);

        let nonce = Aes256Gcm::generate_nonce(OsRng);
        let tag = Aes256Gcm::new(self.key.as_bytes().into())
            .encrypt_in_place_detached(&nonce, ASOC_DATA, &mut self.buf)
            .unwrap();

        self.inner.write_all_vectored(&mut [
            IoSlice::new(as_bytes(&tag)),
            IoSlice::new(as_bytes(&nonce)),
            IoSlice::new(&self.buf),
        ])
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }

    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        unimplemented!()
    }
}

fn read_struct<T>(stream: &mut impl Read) -> io::Result<T> {
    let mut res = mem::MaybeUninit::uninit();
    stream.read_exact(as_mut_bytes(&mut res))?;
    unsafe { res.assume_init() }
}

fn write_struct<T>(stream: &mut impl Write, value: &T) -> io::Result<()> {
    stream.write_all(as_bytes(value))
}

fn as_mut_bytes<T>(value: &mut T) -> &mut [u8] {
    unsafe { slice::from_raw_parts_mut(value as *mut _ as *mut u8, mem::size_of::<T>()) }
}

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { slice::from_raw_parts(value as *const _ as *const u8, mem::size_of::<T>()) }
}

type Username = [u8; 32];
type Postname = [u8; 64];
type Pk = [u8; 32];
type Nonce = u64;

fn username_as_str(name: &Username) -> Option<&str> {
    let len = name.iter().rposition(|&b| b != 0xff)? + 1;
    std::str::from_utf8(&name[..len]).ok()
}

fn str_as_username(name: &str) -> Option<Username> {
    if name.len() > mem::size_of::<Username>() {
        return None;
    }
    let mut buff = [0xffu8; mem::size_of::<Username>()];
    buff[..name.len()].copy_from_slice(name.as_bytes());
    Some(buff)
}

#[repr(packed)]
struct UserProfile {
    name: Username,
    key: ed25519_dalek::SecretKey,
}

#[repr(packed)]
struct UserAuth {
    signature: ed25519_dalek::Signature,
    pk: Pk,
    x: x25519_dalek::PublicKey,
    name: Username,
    nonce: Nonce,
}

impl UserAuth {
    fn sign(UserProfile { name, key }: UserProfile, sx: &x25519_dalek::EphemeralSecret) -> Self {
        let nonce =
            time::SystemTime::now().duration_since(time::SystemTime::UNIX_EPOCH).unwrap().as_secs();
        let mut message = [0; mem::size_of::<Username>() + mem::size_of::<u64>()];
        message[..mem::size_of::<Username>()].copy_from_slice(&name);
        message[mem::size_of::<Username>()..].copy_from_slice(&nonce.to_le_bytes());

        let signing_key = ed25519_dalek::SigningKey::from_bytes(&key);
        let signature = signing_key.sign(&message);
        let pk = ed25519_dalek::VerifyingKey::from(&signing_key).to_bytes();
        let x = x25519_dalek::PublicKey::from(sx);

        Self { signature, pk, x, name, nonce }
    }

    fn verify(
        &self,
        pk: Pk,
        nonce: Nonce,
        sx: x25519_dalek::EphemeralSecret,
    ) -> Result<SharedSecret, ed25519_dalek::SignatureError> {
        if nonce >= self.nonce {
            eprintln!("invalid auth nonce");
            return Err(ed25519_dalek::SignatureError::default());
        }

        let pk = ed25519_dalek::VerifyingKey::from_bytes(&pk)?;

        let mut message = [0; mem::size_of::<Username>() + mem::size_of::<u64>()];
        message[..mem::size_of::<Username>()].copy_from_slice(&self.name);
        message[mem::size_of::<Username>()..].copy_from_slice(&nonce.to_le_bytes());

        pk.verify_strict(&message, &self.signature)?;

        Ok(sx.diffie_hellman(&self.x))
    }
}

#[repr(packed)]
struct ServerAuth {
    signature: ed25519_dalek::Signature,
    x: x25519_dalek::PublicKey,
}

impl ServerAuth {
    fn sign(
        user_auth: &UserAuth,
        sk: &ed25519_dalek::SigningKey,
        x: x25519_dalek::PublicKey,
    ) -> Self {
        let signature = sk.sign(user_auth.x.as_bytes());
        Self { signature, x }
    }

    fn verify(
        &self,
        x: Pk,
        pk: ed25519_dalek::VerifyingKey,
        sx: EphemeralSecret,
    ) -> Result<SharedSecret, ed25519_dalek::SignatureError> {
        pk.verify_strict(&x, &self.signature)?;
        Ok(sx.diffie_hellman(&self.x))
    }
}

struct UserData {
    header: UserHeader,
    post_headers: fs::File,
    posts: fs::File,
}

impl UserData {
    fn load(
        auth: &UserAuth,
        sx: x25519_dalek::EphemeralSecret,
    ) -> io::Result<(Self, SharedSecret)> {
        const HEADER_PATH: &str = "header.bin";
        const POST_HEADERS_PATH: &str = "post-headers.bin";
        const POST_PATH: &str = "posts.bin";

        let mut path = PathBuf::from_iter([
            USER_DATA_DIR.get().unwrap().as_path(),
            username_as_str(&auth.name).ok_or(io::ErrorKind::InvalidData)?.as_ref(),
        ]);

        if path.exists() {
            path.push(HEADER_PATH);
            let mut header_file = fs::File::open(&path).ctx("opening user header file")?;
            let mut header: UserHeader =
                read_struct(&mut header_file).ctx("reading the user header")?;
            path.pop();

            let secret = auth
                .verify(header.pk, header.nonce, sx)
                .map_err(|_| io::ErrorKind::PermissionDenied)
                .ctx("authenticating user")?;

            header.nonce = auth.nonce;
            write_struct(&mut header_file, &header).ctx("saving user nonce")?;

            path.push(POST_HEADERS_PATH);
            let post_headers = fs::File::open(&path).ctx("opening user post header file")?;
            path.pop();

            path.push(POST_PATH);
            let posts = fs::File::open(&path).ctx("opening user post file")?;
            path.pop();

            Ok((Self { header, post_headers, posts }, secret))
        } else {
            let secret = auth
                .verify(auth.pk, 0, sx)
                .map_err(|_| io::ErrorKind::PermissionDenied)
                .ctx("verifiing registratio signature")?;

            fs::create_dir_all(&path).ctx("creating new user directory")?;
            path.push(HEADER_PATH);
            let header =
                UserHeader { pk: auth.pk, nonce: auth.nonce, post_count: 0, runs: 0, imports: 0 };
            fs::write(&path, as_bytes(&header)).ctx("writing new user header")?;
            path.pop();

            path.push(POST_HEADERS_PATH);
            let post_headers = fs::File::create_new(&path).ctx("creating new user post headers")?;
            path.pop();

            path.push(POST_PATH);
            let posts = fs::File::create_new(&path).ctx("creating new user posts")?;
            path.pop();

            Ok((Self { header, post_headers, posts }, secret))
        }
    }
}

#[repr(packed)]
struct UserHeader {
    pk: Pk,
    nonce: Nonce,
    post_count: u32,
    imports: u32,
    runs: u32,
}

#[repr(packed)]
struct PostHeader {
    name: Postname,
    timestamp: u64,
    size: u32,
    offset: u32,
    imports: u32,
    runs: u32,
}
