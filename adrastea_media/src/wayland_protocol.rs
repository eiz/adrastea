/*
 * This file is part of Adrastea.
 *
 * Adrastea is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License as published by the Free Software
 * Foundation, version 3.
 *
 * Adrastea is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along
 * with Adrastea. If not, see <https://www.gnu.org/licenses/>.
 */

use core::ops::Range;
use std::{
    fs::File,
    io::{self, BufReader, IoSlice},
    os::fd::{AsRawFd, OwnedFd, RawFd},
    path::{Path, PathBuf},
};

use adrastea_core::{
    net::{UnixScmListener, UnixScmStream},
    util::round_up,
};
use alloc::{collections::BTreeMap, sync::Arc};
use anyhow::bail;
use byteorder::{ByteOrder, NativeEndian};
use parking_lot::Mutex;
use serde::Deserialize;
use tokio::net::UnixListener;

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq)]
pub enum WaylandDataType {
    #[serde(rename = "int")]
    Int,
    #[serde(rename = "uint")]
    Uint,
    #[serde(rename = "fixed")]
    Fixed,
    #[serde(rename = "string")]
    String,
    #[serde(rename = "object")]
    Object,
    #[serde(rename = "new_id")]
    NewId,
    #[serde(rename = "array")]
    Array,
    #[serde(rename = "fd")]
    Fd,
}

#[derive(Debug, Deserialize)]
pub struct WaylandDescription {
    #[serde(rename = "@summary")]
    pub summary: Option<String>,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct WaylandArg {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@type")]
    pub data_type: WaylandDataType,
    #[serde(rename = "@summary")]
    pub summary: Option<String>,
    #[serde(rename = "@interface")]
    pub interface: Option<String>,
    #[serde(rename = "@allow-null")]
    pub allow_null: Option<bool>,
    #[serde(rename = "@enum")]
    pub r#enum: Option<String>,
}

#[derive(Debug, Deserialize)]
pub enum WaylandMessageType {
    #[serde(rename = "destructor")]
    Destructor,
}

#[derive(Debug, Deserialize)]
pub struct WaylandMessage {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@type")]
    pub r#type: Option<WaylandMessageType>,
    #[serde(rename = "@since")]
    pub since: Option<u32>,
    #[serde(rename = "description")]
    pub description: Option<WaylandDescription>,
    #[serde(rename = "arg")]
    pub args: Option<Vec<WaylandArg>>,
}

#[derive(Debug, Deserialize)]
pub struct WaylandEnumEntry {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@value")]
    pub value: String,
    #[serde(rename = "@summary")]
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct WaylandEnum {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@since")]
    pub since: Option<u32>,
    #[serde(rename = "@bitfield")]
    pub bitfield: Option<bool>,
    #[serde(rename = "description")]
    pub description: Option<WaylandDescription>,
    #[serde(rename = "entry")]
    pub entries: Vec<WaylandEnumEntry>,
}

#[derive(Debug, Deserialize)]
pub enum WaylandInterfaceItem {
    #[serde(rename = "request")]
    Request(WaylandMessage),
    #[serde(rename = "event")]
    Event(WaylandMessage),
    #[serde(rename = "enum")]
    Enum(WaylandEnum),
}

#[derive(Debug, Deserialize)]
pub struct WaylandInterface {
    #[serde(rename = "@version")]
    pub version: u32,
    #[serde(rename = "@name")]
    pub name: String,
    pub description: Option<WaylandDescription>,
    #[serde(rename = "$value")]
    pub items: Option<Vec<WaylandInterfaceItem>>,
}

#[derive(Debug, Deserialize)]
pub struct WaylandProtocol {
    #[serde(rename = "@name")]
    pub name: String,
    pub copyright: Option<String>,
    #[serde(rename = "interface")]
    pub interfaces: Vec<WaylandInterface>,
}

impl WaylandProtocol {
    pub fn load_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        Ok(quick_xml::de::from_reader(BufReader::new(File::open(path)?))?)
    }
}

#[derive(Debug)]
struct ResolvedArg {
    data_type: WaylandDataType,
    interface: Option<InterfaceId>,
    allow_null: bool,
}

#[derive(Debug)]
struct ResolvedMessage {
    since: u32,
    args: Vec<ResolvedArg>,
    message: WaylandMessage,
}

#[derive(Debug)]
struct ResolvedInterface {
    name: String,
    version: u32,
    requests: Vec<ResolvedMessage>,
    events: Vec<ResolvedMessage>,
}

struct WaylandProtocolMapInner {
    interfaces: Vec<ResolvedInterface>,
    interface_lookup: BTreeMap<String, InterfaceId>,
}

fn add_message(collection: &mut Vec<ResolvedMessage>, msg: WaylandMessage) {
    let mut args = vec![];
    for arg in msg.args.as_ref().unwrap_or(&vec![]) {
        args.push(ResolvedArg {
            data_type: arg.data_type,
            interface: None,
            allow_null: arg.allow_null.unwrap_or(false),
        });
    }
    collection.push(ResolvedMessage { since: msg.since.unwrap_or(0), args, message: msg });
}

pub struct WaylandProtocolMapBuilder(WaylandProtocolMapInner);

impl WaylandProtocolMapBuilder {
    pub fn new() -> Self {
        Self(WaylandProtocolMapInner { interfaces: vec![], interface_lookup: BTreeMap::new() })
    }

    pub fn dir<P: AsRef<Path>>(self, path: P) -> anyhow::Result<Self> {
        let mut me = self;
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if entry.file_type()?.is_file() && entry.file_name().to_string_lossy().ends_with(".xml")
            {
                me = me.file(entry.path())?;
            }
        }
        Ok(me)
    }

    pub fn file<P: AsRef<Path>>(mut self, path: P) -> anyhow::Result<Self> {
        let protocol = WaylandProtocol::load_path(path)?;
        for interface in protocol.interfaces {
            if self.0.interface_lookup.contains_key(&interface.name) {
                bail!("duplicate interface {:?}", interface);
            }
            let mut requests = vec![];
            let mut events = vec![];
            if let (Some(items), name) = (interface.items, interface.name) {
                for item in items {
                    match item {
                        WaylandInterfaceItem::Request(m) => add_message(&mut requests, m),
                        WaylandInterfaceItem::Event(m) => add_message(&mut events, m),
                        _ => (),
                    }
                }
                self.0
                    .interface_lookup
                    .insert(name.clone(), InterfaceId(self.0.interfaces.len() as u16));
                self.0.interfaces.push(ResolvedInterface {
                    name,
                    version: interface.version,
                    requests,
                    events,
                });
            }
        }
        Ok(self)
    }

    pub fn build(mut self) -> anyhow::Result<WaylandProtocolMap> {
        let WaylandProtocolMapInner { ref interface_lookup, ref mut interfaces } = self.0;
        for iface in interfaces {
            for msg in iface.requests.iter_mut().chain(iface.events.iter_mut()) {
                let empty_vec = vec![];
                let unresolved_args = msg.message.args.as_ref().unwrap_or(&empty_vec);
                for (arg, unresolved_arg) in msg.args.iter_mut().zip(unresolved_args) {
                    if let Some(interface_name) = unresolved_arg.interface.as_ref() {
                        if let Some(interface_id) = interface_lookup.get(interface_name) {
                            arg.interface = Some(*interface_id);
                        } else {
                            bail!("unknown interface {:?}", interface_name);
                        }
                    }
                }
            }
        }
        if !interface_lookup.contains_key("wl_display") {
            bail!("protocol map must include wl_display");
        }
        Ok(WaylandProtocolMap(Arc::new(self.0)))
    }
}

#[derive(Clone)]
pub struct WaylandProtocolMap(Arc<WaylandProtocolMapInner>);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct InterfaceId(pub u16);

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash, Ord, PartialOrd)]
pub struct ObjectId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash, Ord, PartialOrd)]
pub struct InterfaceVersion(pub u32);

pub struct MessageReader<'a> {
    sender: ObjectId,
    opcode: u16,
    protocol_map: &'a WaylandProtocolMap,
    resolved_interface: &'a ResolvedInterface,
    resolved_message: &'a ResolvedMessage,
    data: &'a [u8],
    fds: &'a mut Vec<Option<OwnedFd>>,
}

impl<'a> MessageReader<'a> {
    pub fn sender(&self) -> ObjectId {
        self.sender
    }

    pub fn opcode(&self) -> u16 {
        self.opcode
    }

    pub fn debug_name(&self) -> String {
        format!(
            "{}({}).{}({})",
            self.resolved_interface.name,
            self.sender.0,
            self.resolved_message.message.name,
            self.opcode,
        )
    }

    pub fn args<'b>(&'b mut self) -> MessageReaderArgs<'b, 'a> {
        MessageReaderArgs { reader: self, buf_pos: 0, fd_pos: 0, n: 0 }
    }
}

#[derive(Debug)]
pub enum MessageReaderValue<'a> {
    Int(i32),
    Uint(u32),
    Fixed(i32), // Q8
    String(&'a str),
    Object(Option<ObjectId>),
    NewId(InterfaceId, ObjectId, InterfaceVersion),
    Array(&'a [u8]),
    Fd(usize),
}

pub struct MessageReaderArg {
    data_type: WaylandDataType,
    interface: Option<InterfaceId>,
    data_range: Range<usize>,
    fd_index: Option<usize>,
}

pub struct MessageReaderArgs<'b, 'a: 'b> {
    reader: &'b mut MessageReader<'a>,
    buf_pos: usize,
    fd_pos: usize,
    n: usize,
}

fn string_field_len(data: &[u8], buf_pos: usize) -> usize {
    4 + round_up(NativeEndian::read_u32(&data[buf_pos..buf_pos + 4]) as usize, 4)
}

impl<'b, 'a> MessageReaderArgs<'b, 'a> {
    pub fn data(&self, arg: &MessageReaderArg) -> &[u8] {
        &self.reader.data[arg.data_range.clone()]
    }

    pub fn value(&self, arg: &MessageReaderArg) -> anyhow::Result<MessageReaderValue<'b>> {
        let data: &'b [u8] = &self.reader.data[arg.data_range.clone()];
        Ok(match arg.data_type {
            WaylandDataType::Int => MessageReaderValue::Int(NativeEndian::read_i32(data)),
            WaylandDataType::Uint => MessageReaderValue::Uint(NativeEndian::read_u32(data)),
            WaylandDataType::Fixed => MessageReaderValue::Fixed(NativeEndian::read_i32(data)),
            WaylandDataType::String => {
                let len = NativeEndian::read_u32(data) as usize;
                let str_data = &data[4..4 + len - 1];
                MessageReaderValue::String(std::str::from_utf8(str_data)?)
            }
            WaylandDataType::Object => {
                let object_id = NativeEndian::read_u32(data);
                MessageReaderValue::Object(if object_id == 0 {
                    None
                } else {
                    Some(ObjectId(object_id))
                })
            }
            WaylandDataType::NewId => {
                if let Some(interface_id) = arg.interface {
                    let version =
                        self.reader.protocol_map.0.interfaces[interface_id.0 as usize].version;
                    MessageReaderValue::NewId(
                        interface_id,
                        ObjectId(NativeEndian::read_u32(data)),
                        InterfaceVersion(version),
                    )
                } else {
                    let len = NativeEndian::read_u32(data) as usize;
                    let str_data = &data[4..4 + len - 1];
                    let len = round_up(len, 4);
                    let interface_name = std::str::from_utf8(str_data)?;
                    let version = NativeEndian::read_u32(&data[4 + len..4 + len + 4]);
                    let object_id = NativeEndian::read_u32(&data[4 + len + 4..4 + len + 8]);
                    let interface_id = (self.reader.protocol_map.0.interface_lookup)
                        .get(interface_name)
                        .ok_or_else(|| anyhow::anyhow!("unknown interface {:?}", interface_name))?;
                    MessageReaderValue::NewId(
                        *interface_id,
                        ObjectId(object_id),
                        InterfaceVersion(version),
                    )
                }
            }
            WaylandDataType::Array => {
                let len = NativeEndian::read_u32(data) as usize;
                MessageReaderValue::Array(&data[4..4 + len])
            }
            WaylandDataType::Fd => MessageReaderValue::Fd(arg.fd_index.unwrap()),
        })
    }

    pub fn take_fd(&mut self, arg: &MessageReaderArg) -> OwnedFd {
        let fd = self.reader.fds[arg.fd_index.unwrap()].take().unwrap();
        fd
    }

    pub fn advance(&mut self) -> Option<MessageReaderArg> {
        let arg_spec = match self.reader.resolved_message.args.get(self.n) {
            Some(arg) => arg,
            None => return None,
        };
        let length = match arg_spec.data_type {
            WaylandDataType::Int => 4,
            WaylandDataType::Uint => 4,
            WaylandDataType::Fixed => 4,
            WaylandDataType::String => string_field_len(&self.reader.data, self.buf_pos),
            WaylandDataType::Object => 4,
            WaylandDataType::NewId => {
                if let Some(_interface) = arg_spec.interface.as_ref() {
                    4
                } else {
                    string_field_len(&self.reader.data, self.buf_pos) + 4 + 4
                }
            }
            WaylandDataType::Array => {
                4 + round_up(
                    NativeEndian::read_u32(&self.reader.data[self.buf_pos..self.buf_pos + 4])
                        as usize,
                    4,
                )
            }
            WaylandDataType::Fd => 0,
        };
        let interface = match arg_spec.data_type {
            WaylandDataType::NewId => arg_spec.interface,
            _ => None,
        };
        let has_fd = arg_spec.data_type == WaylandDataType::Fd;
        let data_range = self.buf_pos..self.buf_pos + length;
        let fd_index = if has_fd {
            let fd_pos = self.fd_pos;
            self.fd_pos += 1;
            Some(fd_pos)
        } else {
            None
        };
        self.buf_pos += length;
        self.n += 1;
        Some(MessageReaderArg { data_type: arg_spec.data_type, interface, data_range, fd_index })
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum WaylandConnectionRole {
    Client,
    Server,
}

pub struct WaylandReceiver {
    inner: Arc<WaylandConnection>,
    cmsg_buf: Vec<u8>,
    rx_buf_max: usize,
    rx_buf_fd_max: usize,
    rx_buf: Vec<u8>,
    rx_buf_fill: usize,
    rx_fd_buf: Vec<Option<OwnedFd>>,
}

impl WaylandReceiver {
    fn new(inner: Arc<WaylandConnection>) -> Self {
        Self {
            inner,
            cmsg_buf: UnixScmStream::alloc_cmsg_buf(),
            rx_buf_max: 65536,
            rx_buf_fd_max: 1024,
            rx_buf_fill: 0,
            rx_buf: vec![0; 65536],
            rx_fd_buf: vec![],
        }
    }

    pub fn message<'a>(&'a mut self) -> anyhow::Result<MessageReader<'a>> {
        if self.rx_buf_fill < 8 {
            bail!("no current message");
        }
        let sender = ObjectId(NativeEndian::read_u32(&self.rx_buf[0..4]));
        let opcode = NativeEndian::read_u16(&self.rx_buf[4..6]);
        let length = NativeEndian::read_u16(&self.rx_buf[6..8]) as usize;
        if self.rx_buf_fill < length {
            bail!("no current message");
        }
        let (resolved_interface, resolved_message) =
            match (self.inner.connection_role, self.inner.handle_table.lock().get(&sender)) {
                (WaylandConnectionRole::Server, Some(interface_id)) => {
                    let interface = &self.inner.protocol_map.0.interfaces[interface_id.0 as usize];
                    let message = &interface.requests[opcode as usize];
                    (interface, message)
                }
                (WaylandConnectionRole::Client, Some(interface_id)) => {
                    let interface = &self.inner.protocol_map.0.interfaces[interface_id.0 as usize];
                    let message = &interface.events[opcode as usize];
                    (interface, message)
                }
                _ => bail!("invalid message sender"),
            };
        Ok(MessageReader {
            sender,
            opcode,
            data: &self.rx_buf[8..length],
            fds: &mut self.rx_fd_buf,
            resolved_interface,
            resolved_message,
            protocol_map: &self.inner.protocol_map,
        })
    }

    pub async fn advance(&mut self) -> anyhow::Result<()> {
        if self.rx_buf_fill > 0 {
            self.consume_message()?;
        }
        self.fill_buffer(8, 0).await?;
        let msg_len = NativeEndian::read_u16(&self.rx_buf[6..8]) as usize;
        self.fill_buffer(msg_len, 0).await?;
        let inner = self.inner.clone(); // TODO this clone is annoying to get rid of =/
        let mut msg = self.message()?;
        let mut args = msg.args();
        while let Some(arg) = args.advance() {
            if arg.data_type == WaylandDataType::NewId {
                let (interface_id, object_id) = match args.value(&arg)? {
                    // TODO we might need to handle versioning here eventually
                    MessageReaderValue::NewId(interface_id, object_id, _) => {
                        (interface_id, object_id)
                    }
                    _ => unreachable!(),
                };
                inner.handle_table.lock().insert(object_id, interface_id);
            }
        }
        Ok(())
    }

    async fn fill_buffer(&mut self, bytes_needed: usize, fds_needed: usize) -> io::Result<()> {
        loop {
            if self.rx_buf_fill >= bytes_needed && self.rx_fd_buf.len() >= fds_needed {
                return Ok(());
            }
            if bytes_needed > self.rx_buf_max || fds_needed > self.rx_buf_fd_max {
                return Err(io::ErrorKind::InvalidInput.into());
            }
            let mut blocking = true;
            loop {
                let buf = &mut self.rx_buf[self.rx_buf_fill..];
                // TODO need an IoSliceMut version of this so we can do the deque thing
                let result = self
                    .inner
                    .stream
                    .recv(buf, &mut self.cmsg_buf, &mut self.rx_fd_buf, blocking)
                    .await;
                match result {
                    Ok(nread) => {
                        if nread == 0 {
                            // we may reach the end of the stream after reading
                            // enough data but before the buffer fill loop
                            // naturally exits.
                            if !blocking {
                                break;
                            }
                            return Err(io::ErrorKind::UnexpectedEof.into());
                        }
                        self.rx_buf_fill += nread;
                        if self.rx_fd_buf.len() > self.rx_buf_fd_max {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                "fd receive buffer overflow",
                            ));
                        }
                        blocking = false;
                    }
                    Err(e) => {
                        if e.kind() == io::ErrorKind::WouldBlock && !blocking {
                            break;
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
        }
    }

    fn consume_message(&mut self) -> anyhow::Result<()> {
        let mut msg = self.message()?;
        let mut consumed_fds = 0;
        let consumed_bytes = msg.data.len() + 8;
        let mut args = msg.args();
        while let Some(arg) = args.advance() {
            if arg.fd_index.is_some() {
                consumed_fds += 1;
            }
        }
        if consumed_bytes > self.rx_buf_fill || consumed_fds > self.rx_fd_buf.len() {
            panic!("message is larger than itself...?");
        }
        self.rx_fd_buf.drain(..consumed_fds);
        self.rx_buf.rotate_left(consumed_bytes);
        self.rx_buf_fill -= consumed_bytes;
        Ok(())
    }
}

pub struct MessageBuilder<'a> {
    conn: &'a WaylandConnection,
    resolved_interface: &'a ResolvedInterface,
    resolved_message: &'a ResolvedMessage,
    sender_id: ObjectId,
    opcode: u16,
    data: &'a mut Vec<u8>,
    fds: &'a mut Vec<RawFd>,
    owned_fds: Vec<OwnedFd>,
    n: usize,
}

impl<'a> MessageBuilder<'a> {
    pub fn int(mut self, value: i32) -> Self {
        assert_eq!(self.resolved_message.args[self.n].data_type, WaylandDataType::Int);
        self.data.extend_from_slice(&value.to_ne_bytes());
        self.n += 1;
        self
    }

    pub fn uint(mut self, value: u32) -> Self {
        assert_eq!(self.resolved_message.args[self.n].data_type, WaylandDataType::Uint);
        self.data.extend_from_slice(&value.to_ne_bytes());
        self.n += 1;
        self
    }

    pub fn fixed(mut self, value: i32) -> Self {
        assert_eq!(self.resolved_message.args[self.n].data_type, WaylandDataType::Fixed);
        self.data.extend_from_slice(&value.to_ne_bytes());
        self.n += 1;
        self
    }

    fn write_string(&mut self, value: &str) {
        self.data.extend_from_slice(&(value.len() as u32 + 1).to_ne_bytes());
        self.data.extend_from_slice(value.as_bytes());
        self.data.push(0);
        self.data.resize(round_up(self.data.len(), 4), 0);
    }

    pub fn string(mut self, value: &str) -> Self {
        assert_eq!(self.resolved_message.args[self.n].data_type, WaylandDataType::String);
        self.write_string(value);
        self.n += 1;
        self
    }

    pub fn object(mut self, value: Option<ObjectId>) -> Self {
        assert_eq!(self.resolved_message.args[self.n].data_type, WaylandDataType::Object);
        self.data.extend_from_slice(&value.unwrap_or(ObjectId(0)).0.to_ne_bytes());
        self.n += 1;
        self
    }

    pub fn new_id(
        mut self, interface: InterfaceId, object_id: ObjectId, version: InterfaceVersion,
    ) -> Self {
        assert_eq!(self.resolved_message.args[self.n].data_type, WaylandDataType::NewId);
        if self.resolved_message.args[self.n].interface.is_some() {
            self.data.extend_from_slice(&object_id.0.to_ne_bytes());
        } else {
            let resolved = &self.conn.protocol_map.0.interfaces[interface.0 as usize];
            println!("  new_id {} {} {}", resolved.name, version.0, object_id.0);
            self.write_string(&resolved.name);
            self.data.extend_from_slice(&version.0.to_ne_bytes());
            self.data.extend_from_slice(&object_id.0.to_ne_bytes());
        }
        self.n += 1;
        self
    }

    pub fn array(mut self, value: &[u8]) -> Self {
        assert_eq!(self.resolved_message.args[self.n].data_type, WaylandDataType::Array);
        self.data.extend_from_slice(&(value.len() as u32).to_ne_bytes());
        self.data.extend_from_slice(value);
        self.data.resize(round_up(self.data.len(), 4), 0);
        self.n += 1;
        self
    }

    pub fn fd(mut self, fd: RawFd) -> Self {
        assert_eq!(self.resolved_message.args[self.n].data_type, WaylandDataType::Fd);
        self.fds.push(fd);
        self.n += 1;
        self
    }

    pub fn fd_owned(mut self, fd: OwnedFd) -> Self {
        assert_eq!(self.resolved_message.args[self.n].data_type, WaylandDataType::Fd);
        self.fds.push(fd.as_raw_fd());
        self.owned_fds.push(fd);
        self.n += 1;
        self
    }

    pub async fn send(self) -> anyhow::Result<()> {
        if self.data.len() > 65535 {
            bail!("message too large");
        }
        let len = self.data.len() as u16;
        NativeEndian::write_u16(&mut self.data[6..8], len);
        let mut dummy_fds = vec![];
        let mut reader = MessageReader {
            sender: self.sender_id,
            opcode: self.opcode,
            data: &self.data[8..],
            fds: &mut dummy_fds,
            resolved_interface: self.resolved_interface,
            resolved_message: self.resolved_message,
            protocol_map: &self.conn.protocol_map,
        };
        let mut args = reader.args();
        while let Some(arg) = args.advance() {
            if arg.data_type == WaylandDataType::NewId {
                let (interface_id, object_id) = match args.value(&arg)? {
                    // TODO versioning
                    MessageReaderValue::NewId(interface_id, object_id, _) => {
                        (interface_id, object_id)
                    }
                    _ => unreachable!(),
                };
                self.conn.handle_table.lock().insert(object_id, interface_id);
            }
        }
        // TODO buffer sends for non-fd-transmitting messages
        self.conn.stream.send(&[IoSlice::new(&self.data)], &self.fds).await?;
        Ok(())
    }
}

pub struct WaylandSender {
    inner: Arc<WaylandConnection>,
    builder_data: Vec<u8>,
    builder_fds: Vec<RawFd>,
}

impl WaylandSender {
    fn new(inner: Arc<WaylandConnection>) -> Self {
        Self { inner, builder_data: vec![], builder_fds: vec![] }
    }

    pub fn message_builder(
        &mut self, sender_id: ObjectId, opcode: u16,
    ) -> anyhow::Result<MessageBuilder> {
        self.builder_data.clear();
        self.builder_fds.clear();
        self.builder_data.extend_from_slice(&sender_id.0.to_ne_bytes());
        self.builder_data.extend_from_slice(&opcode.to_ne_bytes());
        self.builder_data.extend_from_slice(&[0; 2]);
        let interface_id = *(self.inner.handle_table.lock().get(&sender_id))
            .ok_or_else(|| anyhow::anyhow!("invalid sender id"))?;
        let resolved_interface =
            (self.inner.protocol_map.0.interfaces.get(interface_id.0 as usize))
                .ok_or_else(|| anyhow::anyhow!("invalid interface id"))?;
        let message_list = match self.inner.connection_role {
            WaylandConnectionRole::Server => &resolved_interface.events,
            WaylandConnectionRole::Client => &resolved_interface.requests,
        };
        let resolved_message =
            message_list.get(opcode as usize).ok_or_else(|| anyhow::anyhow!("invalid opcode"))?;
        Ok(MessageBuilder {
            resolved_interface,
            resolved_message,
            sender_id,
            opcode,
            conn: &*self.inner,
            data: &mut self.builder_data,
            fds: &mut self.builder_fds,
            owned_fds: vec![],
            n: 0,
        })
    }
}

pub struct WaylandConnection {
    stream: UnixScmStream,
    connection_role: WaylandConnectionRole,
    protocol_map: WaylandProtocolMap,
    handle_table: Mutex<BTreeMap<ObjectId, InterfaceId>>,
}

impl WaylandConnection {
    pub fn new(
        protocol_map: WaylandProtocolMap, stream: UnixScmStream,
        connection_role: WaylandConnectionRole,
    ) -> (WaylandReceiver, WaylandSender) {
        let display_interface = protocol_map.0.interface_lookup.get("wl_display").unwrap();
        let mut handle_table = BTreeMap::new();
        handle_table.insert(ObjectId(1), *display_interface);
        let conn = Arc::new(Self {
            stream,
            connection_role,
            handle_table: Mutex::new(handle_table),
            protocol_map,
        });
        (WaylandReceiver::new(conn.clone()), WaylandSender::new(conn))
    }
}

#[cfg(test)]
mod test {
    use core::time::Duration;

    use adrastea_core::net::UnixScmListener;
    use tempdir::TempDir;
    use tokio::net::UnixListener;
    use wayland_client::{
        protocol::{
            wl_callback::WlCallback,
            wl_compositor::WlCompositor,
            wl_registry::{self, WlRegistry},
        },
        Connection, Dispatch, EventQueue,
    };

    use super::*;

    #[test]
    pub fn load_all_protocols() -> anyhow::Result<()> {
        let proto_map = WaylandProtocolMapBuilder::new()
            .file("/home/eiz/code/wayland/protocol/wayland.xml")?
            .dir("/home/eiz/code/wayland-protocols/stable/xdg-shell")?
            .build()?;
        assert_eq!(proto_map.0.interfaces.len(), 27);
        Ok(())
    }

    #[tokio::test]
    pub async fn wayland_client_connect() -> anyhow::Result<()> {
        struct TestState {
            wl_compositor: Option<WlCompositor>,
            sync_count: u32,
        }
        impl Dispatch<WlRegistry, ()> for TestState {
            fn event(
                state: &mut Self, proxy: &WlRegistry,
                event: <WlRegistry as wayland_client::Proxy>::Event, _data: &(),
                _conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
            ) {
                println!("wl_registry {:?}", event);
                if let wl_registry::Event::Global { name, interface, version } = event {
                    if interface == "wl_compositor" {
                        state.wl_compositor = Some(proxy.bind(name, version, qhandle, ()));
                    }
                }
            }
        }
        impl Dispatch<WlCallback, ()> for TestState {
            fn event(
                state: &mut Self, _proxy: &WlCallback,
                event: <WlCallback as wayland_client::Proxy>::Event, _data: &(),
                _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
            ) {
                state.sync_count += 1;
                println!("wl_callback {:?}", event);
            }
        }
        impl Dispatch<WlCompositor, ()> for TestState {
            fn event(
                _state: &mut Self, _proxy: &WlCompositor,
                event: <WlCompositor as wayland_client::Proxy>::Event, _data: &(),
                _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
            ) {
                println!("wl_compositor {:?}", event);
            }
        }

        let test_dir = TempDir::new("test.wayland_client_connect")?;
        let sockpath = test_dir.path().join("wayland.sock");
        let listener = UnixScmListener::new(UnixListener::bind(&sockpath)?);
        async fn listener_task(listener: UnixScmListener) -> anyhow::Result<()> {
            let proto_map = WaylandProtocolMapBuilder::new()
                .file("/home/eiz/code/wayland/protocol/wayland.xml")?
                .build()?;
            let stream = listener.accept().await?;
            let (mut rx, mut tx) =
                WaylandConnection::new(proto_map.clone(), stream, WaylandConnectionRole::Server);
            rx.advance().await?;
            // get_registry
            let mut msg = rx.message()?;
            assert_eq!(msg.sender(), ObjectId(1));
            assert_eq!(msg.opcode(), 1);
            let mut args = msg.args();
            let arg = args.advance().unwrap();
            let registry_id = match args.value(&arg) {
                Ok(MessageReaderValue::NewId(_interface_id, object_id, _)) => object_id,
                x => panic!("bad message format {x:?}"),
            };
            tx.message_builder(registry_id, 0)?
                .uint(1)
                .string("wl_compositor")
                .uint(6)
                .send()
                .await?;
            // first sync
            rx.advance().await?;
            let mut msg = rx.message()?;
            assert_eq!(msg.sender(), ObjectId(1));
            assert_eq!(msg.opcode(), 0);
            let mut args = msg.args();
            let arg = args.advance().unwrap();
            let callback_id = match args.value(&arg) {
                Ok(MessageReaderValue::NewId(_interface_id, object_id, _)) => object_id,
                x => panic!("bad message format {x:?}"),
            };
            tx.message_builder(callback_id, 0)?.uint(0).send().await?;
            // bind wl_compositor
            rx.advance().await?;
            let mut msg = rx.message()?;
            assert_eq!(msg.sender(), registry_id);
            assert_eq!(msg.opcode(), 0);
            let mut args = msg.args();
            let arg = args.advance().unwrap();
            let compositor_name = match args.value(&arg) {
                Ok(MessageReaderValue::Uint(x)) => x,
                x => panic!("bad message format {x:?}"),
            };
            let arg = args.advance().unwrap();
            let (compositor_interface_id, compositor_id, compositor_version) =
                match args.value(&arg) {
                    Ok(MessageReaderValue::NewId(interface_id, object_id, version)) => {
                        (interface_id, object_id, version)
                    }
                    x => panic!("bad message format {x:?}"),
                };
            let expected_compositor_id = proto_map.0.interface_lookup.get("wl_compositor").unwrap();
            assert_eq!(compositor_name, 1);
            assert_eq!(compositor_interface_id, *expected_compositor_id);
            assert_eq!(compositor_version, InterfaceVersion(6));
            assert_eq!(compositor_id, ObjectId(4));
            // second sync
            rx.advance().await?;
            let mut msg = rx.message()?;
            assert_eq!(msg.sender(), ObjectId(1));
            assert_eq!(msg.opcode(), 0);
            let mut args = msg.args();
            let arg = args.advance().unwrap();
            let callback_id = match args.value(&arg) {
                Ok(MessageReaderValue::NewId(_interface_id, object_id, _)) => object_id,
                x => panic!("bad message format {x:?}"),
            };
            tx.message_builder(callback_id, 0)?.uint(0).send().await?;

            tokio::time::sleep(Duration::from_millis(1000)).await;
            Ok(())
        }
        let jh = tokio::spawn(async move {
            listener_task(listener).await.unwrap();
        });
        tokio::task::spawn_blocking({
            let sockpath = sockpath.to_path_buf();
            move || {
                use std::os::unix::net::UnixStream;
                let sock = UnixStream::connect(&sockpath).unwrap();
                let conn = Connection::from_socket(sock).unwrap();
                let display = conn.display();
                let mut event_queue: EventQueue<TestState> = conn.new_event_queue();
                let handle = event_queue.handle();
                let _registry = display.get_registry(&handle, ());
                let mut state = TestState { wl_compositor: None, sync_count: 0 };
                display.sync(&handle, ());
                while state.sync_count == 0 {
                    event_queue.blocking_dispatch(&mut state).unwrap();
                }
                display.sync(&handle, ());
                while state.sync_count == 1 {
                    event_queue.blocking_dispatch(&mut state).unwrap();
                }
            }
        })
        .await?;
        jh.await?;
        Ok(())
    }
}
