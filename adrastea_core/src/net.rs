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

use std::{
    io::{IoSlice, IoSliceMut},
    os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
    path::Path,
};

use nix::sys::socket::{ControlMessage, MsgFlags};
use tokio::{
    io::unix::AsyncFd,
    net::{UnixListener, UnixStream},
};

pub struct UnixScmListener {
    socket: UnixListener,
}

impl UnixScmListener {
    pub fn new(socket: UnixListener) -> Self {
        Self { socket }
    }

    pub async fn accept(&self) -> Result<UnixScmStream, std::io::Error> {
        let (stream, _) = self.socket.accept().await?;
        Ok(UnixScmStream::new(stream))
    }

    pub fn into_inner(self) -> UnixListener {
        self.socket
    }
}

pub struct UnixScmStream {
    inner: AsyncFd<OwnedFd>,
}

impl UnixScmStream {
    pub fn new(stream: UnixStream) -> Self {
        Self { inner: AsyncFd::new(stream.into_std().unwrap().into()).unwrap() }
    }

    pub async fn connect(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let stream = UnixStream::connect(path).await?;
        Ok(Self::new(stream))
    }

    pub fn alloc_cmsg_buf() -> Vec<u8> {
        nix::cmsg_space!([RawFd; 253])
    }

    pub fn into_unix_stream(self) -> UnixStream {
        UnixStream::from_std(std::os::unix::net::UnixStream::from(self.inner.into_inner())).unwrap()
    }

    // TODO: these should take &mut self and there's likely logic bugs if you do
    // concurrent IO on the same side of the same stream here. We'll need some
    // method of splitting the streams to handle it properly
    pub async fn recv(
        &self, buf: &mut [u8], cmsg_buf: &mut Vec<u8>, fd_out: &mut Vec<Option<OwnedFd>>,
        should_block: bool,
    ) -> Result<usize, std::io::Error> {
        let result = loop {
            let mut guard = self.inner.readable().await?;
            match guard.try_io(|inner| {
                Ok(nix::sys::socket::recvmsg::<()>(
                    inner.as_raw_fd(),
                    &mut [IoSliceMut::new(buf)],
                    Some(cmsg_buf),
                    MsgFlags::MSG_CMSG_CLOEXEC,
                )?)
            }) {
                Ok(result) => break result?,
                Err(_would_block) => {
                    if !should_block {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::WouldBlock,
                            "Would block",
                        ));
                    }
                    continue;
                }
            }
        };
        for cmsg in result.cmsgs() {
            if let nix::sys::socket::ControlMessageOwned::ScmRights(received_fds) = cmsg {
                for raw_fd in received_fds {
                    fd_out.push(unsafe { Some(OwnedFd::from_raw_fd(raw_fd)) })
                }
            }
        }
        Ok(result.bytes)
    }

    pub async fn send(&self, buf: &[IoSlice<'_>], fds: &[RawFd]) -> Result<usize, std::io::Error> {
        if fds.len() > 253 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Too many file descriptors",
            ));
        }
        Ok(loop {
            let mut guard = self.inner.writable().await?;
            match guard.try_io(|inner| {
                let scm_rights = &[ControlMessage::ScmRights(fds)];
                Ok(nix::sys::socket::sendmsg::<()>(
                    inner.as_raw_fd(),
                    buf,
                    if fds.len() > 0 { scm_rights } else { &[] },
                    MsgFlags::empty(),
                    None,
                )?)
            }) {
                Ok(result) => break result?,
                Err(_would_block) => {
                    continue;
                }
            }
        })
    }
}

#[cfg(test)]
mod test {
    use std::{
        fs::File,
        io::{Read, Seek, SeekFrom, Write},
    };

    use super::*;
    use tempdir::TempDir;

    #[tokio::test]
    async fn send_file_and_data() -> anyhow::Result<()> {
        let dir = TempDir::new("adrastea-test")?;
        let sockpath = dir.path().join("socket");
        let listener = UnixScmListener::new(UnixListener::bind(&sockpath)?);
        async fn listener_task(mut listener: UnixScmListener) -> anyhow::Result<()> {
            let mut stream = listener.accept().await?;
            let mut buf = [0u8; 1024];
            let mut cmsg_buf = UnixScmStream::alloc_cmsg_buf();
            let mut fds = vec![];
            let nread = stream.recv(&mut buf, &mut cmsg_buf, &mut fds, true).await?;
            assert_eq!(nread, 4);
            assert_eq!(fds.len(), 1);
            let mut file = File::from(fds.into_iter().next().unwrap().unwrap());
            file.write(&buf[0..nread])?;
            Ok(())
        }
        let jh = tokio::spawn(async move {
            listener_task(listener).await.unwrap();
        });
        let mut client = UnixScmStream::new(UnixStream::connect(&sockpath).await?);
        let buf = [0x42u8; 4];
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(dir.path().join("testfile"))?;
        client.send(&[IoSlice::new(&buf)], &[file.as_raw_fd()]).await?;
        jh.await?;
        let mut out_buf = [0u8; 4];
        file.seek(SeekFrom::Start(0))?;
        let nread = file.read(&mut out_buf)?;
        assert_eq!(nread, 4);
        assert_eq!(buf, out_buf);
        Ok(())
    }
}
