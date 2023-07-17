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
};

use nix::sys::socket::{ControlMessage, MsgFlags, SockaddrStorage};
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

    pub async fn accept(&mut self) -> Result<UnixScmStream, std::io::Error> {
        let (stream, _) = self.socket.accept().await?;
        Ok(UnixScmStream::new(stream)?)
    }

    pub fn into_inner(self) -> UnixListener {
        self.socket
    }
}

pub struct UnixScmStream {
    inner: AsyncFd<OwnedFd>,
}

impl UnixScmStream {
    pub fn new(stream: UnixStream) -> Result<Self, std::io::Error> {
        Ok(Self { inner: AsyncFd::new(stream.into_std()?.into())? })
    }

    pub fn alloc_cmsg_buf() -> Vec<u8> {
        nix::cmsg_space!([RawFd; 253])
    }

    pub async fn recv(
        &mut self, buf: &mut [u8], cmsg_buf: &mut Vec<u8>,
    ) -> Result<(usize, Vec<OwnedFd>), std::io::Error> {
        let result = loop {
            let mut guard = self.inner.readable().await?;
            match guard.try_io(|inner| {
                Ok(nix::sys::socket::recvmsg::<SockaddrStorage>(
                    inner.as_raw_fd(),
                    &mut [IoSliceMut::new(buf)],
                    Some(cmsg_buf),
                    MsgFlags::MSG_CMSG_CLOEXEC,
                )?)
            }) {
                Ok(result) => break result?,
                Err(_would_block) => {
                    continue;
                }
            }
        };
        let mut fds = Vec::new();
        for cmsg in result.cmsgs() {
            if let nix::sys::socket::ControlMessageOwned::ScmRights(received_fds) = cmsg {
                for raw_fd in received_fds {
                    fds.push(unsafe { OwnedFd::from_raw_fd(raw_fd) })
                }
            }
        }
        Ok((result.bytes, fds))
    }

    pub async fn send(
        &mut self, buf: &[IoSlice<'_>], fds: &[RawFd],
    ) -> Result<usize, std::io::Error> {
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
