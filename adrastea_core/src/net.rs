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
    io::IoSliceMut,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
};

use nix::sys::socket::{MsgFlags, SockaddrStorage};
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

    pub async fn recvmsg(
        &mut self, buf: &mut [u8],
    ) -> Result<(usize, Vec<OwnedFd>), std::io::Error> {
        let mut cmsg_space = nix::cmsg_space!([u8; 1024]);
        let result = loop {
            let mut guard = self.inner.readable().await?;
            match guard.try_io(|inner| {
                Ok(nix::sys::socket::recvmsg::<SockaddrStorage>(
                    inner.as_raw_fd(),
                    &mut [IoSliceMut::new(buf)],
                    Some(&mut cmsg_space),
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
}
