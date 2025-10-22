// Copyright 2025 hpc-ops authors

#include "src/communicator/channel.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <string>

namespace hpc {
namespace communicator {

struct MessageHeader {
  int size;
};

Channel::Channel() { socket_ = -1; }

Channel::Channel(int sock) { socket_ = sock; }

Channel::~Channel() { socket_ = -1; }

bool Channel::Send(const std::string &data) {
  MessageHeader header;
  header.size = data.size();

  struct iovec iov[2];
  iov[0].iov_base = &header;
  iov[0].iov_len = sizeof(header);
  iov[1].iov_base = reinterpret_cast<void *>(const_cast<char *>(data.data()));
  iov[1].iov_len = data.size();

  struct msghdr msg = {0};
  msg.msg_iov = iov;
  msg.msg_iovlen = 2;

  ssize_t size = sendmsg(socket_, &msg, 0);

  return size == static_cast<ssize_t>(sizeof(header) + data.size());
}

bool Channel::Recv(std::string *data) {
  // recv MessageHeader
  MessageHeader header = {0};
  ssize_t hd_size = recv(socket_, &header, sizeof(header), 0);
  if (hd_size != sizeof(header)) {
    return false;
  }

  data->resize(header.size);
  uint8_t *ptr = reinterpret_cast<uint8_t *>((data->data()));
  ssize_t size = recv(socket_, ptr, header.size, MSG_WAITALL);

  return size == static_cast<ssize_t>(data->size());
}

bool Channel::SendFd(int fd, const std::string &data) {
  MessageHeader header;
  header.size = data.size();

  ssize_t hd_size = send(socket_, &header, sizeof(header), 0);
  if (hd_size != sizeof(header)) {
    return false;
  }

  struct iovec iov;
  iov.iov_base = reinterpret_cast<void *>(const_cast<char *>(data.data()));
  iov.iov_len = data.size();

  char buf[CMSG_SPACE(sizeof(fd))];
  memset(buf, 0, sizeof(buf));

  struct msghdr msg = {0};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);

  struct cmsghdr *cmsg;
  cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(fd));
  memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));

  msg.msg_controllen = cmsg->cmsg_len;

  ssize_t size = sendmsg(socket_, &msg, 0);

  return size == static_cast<ssize_t>(data.size());
}

bool Channel::RecvFd(int *fd, std::string *data) {
  MessageHeader header;
  ssize_t hd_size = recv(socket_, &header, sizeof(header), MSG_WAITALL);
  if (hd_size != sizeof(header)) {
    return false;
  }

  data->resize(header.size);
  char control_buf[CMSG_SPACE(sizeof(int))];

  struct iovec iov;
  iov.iov_base = reinterpret_cast<void *>(data->data());
  iov.iov_len = data->size();

  struct msghdr msg = {0};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control_buf;
  msg.msg_controllen = sizeof(control_buf);

  ssize_t size = recvmsg(socket_, &msg, MSG_WAITALL);
  if (size != header.size) {
    return false;
  }

  struct cmsghdr *cmsg;
  for (cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
      int *pfd = reinterpret_cast<int *>(CMSG_DATA(cmsg));
      *fd = *pfd;
      break;
    }
  }

  return true;
}

}  // namespace communicator
}  // namespace hpc
