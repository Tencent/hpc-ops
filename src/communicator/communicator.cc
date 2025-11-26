// Copyright 2025 hpc-ops authors

#include "src/communicator/communicator.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/communicator/channel.h"
#include "src/communicator/connector.h"
#include "src/communicator/listener.h"

namespace hpc {
namespace communicator {

Communicator::Communicator(int rank, int world_size, const std::string &comm_name)
    : kRegistery_(comm_name) {
  rank_ = rank;
  world_size_ = world_size;

  root_ = 0;

  bool is_root = (rank_ == root_);
  if (is_root) {
    Listener listener;
    listener.Listen(kRegistery_);

    std::vector<std::shared_ptr<Channel>> channels;
    for (int i = 0; i < world_size_ - 1; ++i) {
      auto channel = listener.Accept();
      channels.push_back(channel);
    }

    // protocol
    // 1. recv client -> root
    for (auto &chan : channels) {
      std::string data;
      chan->Recv(&data);
      int rank = std::stoi(data);

      channels_.insert({rank, chan});
    }

    // 2. send root -> client
    for (auto &[_, chan] : channels_) {
      std::string data = std::to_string(rank);
      chan->Send(data);
    }

  } else {
    auto channel = Connector::Connect(kRegistery_);

    // 1. send meta to root
    {
      std::string data = std::to_string(rank_);
      channel->Send(data);
    }

    // 2. recv meta from root
    {
      std::string data;
      channel->Recv(&data);
      root_ = std::stoi(data);
    }

    channel_ = channel;
  }
}

Communicator::~Communicator() {
  rank_ = -1;
  world_size_ = -1;
}

bool Communicator::Broadcast(const std::string &send_data, std::string *recv_data, int root) {
  // we only have root -> client channel
  // so we first send data to root, then broadcast it
  if (root != root_) {
    std::string rdata;

    // recv data from one client;
    if (rank_ == root) {
      bool ok = channel_->Send(send_data);
      if (!ok) {
        return false;
      }
    } else if (rank_ == root_) {
      bool ok = channels_[root]->Recv(&rdata);
      if (!ok) {
        return false;
      }
    }

    return Broadcast(rdata, recv_data, 0);
  } else {
    bool ok = true;
    if (rank_ == root) {
      *recv_data = send_data;
      for (auto &[rank, chan] : channels_) {
        ok = chan->Send(send_data) && ok;
      }
    } else {
      ok = channel_->Recv(recv_data);
    }

    return ok;
  }
}

bool Communicator::BroadcastFd(const int send_fd, int *recv_fd, const std::string &send_data,
                               std::string *recv_data, int root) {
  // we only have root -> client channel
  // so we first send data to root, then broadcast it
  if (root != root_) {
    int rfd = -1;
    std::string rdata;

    // recv fd and data from one client;
    if (rank_ == root) {
      bool ok = channel_->SendFd(send_fd, send_data);
      if (!ok) {
        return false;
      }
    } else if (rank_ == root_) {
      bool ok = channels_[root]->RecvFd(&rfd, &rdata);
      if (!ok) {
        return false;
      }
    }

    return BroadcastFd(rfd, recv_fd, rdata, recv_data, 0);
  } else {
    bool ok = true;
    if (rank_ == root) {
      *recv_fd = send_fd;
      *recv_data = send_data;
      for (auto &[rank, chan] : channels_) {
        ok = chan->SendFd(send_fd, send_data) && ok;
      }
    } else {
      ok = channel_->RecvFd(recv_fd, recv_data);
    }
    return ok;
  }
}

bool Communicator::Allgather(const std::string &send_data, std::vector<std::string> *recv_datas) {
  bool ok = true;
  recv_datas->clear();
  for (int root = 0; root < world_size_; ++root) {
    std::string recv_data;
    ok = Broadcast(send_data, &recv_data, root) && ok;
    recv_datas->push_back(recv_data);
  }
  return ok;
}

bool Communicator::AllgatherFd(const int send_fd, std::vector<int> *recv_fds,
                               const std::string &send_data, std::vector<std::string> *recv_datas) {
  bool ok = true;
  recv_fds->clear();
  recv_datas->clear();
  for (int root = 0; root < world_size_; ++root) {
    int recv_fd;
    std::string recv_data;
    ok = BroadcastFd(send_fd, &recv_fd, send_data, &recv_data, root) && ok;
    recv_fds->push_back(recv_fd);
    recv_datas->push_back(recv_data);
  }
  return ok;
}

void Communicator::Barrier() {
  if (rank_ == 0) {
    std::string data;
    for (auto &[rank, chan] : channels_) {
      chan->Recv(&data);
    }
  } else {
    std::string data = "C->S";
    channel_->Send(data);
  }

  if (rank_ == 0) {
    std::string data = "S->C";
    for (auto &[rank, chan] : channels_) {
      chan->Send(data);
    }
  } else {
    std::string data;
    channel_->Recv(&data);
  }
}

}  // namespace communicator
}  // namespace hpc
