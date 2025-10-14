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

Communicator::Communicator(int rank, int world_size) {
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
  if (root != root_) {
    throw std::runtime_error("we only support root = 0 yet!");
  }

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

bool Communicator::BroadcastFd(const int send_fd, int *recv_fd, const std::string &send_data,
                               std::string *recv_data, int root) {
  if (root != root_) {
    throw std::runtime_error("we only support root = 0 yet!");
  }

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

void Communicator::Barrier() {
  // TODO(reed): add the barrier
}

}  // namespace communicator
}  // namespace hpc
