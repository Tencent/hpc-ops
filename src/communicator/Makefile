all:test_server test_client test_comm test_multi_comm test_multinode_comm

test_server:test_server.cc channel.cc listener.cc protocol.cc
	g++ -o $@ $^ -std=c++17 -O2 -I../../

test_client:test_client.cc channel.cc connector.cc protocol.cc
	g++ -o $@ $^ -std=c++17 -O2 -I../../

test_comm:test_comm.cc channel.cc connector.cc listener.cc communicator.cc protocol.cc
	g++ -o $@ $^ -std=c++17 -O2 -I../../

test_multi_comm:test_multi_comm.cu multicast_communicator.cc multicast_object_manager.cc channel.cc connector.cc listener.cc communicator.cc protocol.cc
	nvcc -o $@ $^ -arch=sm_90a -std=c++17 -O2 -I../../ -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs/ -lcuda -lcudart

test_multinode_comm:test_multinode_comm.cu multinode_communicator.cc channel.cc connector.cc listener.cc communicator.cc protocol.cc
	nvcc -o $@ $^ -rdc=true -arch=sm_90a -std=c++17 -O2  -I../../ -I../../3rd/ucl/include -I../../3rd/ucl/nvshmem/include \
	../../3rd/ucl/nvshmem/lib/libnvshmem_device.a \
	-L../../3rd/ucl/nvshmem/lib -lnvshmem_host

clean:
	rm test_server test_client test_comm test_multi_comm
