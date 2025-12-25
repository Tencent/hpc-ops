all:test_server test_client test_comm test_multi_comm

test_server:test_server.cc channel.cc listener.cc protocol.cc
	g++ -o $@ $^ -std=c++17 -O2 -I../../

test_client:test_client.cc channel.cc connector.cc protocol.cc
	g++ -o $@ $^ -std=c++17 -O2 -I../../

test_comm:test_comm.cc channel.cc connector.cc listener.cc communicator.cc protocol.cc
	g++ -o $@ $^ -std=c++17 -O2 -I../../

test_multi_comm:test_multi_comm.cu multicast_communicator.cc multicast_object_manager.cc channel.cc connector.cc listener.cc communicator.cc protocol.cc
	nvcc -o $@ $^ -arch=sm_90a -std=c++17 -O2 -I../../ -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs/ -lcuda -lcudart

clean:
	rm test_server test_client test_comm test_multi_comm
