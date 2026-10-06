// The test runner uses half of hardware_concurrency() for CPU workers.
extern "C" int get_nprocs() { return 16; }
extern "C" int get_nprocs_conf() { return 16; }
