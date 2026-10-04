-- DDL copied verbatim from nsys 2025.6.3 exports.
-- Pack tables: runs/attrib-llama/full-a1/decode-8k.sqlite (the real attrib-llama decode-8k trace).
-- Sampling tables: a CPU-only `nsys start --sample=process-tree --backtrace=dwarf` capture of the rehearsal llama server
--   (the attrib-llama traces were taken with --sample=none and have no sampling tables).

CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    demangledName               INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Kernel function name w/ templates
    shortName                   INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Base kernel function name
    mangledName                 INTEGER,                               -- REFERENCES StringIds(id) -- Raw C++ mangled kernel function name
    launchType                  INTEGER,                               -- REFERENCES ENUM_CUDA_KERNEL_LAUNCH_TYPE(id)
    cacheConfig                 INTEGER,                               -- REFERENCES ENUM_CUDA_FUNC_CACHE_CONFIG(id)
    registersPerThread          INTEGER   NOT NULL,                    -- Number of registers required for each thread executing the kernel.
    gridX                       INTEGER   NOT NULL,                    -- X-dimension grid size.
    gridY                       INTEGER   NOT NULL,                    -- Y-dimension grid size.
    gridZ                       INTEGER   NOT NULL,                    -- Z-dimension grid size.
    blockX                      INTEGER   NOT NULL,                    -- X-dimension block size.
    blockY                      INTEGER   NOT NULL,                    -- Y-dimension block size.
    blockZ                      INTEGER   NOT NULL,                    -- Z-dimension block size.
    staticSharedMemory          INTEGER   NOT NULL,                    -- Static shared memory allocated for the kernel (B).
    dynamicSharedMemory         INTEGER   NOT NULL,                    -- Dynamic shared memory reserved for the kernel (B).
    localMemoryPerThread        INTEGER   NOT NULL,                    -- Amount of local memory reserved for each thread (B).
    localMemoryTotal            INTEGER   NOT NULL,                    -- Total amount of local memory reserved for the kernel (B).
    gridId                      INTEGER   NOT NULL,                    -- Unique grid ID of the kernel assigned at runtime.
    sharedMemoryExecuted        INTEGER,                               -- Shared memory size set by the driver.
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    sharedMemoryLimitConfig     INTEGER,                               -- REFERENCES ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG(id)
    qmdBulkReleaseDone          INTEGER,                               -- QMD bulk release done timestamp from CWD events.
    qmdPreexitDone              INTEGER,                               -- QMD pre-exit done timestamp from CWD events.
    qmdLastCtaDone              INTEGER,                               -- QMD last CTA done timestamp from CWD events.
    graphId                     INTEGER,                               -- Kernel graph ID.
    clusterX                    INTEGER,                               -- Cluster X dimension.
    clusterY                    INTEGER,                               -- Cluster Y dimension.
    clusterZ                    INTEGER,                               -- Cluster Z dimension.
    clusterSchedulingPolicy     INTEGER,                               -- Cluster scheduling policy.
    maxPotentialClusterSize     INTEGER,                               -- Maximum potential cluster size.
    maxActiveClusters           INTEGER,                               -- Maximum active clusters.
    sharedMemoryRequestedPercentage   INTEGER                          -- Shared memory requested percentage.
);

CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- ID used to identify events that this function call has triggered.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
    callchainId                 INTEGER                                -- REFERENCES CUDA_CALLCHAINS(id)
);

CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    bytes                       INTEGER   NOT NULL,                    -- Number of bytes transferred (B).
    copyKind                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_MEMCPY_OPER(id)
    deprecatedSrcId             INTEGER,                               -- Deprecated, use srcDeviceId instead.
    srcKind                     INTEGER,                               -- REFERENCES ENUM_CUDA_MEM_KIND(id)
    dstKind                     INTEGER,                               -- REFERENCES ENUM_CUDA_MEM_KIND(id)
    srcDeviceId                 INTEGER,                               -- Source device ID.
    srcContextId                INTEGER,                               -- Source context ID.
    dstDeviceId                 INTEGER,                               -- Destination device ID.
    dstContextId                INTEGER,                               -- Destination context ID.
    migrationCause              INTEGER,                               -- REFERENCES ENUM_CUDA_UNIF_MEM_MIGRATION(id)
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    virtualAddress              INTEGER,                               -- Virtual base address of the page/s being transferred.
    copyCount                   INTEGER                                -- The total number of memcopy operations traced in this record. In CUDA MemcpyBatchAsync APIs, multiple memcpy operations may be batched together for optimization purposes based on certain heuristics.
);

CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    value                       INTEGER   NOT NULL,                    -- Value assigned to memory.
    bytes                       INTEGER   NOT NULL,                    -- Number of bytes set (B).
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    memKind                     INTEGER                                -- REFERENCES ENUM_CUDA_MEM_KIND(id)
);

CREATE TABLE StringIds (
    -- Consolidation of repetitive string values.

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID reference value.
    value                       TEXT      NOT NULL                     -- String value.
);

CREATE TABLE ThreadNames (
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Thread name
    priority                    INTEGER,                               -- Priority of the thread.
    globalTid                   INTEGER                                -- Serialized GlobalId.
);

CREATE TABLE META_DATA_CAPTURE (
    -- information about nsys capture parameters

    name                        TEXT      NOT NULL,                    -- Name of meta-data record
    value                       TEXT                                   -- Value of meta-data record
);

CREATE TABLE COMPOSITE_EVENTS (
    -- Thread sampling events.

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID of the composite event.
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    cpu                         INTEGER,                               -- ID of CPU this thread was running on.
    threadState                 INTEGER,                               -- REFERENCES ENUM_SAMPLING_THREAD_STATE(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    cpuCycles                   INTEGER   NOT NULL                     -- Value of Performance Monitoring Unit (PMU) counter.
);

CREATE TABLE SAMPLING_CALLCHAINS (
    -- Callchain entries obtained from composite events, used to construct function table views.

    id                          INTEGER   NOT NULL,                    -- REFERENCES COMPOSITE_EVENTS(id)
    symbol                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    module                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Module name
    kernelMode                  INTEGER,                               -- True if kernel mode.
    thumbCode                   INTEGER,                               -- True if thumb code.
    unresolved                  INTEGER,                               -- True if the symbol was not resolved.
    specialEntry                INTEGER,                               -- True if artifical entry added during processing callchain.
    originalIP                  INTEGER,                               -- Instruction pointer value.
    unwindMethod                INTEGER,                               -- REFERENCES ENUM_STACK_UNWIND_METHOD(id)
    stackDepth                  INTEGER   NOT NULL,                    -- Zero-base index of the given function in call stack.

    PRIMARY KEY (id, stackDepth)
);
