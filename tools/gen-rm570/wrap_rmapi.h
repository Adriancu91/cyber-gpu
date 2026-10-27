/* Input for bindgen (I8e, "golden image"): the RM API objects and controls the
 * first client creates after GSP_INIT_DONE, and the two RPC wrappers that carry
 * them (GSP_RM_ALLOC, GSP_RM_CONTROL). Needs the RM kernel include paths and
 * flags (generate.sh). Only MIT headers from open-gpu-kernel-modules 570.144. */
#include "nvtypes.h"
#include "vgpu/rpc_headers.h"
#define RPC_STRUCTURES
#include "g_rpc-structures.h"
#include "nvos.h"
#include "class/cl0000.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "class/cl90f1.h"
#include "class/clc56f.h"
#include "class/clc7c0.h"
#include "class/clc7b5.h"
#include "alloc/alloc_channel.h"
#include "ctrl/ctrl0080/ctrl0080fifo.h"
#include "ctrl/ctrl2080/ctrl2080fifo.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"
#include "ctrl/ctrl2080/ctrl2080internal.h"
#include "ctrl/ctrl90f1.h"

/* The two RPC wrappers end in a flexible array (`NvU8 params[]`), which bindgen
 * can only express with unsafe helper code; this crate forbids unsafe code.
 * So clang computes their layout here and bindgen emits it as plain constants
 * (the `params` offset is where the allocation/control parameters start). */
#include <stddef.h>
enum {
    RPC_GSP_RM_ALLOC_HCLIENT = offsetof(rpc_gsp_rm_alloc_v03_00, hClient),
    RPC_GSP_RM_ALLOC_HPARENT = offsetof(rpc_gsp_rm_alloc_v03_00, hParent),
    RPC_GSP_RM_ALLOC_HOBJECT = offsetof(rpc_gsp_rm_alloc_v03_00, hObject),
    RPC_GSP_RM_ALLOC_HCLASS = offsetof(rpc_gsp_rm_alloc_v03_00, hClass),
    RPC_GSP_RM_ALLOC_STATUS = offsetof(rpc_gsp_rm_alloc_v03_00, status),
    RPC_GSP_RM_ALLOC_PARAMS_SIZE = offsetof(rpc_gsp_rm_alloc_v03_00, paramsSize),
    RPC_GSP_RM_ALLOC_FLAGS = offsetof(rpc_gsp_rm_alloc_v03_00, flags),
    RPC_GSP_RM_ALLOC_PARAMS = offsetof(rpc_gsp_rm_alloc_v03_00, params),
    RPC_GSP_RM_ALLOC_SIZEOF = sizeof(rpc_gsp_rm_alloc_v03_00),
    RPC_GSP_RM_CONTROL_HCLIENT = offsetof(rpc_gsp_rm_control_v03_00, hClient),
    RPC_GSP_RM_CONTROL_HOBJECT = offsetof(rpc_gsp_rm_control_v03_00, hObject),
    RPC_GSP_RM_CONTROL_CMD = offsetof(rpc_gsp_rm_control_v03_00, cmd),
    RPC_GSP_RM_CONTROL_STATUS = offsetof(rpc_gsp_rm_control_v03_00, status),
    RPC_GSP_RM_CONTROL_PARAMS_SIZE = offsetof(rpc_gsp_rm_control_v03_00, paramsSize),
    RPC_GSP_RM_CONTROL_FLAGS = offsetof(rpc_gsp_rm_control_v03_00, flags),
    RPC_GSP_RM_CONTROL_PARAMS = offsetof(rpc_gsp_rm_control_v03_00, params),
    RPC_GSP_RM_CONTROL_SIZEOF = sizeof(rpc_gsp_rm_control_v03_00),
};
