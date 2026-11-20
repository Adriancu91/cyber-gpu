// Host tool for the SECOND NAK tool (C-147 point 5): the same job as
// nak_dump.c (ask NVK to compile one SPIR-V compute shader for a chosen chip,
// without a GPU, through the nouveau drm-shim), but it also enables
// VK_KHR_cooperative_matrix + 16-bit floats, and prints the cooperative-matrix
// shapes the driver reports for that chip. nak_dump.c is left untouched so the
// programs built with the first tool (Mesa 25.0.7) keep their checksums.
// Usage: nak_dump_cm shader.spv push_constant_bytes
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s failed: %d\n", #x, r_); return 1; } } while (0)

static const char *ty(VkComponentTypeKHR t) {
    switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
    case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
    case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
    case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
    case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
    default: return "?";
    }
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s shader.spv push_bytes\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc(n);
    if (fread(code, 1, n, f) != (size_t)n) { fprintf(stderr, "read failed\n"); return 1; }
    fclose(f);

    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t count = 1; VkPhysicalDevice pd;
    VkResult er = vkEnumeratePhysicalDevices(inst, &count, &pd);
    if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || count == 0) { fprintf(stderr, "no device (drm-shim?)\n"); return 1; }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    fprintf(stderr, "device: %s\n", props.deviceName);

    // The shapes the driver offers for this chip (what NAK can lower to MMA).
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getcm = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!getcm) { fprintf(stderr, "no vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\n"); return 1; }
    uint32_t ncm = 0; CHECK(getcm(pd, &ncm, NULL));
    VkCooperativeMatrixPropertiesKHR *cm = calloc(ncm ? ncm : 1, sizeof *cm);
    for (uint32_t i = 0; i < ncm; i++) cm[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    CHECK(getcm(pd, &ncm, cm));
    for (uint32_t i = 0; i < ncm; i++)
        fprintf(stderr, "coopmat: M=%u N=%u K=%u A=%s B=%s C=%s R=%s sat=%u\n", cm[i].MSize, cm[i].NSize, cm[i].KSize,
                ty(cm[i].AType), ty(cm[i].BType), ty(cm[i].CType), ty(cm[i].ResultType), cm[i].saturatingAccumulation);
    free(cm);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .cooperativeMatrix = VK_TRUE };
    VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &fcm };
    VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f13,
        .bufferDeviceAddress = VK_TRUE, .shaderFloat16 = VK_TRUE, .shaderInt8 = VK_TRUE, .vulkanMemoryModel = VK_TRUE,
        .storageBuffer8BitAccess = VK_TRUE };
    VkPhysicalDeviceVulkan11Features f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12,
        .storageBuffer16BitAccess = VK_TRUE };
    VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11 };
    f2.features.shaderInt64 = VK_TRUE;
    const char *ext[] = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2, .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = ext };
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, NULL, &dev));

    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)n, .pCode = code };
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = (uint32_t)atoi(argv[2]) };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
        .layout = pl };
    VkPipeline p; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &p));
    fprintf(stderr, "pipeline compiled\n");
    vkDestroyPipeline(dev, p, NULL); vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyShaderModule(dev, sm, NULL); vkDestroyDevice(dev, NULL); vkDestroyInstance(inst, NULL);
    free(code);
    return 0;
}
