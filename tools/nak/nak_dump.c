// Host tool (runs on the build machine, never in the OS): asks Mesa's NVK
// driver to compile one SPIR-V compute shader for a chosen NVIDIA chip,
// without a GPU, through Mesa's nouveau drm-shim. With NAK_DEBUG=print NAK
// prints the encoded SM86 words, which tools/nak/build.sh extracts.
// Usage: nak_dump shader.spv push_constant_bytes
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s failed: %d\n", #x, r_); return 1; } } while (0)

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

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .bufferDeviceAddress = VK_TRUE };
    VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12 };
    f2.features.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
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
