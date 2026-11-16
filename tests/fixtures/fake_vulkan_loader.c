/* Minimal fake Vulkan loader for the honest-degradation regression test.
 * It exports only vkGetInstanceProcAddr, and the function it returns for
 * vkCreateInstance always fails. A host with this loader must report
 * Vulkan as unavailable, never crash by calling an unresolved entry point. */
#include <string.h>

static int fail_create(const void *info, const void *callbacks, void *instance) {
    (void)info;
    (void)callbacks;
    (void)instance;
    return -3; /* VK_ERROR_INITIALIZATION_FAILED */
}

void *vkGetInstanceProcAddr(void *instance, const char *name) {
    (void)instance;
    if (name != 0 && strcmp(name, "vkCreateInstance") == 0) {
        return (void *)fail_create;
    }
    return 0;
}
