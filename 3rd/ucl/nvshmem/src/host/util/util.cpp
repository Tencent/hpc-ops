/****
 * Copyright (c) 2016-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * See License.txt for license information
 ****/

#define __STDC_FORMAT_MACROS 1

#include <execinfo.h>                            // for backtrace
#include <inttypes.h>                            // for PRIu64
#include <sched.h>                               // for sched_getaffi...
#include <signal.h>                              // for signal, SIGSEGV
#include <stdint.h>                              // for uint64_t
#include <stdio.h>                               // for size_t, NULL
#include <stdlib.h>                              // for calloc, exit
#include <string.h>                              // for memcpy, memset
#include <unistd.h>                              // for gethostname
#include <string>                                // for string, basic...
#include "non_abi/nvshmemx_error.h"              // for NVSHMEMI_ERRO...
#include "internal/host/debug.h"                 // for INFO, NVSHMEM...
#include "internal/host/error_codes_internal.h"  // for NVSHMEMI_SUCCESS
#include "internal/host/util.h"                  // for nvshmemu_getHostHash
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for bootstrap_han...
#include <arpa/inet.h>                                                     // for inet_pton
#include <ifaddrs.h>

static void sig_handler(int sig) {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    backtrace_symbols_fd(array, size, STDERR_FILENO);

    exit(1);
}

void setup_sig_handler() { signal(SIGSEGV, sig_handler); }

/* based on DJB2, result = result * 33 + char */
uint64_t nvshmemu_getHostHash() {
    FILE *fp;
    char uuid[1024] = {0};
    uint64_t result = 5381;

    struct ifaddrs *interfaces, *interface;
    bool useIp = false;

    if (getifaddrs(&interfaces) == -1) {
        INFO(NVSHMEM_UTIL, "No socket interface found, use machine-id");
        goto use_machine_id;
    }

    for (interface = interfaces; interface != NULL; interface = interface->ifa_next) {
        if (interface->ifa_name == NULL || interface->ifa_addr == NULL || strcmp(interface->ifa_name, nvshmemi_options.BOOTSTRAP_UID_SOCK_IFNAME) != 0) 
            continue;

        if (interface->ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in* addr = (struct sockaddr_in*)interface->ifa_addr;
            result = static_cast<uint64_t>(ntohl(addr->sin_addr.s_addr));
            useIp = true;
            break;
        }
    }
    freeifaddrs(interfaces);

    if (!useIp) {
        INFO(NVSHMEM_UTIL, "No socket interface found, use machine-id");
        goto use_machine_id;
    }

    INFO(NVSHMEM_UTIL, "ifname: %s, use ip as hostname, result %" PRIu64, nvshmemi_options.BOOTSTRAP_UID_SOCK_IFNAME, result);
    return result;

use_machine_id:
    fp = fopen("/etc/machine-id", "r");
    if (fp == NULL) {
        INFO(NVSHMEM_UTIL, "open uuid file failed, use hostname");
        goto use_hostname;
    }

    if (fgets(uuid, sizeof(uuid), fp) == NULL) {
        INFO(NVSHMEM_UTIL, "get uuid name failed, use hostname");
        fclose(fp);
        goto use_hostname;
    }
    fclose(fp);

    for (int c = 0; c < 1024 && uuid[c] != '\0'; c++) {
      result = ((result << 5) + result) + uuid[c];
    }

    INFO(NVSHMEM_UTIL, "uuid name: %s hash %" PRIu64, uuid, result);

    return result;

use_hostname:
    char hostname[1024];
    int status = 0;

    status = gethostname(hostname, 1024);
    if (status) NVSHMEMI_ERROR_EXIT("gethostname failed \n");

    for (int c = 0; c < 1024 && hostname[c] != '\0'; c++) {
        result = ((result << 5) + result) + hostname[c];
    }

    INFO(NVSHMEM_UTIL, "host name: %s hash %" PRIu64, hostname, result);

    return result;
}

// TODO: force to single node
int nvshmemu_get_num_gpus_per_node() { return 128; }

/* Convert data to a hexadecimal string */
char *nvshmemu_hexdump(void *ptr, size_t len) {
    const char *hex = "0123456789abcdef";

    char *str = (char *)malloc(len * 2 + 1);
    if (str == NULL) return NULL;

    char *ptr_c = (char *)ptr;

    for (size_t i = 0; i < len; i++) {
        str[i * 2] = hex[(ptr_c[i] >> 4) & 0xF];
        str[i * 2 + 1] = hex[ptr_c[i] & 0xF];
    }

    str[len * 2] = '\0';

    return str;
}

/* Wrap 'str' to fit within 'wraplen' columns. Will not break a line of text
 * with no whitespace that exceeds the allowed length. After each line break,
 * insert 'indent' string (if provided).  Caller must free the returned buffer.
 */
char *nvshmemu_wrap(const char *str, const size_t wraplen, const char *indent,
                    const int strip_backticks) {
    const size_t indent_len = indent != NULL ? strlen(indent) : 0;
    size_t str_len = 0, line_len = 0, line_breaks = 0;
    char *str_s = NULL;

    /* Count characters and newlines */
    for (const char *s = str; *s != '\0'; s++, str_len++)
        if (*s == '\n') ++line_breaks;

    /* Worst case is wrapping at 1/2 wraplen plus explicit line breaks. Each
     * wrap adds an indent string. The newline is either already in the source
     * string or replaces a whitespace in the source string */
    const size_t out_len = str_len + 1 + (2 * (str_len / wraplen + 1) + line_breaks) * indent_len;
    char *out = (char *)calloc(out_len, sizeof(char));
    char *str_p = (char *)str;
    std::string statement = "";

    if (out == NULL) {
        fprintf(stderr, "%s:%d Unable to allocate output buffer\n", __FILE__, __LINE__);
        return NULL;
    }

    while (*str_p != '\0' &&
           /* avoid overflowing out */ statement.length() < (out_len - 1)) {
        /* Remember location of last space */
        if (*str_p == ' ') {
            str_s = str_p;
        }
        /* Wrap here if there is a newline */
        else if (*str_p == '\n') {
            str_s = str_p;
            statement += "\n"; /* Append newline and indent */
            if (indent) {
                statement += indent;
            }
            str_p++;
            str_s = NULL;
            line_len = 0;
            continue;
        }

        /* Remove backticks from the input string */
        else if (*str_p == '`' && strip_backticks) {
            str_p++;
            continue;
        }

        /* Reached end of line, try to wrap */
        if (line_len >= wraplen) {
            if (str_s != NULL) {
                str_p = str_s; /* Jump back to last space */
                size_t found =
                    statement.find_last_of(" "); /* Find the last token, remove it from statement as
                                                    it will be appended subsequently */
                auto last_word = statement.substr(found + 1);
                statement.erase(found, found + 1 + last_word.length());
                statement += "\n"; /* Append newline and indent */
                if (indent) {
                    statement += indent;
                }
                str_p++;
                str_s = NULL;
                line_len = 0;
                continue;
            }
        }
        statement += (*str_p);
        str_p++;
        line_len++;
    }

    memset(out, '\0', out_len);
    memcpy(out, statement.c_str(), statement.length());
    return out;
}

/* Output the CPU affinity of the calling thread to the debug log with the
 * provided 'category'.  The 'thread_name' is printed to identify the calling
 * thread.
 */
void nvshmemu_debug_log_cpuset(int category, const char *thread_name) {
    cpu_set_t my_set;

    CPU_ZERO(&my_set);

    int ret = sched_getaffinity(0, sizeof(my_set), &my_set);

    if (ret == 0) {
        char cores_str[1024];
        char *cores_str_wrap;
        int core_count = 0;

        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &my_set)) core_count++;
        }

        size_t off = 0;

        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &my_set)) {
                off += snprintf(cores_str + off, sizeof(cores_str) - off, "%2d ", i);
                if (off >= sizeof(cores_str)) break;
            }
        }

        cores_str_wrap = nvshmemu_wrap(cores_str, /* Line wrap */ 80, /* Indent */ "    ", 0);
        INFO(category, "PE %d (%s) affinity to %d CPUs:\n    %s", nvshmemi_boot_handle.pg_rank,
             thread_name, core_count, cores_str_wrap);
        free(cores_str_wrap);
    }
}

nvshmemResult_t nvshmemu_gethostname(char *hostname, int maxlen) {
    FILE *fp;

    assert(maxlen >= INET_ADDRSTRLEN && maxlen <= 1024);

    static bool initialized = false;
    static char cachedHostname[1024] = {0};
    bool useIp=false;

    if (initialized) {
        strncpy(hostname, cachedHostname, maxlen);
        return NVSHMEMI_SUCCESS;
    }
    
    struct ifaddrs *interfaces, *interface;
    if (getifaddrs(&interfaces) == -1) {
        goto use_machine_id;
    }

    for (interface = interfaces; interface != NULL; interface = interface->ifa_next) {
        if (interface->ifa_name == NULL || interface->ifa_addr == NULL || strcmp(interface->ifa_name, nvshmemi_options.BOOTSTRAP_UID_SOCK_IFNAME) != 0) 
            continue;

        if (interface->ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in* addr = (struct sockaddr_in*)interface->ifa_addr;
            inet_ntop(AF_INET, &addr->sin_addr, hostname, INET_ADDRSTRLEN);
            useIp = true;
            break;
        }
    }
    freeifaddrs(interfaces);

    if (!useIp) {
        goto use_machine_id;
    }
    for (int i = 0; i < maxlen && hostname[i] != '\0'; ++i) {
        if (hostname[i] == '.') {
            hostname[i] = '#';
        }
    }
    goto out;

use_machine_id:
    fp = fopen("/etc/machine-id", "r");
    if (fp == NULL) {
        printf("open uuid file failed, use hostname");
        goto use_real_hostname;
    }

    if (fgets(hostname, maxlen, fp) == NULL) {
      printf("get uuid name failed, use hostname");
      fclose(fp);
      goto use_real_hostname;
    }
    fclose(fp);
    goto out;

use_real_hostname:
    if (gethostname(hostname, maxlen) != 0) {
        strncpy(hostname, "unknown", maxlen);
        return NVSHMEMI_SYSTEM_ERROR;
    }

out:
    int i = 0;
    while ((hostname[i] != '.') && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
    hostname[i] = '\0';
    strncpy(cachedHostname, hostname, maxlen);
    initialized = true;
    return NVSHMEMI_SUCCESS;
}
