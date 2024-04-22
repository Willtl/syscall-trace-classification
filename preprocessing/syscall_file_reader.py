import re

RELEVANT_SYSCALLS = {
    "recvfrom", "write", "ioctl", "read", "sendto", "writev", "close", "socket", "bind", "connect",
    "mkdir", "access", "chmod", "open", "fchown", "rename", "unlink", "umask", "recvmsg", "sendmsg",
    "getdents64", "epoll_wait", "dup", "pread", "pwrit", "fcntl64"
}


class SyscallFileReader:
    def __init__(self, filter_calls=False):
        self.filter_calls = filter_calls

    def read(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Determine the format by checking the first non-empty line
        first_non_empty_line = next((line for line in lines if line.strip()), '').strip()

        # If the line is numeric (possibly space-separated), treat it as a pre-processed sequence
        if first_non_empty_line.replace(' ', '').isnumeric():
            # Assuming pre-processed sequences are space-separated
            syscalls = [int(num) for num in first_non_empty_line.split()]
        else:
            # Define a regular expression pattern for syscalls (assuming syscalls start with word characters and end with '(')
            syscall_pattern = re.compile(r'^\w+\(')

            syscalls = []

            for line in lines:
                # Strip whitespace from the beginning and end of the line
                line = line.strip()

                # Use regular expression to find a match at the start of the line
                match = syscall_pattern.match(line)
                if match:
                    syscall_name = match.group().rstrip('(')  # Extract syscall name and remove trailing '('

                    if self.filter_calls:
                        # Include only relevant syscalls if filtering is enabled
                        if syscall_name in self.RELEVANT_SYSCALLS:
                            syscalls.append(syscall_name)
                    else:
                        # Include all syscalls if no filtering is applied
                        syscalls.append(syscall_name)

        return syscalls