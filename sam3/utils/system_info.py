import psutil
import platform

def available_ram(size_unit="Bytes"):
    vm = psutil.virtual_memory()
    if size_unit == "GB":
        return vm.available / (1024 ** 3)
    elif size_unit == "MB":
        return vm.available / (1024 ** 2)
    elif size_unit == "KB":
        return vm.available / 1024
    return vm.available

def total_ram(size_unit="Bytes"):
    vm = psutil.virtual_memory()
    if size_unit == "GB":
        return vm.total / (1024 ** 3)
    elif size_unit == "MB":
        return vm.total / (1024 ** 2)
    elif size_unit == "KB":
        return vm.total / 1024
    return vm.total

def cpu_usage():
    return psutil.cpu_percent(interval=1)

def cpu_cores():
    return psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)

def get_system_info():
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)

    # ---------------------------
    # RAM Information
    # ---------------------------
    vm = psutil.virtual_memory()

    total_ram_gb = vm.total / (1024 ** 3)
    available_ram_gb = vm.available / (1024 ** 3)
    used_ram_gb = vm.used / (1024 ** 3)

    print("\nRAM Information:")
    print(f"Total RAM:      {total_ram_gb:.2f} GB")
    print(f"Available RAM:  {available_ram_gb:.2f} GB")
    print(f"Used RAM:       {used_ram_gb:.2f} GB")

    # ---------------------------
    # CPU Information
    # ---------------------------
    print("\nCPU Information:")

    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    print(f"Physical Cores: {physical_cores}")
    print(f"Logical Cores:  {logical_cores}")

    freq = psutil.cpu_freq()

    if freq:
        print(f"Current Frequency: {freq.current:.2f} MHz")
        print(f"Min Frequency:     {freq.min:.2f} MHz")
        print(f"Max Frequency:     {freq.max:.2f} MHz")
    else:
        print("CPU frequency information not available.")

    per_core_freq = psutil.cpu_freq(percpu=True)

    print("\nPer-Core Frequencies:")
    for i, core in enumerate(per_core_freq):
        print(f"Core {str(i).zfill(len(str(logical_cores)))}: Current - {core.current:.2f} MHz | Min - {core.min:.2f} MHz | Max - {core.max:.2f} MHz")

    # ---------------------------
    # Extra Info
    # ---------------------------
    print("\nPlatform Info:")
    print(f"Processor: {platform.processor()}")
    print(f"System:    {platform.system()}")
    print(f"Machine:   {platform.machine()}")

    print("\n" + "=" * 50)

    print(f"CPU USAGE: {cpu_usage()}%")


# if __name__ == "__main__":
#     get_system_info()