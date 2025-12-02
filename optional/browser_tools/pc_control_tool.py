# optional/browser_tools/pc_control_tool.py — FULL PC CONTROL
"""
Maven PC Control Tool - System monitoring, cleanup, and management.
Like CCleaner + Afterburner + Task Manager combined.

Requirements:
    pip install psutil GPUtil wmi pywin32

Usage:
    pc: status              - Full system overview
    pc: cpu                 - CPU stats
    pc: gpu                 - GPU stats
    pc: ram                 - Memory stats
    pc: disk                - Disk usage
    pc: processes           - List top processes
    pc: kill <name|pid>     - Kill a process
    pc: startup             - List startup programs
    pc: clean               - Clean temp files
    pc: clean browser       - Clean browser cache
    pc: scan                - Security scan
    pc: network             - Network connections
    pc: temps               - All temperatures
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# System monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# GPU monitoring
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Windows-specific
try:
    import wmi
    import win32com.client
    HAS_WMI = True
except ImportError:
    HAS_WMI = False


def _format_bytes(bytes_val: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU information and usage."""
    if not HAS_PSUTIL:
        return {"error": "psutil not installed"}

    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    cpu_freq = psutil.cpu_freq()

    info = {
        "usage_total": f"{psutil.cpu_percent(interval=0.1):.1f}%",
        "usage_per_core": [f"{p:.1f}%" for p in cpu_percent],
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "frequency_current": f"{cpu_freq.current:.0f} MHz" if cpu_freq else "N/A",
        "frequency_max": f"{cpu_freq.max:.0f} MHz" if cpu_freq else "N/A",
    }

    # Try to get temperature
    if HAS_WMI and sys.platform == "win32":
        try:
            w = wmi.WMI(namespace="root\\wmi")
            temps = w.MSAcpi_ThermalZoneTemperature()
            if temps:
                # Convert from tenths of Kelvin to Celsius
                temp_c = (temps[0].CurrentTemperature / 10) - 273.15
                info["temperature"] = f"{temp_c:.1f}°C"
        except Exception:
            pass

    return info


def _get_gpu_info() -> Dict[str, Any]:
    """Get GPU information and usage (NVIDIA via GPUtil, AMD via WMI)."""
    gpu_info = []

    # Try GPUtil first (NVIDIA GPUs)
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    "name": gpu.name,
                    "id": gpu.id,
                    "vendor": "NVIDIA",
                    "load": f"{gpu.load * 100:.1f}%",
                    "memory_used": f"{gpu.memoryUsed:.0f} MB",
                    "memory_total": f"{gpu.memoryTotal:.0f} MB",
                    "memory_percent": f"{(gpu.memoryUsed / gpu.memoryTotal) * 100:.1f}%" if gpu.memoryTotal else "N/A",
                    "temperature": f"{gpu.temperature}°C" if gpu.temperature else "N/A",
                    "driver": gpu.driver,
                })
        except Exception:
            pass

    # Try WMI for AMD/Intel GPUs (Windows)
    if HAS_WMI and sys.platform == "win32" and not gpu_info:
        try:
            w = wmi.WMI()
            for gpu in w.Win32_VideoController():
                # Get VRAM in MB
                vram_mb = int(gpu.AdapterRAM / (1024 * 1024)) if gpu.AdapterRAM else 0
                vendor = "AMD" if "AMD" in (gpu.Name or "") or "Radeon" in (gpu.Name or "") else \
                         "Intel" if "Intel" in (gpu.Name or "") else "Unknown"
                gpu_info.append({
                    "name": gpu.Name or "Unknown GPU",
                    "vendor": vendor,
                    "driver_version": gpu.DriverVersion or "N/A",
                    "memory_total": f"{vram_mb} MB" if vram_mb > 0 else "N/A",
                    "status": gpu.Status or "N/A",
                    "video_processor": gpu.VideoProcessor or "N/A",
                })
        except Exception as e:
            pass

    if not gpu_info:
        return {"error": "No GPU detected (install GPUtil for NVIDIA or check WMI for AMD)"}

    return {"gpus": gpu_info}


def _get_ram_info() -> Dict[str, Any]:
    """Get RAM information."""
    if not HAS_PSUTIL:
        return {"error": "psutil not installed"}

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    return {
        "total": _format_bytes(mem.total),
        "available": _format_bytes(mem.available),
        "used": _format_bytes(mem.used),
        "percent": f"{mem.percent}%",
        "swap_total": _format_bytes(swap.total),
        "swap_used": _format_bytes(swap.used),
        "swap_percent": f"{swap.percent}%",
    }


def _get_disk_info() -> Dict[str, Any]:
    """Get disk information."""
    if not HAS_PSUTIL:
        return {"error": "psutil not installed"}

    disks = []
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disks.append({
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "fstype": partition.fstype,
                "total": _format_bytes(usage.total),
                "used": _format_bytes(usage.used),
                "free": _format_bytes(usage.free),
                "percent": f"{usage.percent}%",
            })
        except PermissionError:
            continue

    return {"disks": disks}


def _get_processes(top_n: int = 15) -> List[Dict[str, Any]]:
    """Get top processes by CPU/memory usage."""
    if not HAS_PSUTIL:
        return [{"error": "psutil not installed"}]

    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
        try:
            pinfo = proc.info
            processes.append({
                "pid": pinfo['pid'],
                "name": pinfo['name'],
                "cpu": f"{pinfo['cpu_percent']:.1f}%",
                "memory": f"{pinfo['memory_percent']:.1f}%",
                "status": pinfo['status'],
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Sort by CPU usage
    processes.sort(key=lambda x: float(x['cpu'].rstrip('%')), reverse=True)
    return processes[:top_n]


def _kill_process(identifier: str) -> str:
    """Kill a process by name or PID."""
    if not HAS_PSUTIL:
        return "Error: psutil not installed"

    killed = []
    try:
        # Try as PID first
        pid = int(identifier)
        proc = psutil.Process(pid)
        proc.terminate()
        return f"Killed process {proc.name()} (PID: {pid})"
    except ValueError:
        # It's a name, not a PID
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if identifier.lower() in proc.info['name'].lower():
                    proc.terminate()
                    killed.append(f"{proc.info['name']} (PID: {proc.info['pid']})")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except psutil.NoSuchProcess:
        return f"Process {identifier} not found"
    except psutil.AccessDenied:
        return f"Access denied killing {identifier}. Run as administrator."

    if killed:
        return f"Killed: {', '.join(killed)}"
    return f"No process matching '{identifier}' found"


def _get_startup_programs() -> List[Dict[str, str]]:
    """Get startup programs (Windows)."""
    startups = []

    if sys.platform != "win32":
        return [{"error": "Startup detection only supported on Windows"}]

    # Registry locations for startup
    import winreg

    startup_locations = [
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
    ]

    for hive, path in startup_locations:
        try:
            key = winreg.OpenKey(hive, path)
            i = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, i)
                    startups.append({
                        "name": name,
                        "command": value,
                        "location": "HKCU" if hive == winreg.HKEY_CURRENT_USER else "HKLM",
                    })
                    i += 1
                except WindowsError:
                    break
            winreg.CloseKey(key)
        except WindowsError:
            continue

    # Startup folder
    startup_folder = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup"
    if startup_folder.exists():
        for item in startup_folder.iterdir():
            startups.append({
                "name": item.name,
                "command": str(item),
                "location": "Startup Folder",
            })

    return startups


def _clean_temp_files() -> Dict[str, Any]:
    """Clean temporary files."""
    cleaned = {
        "files_deleted": 0,
        "space_freed": 0,
        "errors": [],
    }

    temp_dirs = [
        Path(os.environ.get("TEMP", "")),
        Path(os.environ.get("TMP", "")),
        Path("C:/Windows/Temp") if sys.platform == "win32" else Path("/tmp"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Temp",
    ]

    for temp_dir in temp_dirs:
        if not temp_dir.exists():
            continue

        for item in temp_dir.iterdir():
            try:
                if item.is_file():
                    size = item.stat().st_size
                    item.unlink()
                    cleaned["files_deleted"] += 1
                    cleaned["space_freed"] += size
                elif item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    shutil.rmtree(item)
                    cleaned["files_deleted"] += 1
                    cleaned["space_freed"] += size
            except Exception as e:
                cleaned["errors"].append(str(e)[:50])

    cleaned["space_freed_human"] = _format_bytes(cleaned["space_freed"])
    return cleaned


def _clean_browser_cache() -> Dict[str, Any]:
    """Clean browser caches."""
    cleaned = {
        "browsers": [],
        "total_freed": 0,
    }

    local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
    appdata = Path(os.environ.get("APPDATA", ""))

    browser_caches = {
        "Chrome": local_appdata / "Google/Chrome/User Data/Default/Cache",
        "Edge": local_appdata / "Microsoft/Edge/User Data/Default/Cache",
        "Firefox": appdata / "Mozilla/Firefox/Profiles",
        "Brave": local_appdata / "BraveSoftware/Brave-Browser/User Data/Default/Cache",
    }

    for browser, cache_path in browser_caches.items():
        if not cache_path.exists():
            continue

        try:
            if browser == "Firefox":
                # Firefox has profile folders
                for profile in cache_path.iterdir():
                    cache = profile / "cache2"
                    if cache.exists():
                        size = sum(f.stat().st_size for f in cache.rglob('*') if f.is_file())
                        shutil.rmtree(cache)
                        cleaned["browsers"].append({"name": browser, "freed": _format_bytes(size)})
                        cleaned["total_freed"] += size
            else:
                size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                shutil.rmtree(cache_path)
                cleaned["browsers"].append({"name": browser, "freed": _format_bytes(size)})
                cleaned["total_freed"] += size
        except Exception as e:
            cleaned["browsers"].append({"name": browser, "error": str(e)[:50]})

    cleaned["total_freed_human"] = _format_bytes(cleaned["total_freed"])
    return cleaned


def _security_scan() -> Dict[str, Any]:
    """Basic security scan - check for suspicious processes and connections."""
    results = {
        "suspicious_processes": [],
        "suspicious_connections": [],
        "warnings": [],
    }

    if not HAS_PSUTIL:
        return {"error": "psutil not installed"}

    # Whitelist of legitimate apps that run from AppData/Programs
    # These are safe and should NOT be flagged as warnings
    WHITELISTED_APPS = [
        # Browsers
        "opera", "chrome", "firefox", "brave", "edge", "vivaldi", "arc",
        # Development tools
        "python", "node", "npm", "code", "vscode", "cursor", "windsurf",
        "git", "github", "gitkraken", "sublime", "atom", "neovim",
        # Communication
        "discord", "slack", "telegram", "teams", "zoom", "skype", "signal",
        # Gaming
        "steam", "epic", "gog", "battle.net", "origin", "ubisoft",
        # Utilities
        "ollama", "onedrive", "dropbox", "spotify", "notion", "obsidian",
        "1password", "bitwarden", "lastpass", "authy",
        # System tools
        "powershell", "windowsterminal", "alacritty", "wezterm",
        # Media
        "vlc", "obs", "audacity", "gimp", "blender",
        # Other common apps
        "postman", "insomnia", "docker", "wsl", "nvidia", "amd", "razer",
        "logitech", "corsair", "hyperx", "steelseries",
    ]

    # Known suspicious process patterns
    suspicious_patterns = [
        "cryptominer", "miner", "xmrig", "cgminer",
        "keylogger", "rat", "trojan", "malware",
        "powershell -enc", "cmd /c", "wscript",
    ]

    def is_whitelisted(name: str, exe_path: str) -> bool:
        """Check if a process is whitelisted."""
        name_lower = name.lower()
        exe_lower = exe_path.lower()
        for app in WHITELISTED_APPS:
            if app in name_lower or app in exe_lower:
                return True
        return False

    # Check processes
    for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
        try:
            pinfo = proc.info
            name_lower = (pinfo['name'] or "").lower()
            cmdline = " ".join(pinfo['cmdline'] or []).lower()
            exe = pinfo.get('exe', '') or ''

            for pattern in suspicious_patterns:
                if pattern in name_lower or pattern in cmdline:
                    results["suspicious_processes"].append({
                        "pid": pinfo['pid'],
                        "name": pinfo['name'],
                        "reason": f"Matches pattern: {pattern}",
                    })
                    break

            # Check for hidden/system paths running as user
            # BUT skip whitelisted apps
            if exe and ("temp" in exe.lower() or "appdata" in exe.lower()):
                # Skip if it's a whitelisted legitimate app
                if is_whitelisted(pinfo['name'] or '', exe):
                    continue
                # Executable in temp folder - could be suspicious
                results["warnings"].append({
                    "pid": pinfo['pid'],
                    "name": pinfo['name'],
                    "reason": f"Running from temp location: {exe[:50]}",
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Check network connections
    suspicious_ports = [4444, 5555, 6666, 1337, 31337]  # Common RAT ports

    for conn in psutil.net_connections(kind='inet'):
        try:
            if conn.status == 'ESTABLISHED':
                remote_port = conn.raddr.port if conn.raddr else None
                local_port = conn.laddr.port if conn.laddr else None

                if remote_port in suspicious_ports or local_port in suspicious_ports:
                    results["suspicious_connections"].append({
                        "local": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "N/A",
                        "remote": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "N/A",
                        "reason": f"Suspicious port detected",
                    })
        except Exception:
            continue

    results["scan_time"] = datetime.now().isoformat()
    results["status"] = "clean" if not results["suspicious_processes"] and not results["suspicious_connections"] else "issues_found"

    return results


def _get_network_info() -> Dict[str, Any]:
    """Get network information and connections."""
    if not HAS_PSUTIL:
        return {"error": "psutil not installed"}

    # Network interfaces
    interfaces = []
    for name, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family.name == 'AF_INET':  # IPv4
                interfaces.append({
                    "name": name,
                    "ip": addr.address,
                    "netmask": addr.netmask,
                })

    # Network stats
    net_io = psutil.net_io_counters()
    stats = {
        "bytes_sent": _format_bytes(net_io.bytes_sent),
        "bytes_recv": _format_bytes(net_io.bytes_recv),
        "packets_sent": net_io.packets_sent,
        "packets_recv": net_io.packets_recv,
    }

    # Active connections
    connections = []
    for conn in psutil.net_connections(kind='inet')[:20]:  # Limit to 20
        try:
            connections.append({
                "local": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "N/A",
                "remote": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "N/A",
                "status": conn.status,
                "pid": conn.pid,
            })
        except Exception:
            continue

    return {
        "interfaces": interfaces,
        "stats": stats,
        "connections": connections,
    }


def _get_full_status() -> Dict[str, Any]:
    """Get complete system status."""
    return {
        "cpu": _get_cpu_info(),
        "gpu": _get_gpu_info(),
        "ram": _get_ram_info(),
        "disk": _get_disk_info(),
        "top_processes": _get_processes(5),
    }


def pc(command: str) -> str:
    """
    Maven PC Control - Full system control like CCleaner + Afterburner.

    Commands:
        pc: status          - Full system overview
        pc: cpu             - CPU stats
        pc: gpu             - GPU stats
        pc: ram             - Memory stats
        pc: disk            - Disk usage
        pc: temps           - All temperatures
        pc: processes       - List top processes
        pc: kill <name|pid> - Kill a process
        pc: startup         - List startup programs
        pc: clean           - Clean temp files
        pc: clean browser   - Clean browser cache
        pc: scan            - Security scan
        pc: network         - Network info
    """
    import json

    cmd = command.strip().lower()

    # Status commands
    if cmd in ["status", "overview", "all", ""]:
        return json.dumps(_get_full_status(), indent=2)

    if cmd in ["cpu", "processor"]:
        return json.dumps(_get_cpu_info(), indent=2)

    if cmd in ["gpu", "graphics", "video"]:
        return json.dumps(_get_gpu_info(), indent=2)

    if cmd in ["ram", "memory", "mem"]:
        return json.dumps(_get_ram_info(), indent=2)

    if cmd in ["disk", "disks", "storage", "drives"]:
        return json.dumps(_get_disk_info(), indent=2)

    if cmd in ["temps", "temperature", "temperatures"]:
        temps = {
            "cpu": _get_cpu_info().get("temperature", "N/A"),
            "gpu": _get_gpu_info().get("gpus", [{}])[0].get("temperature", "N/A") if _get_gpu_info().get("gpus") else "N/A",
        }
        return json.dumps(temps, indent=2)

    if cmd in ["processes", "procs", "ps", "tasks"]:
        return json.dumps(_get_processes(15), indent=2)

    if cmd.startswith("kill "):
        target = command[5:].strip()
        return _kill_process(target)

    if cmd in ["startup", "startups", "autorun"]:
        return json.dumps(_get_startup_programs(), indent=2)

    if cmd == "clean":
        return json.dumps(_clean_temp_files(), indent=2)

    if cmd in ["clean browser", "clean browsers", "browser cache"]:
        return json.dumps(_clean_browser_cache(), indent=2)

    if cmd in ["scan", "security", "security scan"]:
        return json.dumps(_security_scan(), indent=2)

    if cmd in ["network", "net", "connections"]:
        return json.dumps(_get_network_info(), indent=2)

    # Help
    return """PC Control Commands:
  status     - Full system overview
  cpu        - CPU usage and stats
  gpu        - GPU usage and temps
  ram        - Memory usage
  disk       - Disk usage
  temps      - All temperatures
  processes  - Top processes
  kill <x>   - Kill process by name/PID
  startup    - Startup programs
  clean      - Clean temp files
  clean browser - Clean browser cache
  scan       - Security scan
  network    - Network connections"""


# CLI testing
if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        print(pc(cmd))
    else:
        print(pc("status"))
