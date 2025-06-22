#!/usr/bin/env python3
"""
Integrated Advanced Shell - Deliverable 4
Complete OS Simulation with Process Management, Memory Management, 
Security, Scheduling, and Advanced I/O Features
"""

import os
import sys
import subprocess
import threading
import time
import signal
import shlex
import tempfile
import hashlib
import uuid
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from collections import deque
import platform

# ============================================================================
# CORE ENUMS AND DATA STRUCTURES
# ============================================================================

class ProcessState(Enum):
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    TERMINATED = "terminated"

class UserRole(Enum):
    ADMIN = "admin"
    STANDARD = "standard"
    GUEST = "guest"

class Permission(Enum):
    READ = "r"
    WRITE = "w" 
    EXECUTE = "x"

class PageReplacementAlgorithm(Enum):
    FIFO = "FIFO"
    LRU = "LRU"

@dataclass
class Process:
    pid: int
    name: str
    state: ProcessState
    burst_time: float
    remaining_time: float
    priority: int
    arrival_time: float = 0.0
    start_time: float = 0.0
    completion_time: float = 0.0
    waiting_time: float = 0.0
    turnaround_time: float = 0.0

@dataclass
class User:
    username: str
    password_hash: str
    role: UserRole
    home_directory: str
    last_login: str

@dataclass
class FilePermissions:
    owner: str
    owner_permissions: Set[Permission]
    group_permissions: Set[Permission]
    other_permissions: Set[Permission]

# ============================================================================
# MEMORY MANAGEMENT SYSTEM
# ============================================================================

class MemoryManager:
    def __init__(self, total_frames=16):
        self.total_frames = total_frames
        self.physical_memory = [None] * total_frames
        self.process_pages = {}  # process_id -> list of pages
        self.page_table = {}     # (process_id, page_number) -> frame_number
        self.algorithm = PageReplacementAlgorithm.FIFO
        self.fifo_queue = deque()
        self.lru_order = []
        self.page_faults = 0
        self.total_accesses = 0

    def display_status(self):
        """Display current memory status"""
        used_frames = sum(1 for frame in self.physical_memory if frame is not None)
        free_frames = self.total_frames - used_frames
        
        print(f"Memory Management Status:")
        print(f"  Total Frames: {self.total_frames}")
        print(f"  Used Frames: {used_frames}")
        print(f"  Free Frames: {free_frames}")
        print(f"  Algorithm: {self.algorithm.value}")
        print(f"  Page Faults: {self.page_faults}")
        print(f"  Total Accesses: {self.total_accesses}")
        
        if self.total_accesses > 0:
            fault_rate = (self.page_faults / self.total_accesses) * 100
            print(f"  Fault Rate: {fault_rate:.1f}%")
        
        if self.process_pages:
            print(f"  Active Processes:")
            for pid, pages in self.process_pages.items():
                print(f"    Process {pid}: {len(pages)} pages")

    def allocate_pages(self, process_id, num_pages):
        """Allocate pages for a process"""
        if process_id not in self.process_pages:
            self.process_pages[process_id] = []
        
        allocated = 0
        for page_num in range(num_pages):
            page_key = (process_id, page_num)
            if page_key not in self.page_table:
                self.process_pages[process_id].append(page_num)
                allocated += 1
        
        print(f"Allocated {allocated} pages for process {process_id}")

    def access_page(self, process_id, page_number):
        """Access a page (with page fault handling)"""
        self.total_accesses += 1
        page_key = (process_id, page_number)
        
        if page_key in self.page_table:
            # Page hit
            frame_number = self.page_table[page_key]
            if self.algorithm == PageReplacementAlgorithm.LRU:
                self._update_lru(frame_number)
            print(f"Page hit: Process {process_id}, Page {page_number} -> Frame {frame_number}")
        else:
            # Page fault
            self.page_faults += 1
            frame_number = self._handle_page_fault(process_id, page_number)
            print(f"Page fault: Process {process_id}, Page {page_number} -> Frame {frame_number}")

    def _handle_page_fault(self, process_id, page_number):
        """Handle page fault by loading page into memory"""
        # Find free frame or replace using algorithm
        frame_number = self._find_free_frame()
        if frame_number is None:
            frame_number = self._replace_page()
        
        # Load page into frame
        page_info = {
            'process_id': process_id,
            'page_number': page_number
        }
        self.physical_memory[frame_number] = page_info
        self.page_table[(process_id, page_number)] = frame_number
        
        # Update algorithm data structures
        if self.algorithm == PageReplacementAlgorithm.FIFO:
            self.fifo_queue.append(frame_number)
        elif self.algorithm == PageReplacementAlgorithm.LRU:
            self.lru_order.append(frame_number)
        
        return frame_number

    def _find_free_frame(self):
        """Find a free frame in physical memory"""
        for i, frame in enumerate(self.physical_memory):
            if frame is None:
                return i
        return None

    def _replace_page(self):
        """Replace a page using the current algorithm"""
        if self.algorithm == PageReplacementAlgorithm.FIFO:
            return self.fifo_queue.popleft()
        elif self.algorithm == PageReplacementAlgorithm.LRU:
            return self.lru_order.pop(0)

    def _update_lru(self, frame_number):
        """Update LRU order for accessed frame"""
        if frame_number in self.lru_order:
            self.lru_order.remove(frame_number)
        self.lru_order.append(frame_number)

    def deallocate_pages(self, process_id):
        """Deallocate all pages for a process"""
        if process_id not in self.process_pages:
            print(f"Process {process_id} has no allocated pages")
            return
        
        # Remove pages from physical memory and page table
        pages_to_remove = []
        for (pid, page_num), frame_num in self.page_table.items():
            if pid == process_id:
                self.physical_memory[frame_num] = None
                pages_to_remove.append((pid, page_num))
                
                # Remove from algorithm data structures
                if frame_num in self.fifo_queue:
                    self.fifo_queue.remove(frame_num)
                if frame_num in self.lru_order:
                    self.lru_order.remove(frame_num)
        
        for page_key in pages_to_remove:
            del self.page_table[page_key]
        
        num_deallocated = len(self.process_pages[process_id])
        del self.process_pages[process_id]
        print(f"Deallocated {num_deallocated} pages for process {process_id}")

    def set_algorithm(self, algorithm):
        """Set page replacement algorithm"""
        self.algorithm = algorithm
        # Reset algorithm data structures
        self.fifo_queue.clear()
        self.lru_order.clear()
        
        # Rebuild data structures for current pages
        for frame_num, page_info in enumerate(self.physical_memory):
            if page_info is not None:
                if algorithm == PageReplacementAlgorithm.FIFO:
                    self.fifo_queue.append(frame_num)
                elif algorithm == PageReplacementAlgorithm.LRU:
                    self.lru_order.append(frame_num)

    def compare_algorithms(self):
        """Compare FIFO vs LRU algorithms"""
        print("Page Replacement Algorithm Comparison:")
        print("Current algorithm:", self.algorithm.value)
        print("FIFO: First In, First Out - Simple but may suffer from Belady's anomaly")
        print("LRU:  Least Recently Used - Better performance but more overhead")
        
        if self.total_accesses > 0:
            print(f"Current fault rate: {(self.page_faults/self.total_accesses)*100:.1f}%")

# ============================================================================
# PROCESS SCHEDULING SYSTEM
# ============================================================================

class ProcessScheduler:
    def __init__(self):
        self.processes = []
        self.completed_processes = []
        self.current_process = None
        self.algorithm = "round_robin"
        self.time_quantum = 2.0
        self.current_time = 0.0
        self.context_switches = 0
        self.running = False
        self.next_pid = 1

    def add_process(self, burst_time, priority):
        """Add a new process to the ready queue"""
        process = Process(
            pid=self.next_pid,
            name=f"Process_{self.next_pid}",
            state=ProcessState.READY,
            burst_time=burst_time,
            remaining_time=burst_time,
            priority=priority,
            arrival_time=self.current_time
        )
        self.processes.append(process)
        self.next_pid += 1
        print(f"Added {process.name} (PID {process.pid}) with burst time {burst_time}s, priority {priority}")

    def set_round_robin(self, quantum):
        """Set scheduler to Round Robin"""
        self.algorithm = "round_robin"
        self.time_quantum = quantum
        print(f"Scheduler set to Round Robin with quantum {quantum}s")

    def set_priority_scheduling(self):
        """Set scheduler to Priority-based"""
        self.algorithm = "priority"
        print("Scheduler set to Priority-based scheduling")

    def start_scheduling(self):
        """Start the scheduler"""
        self.running = True
        print("Process scheduler started")

    def stop_scheduling(self):
        """Stop the scheduler"""
        self.running = False
        print("Process scheduler stopped")

    def get_status(self):
        """Get current scheduler status"""
        status = f"Process Scheduler Status:\n"
        status += f"  Algorithm: {self.algorithm}\n"
        if self.algorithm == "round_robin":
            status += f"  Time Quantum: {self.time_quantum}s\n"
        status += f"  Running: {self.running}\n"
        status += f"  Current Time: {self.current_time:.1f}s\n"
        status += f"  Ready Processes: {len(self.processes)}\n"
        status += f"  Completed: {len(self.completed_processes)}\n"
        
        if self.current_process:
            status += f"  Current Process: PID {self.current_process.pid}\n"
        
        return status

    def get_metrics(self):
        """Get performance metrics"""
        if not self.completed_processes:
            return "No completed processes to analyze"
        
        total_turnaround = sum(p.turnaround_time for p in self.completed_processes)
        total_waiting = sum(p.waiting_time for p in self.completed_processes)
        count = len(self.completed_processes)
        
        metrics = f"Scheduler Performance Metrics:\n"
        metrics += f"  Completed Processes: {count}\n"
        metrics += f"  Average Turnaround Time: {total_turnaround/count:.2f}s\n"
        metrics += f"  Average Waiting Time: {total_waiting/count:.2f}s\n"
        metrics += f"  Context Switches: {self.context_switches}\n"
        
        return metrics

    def simulate_workload(self, duration):
        """Simulate scheduler workload"""
        print(f"Simulating workload for {duration} seconds...")
        
        # Add sample processes if none exist
        if not self.processes and not self.current_process:
            self.add_process(3.0, 1)
            self.add_process(2.0, 2)
            self.add_process(4.0, 1)
        
        self.start_scheduling()
        start_time = self.current_time
        
        while self.current_time < start_time + duration and (self.processes or self.current_process):
            self._schedule_step()
            time.sleep(0.1)
        
        print(f"Simulation completed. Total time: {self.current_time:.1f}s")

    def _schedule_step(self):
        """Execute one scheduling step"""
        if not self.running:
            return
        
        if not self.current_process and self.processes:
            # Select next process
            if self.algorithm == "priority":
                self.processes.sort(key=lambda p: p.priority)
            
            self.current_process = self.processes.pop(0)
            self.current_process.state = ProcessState.RUNNING
            self.current_process.start_time = self.current_time
            self.context_switches += 1
        
        if self.current_process:
            # Execute current process
            execution_time = min(0.5, self.current_process.remaining_time)
            if self.algorithm == "round_robin":
                execution_time = min(execution_time, self.time_quantum)
            
            self.current_process.remaining_time -= execution_time
            self.current_time += execution_time
            
            # Check if process completed
            if self.current_process.remaining_time <= 0:
                self.current_process.completion_time = self.current_time
                self.current_process.turnaround_time = self.current_process.completion_time - self.current_process.arrival_time
                self.current_process.waiting_time = self.current_process.turnaround_time - self.current_process.burst_time
                self.current_process.state = ProcessState.TERMINATED
                self.completed_processes.append(self.current_process)
                self.current_process = None
            elif self.algorithm == "round_robin" and execution_time >= self.time_quantum:
                # Time quantum expired, preempt process
                self.current_process.state = ProcessState.READY
                self.processes.append(self.current_process)
                self.current_process = None

# ============================================================================
# PROGRAM COUNTER SIMULATION
# ============================================================================

class ProgramCounter:
    def __init__(self):
        self.pc = 0
        self.running = False
        self.instructions = []

    def start(self, start_address=0):
        """Start program execution"""
        self.pc = start_address
        self.running = True
        print(f"Program Counter started at address {start_address}")

    def stop(self):
        """Stop program execution"""
        self.running = False
        print(f"Program Counter stopped at address {self.pc}")

    def get_current_value(self):
        """Get current PC value"""
        return self.pc

    def increment(self):
        """Increment program counter"""
        if self.running:
            self.pc += 1

    def jump(self, address):
        """Jump to specific address"""
        if self.running:
            self.pc = address
            print(f"Jumped to address {address}")

# ============================================================================
# SECURITY MANAGEMENT SYSTEM
# ============================================================================

class SecurityManager:
    def __init__(self):
        self.users = {}
        self.current_user = None
        self.file_permissions = {}
        self._initialize_default_users()

    def _initialize_default_users(self):
        """Initialize default users"""
        # Create default users
        self.users["admin"] = User(
            username="admin",
            password_hash=self._hash_password("admin123"),
            role=UserRole.ADMIN,
            home_directory="/home/admin",
            last_login=""
        )
        
        self.users["user"] = User(
            username="user", 
            password_hash=self._hash_password("user123"),
            role=UserRole.STANDARD,
            home_directory="/home/user",
            last_login=""
        )
        
        self.users["guest"] = User(
            username="guest",
            password_hash=self._hash_password("guest123"), 
            role=UserRole.GUEST,
            home_directory="/home/guest",
            last_login=""
        )

    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate(self, username, password):
        """Authenticate user"""
        if username not in self.users:
            return False
        
        user = self.users[username]
        password_hash = self._hash_password(password)
        
        if user.password_hash == password_hash:
            self.current_user = user
            user.last_login = time.strftime("%Y-%m-%d %H:%M:%S")
            return True
        
        return False

    def create_user(self, username, password, role_str):
        """Create new user"""
        if username in self.users:
            print(f"User {username} already exists")
            return False
        
        try:
            role = UserRole(role_str.lower())
        except ValueError:
            print(f"Invalid role: {role_str}")
            return False
        
        self.users[username] = User(
            username=username,
            password_hash=self._hash_password(password),
            role=role,
            home_directory=f"/home/{username}",
            last_login=""
        )
        
        print(f"User {username} created successfully with role {role_str}")
        return True

    def set_file_permissions(self, filepath, permissions_str):
        """Set file permissions (rwx format)"""
        if len(permissions_str) != 9:
            print("Invalid permission format. Use rwxrwxrwx format")
            return False
        
        # Parse permission string
        owner_perms = self._parse_permission_triple(permissions_str[0:3])
        group_perms = self._parse_permission_triple(permissions_str[3:6])
        other_perms = self._parse_permission_triple(permissions_str[6:9])
        
        self.file_permissions[filepath] = FilePermissions(
            owner=self.current_user.username,
            owner_permissions=owner_perms,
            group_permissions=group_perms,
            other_permissions=other_perms
        )
        
        return True

    def _parse_permission_triple(self, perm_str):
        """Parse rwx permission triple"""
        perms = set()
        if perm_str[0] == 'r':
            perms.add(Permission.READ)
        if perm_str[1] == 'w':
            perms.add(Permission.WRITE)
        if perm_str[2] == 'x':
            perms.add(Permission.EXECUTE)
        return perms

    def check_file_permission(self, filepath, operation):
        """Check if current user has permission for file operation"""
        if not self.current_user:
            return False
        
        # Admin can do everything
        if self.current_user.role == UserRole.ADMIN:
            return True
        
        # If no explicit permissions set, allow for now
        if filepath not in self.file_permissions:
            return True
        
        file_perm = self.file_permissions[filepath]
        
        # Check permissions based on user type
        if file_perm.owner == self.current_user.username:
            # Owner permissions
            if operation == 'read':
                return Permission.READ in file_perm.owner_permissions
            elif operation == 'write':
                return Permission.WRITE in file_perm.owner_permissions
            elif operation == 'execute':
                return Permission.EXECUTE in file_perm.owner_permissions
        else:
            # Other permissions
            if operation == 'read':
                return Permission.READ in file_perm.other_permissions
            elif operation == 'write':
                return Permission.WRITE in file_perm.other_permissions
            elif operation == 'execute':
                return Permission.EXECUTE in file_perm.other_permissions
        
        return False

# ============================================================================
# INTEGRATED ADVANCED SHELL
# ============================================================================

class IntegratedAdvancedShell:
    def __init__(self):
        # Initialize all subsystems
        self.memory_manager = MemoryManager()
        self.process_scheduler = ProcessScheduler()
        self.program_counter = ProgramCounter()
        self.security_manager = SecurityManager()
        
        # Shell state
        self.running = False
        self.jobs = {}
        self.job_counter = 1
        self.history = []
        self.is_windows = platform.system() == "Windows"
        
        # Built-in commands
        self.built_in_commands = {
            'cd', 'pwd', 'exit', 'echo', 'clear', 'ls', 'cat', 'mkdir', 'rmdir',
            'rm', 'touch', 'kill', 'jobs', 'fg', 'bg', 'help', 'whoami', 'logout',
            'create_user', 'chmod', 'su', 'mem_status', 'mem_alloc', 'mem_access',
            'mem_dealloc', 'mem_algorithm', 'schedule_rr', 'schedule_priority',
            'add_process', 'scheduler_status', 'scheduler_metrics', 'stop_scheduler',
            'pc_start', 'pc_stop', 'pc_status', 'simulate_workload', 'compare_algorithms',
            'security_help', 'scheduler_help', 'memory_help', 'grep', 'head', 'tail',
            'wc', 'create_sample_data', 'ps', 'show_files', 'test_redirect'
        }

    def run(self):
        """Main shell execution loop"""
        # Handle login
        if not self.login_prompt():
            return
        
        self.running = True
        
        # Main command loop
        while self.running:
            self.update_job_status()
            
            try:
                command_line = input(self.display_prompt())
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                continue
            
            if not command_line.strip():
                continue
            
            # Check for background execution
            background = command_line.endswith(" &")
            if background:
                command_line = command_line[:-2].strip()
            
            # Handle output redirection
            redirect_file = None
            append_mode = False
            
            if '>>' in command_line:
                parts = command_line.split('>>', 1)
                command_line = parts[0].strip()
                redirect_file = parts[1].strip()
                append_mode = True
            elif '>' in command_line:
                parts = command_line.split('>', 1)
                command_line = parts[0].strip()
                redirect_file = parts[1].strip()
                append_mode = False
            
            # Execute command
            if redirect_file:
                self.execute_with_redirection(command_line, redirect_file, append_mode, background)
            else:
                commands = self.parse_command_with_pipes(command_line)
                self.execute_piped_commands(commands, background)

    def login_prompt(self):
        """Handle user login"""
        print("Please login to continue...")
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            try:
                username = input("Username: ")
                password = input("Password: ")
                
                if self.security_manager.authenticate(username, password):
                    user = self.security_manager.current_user
                    session_token = str(uuid.uuid4())[:16]
                    
                    print("Login successful. Welcome, {}!".format(username))
                    print(f"Welcome to the Advanced Shell, {username}!")
                    print(f"Role: {user.role.value.title()}")
                    print(f"Session Token: {session_token}")
                    print("Type 'help' to see available commands")
                    return True
                else:
                    attempts += 1
                    remaining = max_attempts - attempts
                    if remaining > 0:
                        print(f"Invalid credentials. {remaining} attempts remaining.")
                    else:
                        print("Maximum login attempts exceeded.")
                        
            except (EOFError, KeyboardInterrupt):
                print("\nLogin cancelled.")
                return False
        
        return False

    def display_prompt(self):
        """Display command prompt"""
        user = self.security_manager.current_user
        hostname = "Mandodari"
        current_dir = os.getcwd()
        
        if self.is_windows:
            # Windows-style prompt
            prompt = f"{user.username}@{hostname}:{current_dir}# " if user.role == UserRole.ADMIN else f"{user.username}@{hostname}:{current_dir}$ "
        else:
            # Unix-style prompt
            prompt = f"{user.username}@{hostname}:~# " if user.role == UserRole.ADMIN else f"{user.username}@{hostname}:~$ "
        
        return prompt

    def parse_command_with_pipes(self, command_line):
        """Parse command line with pipe support"""
        if not command_line.strip():
            return []
        
        # Split by pipes
        pipe_commands = command_line.split('|')
        parsed_commands = []
        
        for cmd in pipe_commands:
            cmd = cmd.strip()
            if cmd:
                try:
                    parts = shlex.split(cmd)
                    parsed_commands.append(parts)
                except:
                    parsed_commands.append(cmd.split())
        
        return parsed_commands

    def execute_with_redirection(self, command_line, redirect_file, append_mode, background):
        """Execute command with output redirection"""
        # Check write permission
        if not self.security_manager.check_file_permission(redirect_file, 'write'):
            print(f"Permission denied: cannot write to {redirect_file}")
            return
        
        # Parse command
        parts = command_line.strip().split()
        if not parts:
            return
        
        command = parts[0]
        args = parts[1:]
        
        # Capture output
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        
        try:
            with redirect_stdout(output_buffer):
                if command == "echo":
                    text = " ".join(args)
                    print(text)
                elif command == "ls":
                    self.list_directory(args)
                elif command == "pwd":
                    print(os.getcwd())
                elif command == "whoami":
                    print(self.security_manager.current_user.username)
                elif command == "mem_status":
                    self.memory_manager.display_status()
                elif command == "scheduler_status":
                    print(self.process_scheduler.get_status())
                elif command == "scheduler_metrics":
                    print(self.process_scheduler.get_metrics())
                else:
                    # Try built-in command
                    if not self.execute_built_in_command(command, args):
                        # External command
                        self.execute_external_with_redirection(command, args, redirect_file, append_mode)
                        return
        except Exception as e:
            output_buffer.write(f"Error: {e}\n")
        
        # Write to file
        output = output_buffer.getvalue()
        mode = 'a' if append_mode else 'w'
        
        try:
            with open(redirect_file, mode) as f:
                f.write(output)
            print(f"Output redirected to {redirect_file}")
        except Exception as e:
            print(f"Error writing to file: {e}")

    def execute_external_with_redirection(self, command, args, redirect_file, append_mode):
        """Execute external command with redirection"""
        try:
            mode = 'a' if append_mode else 'w'
            with open(redirect_file, mode) as f:
                if self.is_windows:
                    subprocess.run(f"{command} {' '.join(args)}", stdout=f, shell=True)
                else:
                    subprocess.run([command] + args, stdout=f)
            print(f"Output redirected to {redirect_file}")
        except Exception as e:
            print(f"Error: {e}")

    def execute_piped_commands(self, commands, background=False):
        """Execute piped commands"""
        if not commands:
            return

        if len(commands) == 1:
            # Single command
            command, args = commands[0][0], commands[0][1:]
            if not self.execute_built_in_command(command, args):
                self.execute_external_command(command, args, background)
            return

        # Multiple commands with piping
        processes = []
        prev_stdout = None
        
        try:
            for i, cmd_parts in enumerate(commands):
                command = cmd_parts[0]
                args = cmd_parts[1:] if len(cmd_parts) > 1 else []
                
                if self.is_built_in_command(command):
                    # Handle built-in command in pipe
                    output = self.execute_built_in_with_pipe(command, args, prev_stdout)
                    if i < len(commands) - 1:
                        # Create temp file for output
                        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
                        temp_file.write(output)
                        temp_file.close()
                        prev_stdout = open(temp_file.name, 'r')
                    else:
                        print(output, end='')
                else:
                    # External command
                    stdin = prev_stdout
                    stdout = subprocess.PIPE if i < len(commands) - 1 else None
                    
                    process = subprocess.Popen(
                        [command] + args,
                        stdin=stdin,
                        stdout=stdout,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        shell=self.is_windows
                    )
                    
                    processes.append(process)
                    
                    if prev_stdout and prev_stdout != subprocess.PIPE:
                        prev_stdout.close()
                    
                    prev_stdout = process.stdout
            
            # Wait for completion
            for process in processes:
                process.wait()
                
        except Exception as e:
            print(f"Error in piped commands: {e}")
        finally:
            if prev_stdout and prev_stdout != subprocess.PIPE:
                prev_stdout.close()

    def execute_built_in_command(self, command, args):
        """Execute built-in commands"""
        if command == "exit" or command == "logout":
            self.running = False
            return True
        elif command == "help":
            self.show_help()
            return True
        elif command == "echo":
            text = " ".join(args)
            print(text)
            return True
        elif command == "pwd":
            print(os.getcwd())
            return True
        elif command == "cd":
            self.change_directory(args)
            return True
        elif command == "ls":
            self.list_directory(args)
            return True
        elif command == "cat":
            self.display_file_contents(args)
            return True
        elif command == "clear":
            os.system('cls' if self.is_windows else 'clear')
            return True
        elif command == "whoami":
            print(self.security_manager.current_user.username)
            return True
        elif command == "create_user":
            if len(args) >= 3:
                self.security_manager.create_user(args[0], args[1], args[2])
            else:
                print("Usage: create_user <username> <password> <role>")
            return True
        elif command == "chmod":
            if len(args) >= 2:
                self.security_manager.set_file_permissions(args[0], args[1])
                print(f"Permissions set for {args[0]}")
            else:
                print("Usage: chmod <file> <permissions>")
            return True
        elif command == "su":
            if args:
                username = args[0]
                password = input(f"Password for {username}: ")
                if self.security_manager.authenticate(username, password):
                    print(f"Login successful. Welcome, {username}!")
                    print(f"Switched to user {username}")
                else:
                    print("Authentication failed")
            return True
        elif command == "mem_status":
            self.memory_manager.display_status()
            return True
        elif command == "mem_alloc":
            if len(args) >= 2:
                self.memory_manager.allocate_pages(int(args[0]), int(args[1]))
            else:
                print("Usage: mem_alloc <process_id> <pages>")
            return True
        elif command == "mem_access":
            if len(args) >= 2:
                self.memory_manager.access_page(int(args[0]), int(args[1]))
            else:
                print("Usage: mem_access <process_id> <page_number>")
            return True
        elif command == "mem_dealloc":
            if args:
                self.memory_manager.deallocate_pages(int(args[0]))
            else:
                print("Usage: mem_dealloc <process_id>")
            return True
        elif command == "mem_algorithm":
            if args and args[0].upper() in ["FIFO", "LRU"]:
                algorithm = PageReplacementAlgorithm.FIFO if args[0].upper() == "FIFO" else PageReplacementAlgorithm.LRU
                self.memory_manager.set_algorithm(algorithm)
                print(f"Algorithm set to {args[0].upper()}")
            else:
                print("Usage: mem_algorithm <FIFO|LRU>")
            return True
        elif command == "compare_algorithms":
            self.memory_manager.compare_algorithms()
            return True
        elif command == "schedule_rr":
            if args:
                quantum = float(args[0])
                self.process_scheduler.set_round_robin(quantum)
            else:
                print("Usage: schedule_rr <time_quantum>")
            return True
        elif command == "schedule_priority":
            self.process_scheduler.set_priority_scheduling()
            return True
        elif command == "add_process":
            if len(args) >= 2:
                self.process_scheduler.add_process(float(args[0]), int(args[1]))
            else:
                print("Usage: add_process <burst_time> <priority>")
            return True
        elif command == "scheduler_status":
            print(self.process_scheduler.get_status())
            return True
        elif command == "scheduler_metrics":
            print(self.process_scheduler.get_metrics())
            return True
        elif command == "simulate_workload":
            duration = int(args[0]) if args else 5
            self.process_scheduler.simulate_workload(duration)
            return True
        elif command == "stop_scheduler":
            self.process_scheduler.stop_scheduling()
            return True
        elif command == "pc_start":
            address = int(args[0]) if args else 0
            self.program_counter.start(address)
            return True
        elif command == "pc_stop":
            self.program_counter.stop()
            return True
        elif command == "pc_status":
            print(f"Program Counter: {self.program_counter.get_current_value()}")
            return True
        elif command == "test_redirect":
            self.test_redirect_command()
            return True
        elif command == "show_files":
            self.show_files_command()
            return True
        
        return False

    def execute_built_in_with_pipe(self, command, args, stdin_file):
        """Execute built-in command for pipe processing"""
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer):
            if command == "echo":
                print(" ".join(args))
            elif command == "ls":
                self.list_directory(args)
            elif command == "cat":
                if stdin_file:
                    content = stdin_file.read()
                    print(content, end='')
                else:
                    self.display_file_contents(args)
            elif command == "grep":
                self.grep_command(args, stdin_file)
            elif command == "head":
                self.head_command(args, stdin_file)
            elif command == "tail":
                self.tail_command(args, stdin_file)
            elif command == "wc":
                self.wc_command(args, stdin_file)
            elif command == "whoami":
                print(self.security_manager.current_user.username)
            elif command == "mem_status":
                self.memory_manager.display_status()
            elif command == "scheduler_status":
                print(self.process_scheduler.get_status())
        
        return output_buffer.getvalue()

    def is_built_in_command(self, command):
        """Check if command is built-in"""
        return command in self.built_in_commands

    def execute_external_command(self, command, args, background=False):
        """Execute external command"""
        try:
            if background:
                process = subprocess.Popen([command] + args)
                job_id = self.job_counter
                self.jobs[job_id] = {
                    'pid': process.pid,
                    'command': f"{command} {' '.join(args)}",
                    'process': process
                }
                self.job_counter += 1
                print(f"[{job_id}] {process.pid}")
            else:
                subprocess.run([command] + args)
        except FileNotFoundError:
            print(f"Command not found: {command}")
        except Exception as e:
            print(f"Error: {e}")

    def update_job_status(self):
        """Update background job status"""
        completed_jobs = []
        for job_id, job_info in self.jobs.items():
            if job_info['process'].poll() is not None:
                completed_jobs.append(job_id)
        
        for job_id in completed_jobs:
            job_info = self.jobs.pop(job_id)
            print(f"[{job_id}] Done\t{job_info['command']}")

    # File operation methods
    def change_directory(self, args):
        """Change directory"""
        if not args:
            directory = os.path.expanduser("~")
        else:
            directory = args[0]
        
        try:
            os.chdir(directory)
        except FileNotFoundError:
            print(f"cd: {directory}: No such file or directory")
        except PermissionError:
            print(f"cd: {directory}: Permission denied")

    def list_directory(self, args):
        """List directory contents"""
        try:
            directory = args[0] if args else "."
            items = os.listdir(directory)
            for item in sorted(items):
                print(item)
        except Exception as e:
            print(f"ls: {e}")

    def display_file_contents(self, args):
        """Display file contents"""
        if not args:
            print("cat: missing file argument")
            return
        
        for filename in args:
            if not self.security_manager.check_file_permission(filename, 'read'):
                print(f"Permission denied: read access to {filename}")
                continue
            
            try:
                with open(filename, 'r') as f:
                    print(f.read(), end='')
            except FileNotFoundError:
                print(f"cat: {filename}: No such file or directory")
            except Exception as e:
                print(f"cat: {filename}: {e}")

    def grep_command(self, args, stdin_file):
        """Grep command implementation"""
        if not args:
            print("grep: missing pattern")
            return
        
        pattern = args[0]
        
        if stdin_file:
            for line in stdin_file:
                if pattern in line:
                    print(line, end='')
        elif len(args) > 1:
            for filename in args[1:]:
                try:
                    with open(filename, 'r') as f:
                        for line in f:
                            if pattern in line:
                                print(line, end='')
                except Exception as e:
                    print(f"grep: {e}")

    def head_command(self, args, stdin_file):
        """Head command implementation"""
        lines = 10
        if args and args[0].startswith('-'):
            try:
                lines = int(args[0][1:])
                args = args[1:]
            except:
                pass
        
        if stdin_file:
            for i, line in enumerate(stdin_file):
                if i >= lines:
                    break
                print(line, end='')
        elif args:
            for filename in args:
                try:
                    with open(filename, 'r') as f:
                        for i, line in enumerate(f):
                            if i >= lines:
                                break
                            print(line, end='')
                except Exception as e:
                    print(f"head: {e}")

    def tail_command(self, args, stdin_file):
        """Tail command implementation"""
        lines = 10
        if args and args[0].startswith('-'):
            try:
                lines = int(args[0][1:])
                args = args[1:]
            except:
                pass
        
        if stdin_file:
            all_lines = stdin_file.readlines()
            for line in all_lines[-lines:]:
                print(line, end='')
        elif args:
            for filename in args:
                try:
                    with open(filename, 'r') as f:
                        all_lines = f.readlines()
                        for line in all_lines[-lines:]:
                            print(line, end='')
                except Exception as e:
                    print(f"tail: {e}")

    def wc_command(self, args, stdin_file):
        """Word count command"""
        def count_file(content):
            lines = content.count('\n')
            words = len(content.split())
            chars = len(content)
            return lines, words, chars
        
        if stdin_file:
            content = stdin_file.read()
            lines, words, chars = count_file(content)
            print(f"{lines:8d}{words:8d}{chars:8d}")
        elif args:
            for filename in args:
                try:
                    with open(filename, 'r') as f:
                        content = f.read()
                        lines, words, chars = count_file(content)
                        print(f"{lines:8d}{words:8d}{chars:8d} {filename}")
                except Exception as e:
                    print(f"wc: {e}")

    def test_redirect_command(self):
        """Test redirection functionality"""
        try:
            test_content = "This is a test file created by the shell\n"
            with open("test_file.txt", "w") as f:
                f.write(test_content)
            print("✓ File creation test: SUCCESS")
            
            if os.path.exists("test_file.txt"):
                print("✓ File verification: SUCCESS")
                with open("test_file.txt", "r") as f:
                    content = f.read()
                    print(f"✓ Content: '{content.strip()}'")
                os.remove("test_file.txt")
                print("✓ Cleanup: SUCCESS")
            else:
                print("✗ File verification: FAILED")
        except Exception as e:
            print(f"✗ Test failed: {e}")

    def show_files_command(self):
        """Show files with permission info"""
        try:
            files = os.listdir('.')
            print(f"{'Filename':<20} {'Type':<10} {'Size':<8}")
            print("-" * 40)
            
            for item in sorted(files):
                if os.path.isfile(item):
                    size = os.path.getsize(item)
                    print(f"{item:<20} {'File':<10} {size:<8}")
                else:
                    print(f"{item:<20} {'Directory':<10} {'-':<8}")
        except Exception as e:
            print(f"Error listing files: {e}")

    def show_help(self):
        """Display help information"""
        print("Advanced Integrated Shell - Help")
        print("=" * 50)
        print()
        print("FILE OPERATIONS:")
        print("  ls [dir]           - List directory contents")
        print("  cat <file>         - Display file contents")
        print("  cd <dir>           - Change directory")
        print("  pwd                - Print working directory")
        print("  touch <file>       - Create empty file")
        print("  mkdir <dir>        - Create directory")
        print("  rm <file>          - Remove file")
        print()
        print("SECURITY:")
        print("  whoami             - Show current user")
        print("  su <user>          - Switch user")
        print("  create_user <name> <pass> <role> - Create user")
        print("  chmod <file> <perm> - Set file permissions")
        print()
        print("MEMORY MANAGEMENT:")
        print("  mem_status         - Show memory status")
        print("  mem_alloc <pid> <pages> - Allocate memory")
        print("  mem_access <pid> <page> - Access page")
        print("  mem_dealloc <pid>  - Deallocate memory")
        print("  mem_algorithm <FIFO|LRU> - Set algorithm")
        print("  compare_algorithms - Compare algorithms")
        print()
        print("PROCESS SCHEDULING:")
        print("  schedule_rr <quantum> - Set Round Robin")
        print("  schedule_priority  - Set Priority scheduling")
        print("  add_process <time> <priority> - Add process")
        print("  scheduler_status   - Show scheduler status")
        print("  scheduler_metrics  - Show performance metrics")
        print("  simulate_workload <duration> - Run simulation")
        print("  stop_scheduler     - Stop scheduler")
        print()
        print("PROGRAM COUNTER:")
        print("  pc_start [addr]    - Start program counter")
        print("  pc_stop            - Stop program counter")
        print("  pc_status          - Show PC value")
        print()
        print("PIPING & REDIRECTION:")
        print("  cmd1 | cmd2        - Pipe output")
        print("  cmd > file         - Redirect output")
        print("  cmd >> file        - Append output")
        print("  grep <pattern>     - Search text")
        print("  head [-n]          - Show first lines")
        print("  tail [-n]          - Show last lines")
        print("  wc                 - Count words/lines")
        print()
        print("SYSTEM:")
        print("  help               - Show this help")
        print("  clear              - Clear screen")
        print("  exit               - Exit shell")
        print("  test_redirect      - Test file redirection")
        print("  show_files         - Show files with info")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    shell = IntegratedAdvancedShell()
    shell.run()