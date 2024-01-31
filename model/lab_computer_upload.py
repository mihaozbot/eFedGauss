import paramiko #pip install paramiko
import tarfile
import os
import sys

def test_ssh_connection(hostname, port, username, password):
    try:
        # Create an SSH client instance
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the server
        ssh.connect(hostname, port=port, username=username, password=password)
        
        print("Connection successful.")

        # Close the connection
        ssh.close()
    except Exception as e:
        print(f"Connection failed: {e}")

# Function to create a tarball from a source directory
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for file in files:
                if not file.startswith('.') and not file.endswith('.pyc'):
                    file_path = os.path.join(root, file)
                    tar.add(file_path, arcname=os.path.relpath(file_path, start=source_dir))

# Function to show upload progress
def print_progress(transferred, total):
    progress_percentage = (transferred / total) * 100
    sys.stdout.write(f"Transfer Progress: {progress_percentage:.2f}%\r")
    sys.stdout.flush()

# Function to transfer a file via SFTP
def transfer_file(local_path, remote_path, hostname, port, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port=port, username=username, password=password)

    sftp = ssh.open_sftp()
    sftp.put(local_path, remote_path, callback=print_progress)
    sftp.close()
    ssh.close()

# Function to execute commands remotely via SSH
def execute_commands_on_server(hostname, port, username, password, remote_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port=port, username=username, password=password)

    extraction_directory = '/path/to/extraction_directory'  # Modify this path as needed

    commands = [
        f"mkdir -p {extraction_directory}",
        f"tar -xvzf {remote_path} -C {extraction_directory}",
    ]

    for command in commands:
        stdin, stdout, stderr = ssh.exec_command(command)
        print(stdout.read().decode())
        err = stderr.read().decode()
        if err:
            print("Error:", err)

    ssh.close()

# Parameters for the target machine
hostname = '192.168.222.12' 
port = 22
username = 'OptiTrack' 
password = '2024'  
remote_path = '/remote/path/eFedGauss.tar.gz'  

test_ssh_connection(hostname, port, username, password)

# Create a tarball from the source directory
source_folder = r'C:\Users\Miha\OneDrive - Univerza v Ljubljani\Doktorski_studij\Delo\eGAUSSp_Python'
output_tarball = 'eFedGauss.tar.gz'
make_tarfile(output_tarball, source_folder)

# Transfer the tarball to the remote machine
transfer_file(output_tarball, remote_path, hostname, port, username, password)

# Execute commands on the remote machine
execute_commands_on_server(hostname, port, username, password, remote_path)
