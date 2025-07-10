#!/usr/bin/env bash


# Get Ubuntu version
ubuntu_version=$(lsb_release -rs | cut -d. -f1)

# Get current GCC major version
gcc_version=$(gcc -dumpversion | cut -d. -f1)

echo "Detected Ubuntu version: $ubuntu_version"
echo "Detected GCC version: $gcc_version"

# Check conditions
if [[ "$ubuntu_version" == "20" && "$gcc_version" -lt 11 ]]; then
    # This won't change your system gcc version!
    # To activate GCC 11, you can use the following commands:
    # export CC=/usr/bin/gcc-11
    # export CXX=/usr/bin/g++-11
    echo "Installing GCC 11 and G++ 11 from PPA..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt update
    sudo apt install -y gcc-11 g++-11
    echo "GCC 11 and G++ 11 installed successfully."
else
    echo "No installation needed. Either Ubuntu is not 20.04 or GCC >= 11 is already installed."
fi