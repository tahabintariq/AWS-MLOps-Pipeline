
#!/bin/bash
# This script is idempotent and handles environment setup on the EBS volume

set -e

echo "Environment Setup"

# 1. Update package list
sudo apt-get update -y

# 2. Install Python3, pip, and virtualenv tools [cite: 66]
sudo apt-get install -y python3 python3-pip python3-venv

# 3. Define path on EBS volume (Crucial for Task 1 & 3 requirements) [cite: 29, 32]
VENV_PATH="/mnt/ml-data/ml_venv"

# 4. Create virtual environment if it doesn't exist [cite: 67, 70]
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
    echo "Environment created at $VENV_PATH"
else
    echo "Environment already exists at $VENV_PATH"
fi

# 5. Activate environment and upgrade pip 
source "$VENV_PATH/bin/activate"
pip install --upgrade pip --quiet

# 6. Install required ML libraries [cite: 68]
# pip install is naturally idempotent (it won't re-install if version matches)
pip install numpy pandas scikit-learn matplotlib seaborn joblib --quiet

echo "Setup Complete"
echo "To activate: source $VENV_PATH/bin/activate"
