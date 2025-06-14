echo "Creating virtual environment..."
conda create -y -n lerobot python=3.10
conda activate lerobot

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-pyaudio \
    portaudio19-dev \
    build-essential \
    git

git clone https://github.com/huggingface/lerobot.git
cd lerobot

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installation complete!"