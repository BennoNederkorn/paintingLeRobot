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

echo "Installing lerobot dependencies..."
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
cd ..

echo "Installing our dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installation complete!"