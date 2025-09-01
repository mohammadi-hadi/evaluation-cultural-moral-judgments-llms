#!/bin/bash
# Setup script for Azure/SURF Jupyter environment
# This script sets up automatic syncing with GitHub

echo "======================================"
echo "Azure/SURF Server Setup for Project06"
echo "======================================"

# 1. Clone the repository (if not already cloned)
if [ ! -d "Project06" ]; then
    echo "Cloning repository from GitHub..."
    git clone https://github.com/mohammadi-hadi/moral-alignment-pipeline.git Project06
    cd Project06
else
    echo "Repository already exists. Updating..."
    cd Project06
    git pull origin main
fi

# 2. Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create data directories
echo ""
echo "Creating directory structure..."
mkdir -p outputs/{scores,traces,prompts,evaluations,visualizations,experiments,cross_evaluation,cache}
mkdir -p outputs/scores/{logprob,direct,cot}
mkdir -p outputs/traces/{raw,processed}
mkdir -p outputs/prompts/{templates,filled}
mkdir -p outputs/evaluations/{peer,human}
mkdir -p outputs/visualizations/{figures,reports}
mkdir -p outputs/experiments/{runs,logs}
mkdir -p sample_data

# 4. Create sync script for automatic updates
echo ""
echo "Creating sync script..."
cat > sync_with_github.sh << 'EOF'
#!/bin/bash
# Sync script to pull latest changes from GitHub

cd ~/Project06

# Stash any local changes
git stash

# Pull latest changes
echo "Pulling latest changes from GitHub..."
git pull origin main

# Apply stashed changes (if any)
git stash pop 2>/dev/null

echo "Sync complete!"
echo "Last update: $(date)"
EOF

chmod +x sync_with_github.sh

# 5. Create Jupyter notebook launcher
echo ""
echo "Creating Jupyter notebook launcher..."
cat > launch_notebook.sh << 'EOF'
#!/bin/bash
# Launch Jupyter notebook with the moral alignment pipeline

cd ~/Project06

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Launch Jupyter
echo "Starting Jupyter Notebook..."
jupyter notebook moral_alignment_complete.ipynb --ip=0.0.0.0 --no-browser
EOF

chmod +x launch_notebook.sh

# 6. Create environment file template
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env template..."
    cat > .env.template << 'EOF'
# API Keys - Add your keys here
OPENAI_API_KEY=your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
COHERE_API_KEY=your-cohere-key-here
MISTRAL_API_KEY=your-mistral-key-here

# Optional: Model cache directory
MODEL_CACHE_DIR=/path/to/model/cache
EOF
    echo "⚠️  Remember to create .env file with your API keys!"
fi

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Add your API keys to .env file"
echo "2. Run ./sync_with_github.sh to sync with GitHub"
echo "3. Run ./launch_notebook.sh to start Jupyter"
echo "4. Or run: python run_experiments.py --profile lightweight"
echo ""
echo "To keep synced with GitHub, run periodically:"
echo "  ./sync_with_github.sh"