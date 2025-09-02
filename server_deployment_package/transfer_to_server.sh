#!/bin/bash

# Transfer Script - Send deployment package to server
# Edit the SERVER variable below with your server details

# EDIT THIS: Your server connection details
SERVER="your-username@your-server-ip"
SERVER_PATH="/tmp/moral_alignment_deployment"

echo "================================================"
echo "TRANSFER DEPLOYMENT PACKAGE TO SERVER"
echo "================================================"
echo ""

# Check if server variable was edited
if [ "$SERVER" == "your-username@your-server-ip" ]; then
    echo "ERROR: Please edit this script and set your server details"
    echo "       Edit the SERVER variable at the top of this file"
    echo ""
    echo "Example:"
    echo "  SERVER=\"ubuntu@192.168.1.100\""
    echo "  or"
    echo "  SERVER=\"researcher@gpu-server.university.edu\""
    exit 1
fi

echo "Server: $SERVER"
echo "Destination: $SERVER_PATH"
echo ""

# Create tarball
echo "Creating deployment package..."
cd ..
tar -czf server_deployment.tar.gz server_deployment_package/
echo "✅ Package created: server_deployment.tar.gz"

# Get package size
SIZE=$(du -h server_deployment.tar.gz | cut -f1)
echo "📦 Package size: $SIZE"
echo ""

# Transfer to server
echo "Transferring to server..."
scp server_deployment.tar.gz $SERVER:$SERVER_PATH.tar.gz

if [ $? -eq 0 ]; then
    echo "✅ Transfer complete!"
    echo ""
    echo "Next steps:"
    echo "1. SSH to your server:"
    echo "   ssh $SERVER"
    echo ""
    echo "2. Extract the package:"
    echo "   cd /tmp"
    echo "   tar -xzf moral_alignment_deployment.tar.gz"
    echo "   cd server_deployment_package"
    echo ""
    echo "3. Run setup:"
    echo "   chmod +x setup_server.sh"
    echo "   ./setup_server.sh"
    echo ""
    echo "4. Start evaluation:"
    echo "   cd /data/storage_4_tb/moral-alignment-pipeline"
    echo "   python run_evaluation.py --models qwen2.5-32b --samples 100"
else
    echo "❌ Transfer failed!"
    echo "   Please check your server connection and try again"
fi

# Clean up
rm -f ../server_deployment.tar.gz

echo ""
echo "================================================"