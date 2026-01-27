#!/bin/bash
# Setup script for Agent Consistency Research Project
# This script helps you set up your API keys as environment variables

echo "Setting up environment variables for Agent Consistency Research..."

# Add to your shell profile (zsh for macOS)
SHELL_PROFILE="$HOME/.zshrc"

# OpenAI API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "To set your OpenAI API key permanently, add this line to $SHELL_PROFILE:"
    echo 'export OPENAI_API_KEY="your_openai_api_key_here"'
    echo ""
    echo "Or run this command now (temporary for this session):"
    echo 'export OPENAI_API_KEY="your_openai_api_key_here"'
    echo ""
fi

# Together AI API Key
if [ -z "$TOGETHER_API_KEY" ]; then
    echo ""
    echo "To set your Together AI API key permanently, add this line to $SHELL_PROFILE:"
    echo 'export TOGETHER_API_KEY="your_together_api_key_here"'
    echo ""
    echo "Or run this command now (temporary for this session):"
    echo 'export TOGETHER_API_KEY="your_together_api_key_here"'
    echo ""
fi

# Check if keys are set
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✓ OPENAI_API_KEY is set"
else
    echo "✗ OPENAI_API_KEY is not set"
fi

if [ -n "$TOGETHER_API_KEY" ]; then
    echo "✓ TOGETHER_API_KEY is set"
else
    echo "✗ TOGETHER_API_KEY is not set"
fi

echo ""
echo "To test the agent, run: python agent.py"

