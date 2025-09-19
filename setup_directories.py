"""
Directory Setup Script
=====================
Creates all necessary directories for the project
"""

import os

def create_project_directories():
    """Create all necessary directories for the project"""
    
    directories = [
        'data/raw',
        'data/processed', 
        'results/plots',
        'models',
        'dashboard',
        'config',
        'src',
        'notebooks',
        'tests'
    ]
    
    print("Creating project directories...")
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created: {directory}")
        except Exception as e:
            print(f"✗ Error creating {directory}: {e}")
    
    print("Directory setup completed!")

if __name__ == "__main__":
    create_project_directories()
