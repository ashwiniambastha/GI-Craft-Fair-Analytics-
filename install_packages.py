"""
Package Installation Script
===========================
Automatically installs required packages for the GI Craft Fair Analytics project
"""

import subprocess
import sys
import importlib

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_packages():
    """Check for required packages and install missing ones"""
    
    # Essential packages (minimum required)
    essential_packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'xgboost',
        'lightgbm',
        'catboost',
        'lifelines',
        'plotly',
        'streamlit',
        'pyyaml',
        'joblib'
    ]
    
    # Optional packages (nice to have)
    optional_packages = [
        'shap',
        'boruta',
        'statsmodels',
        'scipy',
        'tqdm'
    ]
    
    print("🔍 Checking required packages...")
    print("=" * 50)
    
    missing_essential = []
    missing_optional = []
    
    # Check essential packages
    for package in essential_packages:
        try:
            # Handle special cases
            if package == 'scikit-learn':
                importlib.import_module('sklearn')
            elif package == 'pyyaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Missing")
            missing_essential.append(package)
    
    # Check optional packages
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} (optional)")
        except ImportError:
            print(f"⚠ {package} (optional) - Missing")
            missing_optional.append(package)
    
    # Install missing essential packages
    if missing_essential:
        print(f"\n📦 Installing {len(missing_essential)} essential packages...")
        for package in missing_essential:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
    
    # Ask about optional packages
    if missing_optional:
        print(f"\n📦 Found {len(missing_optional)} optional packages:")
        for package in missing_optional:
            print(f"  - {package}")
        
        response = input("\nInstall optional packages? (y/n): ").lower()
        if response in ['y', 'yes']:
            for package in missing_optional:
                print(f"Installing {package}...")
                if install_package(package):
                    print(f"✓ {package} installed successfully")
                else:
                    print(f"✗ Failed to install {package}")
    
    print("\n🎉 Package installation completed!")
    
    # Final verification
    print("\n🔍 Final verification...")
    all_packages = essential_packages + optional_packages
    working_packages = []
    still_missing = []
    
    for package in all_packages:
        try:
            if package == 'scikit-learn':
                importlib.import_module('sklearn')
            elif package == 'pyyaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            working_packages.append(package)
        except ImportError:
            still_missing.append(package)
    
    print(f"✓ Working packages: {len(working_packages)}")
    if still_missing:
        print(f"✗ Still missing: {still_missing}")
        print("\nYou may need to install these manually:")
        for package in still_missing:
            print(f"  pip install {package}")
    
    return len(still_missing) == 0

def install_from_requirements():
    """Install packages from requirements.txt"""
    print("📦 Installing from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    except FileNotFoundError:
        print("⚠ requirements.txt not found, installing packages individually")
        return False

def main():
    """Main installation process"""
    print("GI Craft Fair Analytics - Package Installation")
    print("=" * 50)
    
    # Option 1: Try installing from requirements.txt
    if install_from_requirements():
        print("✓ All packages installed from requirements.txt")
    else:
        # Option 2: Install packages individually
        print("📦 Installing packages individually...")
        check_and_install_packages()
    
    print("\n🚀 Ready to run the analytics pipeline!")
    print("Next steps:")
    print("1. python run_complete_analysis.py")
    print("2. Or run individual modules from the src/ directory")

if __name__ == "__main__":
    main()
