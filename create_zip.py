#!/usr/bin/env python3
"""
Create deployment zip file excluding venv folder
"""

import os
import zipfile
from pathlib import Path

def create_deployment_zip():
    """Create a zip file of the project excluding venv folder."""
    
    # Get the current directory (should be parent of itfn_ai_challenge)
    current_dir = Path.cwd()
    project_dir = current_dir / "itfn_ai_challenge"
    zip_path = current_dir / "itfn_ai_challenge_deployment.zip"
    
    print(f"📦 Creating deployment zip file...")
    print(f"📁 Project directory: {project_dir}")
    print(f"📄 Zip file: {zip_path}")
    
    if not project_dir.exists():
        print(f"❌ Project directory not found: {project_dir}")
        return False
    
    # Files/folders to exclude
    exclude_patterns = {
        "venv",
        "__pycache__",
        "*.pyc",
        ".git",
        ".DS_Store",
        "Thumbs.db"
    }
    
    def should_exclude(file_path):
        """Check if file/folder should be excluded."""
        name = file_path.name
        parent = file_path.parent.name
        
        # Exclude venv folder and its contents
        if "venv" in str(file_path):
            return True
            
        # Exclude other patterns
        for pattern in exclude_patterns:
            if pattern in name or pattern in parent:
                return True
                
        return False
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            file_count = 0
            
            # Walk through all files in the project directory
            for root, dirs, files in os.walk(project_dir):
                # Remove excluded directories from dirs list to prevent walking into them
                dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d)]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    if not should_exclude(file_path):
                        # Calculate relative path from project directory
                        arcname = file_path.relative_to(project_dir.parent)
                        zipf.write(file_path, arcname)
                        file_count += 1
                        print(f"✅ Added: {arcname}")
            
            print(f"\n🎉 Successfully created zip file with {file_count} files!")
            print(f"📄 Zip file location: {zip_path}")
            
            # Show zip file size
            zip_size = zip_path.stat().st_size
            print(f"📊 Zip file size: {zip_size:,} bytes ({zip_size / (1024*1024):.1f} MB)")
            
            return True
            
    except Exception as e:
        print(f"❌ Error creating zip file: {e}")
        return False

if __name__ == "__main__":
    success = create_deployment_zip()
    if success:
        print("\n✅ Deployment zip file ready for judges!")
    else:
        print("\n❌ Failed to create deployment zip file.")
