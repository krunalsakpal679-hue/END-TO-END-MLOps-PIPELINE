"""
create_users.py

Utility to generate HTTP Basic Auth credentials using bcrypt hashing
for Nginx to secure the MLflow backend.
"""

import os
import secrets
import string
import sys
import subprocess

# Ensure bcrypt is available
try:
    import bcrypt
except ImportError:
    print("Installing bcrypt for secure password hashing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bcrypt"])
    import bcrypt

def generate_password(length=12):
    """Generates a highly secure random 12-character password."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for i in range(length))

def main():
    user_list_str = os.getenv("USER_LIST")
    if not user_list_str:
        print("Error: USER_LIST environment variable not set.")
        print("Example: set USER_LIST=admin,ds1,readonly")
        sys.exit(1)
        
    usernames = [u.strip() for u in user_list_str.split(",") if u.strip()]
    if not usernames:
        print("Error: No valid usernames found in USER_LIST.")
        sys.exit(1)

    htpasswd_path = ".htpasswd"
    passwords_path = "generated_passwords.txt"
    
    htpasswd_lines = []
    generated_passwords = {}
    
    for username in usernames:
        password = generate_password()
        
        # Hash password using modern bcrypt with 12 rounds
        hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))
        hashed_str = hashed_bytes.decode('utf-8')
        
        # Nginx expects: username:hashed_password
        htpasswd_lines.append(f"{username}:{hashed_str}")
        generated_passwords[username] = password
        
    # Save the bcrypt-hashed htpasswd file locally
    # Note: In production, this file gets mounted/moved to /etc/nginx/.htpasswd
    with open(htpasswd_path, "w") as f:
        f.write("\n".join(htpasswd_lines) + "\n")
    
    # Save plaintext passwords securely
    with open(passwords_path, "w") as f:
        f.write("=== MLflow Generated Credentials ===\n")
        for user, pw in generated_passwords.items():
            f.write(f"Username: {user} | Password: {pw}\n")
            
    # Automatically append to .gitignore if not present to prevent accidental commits
    gitignore_path = ".gitignore"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            content = f.read()
        if "generated_passwords.txt" not in content:
            with open(gitignore_path, "a") as f:
                f.write("\n# Sensitive Auth Data\n")
                f.write("generated_passwords.txt\n")
                f.write(".htpasswd\n")
    else:
        with open(gitignore_path, "w") as f:
            f.write("generated_passwords.txt\n")
            f.write(".htpasswd\n")

    print(f"Created {len(usernames)} users. Passwords saved to {passwords_path} — keep this safe!")

if __name__ == "__main__":
    main()
