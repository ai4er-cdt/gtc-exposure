import virtualenv
import pip
import os

# Define and create the base directory install virtual environments
venvs_dir = os.path.join(os.path.expanduser("~"), "nb-venvs")

# Since you only need to create the virtual environment once, check for its existence before trying to create it
if not os.path.isdir(venvs_dir):
    os.makedirs(venvs_dir)

# Define the venv directory
venv_dir = os.path.join(venvs_dir, 'venv-notebook')

if not os.path.isdir(venv_dir):
    # Create the virtual environment
    print(f'[INFO] Creating: virtual env at: {venv_dir}')
    virtualenv.create_environment(venv_dir)


# Activate the venv, making use of the `activate_this.py` file within the venv.
activate_file = os.path.join(venv_dir, "bin", "activate_this.py" )
exec(open(activate_file).read(), dict(__file__=activate_file))
    
pip.main(["install", "--prefix", venv_dir, "-r", "requirements.txt", '-q'])





