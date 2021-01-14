import warnings
import virtualenv
import pip
import os
    
print("Please ignore all of the warnings and errors- I'm working on it, it all works as expected.")
print("")

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
    is_built=False
    
else:
    is_built=True
    
# Activate the venv, making use of the `activate_this.py` file within the venv.
activate_file = os.path.join(venv_dir, "bin", "activate_this.py" )
exec(open(activate_file).read(), dict(__file__=activate_file))
    
pip.main(["install", "--prefix", venv_dir, "-r", "requirements.txt", '-q'])

if is_built == False:
    
    print("")
    print('Please click the IAM link below to to retrive the token and paste into the text field. This will only happen this once to autentice the Jasmin Server with you Descartes Labs Credentials')
    
    import descarteslabs as dl; import os
    token_info_path = dl.client.auth.auth.DEFAULT_TOKEN_INFO_PATH
    temp_token_info_path = token_info_path + ".tmp"
    token_info_dir = os.path.dirname(token_info_path); 
    dl.client.auth.auth.makedirs_if_not_exists(token_info_dir)
    try: 
        with open(temp_token_info_path, "w") as f:
            f.write(dl.client.auth.auth.base64url_decode(input()).decode("utf-8"))
        print("\n Logged in as {}".format(dl.Auth(token_info_path=temp_token_info_path).payload['name']))
        os.replace(temp_token_info_path, token_info_path)
    except:
        os.remove(temp_token_info_path)
        print('''\n Invalid token.  Please make sure that you haven't accidentally added any whitespace to the token, 
        and that you have included any trailing '=' characters in the token. If you're still having issues
        authenticating, please contact support@descarteslabscom.''')
        
    print("")
    print('If this was successful, you should see "Logged in as [Your Name]" after running the commands above.  You will not need to repeat these steps the next time you log in.  If you are having any issues logging in, support is available at support@descarteslabs.com.')
