### Descartes Labs Example Notebooks.

They are now editable and can be used on your own machine using the ```decarteslabs``` python library.

First time you import the library you must authenticate as in the Descartes Labs Workbench

Log in to [IAM](https://iam.descarteslabs.com/auth/login?refresh_token=true&destination=/auth/refresh_token) with your username/password.

Run the following as a code cell and paste the token into the pop up box. You will only need to do this once, not everytime like with Google Earth Engine:

```import descarteslabs as dl; import os
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
```

Now you can just ```import decarteslabs as dl```.
