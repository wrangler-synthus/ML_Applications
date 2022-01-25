The dataset was very large to upload as a folder.
Perform the following steps:

! pip install kaggle


!mkdir ~/.kaggle
!touch ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

- Fill the username and api-key from your account.

api_token = {"username":"username","key":"api-key"}

import json

with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

!kaggle competition download Diabetic Retinopathy Detection

Upload the train part of the notebook in this folder.
