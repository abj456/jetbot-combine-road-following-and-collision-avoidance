import requests
import json
import base64

api = 'http://35.203.156.155:5000/predict'
image_file = '/home/server/flask_server/r_and_c/data/dataset_xy/xy_039_080_8f5403c8-e082-11eb-849f-72b5f773b75d.jpg'
with open(image_file, 'rb') as f:
    im_bytes = f.read()
im_b64 = base64.b64encode(im_bytes).decode("utf-8")
headers = {'Content-type': 'application/json', 'Accept':'text/plain'}
payload = json.dumps({"image": im_b64, "other_key": "value"})

req = requests.post(api, data=payload, headers=headers)
print(json.loads(req.content)['road following y_r'])