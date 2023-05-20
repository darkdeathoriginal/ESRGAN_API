import os.path as osp
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from flask import *
import json
import io
from gevent.pywsgi import WSGIServer


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  
model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'
def start():
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    idx = 0
    path ="LR/uploaded_file.jpg"
    print(path)
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/result.png'.format(base), output)




@app.route("/",methods=['GET'])
def home_page():
    return json.dumps({"message":"message recieved"})

@app.route('/', methods=['POST'])
def upload_file():
    print(request.files)
    if 'test' not in request.files:
        return 'No file uploaded', 400

    file = request.files['test']
    file.save('LR/uploaded_file.jpg')
    start()
    with open('results/result.png', 'rb') as f:
        image_bytes = f.read()

    # Return the image file as a response
    return send_file(
        io.BytesIO(image_bytes),
        mimetype='image/png'
    )


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 7000), app)
    http_server.serve_forever()