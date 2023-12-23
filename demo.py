import gradio as gr
import requests
import cv2
import os
import numpy as np

alpr_count = 0

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
	# Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    return img

def alpr(frame):
    global alpr_count
    
    alpr_count = alpr_count + 1
    print("alpr_count", alpr_count) 
    url = "http://127.0.0.1:8080/alpr"
    file = {'file': open(frame, 'rb')}

    r = requests.post(url=url, files=file)
   
    alpr_output = None
    
    try:
        image = cv2.imread(frame, cv2.IMREAD_COLOR)
        if image is None:
            print('image is null')
            sys.exit()
        image = cv2.resize(image, (1024, 640))
        for alpr in r.json().get('alprs'):
            print(alpr.get('recog_results'), alpr.get('recog_boxes'))
            image = plot_one_box(alpr.get('recog_boxes'), image, label=alpr.get('recog_results'), color=[0, 255, 0], line_thickness=1)
            cv2.rectangle(image, (alpr.get('vehicle_boxes')[0], alpr.get('vehicle_boxes')[1]), (alpr.get('vehicle_boxes')[2], alpr.get('vehicle_boxes')[3]), (200, 200, 200), 2)    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        alpr_output = image.copy()
        
    except:
        pass

    return alpr_output

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # KBY-AI Online Test Mode
    We offer SDKs for face recognition, liveness detection, ID card recognition and ANPR/ALPR.
    """
    )

    with gr.TabItem("ANPR/ALPR"):
        with gr.Row():
            with gr.Column():
                alpr_image_input = gr.Image(type='filepath', height=300)
                gr.Examples(['test/test1.jpg', 'test/test2.jpg', 'test/sample1_highres.jpg'], 
                            inputs=alpr_image_input)
                alpr_confirmation_button = gr.Button("Confirm")
            with gr.Column():
                alpr_output = gr.Image(type="numpy")
                
        alpr_confirmation_button.click(alpr, inputs=alpr_image_input, outputs=alpr_output)    

#demo.launch(server_name="127.0.0.1", server_port=443, ssl_certfile="kby-ai.crt",
#                        ssl_keyfile="kby-ai.key", ssl_verify=True)
demo.launch(server_name="0.0.0.0", server_port=8000)