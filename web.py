from flask import Flask, render_template, request, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
from geo import getLocation, getAddress
import matplotlib.pyplot as plt
from model import load_model
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import faiss
import py_vncorenlp
import torch
app = Flask(__name__)


UPLOAD_FOLDER = './static/images'
# UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


def calculateDistanceBetween2Points(lat1, logt1, lat2, logt2):
    lat1 = radians(lat1)
    logt1 = radians(logt1)
    lat2 = radians(lat2)
    logt2 = radians(logt2)
    dlon = logt2 - logt1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calculateDistance(df, lat, lng):
    lats, lngs = df['Latitude'].tolist(), df['Longitude'].tolist()
    distance, address = [], []
    for i in range(len(lats)):
        dis = calculateDistanceBetween2Points(lats[i], lngs[i], lat, lng)
        distance.append(round(dis, 2))
        try:
            add = getAddress(lats[i], lngs[i])
            address.append(add)
        except:
            print(lats[i], lngs[i])
    return distance, address


def addKm(distance):
    for dis in distance:
        dis = str(dis) + " km"
    return distance


def removeDuplicate(labels: list) -> list:
    set_ = set()
    ans = []
    for label in labels:
        if label not in set_:
            set_.add(label)
            ans.append(label)
    return ans


def readData():
    db_img = faiss.read_index('./data/index_image.faiss')
    db_text = faiss.read_index('./data/index_text.faiss')
    df_image = pd.read_csv("./database_image.csv")
    df_text = pd.read_csv("./data/database_text.csv")
    return db_img, db_text, df_image, df_text
def query(text = None, image = None):
    lat, longt, location = getLocation()
    db_img, db_text, df_image, df_text = readData()
    model, processor, tokenizer = load_model(weight = "./weights/vnfood_clip_v1.pth")
    model.eval()
    process = None
    input_features = None
    D, I = None, None
    results = None
    if image != None:
        process = processor(image, return_tensors = "pt")
        image_input = process["pixel_values"]
        with torch.no_grad():
            input_features = model.get_image_features(image_input)
        input_features = input_features.cpu().numpy()
        faiss.normalize_L2(input_features)
        D, I = db_text.search(input_features, k = 5)
        results = df_text.iloc[I[0]]['label'].tolist()
    else:
        save_dir = './segment_text' # Load the word and sentence segmentation component
        rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir = save_dir)
        text = rdrsegmenter.word_segment(text)
        token = tokenizer(text = text, return_tensors = 'pt', max_length = 256, truncation = True, padding = 'max_length')
        with torch.no_grad():
            input_features = model.get_text_features(token['input_ids'], token['attention_mask'])
        input_features = input_features.cpu().numpy()
        faiss.normalize_L2(input_features)
        D, I = db_img.search(input_features, k = 5)
        results = df_image.iloc[I[0]]['label'].tolist()
    true_label = removeDuplicate(results)
    print(results)
    df = pd.read_csv("./data/data3.csv")
    filtered_df = df[df['Label'].isin(true_label)]
    for i in range(len(filtered_df)):
        lat_, longt_ = filtered_df.iloc[i]['Latitude'], filtered_df.iloc[i]['Longitude']
        dis = calculateDistanceBetween2Points(lat_, longt_, lat, longt)
        distances.append(round(dis, 2))
    distances = addKm(distances)
    filtered_df["Distance"] = distances
    filtered_df['rank'] = filtered_df.groupby('Label')['Distance'].rank(method='dense', ascending=True)

    filtered_df['label_order'] = pd.Categorical(filtered_df['Label'], categories = true_label, ordered=True)
    filtered_df = filtered_df.sort_values(['label_order', 'rank']).drop(columns='label_order')
    data = filtered_df[['Image', 'Information', 'Rating', 'Address', 'Distance']]
    data['Rating'] = data['Rating'].astype(float)
    data = data.to_dict(orient = 'records')
    return data, location



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        file = request.files['image']
        image_url = None
        image = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_url = url_for('uploaded_file', filename=filename)
            name_img = image_url.split('/')[-1]
            image = Image.open(UPLOAD_FOLDER + '/' + name_img)

        if not text and not image_url:
            lat, lng, location = getLocation()
            df = pd.read_csv("D:/Downloads/data1.csv")
            distance, address = calculateDistance(df = df, lat = lat, lng = lng)
            distance = sorted(distance)
            distance = addKm(distance = distance)
            df['Distance'] = distance
            df['Address'] = address
            data = df[['Image', 'Information', 'Rating', 'Address', 'Distance']]
            data['Rating'] = data['Rating'].astype(float)
            data.sort_values("Rating", ascending = False)
            data = data.to_dict(orient = "records")
            return render_template('index.html', data = data, location = location)
        data, location = query(text = text, image = image)
        
        return render_template('index.html', text = text, image_url = image_url, data = data, location = location)

    return render_template('index.html', text = None, image_url = None)

@app.route('/image/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
