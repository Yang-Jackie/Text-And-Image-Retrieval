from flask import Flask, render_template, Response, request, send_file, jsonify
from pymongo import MongoClient
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import glob
import json

from utils.query_processing import Translation
from utils.faiss import Myfaiss

EXTRACTED_PATH = "E:/aic2024/data/extracted"

app = Flask(__name__, template_folder="templates")

with open("id2frames.json") as json_file:
    json_list = json.load(json_file)

DictImagePath = {}

for item in json_list:
    DictImagePath[int(item["_id"])] = EXTRACTED_PATH + "/" + item["path"]

LenDictPath = len(DictImagePath)
bin_file = "faiss.bin"
MyFaiss = Myfaiss(bin_file, DictImagePath, "cpu", Translation())
########################


@app.route("/home")
@app.route("/")
def thumbnailimg():
    print("load_iddoc")

    pagefile = []
    index = request.args.get("index")
    if index == None:
        index = 0
    else:
        index = int(index)

    imgperindex = 100

    # imgpath = request.args.get('imgpath') + "/"
    pagefile = []

    page_filelist = []
    list_idx = []

    if LenDictPath - 1 > index + imgperindex:
        first_index = index * imgperindex
        last_index = index * imgperindex + imgperindex

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index])
            list_idx.append(tmp_index)
            tmp_index += 1
    else:
        first_index = index * imgperindex
        last_index = LenDictPath

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index])
            list_idx.append(tmp_index)
            tmp_index += 1

    for imgpath, id in zip(page_filelist, list_idx):
        pagefile.append({"imgpath": imgpath, "id": id})

    data = {"num_page": int(LenDictPath / imgperindex) + 1, "pagefile": pagefile}

    return render_template("home.html", data=data)


@app.route("/imgsearch")
def image_search():
    print("image search")
    pagefile = []
    id_query = int(request.args.get("imgid"))
    _, list_ids, _, list_image_paths = MyFaiss.image_search(id_query, k=50)

    imgperindex = 100

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({"imgpath": imgpath, "id": int(id)})

    data = {"num_page": int(LenDictPath / imgperindex) + 1, "pagefile": pagefile}

    return render_template("home.html", data=data)


@app.route("/textsearch")
def text_search():
    print("text search")

    pagefile = []
    text_query = request.args.get("textquery")
    _, list_ids, _, list_image_paths = MyFaiss.text_search(text_query, k=50)

    imgperindex = 100

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({"imgpath": imgpath, "id": int(id)})

    data = {
        "query": text_query,
        "num_page": int(LenDictPath / imgperindex) + 1,
        "pagefile": pagefile,
    }

    return render_template("home.html", data=data)


@app.route("/get_img")
def get_img():
    # print("get_img")
    fpath = request.args.get("fpath")
    # fpath = fpath
    list_image_name = fpath.split("/")
    image_name = "/".join(list_image_name[-2:])

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jpg")
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.resize(img, (640, 360))

    # print(img.shape)
    # img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode(".jpg", img)
    return Response(
        (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=1111)
