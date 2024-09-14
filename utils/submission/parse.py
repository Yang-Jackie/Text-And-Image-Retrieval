import json
import csv

QNA_ANSWER = ''

with open('./raw.json', 'r') as f:
    obj = json.load(f)

with open('./submission.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'frame_idx'])
    for entry in obj:
        parts = entry['imgpath'].split('/')
        file = parts[-3] + "_" + parts[-2]
        writer.writerow({
            'file': file,
            'frame_idx': entry['frame_id'],
        })