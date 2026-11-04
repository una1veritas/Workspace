'''
Created on 2026/11/05

@author: sin
'''
import cv2
from retinaface import RetinaFace
import os

if __name__ == '__main__':
    folder_path = "/Users/sin/Library/Mobile Documents/com~apple~CloudDocs/Documents/data/face_detect"
    image_filename = "L1040669_3k.JPG"
    image_path = os.path.join(folder_path, image_filename)  # 4/3 デジカメから書き出したJPEG画像
    
    score_threshold = 0.5  # 信頼度の閾値を設定
    
    # 顔検出を実行 (内部で重みファイルが自動ダウンロードされます)
    print("検出処理中...")
    faces = RetinaFace.detect_faces(image_path)
    
    if isinstance(faces, dict):
        print(f"検出成功: {len(faces)} 個の顔を検出しました。")
        
        # 元画像に枠を描画して確認用画像を出力
        boxes = list()
        img = cv2.imread(image_path)
        for key, identity in faces.items():
            box = identity["facial_area"]  # [x_min, y_min, x_max, y_max]
            score = identity["score"]
            
            # 信頼度閾値（例: 0.5以上）で描画
            if score >= score_threshold:
                boxes.append(box)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        output_filename = image_filename.split(".")[0] + "_result.jpg"
        cv2.imwrite(output_filename, img)
        print(f"結果画像を {output_filename} として保存しました。")
        cvs_filename = image_filename.split(".")[0] + "_result.cvs"
        with open(cvs_filename, "w") as f:
            f.write("x_min,y_min,x_max,y_max\n")
            for box in boxes:
                f.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")
        print(f"検出結果の座標を {cvs_filename} として保存しました。")
    else:
        print("顔が検出されませんでした。")