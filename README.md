# fairmot_ynet
Combine object dection &amp; tracking &amp; trajectory prediction

![ynet_result_front_line](https://user-images.githubusercontent.com/33422418/215961234-bac69767-471a-4aa3-b963-a6168a1a5347.jpg)

## Addtional file
[Additional](https://github.com/28598519a/fairmot_ynet/releases/tag/Additional)
## Usage
    conda activate ynet
    cd FairMOT
    python src/demo.py mot --load_model ./fairmot_dla34.pth --conf_thres 0.4 --input-video ./videos/test.mp4 --output-root ./demo
    python data_to_pickle.py
    cd ../ynet
    python demo.py


## Env
    conda env create -f ynet.yml
Manual install cython_bbox、DCNv2

    conda env update -f ynet.yml
