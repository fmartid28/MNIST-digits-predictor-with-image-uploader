# MNIST-digits-predictor-with-image-uploader
Convolutional neural network trained on MNIST. 

A (locally hosted) web interface will provide the file handling.
Application interface is implemented using streamlit.


Model and model weights made available for inference in `models/`.

The user just needs to provide an image of a single handwritten number for upload.

## Requirements
Create a virtual environment using python 3.12.
```python3
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Folder structure
app/

* app.py (streamlit app)

notebook/

* test.ipynb (contains experimentation code prior to splitting off into individual files (model, train, predict))

src/

* model.py (defines CNN architecture)

* train.py (defines training and weight archiving to "models/".)

* predict.py (Code to carry our the inference of a digit.)

models/

* *.pth (neural network weights. Pulled by predict and app.)
## Running the program
If youd like to generate your own weights file:
```python3
python src/train.py
```

If you want to just run the streamlit app, or you've generate the weights already and want to host it on streamlit:
```python3
streamlit run app/app.py
````
