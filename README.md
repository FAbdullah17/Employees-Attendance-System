# Employees Attendance System 🕒

A facial recognition–based attendance system using **FaceNet embeddings** and an **SVM classifier** to identify employees with high accuracy. The system is built using OpenCV, Keras-FaceNet, and scikit-learn, and is optimized for practical deployment in workplace environments.

---

## Project Overview

This repository provides an end-to-end pipeline for building, training, and deploying an attendance tracking system using facial recognition. The project includes:

* **Data Preprocessing** (`data_preprocessing.ipynb`):
  • Extracts faces from raw images using Haar cascades
  • Converts images to grayscale, resizes to 160×160
  • Saves structured train/test arrays and labels

* **Embedding & Classification** (`model.ipynb`):
  • Loads images and labels
  • Extracts 512-d embeddings using FaceNet
  • Trains and evaluates an SVM model on the embeddings
  • Reports accuracy and saves the classifier

* **Face Detection** (`haarcascade_frontalface.xml`): Pretrained cascade classifier.

* **Dataset Folder** (`DataSet/`): One folder per employee name with their face images.

* **Processed Data** (`data/processed/`): Contains `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`.

---

## Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/FAbdullah17/Employees-Attendance-System.git
   cd Employees-Attendance-System
   ```

2. **Install required packages**

   ```bash
   pip install opencv-python-headless numpy pandas scikit-learn keras-facenet tensorflow matplotlib tqdm
   ```

3. **Dataset structure**

   ```text
   DataSet/
     ├── Alice/
     │   ├── img1.jpg
     │   └── img2.jpg
     └── Bob/
         ├── img1.jpg
         └── img2.jpg
   ```

4. **Run preprocessing**
   Launch and run `data_preprocessing.ipynb` to detect and save face crops and labels.

5. **Train the SVM classifier**
   Launch and run `model.ipynb` to extract embeddings and train the classifier.

---

## Notebook: data\_preprocessing.ipynb

1. **Initialize**

   ```python
   import cv2, os, numpy as np
   from sklearn.model_selection import train_test_split
   ```

2. **Load dataset & detect faces**

   * Grayscale conversion
   * Resize each face to 160×160
   * Use Haar cascade to extract one face per image

3. **Split & save data**

   ```python
   np.save('X_train.npy', X_train)
   np.save('X_test.npy', X_test)
   np.save('y_train.npy', y_train)
   np.save('y_test.npy', y_test)
   ```

---

## Notebook: model.ipynb

1. **Import libraries & load FaceNet**

   ```python
   from keras_facenet import FaceNet
   embedder = FaceNet()
   ```

2. **Load preprocessed data**

   ```python
   X_train = np.load('X_train.npy')
   y_train = np.load('y_train.npy')
   ```

3. **Compute embeddings**

   ```python
   embeddings_train = embedder.embeddings(X_train)
   embeddings_test = embedder.embeddings(X_test)
   ```

4. **Train SVM classifier**

   ```python
   from sklearn.svm import SVC
   model = SVC(kernel='linear', probability=True)
   model.fit(embeddings_train, y_train)
   ```

5. **Evaluate performance**

   ```python
   y_pred = model.predict(embeddings_test)
   accuracy_score(y_test, y_pred)
   ```

6. **Save the model**

   ```python
   import pickle
   with open('svm_model.pkl', 'wb') as f:
       pickle.dump(model, f)
   ```

---

## Model Performance

* **Embedding model**: FaceNet (512-d vectors)
* **Classifier**: Support Vector Machine (linear kernel)
* **Accuracy**: High accuracy across test set with clean class separation

---

## Usage & Deployment

* Capture live video using OpenCV and detect faces
* Pass face to FaceNet for embedding
* Predict employee using trained SVM
* Log attendance in CSV or database

---

## Future Enhancements

* Add GUI for user interaction
* Extend dataset with more subjects
* Integrate cloud storage for logs
* Real-time attendance dashboard

---

## Contributing

Open issues or submit PRs. All contributions are appreciated.

---

## License

MIT License. See [LICENSE](LICENSE) for full text.
