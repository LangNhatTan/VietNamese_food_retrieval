# Introduction
This is a project about Vietnamese cuisine retrieval focused on multimodal model for querying. The script now utilize a new model base on CLIP model architecture. Our model has accuracy higher than CLIP (openAI), BLIP (Salesforce) and ALIGN (Kakao Brain).
# Implementation
1. Download the source code from this branch in the repository.
2. Install the required libraries from the `requirements.txt` file.
# Usage
1. Open file `traininng-model.ipynb` for training model and export file `.pth`.
2. Open file `database_image.csv` in folder data and rename the path in column **image**.
3. Run this script:
```python
python .\web.py --weight path/to/weight/model.pth
```
## Main Libraries Used
1. **cv2**: Image processing and computer vision.
2. **torch**: Neural network processing.
3. **numpy**: Processing the numerical data.
4. **huggingface**: Loading vision and text model.
5. **FAISS**: storing and searching vector embedding.
## Contribution
If you would like to contribute to the project, please create a pull request and clearly describe the changes you want to make.

## Authors
- **Lăng Nhật Tân**
