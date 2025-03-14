# crypto-sentiment-classification
This project focuses on classifying online comments as either positive or negative based on sentiment. Using a labeled dataset of past comments, the model learns sentiment patterns and predicts whether new, unseen text expresses a negative or positive sentiment.

## Key Features

✅ Preprocessed dataset with labeled sentiment (0 = Negative, 1 = Positive)  
✅ Machine learning and deep learning approaches for text classification  
✅ Data augmentation techniques to improve model performance  
✅ Future predictions on real-time or batch text data 

## Use Cases

📢 Social media sentiment analysis  
💬 Customer feedback classification  
⚠️ Automated content moderation  


🔹 Tech Stack: Python, TensorFlow, NLP techniques, FastApi for Realtime Inferencing

## Data  
This model training is done with sample 562 records collected from Reddit. There are 2 possible Sentiment Labels Positive/Negative  

**📥 Download Dataset**
[Click here to download the dataset](https://raw.githubusercontent.com/your-username/your-repo/main/data/dataset.csv)

**Training the model**
Used a sentance transformer "all-mpnet-base-v2" for encoding the text and a shallow neural network to perform the predictions  
To train with your customer dataset can use csv file as in the below format  

**sample data**  
```
Comment,Sentiment
today was my luckey day, Positive
all market fell today, Negative
```
we can use the train.py module to train the data the data by using below command and parameters:
```
python3 train.py --input_file="/data/raw/crypto_currency_sentiment_dataset.csv" --output_dir="/models/model" --epochs=50 --batch_size=8
```
## To Run and Inference below specified docker configurations can be used
