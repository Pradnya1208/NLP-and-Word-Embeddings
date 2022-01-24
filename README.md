<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">NLP and Word Embeddings</div>
<div align="center"><img src="https://github.com/Pradnya1208/NLP-and-Word-Embeddings/blob/main/output/intro.gif?raw=true" width="60%"></div>


## Overview:
Natural language processing (NLP) is a field of computer science, artificial intelligence concerned with the interactions between computers and human (natural) languages, and, in particular, concerned with programming computers to fruitfully process large natural language data.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.

## Dataset:
[IMDB Dataset](http://www.imdb.com/)<br>
A common dataset for sentiment analysis is the corpus of Internet Movie Database (IMDB) movie reviews. Since each review comes with a text and a numerical rating, the number of stars, it is easy to label the training data. In the IMDB dataset, movie reviews that gave less then five stars where labeled negative while movies that gave more than seven stars where labeled positive (IMDB works with a ten star scale).

### Sentiment analysis with the IMDB dataset:
Sentiment analysis is about judging how positive or negative the tone in a document is. The output of a sentiment analysis is a score between zero and one, where one means the tone is very positive and zero means it is very negative. Sentiment analysis is used for trading quite frequently. For example the sentiment of quarterly reports issued by firms is automatically analyzed to see how the firm judges its own position. Sentiment analysis is also applied to the tweets of traders to estimate an overall market mood. Today, there are many data providers that offer sentiment analysis as a service.

In principle, training a sentiment analysis model works just like training a binary text classifier. The text gets classified into positive (1) or not positive (0). This works exactly like other binary classification only that we need some new tools to handle text.

### Tokenizing text:
Computers can not work with words directly. To them, a word is just a meaningless row of characters. To work with words, we need to turn words into so called 'Tokens'. A token is a number that represents that word. Each word gets assigned a token. Tokens are usually assigned by word frequency. The most frequent words like 'a' or 'the' get tokens like 1 or 2 while less often used words like 'profusely' get assigned very high numbers.

We can tokenize text directly with Keras. When we tokenize text, we usually choose a maximum number of words we want to consider, our vocabulary so to speak. This prevents us from assigning tokens to words that are hardly ever used, mostly because of typos or because they are not actual words or because they are just very uncommon. This prevents us from over fitting to texts that contain strange words or wired spelling errors. Words that are beyond that cutoff point get assigned the token 0, unknown.

### Embeddings:
As the attuned reader might have already guessed, words and word tokens are categorical features. As such, we can not directly feed them into the neural net. Just because a word has a larger token value, it does not express a higher value in any way. It is just a different category. Previously, we have dealt with categorical data by turning it into one hot encoded vectors. But for words, this is impractical. Since our vocabulary is 10,000 words, each vector would contain 10,000 numbers which are all zeros except for one. This is highly inefficient. Instead we will use an embedding.

Embeddings also turn categorical data into vectors. But instead of creating a one hot vector, we create a vector in which all elements are numbers.
In practice, embeddings work like a look up table. For each token, they store a vector. When the token is given to the embedding layer, it returns the vector for that token and passes it through the neural network. As the network trains, the embeddings get optimized as well. Remember that neural networks work by calculating the derivative of the loss function with respect to the parameters (weights) of the model. Through backpropagation we can also calculate the derivative of the loss function with respect to the input of the model. Thus we can optimize the embeddings to deliver ideal inputs that help our model.

In practice it looks like this: We have to specify how large we want the word vectors to be. A 50 dimensional vector is able to capture good embeddings even for quite large vocabularies. We also have to specify for how many words we want embeddings and how long our sequences are.

## Implementation:

**Libraries:**  `NumPy`  `pandas` `sklearn`  `Matplotlib` `tensorflow` `keras` `plotly`


## Model training and evaluation:
```
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

embedding_dim = 50

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 100, 50)           500000    
_________________________________________________________________
flatten_1 (Flatten)          (None, 5000)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                160032    
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 660,065
Trainable params: 660,065
Non-trainable params: 0
_________________________________________________________________
```
You can see that the embedding layer has 500,000 trainable parameters, that is 50 parameters for each of the 10K words.

```
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
```
```
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
```
Note that training your own embeddings is prone to over fitting. As you can see our model archives 100% accuracy on the training set but only 83% accuracy on the validation set. A clear sign of over fitting. In practice it is therefore quite rare to train new embeddings unless you have a massive dataset. Much more commonly, pre trained embeddings are used. A common pretrained embedding is GloVe, Global Vectors for Word Representation. It has been trained on billions of words from Wikipedia and the Gigaword 5 dataset, more than we could ever hope to train from our movie reviews.

<br>

#### We can create an embedding matrix holding all word vectors

This embedding matrix can be used as weights for the embedding layer. This way, the embedding layer uses the pre trained GloVe weights instead of random ones. We can also set the embedding layer to not trainable. This means, Keras won't change the weights of the embeddings while training which makes sense since our embeddings are already trained.

```
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen, weights = [embedding_matrix], trainable = False))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 100, 100)          1000000   
_________________________________________________________________
flatten_2 (Flatten)          (None, 10000)             0         
_________________________________________________________________
dense_3 (Dense)              (None, 32)                320032    
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 33        
=================================================================
Total params: 1,320,065
Trainable params: 320,065
Non-trainable params: 1,000,000
```
#### Notice that we now have far fewer trainable parameters.

### Using the trained model:
To determine the sentiment of a text, we can now use our trained model.
```
my_text = 'I love dogs. Dogs are the best. They are lovely, cuddly animals that only want the best for humans.'

seq = tokenizer.texts_to_sequences([my_text])
print('raw seq:',seq)
seq = pad_sequences(seq, maxlen=maxlen)
print('padded seq:',seq)
prediction = model.predict(seq)
print('positivity:',prediction)
```
```
raw seq: [[10, 116, 2515, 2515, 23, 1, 115, 33, 23, 1332, 1388, 12, 61, 178, 1, 115, 15, 1706]]
padded seq: [[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0   10  116
  2515 2515   23    1  115   33   23 1332 1388   12   61  178    1  115
    15 1706]]
positivity: [[ 0.71884662]]
```

### Learnings:
`Word embeddings` `Tokenization` `NLP` 






## References:
[NLP and Topic Modelling](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial)<br>
[Approaching (Almost) Any NLP Problem on Kaggle](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle)<br>
[Understanding RNN](https://medium.com/mindorks/understanding-the-recurrent-neural-network-44d593f112a2)<br>
[Word Embeddings](https://machinelearningmastery.com/what-are-word-embeddings/)<br>

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner


[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]
