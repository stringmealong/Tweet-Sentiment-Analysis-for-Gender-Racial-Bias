# Tweet-Sentiment-Analysis-for-Gender-Racial-Bias

#This project, completed while taking Big Data Analytics (IST 718)
under the supervision of Dr. Daniel Acuna, aimed to identify
gender and racial bias by utilizing various sentiment analysis
models. The specific question the project proposed to answer was
the following: How is sentiment intensity affected by the implied
race and/or gender of a name/pronoun when used in a sentence?
In order to answer this question, model training was performed
on a dataset consisting of tweets from the social media platform,
Twitter, along with corresponding polarity scores that indicated
sentiment. This application required the cleaning and
preprocessing of data. Feature engineering was performed using
the MinMaxScaler package to ensure that range of features was
evenly scaled across datasets after vectorization [of features].
Sklearn's Binarizer package was also used in order to create
binarized labels, such that the polarities were classified as either
0 or 1 at the polarity threshold of 0.5. Above this threshold,
sentiment was deemed "positive", whilst below "negative".
Since the tweets would be classified as either negative or
positive, binary classification models were chosen for this
analysis. Random Forest, Naive Bayes, and Logistic Regression
models were built using Python, and were evaluated using both
Accuracy and F1 scores. Of the models used, the Logistic
Regression model's performance was the most optimal, with an
accuracy of approx. 95.03%, an F1 score for positive tweets of
approx. 91.26%, and an F1 score for negative tweets of approx.
88.93%.
The final results indicated that the greatest bias was gender
based, as a significantly larger number of tweets regarding female
(rather than male) names and pronouns were indicated to have
"negative" sentiment. Moreover, typically African-American names
also had a higher indication of "negative" sentiment than those of
other ethnicities. Thus, the African-American female group was
shown to have endured the greatest amount of negative bias.
