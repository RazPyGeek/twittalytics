#!/usr/bin/env
'''
This is the Twittalytics App. This App serves as a marketing campaign guide.
Find out how well your next brand tweet will perform before you even post it!
Hashbang above allows this script to run on any operating system.
'''

from scipy.special import softmax
import streamlit as st, pandas as pd, joblib, numpy as np
from transformers import AutoTokenizer,AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification


# Reads in saved classification model

load_rf = joblib.load('best_model/rf_model.joblib')

# Custom Functions
#st.sidebar.header('Input Features')
convert = {'Yes':1,
			'No':0
	
}

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# App header
st.title(':bird: :red[Twittalytics App]')
# Info tooltip
st.info("""This app predicts the **Retweet Probability** of your twitter post!
Some training data  was obtained from the 
[HuggingFace library](https://huggingface.co/datasets/mesolitica/snapshot-twitter-2022-09-03/tree/main).
You can either enter features as input from the sidebar or upload a csv file containing rows of features with a header.
""", icon= None)



st.divider()  # Draws a horizontal rule

uploaded_file = st.file_uploader(":red[**Upload A CSV File**]")

if uploaded_file is not None:
  	try:
  		# Can be used wherever a "file-like" object is accepted:
	    dataframe = pd.read_csv(uploaded_file)
	    st.success('File Imported Successfully!', icon="✅")
	    st.write(dataframe.head(5))

  	except Exception as e:

  		st.error(':red[Error:] Wrong File Type', icon="🚨")

  	input_df = dataframe.iloc[:, :-1]
else:
	
	def user_input_features():

		Tweet_Text = st.sidebar.text_input(label='**Enter Tweet:**', placeholder='Life of Brian #DataScience',max_chars=120)
		Veri = st.sidebar.radio('**Account Verification Status:**',('Yes','No'), horizontal= True)
		Account_Age = st.sidebar.slider('**Age Of Twitter Account (Years):**',1,17,3)
		Followers_Count = st.sidebar.slider('**Number Of Followers:**',0,108900197,100)
		Favorite_Count = st.sidebar.slider('**No. Times Tweet Marked As Favorite:**',0,2399697,100)
		Status_Count = st.sidebar.slider('**No. Tweets Posted By Account:**',1,2567806,100)
		Listed_Count = st.sidebar.slider('**No. Twitter List You Belong To:**',0,133955,100)
		Favourites_Count = st.sidebar.slider('**No. Tweets Marked As Favorite:**',0,1770752,100)
		Friends_Count = st.sidebar.slider('**No. Of Accounts Followed:**',0,892016,100)
		Has_Media = st.sidebar.radio('**Will There Be Media Within The Tweet?**',('Yes','No'), horizontal= True)
		Hashtags = st.sidebar.radio('**Does The Post Contain Has Tags?**',('Yes','No'), horizontal= True)

		#Convert Bool Values
		Verified = convert[Veri]
		Tweet_Has_Media = convert[Has_Media]
		Has_Hashtags = convert[Hashtags]

		negative,neutral,positive = 0,0,0

		'''Converting Tweet Column To sentiment polarity using
		the latest RoBERTa model'''

		roberta_model = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
		tokenizer = AutoTokenizer.from_pretrained(roberta_model)
		config = AutoConfig.from_pretrained(roberta_model)

		# PT
		model = AutoModelForSequenceClassification.from_pretrained(roberta_model)

		text = Tweet_Text
		text = preprocess(text)
		encoded_input = tokenizer(text, return_tensors='pt')
		output = model(**encoded_input)
		scores = output[0][0].detach().numpy()
		scores = softmax(scores)
		ranking = np.argsort(scores)
		ranking = ranking[::-1]

		for i in range(scores.shape[0]):
			l = config.id2label[ranking[i]]
			s = scores[ranking[i]]
			if l == 'neutral':
				neutral = s
			elif l == 'positive':
				positive = s
			else:
				negative = s


		data = {'Followers_Count': Followers_Count,
				'Favorite_Count': Favorite_Count,
				'Status_Count': Status_Count,
				'Verified': Verified,
				'Listed_Count': Listed_Count,
				'Favourites_Count': Favourites_Count,
				'Friends_Count':Friends_Count,
				'neutral': neutral,
				'positive': positive,
				'negative': negative,
				'Age_Of_Account': Account_Age,
				'Tweet_Has_Media': Tweet_Has_Media,
				'Has_Hashtags': Has_Hashtags
				}
		features = pd.DataFrame(data,index=[0])
		return features

	input_df = user_input_features()


st.divider()  # Draws a horizontal rule

if st.button('**Predict**', type = "primary",):
	# Apply model to make predictions
	prediction = load_rf.predict(input_df )
	prediction_proba = load_rf.predict_proba(input_df )


	tab1, tab2,  = st.tabs(["Prediction", "Prediction Probability"])

	with tab1:
		st.subheader('Prediction')
		retweet_class = np.array(['Class 0: 0 Retweets','Class 1: 1 to 100 Retweets', 'Class 2: 101 to 300 Retweets',
									'Class 3: 301 to 1000 Retweets', 'Class 4: 1001+ Retweets'])
		st.write(retweet_class[prediction])

	with tab2:
		st.subheader('Prediction Probability')
		st.write(prediction_proba)
		with st.expander("See explanation"):
			st.write("""
			    This tab displays the probability of a tweet belonging to each of the four retweet classes.
			    The retweet_class with the highest probability among the four will be considered as the predicted class for the tweet. 
			    This probability-based approach enables us to make more confident predictions and helps in determining the 
			    most likely retweet class for each individual tweet.
			""")
			
