#!/usr/bin/env
'''
This is the Twittalytics App. This App serves as a marketing campaign guide.
Find out how well your next brand tweet will perform before you even post it!
Hashbang above allows this script to run on any operating system.
'''

from scipy.special import softmax
import sklearn
import torch
import streamlit as st, pandas as pd, joblib, numpy as np
from transformers import AutoTokenizer,AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification


# Reads in saved classification model

load_rf = joblib.load(f"best_model/rf_model.joblib")

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

[Sample CSV Input File](https://github.com/RazPyGeek/Twittalytics-Data/blob/main/app-test-data.csv)

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
		Followers_Count = st.sidebar.number_input('**Number Of Followers:**',min_value = 0, max_value = 108900197, value = 100, 
			help = 'Enter any number between 0 and 108900197', step = 500)
		Favorite_Count = st.sidebar.number_input('**Tweet Favorite Count:**',min_value = 0, max_value = 2399697, value = 100, 
			help = 'Enter any number between 0 and 2399697', step = 500)
		Favourites_Count = st.sidebar.number_input('**No. Tweets Marked As Favorite:**',min_value = 0, max_value = 1770752, value = 100, 
			help = 'Enter any number between 0 and 1770752', step = 500)
		Status_Count = st.sidebar.number_input('**No. Tweets Posted By Account:**',min_value = 1, max_value = 2567806, value = 100, 
			help = 'Enter any number between 0 and 2567806', step = 500)
		Listed_Count = st.sidebar.number_input('**No. Twitter List You Belong To:**',min_value = 0, max_value = 133955, value = 100, 
			help = 'Enter any number between 0 and 133955', step = 500)
		Friends_Count = st.sidebar.number_input('**No. Of Accounts Followed:**',min_value = 0, max_value = 892016, value = 100, 
			help = 'Enter any number between 0 and 892016', step = 500)
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
		with st.expander("Tips & Recommendations"):
			if prediction == 'Class 0':
				st.write("""
			    * Engage more with your followers to increase your tweet visibility.
				* Include relevant hashtags to expand the reach of your tweet.
				* Consider adding images or videos to your tweets to make them more engaging.
				* Aim for a moderate number of favorites and status counts to maintain a balanced engagement.

				**These tips are not absolute. Customize your strategies based on your target audience, 
				industry trends, and the evolving nature of social media.**
			""")
			elif prediction == 'Class 1':
				st.write("""
			    * **Verification or Quality:** Verification adds credibility; if not, focus on producing high-quality content.
				* **Followers Count:** Aim for a Followers Count up to 78,064,408 to expand your reach.
				* **Favorite Counts:** Share content with Favorite Counts up to 10,673 to encourage engagement.
				* **Status Count:** Maintain an active presence with a Status Count up to 2,113,577.
				* **Listed Count:** Incorporate hashtags and trends to build a Listed Count up to 133,955.
				* **Media:** Incorporate images or videos in your tweets for higher engagement.
				* **Favourites Count:** Aim for a Favourites Count up to 1,770,752 to showcase your interests.
				* **Friends Count:** Keep your Friends Count balanced up to 598,203 for optimal interactions.

				**These tips are not absolute. Customize your strategies based on your target audience, 
				industry trends, and the evolving nature of social media.**
			""")
			elif prediction == 'Class 2':
				st.write("""
				* **Followers Count:** Aim for a Followers Count up to 108,900,197 for broader reach.
				* **Favorite Counts:** Share tweets with Favorite Counts up to 21,060 for better engagement.
				* **Status Count:** Maintain an active presence with a Status Count up to 1,253,906.
				* **Listed Count:** Incorporate relevant hashtags to build a Listed Count up to 133,940.
				* **Media:** Include images or videos in your tweets for higher engagement.
				* **Favourites Count:** Aim for a Favourites Count up to 1,423,877 to showcase your interests.
				* **Friends Count:** Keep your Friends Count balanced up to 892,016 for optimal interactions.

				**These tips are not absolute. Customize your strategies based on your target audience, 
				industry trends, and the evolving nature of social media.**
			""")
			elif prediction == 'Class 3':
				st.write("""
				* **Hashtags:** Use trending hashtags to increase tweet visibility.
				* **Followers Count:** Aim for a Followers Count up to 108,900,155 for broader reach.
				* **Favorite Counts:** Share content with Favorite Counts up to 35,949 for better engagement.
				* **Status Count:** Maintain an active presence with a Status Count up to 1,436,687.
				* **Listed Count:** Including hashtags can build a Listed Count up to 128,199 for better visibility.
				* **Media:** Incorporate relevant images or videos to enhance engagement.
				* **Favourites Count:** Aim for a Favourites Count up to 787,634 to showcase your interests.
				* **Friends Count:** Keep your Friends Count balanced up to 891,835 for optimal interactions.

				**These tips are not absolute. Customize your strategies based on your target audience, 
				industry trends, and the evolving nature of social media.**
			""")
			else:
				st.write("""
			    * **Compelling Content:** Focus on creating compelling and shareable content to encourage retweets.
				* **Trending Hashtags:** Utilize popular and trending hashtags to increase visibility.
				* **Followers Count:** Aim for a Followers Count up to 108,898,870 for broader reach.
				* **Favorite Counts:** Share content with Favorite Counts up to 2,399,697 for better engagement.
				* **Status Count:** Maintain an active presence with a Status Count up to 1,346,173.
				* **Listed Count:** Including relevant hashtags can build a Listed Count up to 128,201 for better visibility.
				* **Media:** Incorporate images or videos to enhance the visual appeal of your tweets.
				* **Favourites Count:** Aim for a Favourites Count up to 900,465 to showcase your interests.
				* **Friends Count:** Keep your Friends Count balanced up to 891,971 for optimal interactions.

				**These tips are not absolute. Customize your strategies based on your target audience, 
				industry trends, and the evolving nature of social media.**
			""")

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
			
