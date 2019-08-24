from django.shortcuts import render
from django.http import JsonResponse
from random import randint
from django.db.models.aggregates import Count
import tweepy
import re
import pickle
from sklearn.externals import joblib

import os
from django.conf import settings

from .models import Random

def training(request):
	return render(request, 'screens/training.html')
	
def performance(request):
	return render(request, 'screens/performance.html')

def index(request):
	return render(request, 'screens/index.html')

def next(request, step):
	if request.method == "POST":
		input = request.POST['input']
	
	# Begin
	if step == 1:
		stage1 = '<br>Starting...<br><b>Preprocessing Stage<b><br><br>'
		return JsonResponse({'msgs': stage1, 'current': input})

	# Removing HTML Tags
	elif step == 2:
		c1 = len(input)
		def cleanhtml(raw_html):
			cleanr = re.compile('<.*?>') 
			cleantext = re.sub(cleanr, '', raw_html)
			return cleantext

		clean = cleanhtml(input)
		c2 = len(clean)
		if c2<c1:
			stage2 = '<b>1. Removing HTML Tags:</b><br>Tags Found<br>Cleaning...<br>'+clean
		else:
			stage2 = '<b>1. Removing HTML Tags:</b><br>No Tag Found<br>'+clean

		return JsonResponse({'msgs': stage2, 'current': clean})

	# Removing Links/URLs
	elif step == 3:
		c1 = len(input)
		clean = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', input)
		c2 = len(clean)

		if c2<c1:
			stage3 = '<br><br><b>2. Removing URLs:</b><br>Links Found<br>Cleaning...<br>'+clean
		else:
			stage3 = '<br><br><b>2. Removing URLs:</b><br>No Link Found<br>'+clean

		return JsonResponse({'msgs': stage3, 'current': clean})
	
	# LowerCase
	elif step == 4:
		string = input.lower()
		stage4 = '<br><br><b>3. Converting string to lowercase:</b><br>'+string

		return JsonResponse({'msgs': stage4, 'current': string})
	
	# Predictions
	elif step == 5:
		clf = joblib.load(os.path.join(settings.PROJECT_ROOT, 'classifiers/nb.pkl'))
		li = []
		li.append(input)

		prediction = clf.predict(li)[0]

		if prediction == 1:
			res = 'Sarcastic'
		else:
			res = 'Not Sarcastic'

		result = '<br><br><b>Predictions:</b><br>1. Naive Bayes<br>'+res
		return JsonResponse({'prediction': result})

	elif step == 6:
		clf = joblib.load(os.path.join(settings.PROJECT_ROOT, 'classifiers/svm.pkl'))
		li = []
		li.append(input)

		prediction = clf.predict(li)[0]

		if prediction == 1:
			res = 'Sarcastic'
		else:
			res = 'Not Sarcastic'

		result = '<br><br>2. SVM<br>'+res
		return JsonResponse({'prediction': result})

	elif step == 7:
		clf = joblib.load(os.path.join(settings.PROJECT_ROOT, 'classifiers/rf.pkl'))
		li = []
		li.append(input)
		prediction = clf.predict(li)[0]

		if prediction == 1:
			res = 'Sarcastic'
		else:
			res = 'Not Sarcastic'

		result = '<br><br>3. Random Forest<br>'+res
		return JsonResponse({'prediction': result})

def get_random(request):
	count = Random.objects.aggregate(count=Count('id'))['count']
	random_index = randint(0, count - 1)
	string = Random.objects.all()[random_index]
	string = str(string)

	return JsonResponse({'string': string})
	
def get_tweet(request):
	#Access tokens 
	keyword = request.POST['keyword']

	auth = tweepy.auth.OAuthHandler('M0WEJ1Hqtr07RANJ3NufEpw4F', 'fn5Tm6yd0AXLHDPRRPP3CU3L18oMq8lpm98ZrQidQ9klNsJiii')
	auth.set_access_token('3271024165-LVAdlJcpS07dv8C1gcf1X9ROVApV2RhnC9Lew1Q', '1ZNzXgeboqFSXZXthVWKGm1rRapNyPMwvBblN9djWLLqb')

	api = tweepy.API(auth)

	results = api.search(q=keyword, lang="en", count=1, tweet_mode="extended")
	
	for tweet in results:
		string = tweet._json['full_text']
	
	#clean
	if 'RT ' in string:
		for tweet in results:
			string = tweet._json['retweeted_status']['full_text']
		# string = string.split('RT ')[1]
	else:
		for tweet in results:
			string = tweet._json['full_text']
	
	string = ' '.join(word for word in string.split() if word[0]!='@')
	return JsonResponse({'string': string})