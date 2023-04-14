import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import io
from io import BytesIO
from pathlib import Path
import recordlinkage
import numpy as np
import math
from PIL import Image
import plotly.express as px
#from selenium import webdriver
#from selenium.webdriver.chrome.options import Options
#from selenium.webdriver.common.keys import Keys
#from bs4 import BeautifulSoup
import re
import preprocessor as prep
from nameparser import HumanName
from pyxlsb import open_workbook as open_xlsb
import warnings
warnings.filterwarnings("ignore")
from time import sleep
from wordcloud import WordCloud, STOPWORDS
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
#import folium
# d_c_base_spacy_model = spacy.load("en_ner_bc5cdr_md")
# pgo_base_spacy_model = spacy.load("en_core_web_sm")
from keras.preprocessing.text import Tokenizer
from pathlib import Path
import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')
#from nltk.tokenize import word_tokenize


import tensorflow as tf
import seaborn as sns
np.random.seed()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import timedelta
import tweepy
from nameparser import HumanName
import datetime as dt
from time import sleep
import gender_guesser.detector as gender
g = gender.Detector()
#from pyxlsb import open_workbook as open_xlsb
import altair as alt

@st.cache(allow_output_mutation=True)
def upload1(data, data2):
	col1, col2 = st.columns(2)
	with st.spinner("Uploading file"):
		with col1:
			if data is not None:
				df = pd.read_excel(data)
				df = df.astype(str)
		with col2:
			df2 = pd.DataFrame()			
			if data2 is not None:
				df2 = pd.read_excel(data2)
				df2 = df2.astype(str)
						
	return df, df2
		
def dataframe_head(df):
	st.dataframe(df.head())
	with st.expander("Show Statistics"):
		buffer = io.StringIO() 
		df.info(buf=buffer) 
		s = buffer.getvalue() 
		st.text(s)

def name_cleaning(df):
    df1 = df.select_dtypes(exclude=[np.number])
    df2 = df.select_dtypes(include=[np.number])
    df1 = df1.astype('str')
    df1 = df1.replace(r'[^a-zA-Z ]', '', regex=True).replace("'", '')
    df1 = df1.apply(lambda x: x.astype(str).str.lower())
    df1 = df1.replace(['miss ','mr ','ms', 'dr', 'shri', 'shree', 'smt', 'ms'],'', regex=True)
    df1 = df1.replace(['bhai', 'bhau', 'bhoi', 'bai', 'kumar', 'kumr', 'kmr','ben', 'dei', 'devi', 'debi', 'kumaar', 'saheb'],
                    '', regex=True)
    df1 = df1.replace(['nan','None'], '')
    df1.append(df2, ignore_index=True)
    return df1

def header_1(url):
     st.markdown(f'<p style="color:#00008B;font-size:36px;border-radius:2%;text-align:center;">{url}</p>', unsafe_allow_html=True)
def header_2(url):
     st.markdown(f'<p style="color:#000000;font-size:18px;border-radius:2%;text-align:center;">{url}</p>', unsafe_allow_html=True)
def header_3(url):
     st.markdown(f'<p style="color:#000000;font-size:36px;border-radius:2%;text-align:center;">{url}</p>', unsafe_allow_html=True)


def calcscore(hcp_first, hcp_middle, hcp_last, hcp_title, hcp_gender, hcp_city, hcp_state, hcp_specialty, sm_first, sm_middle, sm_last, sm_title, sm_gender, sm_city, sm_state, sm_specialty):
	score_list_g1 = {'First':28, 'Middle':4, 'Last':33}
	score_list_g2 = {'Gender':10, 'Title':5, 'City':5, 'State':5, 'Specialty':10}
	final_score_list_g1 = {'First':0, 'Middle':0, 'Last':0}
	final_score_list_g2 = {'Gender':0, 'Title':0, 'City':0, 'State':0, 'Specialty':0}

	if ((hcp_first=="") or (sm_first=="")):
		final_score_list_g1['First'] = 0
	else:
		final_score_list_g1['First'] = score_list_g1['First']

	if ((hcp_middle=="") or (sm_middle=="")):
		final_score_list_g1['Middle'] = 0
	else:
		final_score_list_g1['Middle'] = score_list_g1['Middle']

	if ((hcp_last=="") or (sm_last=="")):
		final_score_list_g1['Last'] = 0
	else:
		final_score_list_g1['Last'] = score_list_g1['Last']

	weight_divided_g1 = sum(score_list_g1.values()) - sum(final_score_list_g1.values())
	num_non_null_g1 = 0
	for value in final_score_list_g1.values():
		if value != 0:
			num_non_null_g1 = num_non_null_g1+1
	for key, value in final_score_list_g1.items():
		if value != 0:
			final_score_list_g1[key] = final_score_list_g1[key]+(weight_divided_g1/num_non_null_g1)
	

	if ((hcp_title=="") or (sm_title=="")):
		final_score_list_g2['Title'] = 0
	else:
		final_score_list_g2['Title'] = score_list_g2['Title']

	if ((hcp_gender=="") or (sm_gender=="")):
		final_score_list_g2['Gender'] = 0
	else:
		final_score_list_g2['Gender'] = score_list_g2['Gender']

	if ((hcp_city=="") or (sm_city=="")):
		final_score_list_g2['City'] = 0
	else:
		final_score_list_g2['City'] = score_list_g2['City']
	    
	if ((hcp_state=="") or (sm_state=="")):
		final_score_list_g2['State'] = 0
	else:
		final_score_list_g2['State'] = score_list_g2['State']

	if ((hcp_specialty=="") or (sm_specialty=="")):
		final_score_list_g2['Specialty'] = 0
	else:
		final_score_list_g2['Specialty'] = score_list_g2['Specialty']
		
	weight_divided_g2 = sum(score_list_g2.values()) - sum(final_score_list_g2.values())
	num_non_null_g2 = 0
	for value in final_score_list_g2.values():
	    if value != 0:
	        num_non_null_g2 = num_non_null_g2+1
	for key, value in final_score_list_g2.items():
	    if value != 0:
	        final_score_list_g2[key] = final_score_list_g2[key]+(weight_divided_g2/num_non_null_g2)
	final_score_list_g1.update(final_score_list_g2)
	
	return final_score_list_g1
	

@st.cache(allow_output_mutation=True, persist=True)
def getoutput(df, df2, threshold1, threshold2):
	with st.spinner(text="Matching the social profiles, Please wait..."):
		df[['PersonFirstName','PersonLastName','PersonTitleCode']] = df[['PersonFirstName','PersonLastName','PersonTitleCode']].apply(lambda x: x.astype(str).str.lower())
		df2[['first','last','title']] = df2[['first','last','title']].apply(lambda x: x.astype(str).str.lower())
		df = df.replace(['nan','None'], '')
		df2 = df2.replace(['nan','None'], '')
		df.insert(len(df.columns), 'SrNum1', range(0, 0 + len(df)))
		df.set_index('SrNum1')
		df2.insert(len(df2.columns), 'SrNum2', range(0, 0 + len(df2)))
		df2.set_index('SrNum2')
		Oncology_keywords = ['oncology', 'oncologist', 'onco ', 'onc ', 'hematology', 'haematology', 'hematologist', 'haematologist','hem/onc',
		'cancer', 'lymphoma', 'chemo', 'chemotherapy', 'haem', 'leukaemia', 'leukemia', 'tumor', 'tumour', 'sarcoma', 'myeloma', 'hemato', 'cell therapy']
		df2['Desc'] = df2['Desc'].str.replace('#',' ')
		df2['Desc'] = df2['Desc'].astype(str)
		#df2['latest_tweets'] = df2['latest_tweets'].str.replace('#','')
		#df2['latest_tweets'] = df2['latest_tweets'].astype(str)
		def search_tag(s, tags):
			result = []
			s = s.lower()
			for each in tags:
			    if each.lower() in s:
			        result.append(each)
			if len(result)>0:
			    x = 'Oncology'
			else:
			    x = ''
			return x
		df2['specialty_extracted'] = df2['Desc'].apply(lambda x: search_tag(x, Oncology_keywords))
		#st.dataframe(df2.head(10))
		#df2['Hashtags'] = df2['latest_tweets'].apply(lambda x: search_tag(x, Oncology_keywords))

		#df['Specialty'] = df['PrimarySpecialtyName'].replace(['Hematology/Oncology','Medical Oncology','Pediatric Hematology/Oncology','Surgical Oncology','Gynecological Oncology','Musculoskeletal Oncology','Advanced Surgical Oncology-ASO'],
		#	['Oncology','Oncology','Oncology','Oncology','Oncology','Oncology','Oncology',])

		df['Specialty'] = df['PrimarySpecialtyName'].apply(lambda x:'Oncology' if 'oncology' in x.lower() else x) 
		df['Specialty'] = df['Specialty'].apply(lambda x:'Oncology' if 'hematology' in x.lower() else x) 
		df['Specialty'] = df['Specialty'].apply(lambda x:'Oncology' if 'haematology' in x.lower() else x) 

		df['Specialty'] = 'Oncology'

		df2.loc[(df2['specialty_extracted'] == 'Oncology') 
		#| (df2['Hashtags'] == 'Oncology')
		, 'Specialty'] = 'Oncology'
		df2['Specialty'] = df2['Specialty'].replace(['nan','None',np.nan], '')

		df2['City'] = df2['City'].replace(['LA','NY'],['LOS ANGELES','NEW YORK'])

		#st.write(len(df2[df2['Specialty']=='Oncology']))
		us_state_to_abbrev = {
		    "Alabama": "AL",
		    "Alaska": "AK",
		    "Arizona": "AZ",
		    "Arkansas": "AR",
		    "California": "CA",
		    "Colorado": "CO",
		    "Connecticut": "CT",
		    "Delaware": "DE",
		    "Florida": "FL",
		    "Georgia": "GA",
		    "Hawaii": "HI",
		    "Idaho": "ID",
		    "Illinois": "IL",
		    "Indiana": "IN",
		    "Iowa": "IA",
		    "Kansas": "KS",
		    "Kentucky": "KY",
		    "Louisiana": "LA",
		    "Maine": "ME",
		    "Maryland": "MD",
		    "Massachusetts": "MA",
		    "Michigan": "MI",
		    "Minnesota": "MN",
		    "Mississippi": "MS",
		    "Missouri": "MO",
		    "Montana": "MT",
		    "Nebraska": "NE",
		    "Nevada": "NV",
		    "New Hampshire": "NH",
		    "New Jersey": "NJ",
		    "New Mexico": "NM",
		    "New York": "NY",
		    "North Carolina": "NC",
		    "North Dakota": "ND",
		    "Ohio": "OH",
		    "Oklahoma": "OK",
		    "Oregon": "OR",
		    "Pennsylvania": "PA",
		    "Rhode Island": "RI",
		    "South Carolina": "SC",
		    "South Dakota": "SD",
		    "Tennessee": "TN",
		    "Texas": "TX",
		    "Utah": "UT",
		    "Vermont": "VT",
		    "Virginia": "VA",
		    "Washington": "WA",
		    "West Virginia": "WV",
		    "Wisconsin": "WI",
		    "Wyoming": "WY",
		    "District of Columbia": "DC",
		    "American Samoa": "AS",
		    "Guam": "GU",
		    "Northern Mariana Islands": "MP",
		    "Puerto Rico": "PR",
		    "United States Minor Outlying Islands": "UM",
		    "U.S. Virgin Islands": "VI",
		}

		df2 = df2.replace({'State code':us_state_to_abbrev})
		#st.dataframe(df2)
		#df2 = df2.loc[(df2['State code'].str.len() == 2) | (df2['State code'] == "")]
		#st.write(len(df2))

		indexer = recordlinkage.Index()
		indexer.sortedneighbourhood(left_on='PersonFirstName', right_on='first', window=25)
		candidates1 = indexer.index(df, df2)
		candidates1 = candidates1.to_frame(index=True)

		indexer.sortedneighbourhood(left_on='PersonLastName', right_on='last', window=25)
		candidates2 = indexer.index(df, df2)
		candidates2 = candidates2.to_frame(index=True)


		candidates = pd.concat([candidates1,candidates2])
		candidates = candidates.drop_duplicates([0,1])
		candidates = pd.MultiIndex.from_frame(candidates)
		

		#select features for string matching
		selectedfeatures1 = ['PersonFirstName','PersonMiddleName','PersonLastName','PersonTitleCode','PersonGender','CityName','State','Specialty']
		selectedfeatures = ['first','middle','last','title','Gender','City','State code','Specialty']
		lengh= len(selectedfeatures)
		#save orginal data for final display
		df1 = df.copy()
		df3 = df2.copy()
		#clean textual data

		df[selectedfeatures1] = name_cleaning(df[selectedfeatures1])
		df2[selectedfeatures] = name_cleaning(df2[selectedfeatures])

		compare = recordlinkage.Compare()
		for i in range(len(selectedfeatures)):
			compare.string(selectedfeatures1[i],
				               selectedfeatures[i],
				               #threshold=0.95,
				               label=selectedfeatures1[i],
				                method='levenshtein'
				              )
		features = compare.compute(candidates, df, df2)
		features.loc[features['State']<1, 'State'] = 0
		features.loc[features['CityName']<1, 'CityName'] = 0

		potential_matches = features.reset_index()
		potential_matches = potential_matches.add_suffix('_score')
		potential_matches.rename(columns={'0_score':'SrNum1', '1_score':'SrNum2'}, inplace=True)

		#potential_matches = potential_matches[(potential_matches['PersonFirstName_score']>=0.4) & (potential_matches['PersonLastName_score']>=0.4)]

		df1['SrNum1']=df1['SrNum1'].astype(int)
		df3['SrNum2']=df3['SrNum2'].astype(int)
		selectedfeatures1.append("SrNum1")
		selectedfeatures1.append("NPI")
		selectedfeatures.append("SrNum2")
		selectedfeatures.append('handle')
		
		df1 = df1.filter(selectedfeatures1)
		df3 = df3.filter(selectedfeatures)

		potential_matches = pd.merge(potential_matches, df1.add_suffix('_hcp'), left_on='SrNum1',right_on='SrNum1_hcp', how='left')
		potential_matches = pd.merge(potential_matches, df3.add_suffix('_sm'), left_on='SrNum2', right_on='SrNum2_sm', how='left')
		#st.write(potential_matches.columns)
		potential_matches['PersonTitleCode_score'] = ''
		#potential_matches.loc[potential_matches['title_sm'].str.contains('MD') & potential_matches['PersonTitleCode_hcp'].str.contains('MD'), 'PersonTitleCode_score'] = 1
		#potential_matches.loc[potential_matches['title_sm'].str.contains('DO') & potential_matches['PersonTitleCode_hcp'].str.contains('DO'), 'PersonTitleCode_score'] = 1
		#potential_matches.loc[potential_matches['PersonTitleCode_score']!=1, 'PersonTitleCode_score'] = 0
		potential_matches['PersonTitleCode_hcp'] = potential_matches['PersonTitleCode_hcp'].apply(lambda x: x.split())
		potential_matches['title_sm'] = potential_matches['title_sm'].apply(lambda x: x.split())
		for i in range(len(potential_matches)):
			set1 = set(potential_matches['PersonTitleCode_hcp'][i])
			set2 = set(potential_matches['title_sm'][i])
			intsctn = set1.intersection(set2)
			if len(intsctn)>0:
				potential_matches['PersonTitleCode_score'][i]=1
			else:
				potential_matches['PersonTitleCode_score'][i]=0

		potential_matches['Score'] = 0

		for index, row in potential_matches.iterrows():
			score_list_static= {'First':28, 'Middle':4, 'Last':33, 'Gender':10, 'Title':5, 'City':5, 'State':5, 'Specialty':10}
			#score_list = calcscore(row['PersonFirstName_hcp'],row['PersonMiddleName_hcp'],row['PersonLastName_hcp'],row['PersonTitleCode_hcp'],row['PersonGender_hcp'],row['CityName_hcp'],row['State_hcp'],row['Specialty_hcp'],row['first_sm'],row['middle_sm'],row['last_sm'],row['title_sm'],row['Gender_sm'],row['City_sm'],row['State code_sm'],row['Specialty_sm'])
			#potential_matches.at[index, 'Score_dynamic'] = row['PersonFirstName_score']*score_list.get('First')+row['PersonMiddleName_score']*score_list.get('Middle')+row['PersonLastName_score']*score_list.get('Last')+row['PersonGender_score']*score_list.get('Gender')+row['PersonTitleCode_score']*score_list.get('Title')+row['CityName_score']*score_list.get('City')+row['State_score']*score_list.get('State')+row['Specialty_score']*score_list.get('Specialty')
			potential_matches.at[index, 'Score'] = row['PersonFirstName_score']*score_list_static.get('First')+row['PersonMiddleName_score']*score_list_static.get('Middle')+row['PersonLastName_score']*score_list_static.get('Last')+row['PersonGender_score']*score_list_static.get('Gender')+row['PersonTitleCode_score']*score_list_static.get('Title')+row['CityName_score']*score_list_static.get('City')+row['State_score']*score_list_static.get('State')+row['Specialty_score']*score_list_static.get('Specialty')

			
		#potential_matches['Score'] = (potential_matches['Score_static']+potential_matches['Score_dynamic'])/2
		#st.dataframe(potential_matches.head())
		potential_matches = potential_matches.sort_values(['SrNum1','Score'], ascending=[True,False])

		final_merge = potential_matches

		test = final_merge.groupby('SrNum1')['Score'].agg(max)
		test= test.reset_index()
		test.rename(columns={'Score':'Max HCP Score'}, inplace=True)
		final_merge = pd.merge(final_merge, test, on='SrNum1', how='left')
		final_merge.loc[final_merge['Max HCP Score']==final_merge['Score'], 'Highest'] = 1
		final_merge.loc[((final_merge['Max HCP Score']-final_merge['Score']>0) & (final_merge['Max HCP Score']-final_merge['Score']<=threshold2)), 'NearBy'] = 1
		final_merge[['Max HCP Score','Highest', 'NearBy']].fillna(0, inplace=True)
		test2 = final_merge.groupby('SrNum1')['NearBy'].agg(max)
		test2= test2.reset_index()
		test2.rename(columns={'NearBy':'HasNearBy'}, inplace=True)
		final_merge = pd.merge(final_merge, test2, on='SrNum1', how='left')
		final_merge.loc[((final_merge['Max HCP Score']>=threshold1) & (final_merge['HasNearBy']!=1) & (final_merge['Highest']==1)), 'Category'] = 'Match'
		final_merge.loc[((final_merge['Max HCP Score']>=threshold1) & (final_merge['HasNearBy']==1) & ((final_merge['Highest']==1) | (final_merge['NearBy']==1))), 'Category'] = 'Manual'
		final_merge.loc[((final_merge['Max HCP Score']<threshold1) & (final_merge['Max HCP Score']>=threshold1-10) & ((final_merge['Highest']==1) | (final_merge['NearBy']==1))), 'Category'] = 'Manual'
		final_merge.loc[((final_merge['Max HCP Score']<threshold1-10) & ((final_merge['Highest']==1) | (final_merge['NearBy']==1))), 'Category'] = 'Non Match'
		#st.dataframe(final_merge)
		#final_merge["Probabilty"]=(final_merge["Score"]/lengh)*100
		#final_merge["Probabilty"]=final_merge["Score"]
		#final_merge.drop(['Score'], inplace=True, axis=1)
		final_merge['handle'] = '@' + final_merge['handle_sm'].astype(str)
		final_merge['Source'] = 'twitter'
		final_merge_match = final_merge[final_merge['Category'] == 'Match']
		final_merge_manual = final_merge[final_merge['Category'] == 'Manual']
		final_merge_non_match = final_merge[final_merge['Category'] == 'Non Match']
		
		lenofcandidates=len(final_merge_match)+len(final_merge_non_match)
		
	return final_merge, final_merge_match,final_merge_manual,final_merge_non_match,potential_matches, candidates, df, df2

def view_data(df, df2, threshold1,threshold2,final_merge, final_merge_match,final_merge_manual,final_merge_non_match,potential_matches, candidates):
	totalrecords=len(df)
	st1 = st.container()
	st2 = st.container()
	st3 = st.container()
	with st1:
		header_3('Results:')
		col1, col2, col3= st.columns(3)
		with col1:
			header_2("Number of HCP Records")
		with col2:
			header_2("Number of Social Profiles")
		#with col3:
			#header_2("Threshold")
		with col3:
			header_2("Number of Candidate Pairs")
		with col1:
			header_1(str(len(df)))
		with col2:
			header_1(str(len(df2)))
		#with col3:
		#	header_1(str(threshold1))
		with col3:
			header_1(str(len(candidates)))

		col1, col2, col3 = st.columns(3)
		with col1:
			header_2("Number of Matching Profiles")
			#st.text("\n")
			#st.text("\n")
		with col2:
			header_2("Number of Manual Review Profiles")
		with col3:
			header_2("Number of Non Matching Profiles")
		with col1:
			header_1(str(len(final_merge_match['SrNum1_hcp'].unique())))
		with col2:
			header_1(str(len(final_merge_manual['SrNum1_hcp'].unique())))
		with col3:
			header_1(str(len(final_merge_non_match['SrNum1_hcp'].unique())))

		st.cache()
	hide_dataframe_row_index = """
	            <style>
	            .row_heading.level0 {display:none}
	            .blank {display:none}
	            </style>
	            """
	st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;padding-left:20px;} </style>', unsafe_allow_html=True)
	header_3("View Data")
	viewradio = st.radio("",('Match','Manual','Non Match'))
	if viewradio == "Match":
		st.caption("All the HCPs mapped with score greater than {}% having no other candidate within -{}% window".format(threshold1,threshold2))
		st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
		st.dataframe(final_merge_match[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp','first_sm','middle_sm','last_sm','title_sm','Gender_sm','City_sm','State code_sm','Specialty_sm','handle','Score']])
	if viewradio == "Manual":
		st.caption("All the HCPs mapped with score greater than {}% and having candidates within -{}% window".format(threshold1,threshold2))
		st.caption("All the HCPs mapped with score between {}% and {}%".format(threshold1,threshold1-10))
		st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
		st.dataframe(final_merge_manual[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp','first_sm','middle_sm','last_sm','title_sm','Gender_sm','City_sm','State code_sm','Specialty_sm','handle','Score']])
	if viewradio == "Non Match":
		st.caption("All the HCPs mapped with score less than {}%".format(threshold1-10))
		st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
		final_merge_non_match1 = final_merge_non_match.drop_duplicates(['NPI_hcp']) 
		final_merge_non_match1 = final_merge_non_match1[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp']]
		st.dataframe(final_merge_non_match1)
	
	col5,col6,col7,col8 = st.columns(4)
	final_merge_match.replace({"nan":np.nan},inplace=True)
	final_merge_match = final_merge_match[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp','first_sm','middle_sm','last_sm','title_sm','Gender_sm','City_sm','State code_sm','Specialty_sm','handle','Score','Source']]
	final_merge_manual.replace({"nan":np.nan},inplace=True)
	final_merge_manual = final_merge_manual[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp','first_sm','middle_sm','last_sm','title_sm','Gender_sm','City_sm','State code_sm','Specialty_sm','handle','Score','Source']]
	final_merge_manual1 = final_merge_manual
	final_merge_manual1['Valid (Enter Y where valid)'] = ''
	final_merge_non_match.replace({"nan":np.nan},inplace=True)
	final_merge_non_match1 = final_merge_non_match[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp','first_sm','middle_sm','last_sm','title_sm','Gender_sm','City_sm','State code_sm','Specialty_sm','handle','Score','Source']]
	potential_matches.replace({"nan":np.nan},inplace=True)
	final_merge1 = final_merge[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp','first_sm','middle_sm','last_sm','title_sm','Gender_sm','City_sm','State code_sm','Specialty_sm','handle','Score','Category','Source']]
	with col5:
		st.download_button(label="Download Match data",data=final_merge_match.to_csv(index=False).encode('utf-8'),file_name="Matching HCPs Twitter API.csv",mime='text/csv')
		
	with col6:
		st.download_button(label="Download Manual Review data",data=final_merge_manual1.to_csv(index=False).encode('utf-8'),file_name="Manual Review HCPs Twitter API.csv",mime='text/csv')
	
	with col7:	
		st.download_button(label="Download non Match data",data=final_merge_non_match1.to_csv(index=False).encode('utf-8'),file_name="Non Matching HCPs Twitter API.csv",mime='text/csv')
	
	with col8:
		st.download_button(label="Download All HCPs data",data=final_merge1.to_csv(index=False).encode('utf-8'),file_name="All HCPs Twitter API.csv",mime='text/csv')


	#st.download_button(label="Download candidate data",data=potential_matches.to_csv(index=False).encode('utf-8'),file_name="Candidate.csv",mime='text/csv')

	
	return final_merge_match,final_merge_manual,final_merge_non_match

def view_data2(df, final_merge_match, final_merge_non_match, threshold1,threshold2):
	totalrecords=len(df)
	header_3('After Manual Review:')
	col1, col2, col3= st.columns(3)
	with col1:
		header_2("Total Number of HCPs")
		st.text("\n")
		st.text("\n")
	with col2:
		header_2("Number of Matching Profiles")
		st.text("\n")
		st.text("\n")
	with col3:
		header_2("Number of Non Matching Profiles")
	with col1:
		header_1(str(len(df)))
	with col2:
		header_1(str(len(final_merge_match['SrNum1_hcp'].unique())))
	with col3:
		header_1(str(len(final_merge_non_match['SrNum1_hcp'].unique())))

	hide_dataframe_row_index = """
	            <style>
	            .row_heading.level0 {display:none}
	            .blank {display:none}
	            </style>
	            """
	st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;padding-left:20px;} </style>', unsafe_allow_html=True)
	viewradio = st.radio("",('Match','Non Match'), key='2')
	if viewradio == "Match":
		st.dataframe(final_merge_match[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp','first_sm','middle_sm','last_sm','title_sm','Gender_sm','City_sm','State code_sm','Specialty_sm','handle','Score']])
	if viewradio == "Non Match":
		final_merge_non_match1 = final_merge_non_match.drop_duplicates(['NPI_hcp'])
		final_merge_non_match1 = final_merge_non_match1[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp']]
		st.dataframe(final_merge_non_match1)
	final_merge_match1 = final_merge_match
	final_merge_match1['Category'] = 'Match'
	final_merge_non_match1 = final_merge_non_match
	final_merge_non_match1['Category'] = 'Non Match'
	finalfile1 = final_merge_match1.append(final_merge_non_match1).drop_duplicates(['SrNum1_hcp'])
	col5,col6,col7 = st.columns(3)
	with col5:
		st.download_button(label="Download Match data",data=final_merge_match.to_csv(index=False).encode('utf-8'),file_name="Matching HCPs After Review.csv",mime='text/csv')
		
	with col6:	
		st.download_button(label="Download Non Match data",data=final_merge_non_match1.to_csv(index=False).encode('utf-8'),file_name="Non Matching HCPs After Review.csv",mime='text/csv')
	
	with col7:
		st.download_button(label="Download All HCP data",data=finalfile1.to_csv(index=False).encode('utf-8'),file_name="All HCPs After Review.csv",mime='text/csv')
	
#@st.cache(persist=True)
def manual_file_upload(manualfile):
	
	manualfile['Valid'] = manualfile['Valid (Enter Y where valid)'].replace({'Y':1,'N':0})
	manualfile[['Valid']].fillna(0, inplace=True)
	test4 = manualfile.groupby('SrNum1_hcp')['Valid'].agg(sum)
	test4= test4.reset_index()
	test4.rename(columns={'Valid':'ValidCount'}, inplace=True)
	manualfile = pd.merge(manualfile,test4,on='SrNum1_hcp', how='left')
	if (manualfile['ValidCount'].unique() > 1).any():
		errordf = manualfile[manualfile['ValidCount']>1]
		st.error("Please select only 1 valid candidate for below HCPs in the CSV file")
		st.dataframe(errordf[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonLastName_hcp']].drop_duplicates())
		st.stop()
	test3 = manualfile.groupby('SrNum1_hcp')['Valid'].agg(max)
	test3= test3.reset_index()
	test3.rename(columns={'Valid':'HasValid'}, inplace=True)
	manualfile = pd.merge(manualfile,test3,on='SrNum1_hcp', how='left')
	manualfile.loc[(manualfile['HasValid']==1) & (manualfile['Valid']==1),'File'] = 'Match'
	manualfile.loc[(manualfile['HasValid']==1) & (manualfile['Valid']!=1),'File'] = 'Discard'
	manualfile.loc[(manualfile['HasValid']!=1),'File'] = 'Non Match'
	st.write()
	return manualfile

@st.cache(persist=True,allow_output_mutation=True)
def profilesearch(df):

	first = df.PersonFirstName
	last = df.PersonLastName
	f_name = first+" "+last

	chrom_opt = Options()
	chrom_opt.add_argument("--disable-extensions")
	chrom_opt.add_argument("--disable-gpu")
	chrom_opt.headless = True
	driver = webdriver.Chrome(chrome_options=chrom_opt)
	handle_list = []
	name = []

	with st.spinner(text="Fetching relevant profiles using combination of \nHCP FirstName and Last Name, Please wait...."):	
		for i in f_name:
		    url = "https://twitter.com/search?q=({})%20&src=typed_query&f=user".format(i)
		    #print(url)
		    driver.get(url)
		    sleep(2)
		    last_height = 0
		    for p in range(10):
		        soup = BeautifulSoup(driver.page_source,"lxml")
		        for div in soup.find_all("div",class_="css-901oao css-1hf3ou5 r-14j79pv r-18u37iz r-37j5jr r-a023e6 r-16dba41 r-rjixqe r-bcqeeo r-qvutc0"):
		            handle = div.text
		            handle = div.text.strip()
		            #print(handle)
		            handle_list.append(handle)
		            name.append(i)
		        driver.execute_script(f"window.scrollTo(0,document.body.scrollHeight);")
		        sleep(2)
		        current_height = driver.execute_script("return document.body.scrollHeight")
		        if current_height == last_height:
		            break
		        last_height = current_height

		# for j,k in zip(first,last):
		#     url = "https://twitter.com/search?q={}%20OR%20{}%20&src=typed_query&f=user".format(j,k)
		#     #print(url)
		#     driver.get(url)
		#     sleep(2)
		#     last_height = 0
		#     for p in range(10):
		#         soup = BeautifulSoup(driver.page_source,"lxml")
		#         for div in soup.find_all("div",class_="css-901oao css-1hf3ou5 r-14j79pv r-18u37iz r-37j5jr r-a023e6 r-16dba41 r-rjixqe r-bcqeeo r-qvutc0"):
		#             handle = div.text
		#             handle = div.text.strip()
		#             #print(handle)
		#             handle_list.append(handle)
		#             name.append(j+' '+k)
		#         driver.execute_script(f"window.scrollTo(0,document.body.scrollHeight);")
		#         sleep(2)
		#         current_height = driver.execute_script("return document.body.scrollHeight")
		#         if current_height == last_height:
		#             break
		#         last_height = current_height

	d = pd.DataFrame()
	d['name'] = name
	d["handle_list"] = handle_list
	d.drop_duplicates(subset=['handle_list'],inplace=True)

	return d

def datacleaning(df):
	st.dataframe(df.head())
	# st.caption("Features considered for preprocessing:")
	# st.caption("Full Name, City Code, Professions, Hastags")
	# st.caption("preprocessing involves:")
	# st.caption("1.Removal of duplicacy")
	# st.caption("2.Cleaning and Categoring Full Name to First, Middle and Last Name")
	# st.caption("3.Separation of titles from Full Name")
	# st.caption("4.Converstion of city code to state")
	with st.spinner("Data cleaning in progress, please wait...."):
		d=pd.DataFrame()
		d_author = []
		d_hastag = []

		with st.spinner("Identifying duplicate profiles and cleaning..."):
			
			for i in df['Author'].unique():
				d_author.append(i)
				d_hastag.append(", ".join(list(df[df.Author == i]['Hashtags'].dropna())))
			    
			d['Author']=d_author
			d['All_hastags'] = d_hastag

			df1 = pd.merge(df,d,how='inner',on=['Author'])
			df = df1.drop_duplicates(subset=['Author'])
			df.reset_index(drop=True,inplace=True)

		def namecleaning(x):
			x= x.split(maxsplit=1)[1]
			x= re.sub('[^\w\s,-]', '', x)
			x= p.clean(x)
			return x

		def citycode(x):
			try:
				x = x.split(".")[1]
				return x
			except:
				return None
		with st.spinner("Cleaning the profile names"):    
			df['Full Name cleaned'] = df['Full Name'].apply(namecleaning)
		lst = ["AACC","AAHIVP","AAS","ABAAHP","ABD","ABLM","ABOM","ABPP","ACCNSP","ACHPN","ACS","AFAASLD","AGCNSBC","AGNPC","AGPCNP","AIBVRC","ANPBC","AOCNP","AOCNS","APRN","APSW","ASCP","ATC","ATSF","BCACP","BCCCP","BCCP","BCIDP","BCOP","BCPA","BCPPS","BCPS","BSc","BSN","CACP","CAQSM","CBM","CCAP","CCCSLP","CCDS","CCHP","CCRC","CCRN","CCRNCSC","CCRP","CCSP","CCTP","CDCES","CDE","CDN","CEHP","CFLE","CFPS","CGC","CHFN","CIC","CIH","CKNS","CLS","CLT","CMD","CMN","CMPC","CNE","CNM","CNS","CPA","CPC","CPCS","CPE","CPH","CPhT","CPI","CPN","CPNPAC","CPNPPC","CPT","CRNA","CRNP","CRS","CRT","CSAT","CSCN","CSCS","CSM","CSR","CSSD","CST","CV","DABAT","DABCC","DABOM","DABR","DABS","DC","DD","DDS","DFAPA","Dipl","DMD","DMedSc","DNP","DO","DPM","DPT","Dr","DS","DVM","EdD","EEG","ELS","EMT","EMTP","es","Esq","FAAD","FAAFP","FAAHKS","FAAHPM","FAAN","FAANA","FAANS","FAAOS","FAAP","FABNO","FACC","FACE","FACEP","FACG","FACHE","FACOG","FACOP","FACOS","FACP","FACPM","FACR","FACS","FAEMS","FAHA","FAIUM","FAMIA","FAND","FAOA","FASA","FASCO","FASCP","FASCRS","FASE","FASN","FAST","FCAP","FCCM","FCCP","FESC","FHFSA","FHRS","FIDSA","FNKF","FNLA","FNLDI","FNP","FNPBC","FNPC","FNPS","FPC","FPMRS","FRCP","FRCPC","FRCSC","FRCSEd","FSCAI","FSCCT","FSCMR","FSIR","FSVM","FSVS","GC","GCDF","GED","HMDC","IBCLC","IFMCP","IHC","INHC","ISE","IV","JD","AFCON","Jr","LAc","law","LCAC","LCGC","LCPC","LCSW","LD","LDN","LE","LICSW","LLC","LMSW","LMT","LP","LPC","LPCC","LPN","LRCPI","LRCSI","MA","MACC","MACP","MAM","MAN","MAS","MBA","MBBS","MBI","MBMS","MBS","MCh","md","MD","MDiv","Mecca","MEd","MEHP","MFA","MHA","MHPE","MHR","MHS","MHSc","MIAC","MLA","MLS","MLSASCP","MM","MMM","MMSc","MOT","MPA","MPE","mph","MPhil","MPHTM","MPP","MRC","MRCSI","MS","MSc","MSCE","MSCI","MSCN","MSCR","MSCS","MSHS","MSME","MSN","MSPH","MSW","myself","NBCT","NCC","NCSN","NCSP","NCTTP","ND","NJWO","NP","NPC","NRP","Nurse","NVRN","OCDT","OCN","OCS","OD","ONC","OT","PAC","PCS","PE","Pharm","PharmD","PhD","PhDoubleDs","PHN","PNPBC","Professor","Psychologist","PsyD","PT","QIAASCP","Quichocho","RBT","RD","RDH","RDN","RhMSUS","RKmd","RN","RNBC","RNCOB","RPh","RPVI","RRT","RVT","Science","SCRN","Sr","ThM","VP","VTILVOT","WCC","WHNPBC","do"]

		fullname=[]
		main_title=[]
		with st.spinner("Seperating out the titles from profile Names"):
			for j in range(len(df)):
				name=[]
				title=[]
				for i in df['Full Name cleaned'][j].split():
					i = re.sub('[^\w\s]', '', i)
					if i in lst:
						title.append(i)
					else:
						name.append(i)
				fullname.append(" ".join(name))
				main_title.append(" ".join(title))

		with st.spinner("Categories Full Name to First,Middle and Last Name"):    
			df['newfullname'] = fullname
			df['title'] = main_title

			df["First Name"] = df["newfullname"].apply(lambda x: HumanName(x).first)
			df["Middle Name"] = df["newfullname"].apply(lambda x: HumanName(x).middle)
			df["Last Name"] = df["newfullname"].apply(lambda x: HumanName(x).last)

			df.Gender.replace({'male':'M','female':'F','unknown':np.nan},inplace=True)
			df['State code'] = df['City Code'].apply(citycode)
		
		with st.spinner("Separating out the Job titles"):
			profession_1 = []
			profession_2= []
			profession_3= []
			Job_title_1= []
			Job_title_2= []
			Job_title_3= []

			for i in df.Professions:
			    
			    try:
			        l = i.split("},")
			        if len(l)==1:
			            prof1 = l[0].split(',')[0].split("=")[1].strip("}")
			            job1 = l[0].split(',')[1].split("=")[1].strip("}")
			            prof2 = None
			            job2 = None
			            prof3 = None
			            job3 = None

			        elif len(l)==2:
			            prof1 = l[0].split(',')[0].split("=")[1].strip("}")
			            job1 = l[0].split(',')[1].split("=")[1].strip("}")
			            prof2 = l[1].split(',')[0].split("=")[1].strip("}")
			            job2 = l[1].split(',')[1].split("=")[1].strip("}")
			            prof3 = None
			            job3 = None
			            

			        elif len(l)==3:
			            prof1 = l[0].split(',')[0].split("=")[1].strip("}")
			            job1 = l[0].split(',')[1].split("=")[1].strip("}")
			            prof2 = l[1].split(',')[0].split("=")[1].strip("}")
			            job2 = l[1].split(',')[1].split("=")[1].strip("}")
			            prof3 = l[2].split(',')[0].split("=")[1].strip("}")
			            job3 = l[2].split(',')[1].split("=")[1].strip("}")


			        else:
			            prof1 = l[0].split(',')[0].split("=")[1].strip("}")
			            job1 = l[0].split(',')[1].split("=")[1].strip("}")
			            prof2 = None
			            job2 = None
			            prof3 = None
			            job3 = None
			        
			        profession_1.append(prof1)
			        profession_2.append(prof2)
			        profession_3.append(prof3)
			        Job_title_1.append(job1)
			        Job_title_2.append(job2)
			        Job_title_3.append(job3)
			    
			    except:
			        profession_1.append(None)
			        profession_2.append(None)
			        profession_3.append(None)
			        Job_title_1.append(None)
			        Job_title_2.append(None)
			        Job_title_3.append(None)
			        

			df['Profession_1'] = profession_1
			df['Job_title_1'] = Job_title_1
			df['Profession_2'] = profession_2
			df['Job_title_2'] = Job_title_2
			df['Profession_3'] = profession_3
			df['Job_title_3'] = Job_title_3
	st.success("Data Cleaning is successful")
	# st.dataframe(df.head())	
	return df

def to_excel(df):
	output = BytesIO()
	writer = pd.ExcelWriter(output, engine='xlsxwriter')
	df.to_excel(writer, index=False, sheet_name='Sheet1')
	workbook = writer.book
	worksheet = writer.sheets['Sheet1']
	format1 = workbook.add_format({'num_format': '0.00'}) 
	worksheet.set_column('A:A', None, format1)  
	writer.save()
	processed_data = output.getvalue()
	return processed_data

image = Image.open('Indegene_Logo.png')
st.sidebar.image(image, width=200, clamp=False, channels="RGB", output_format="auto")
activities = ["Data Collection", "Social Media Handle Mapping", "Affinity Calculation","Insights"]
choice = st.sidebar.selectbox("Select Activity", activities, key='sel1')


#@st.cache
# def search_tag(s, tags):

#     s = s.lower()
#     s = re.sub(r'[^a-zA-Z ]', '', s)
#     flg = '0'
#     for each in tags:
#         if each.lower() in s:
#             flg = '1'
#             break
#     return flg

@st.cache()
def twittwe_api(sm_handles):

	with st.spinner("Collecting profile information from twitter"):
		client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAABYZgwEAAAAAb4tPJ1oHg5LIABwh6d9LWszHsho%3DUm7PFhkU4uuBrVPq3PT3jgUZxeUsJb52zaLlosv7pXLxv9ch93')
		#sm_handles.handle_list
		d = pd.DataFrame()
		d['Final Twitter Handles'] = sm_handles.handle_list.apply(lambda x:x.lstrip('@'))
		d['Final Twitter Handles'] = d['Final Twitter Handles'].apply(lambda x:x.lstrip('@'))
		d = d.loc[d['Final Twitter Handles'].str.len() <= 15]
		d.reset_index(drop=True,inplace=True)

		def parse_twitter(usernames):
		    twitter_user_id = []
		    handle = []
		    location = []
		    twitter_joined = []
		    description = []
		    image = []
		    user_name = []
		    followers = []
		    followings = []
		    total_no_tweets = []
		    #latest_tweets = []
		    users = client.get_users(usernames=usernames,user_fields=['id','location','created_at','description','profile_image_url','name','public_metrics'])
		    for user in users.data:
		        handle.append(user.username)
		        twitter_user_id.append(user.id)
		        user_id = user.id
		        location.append(user.location)
		        twitter_joined.append(user.created_at)
		        description.append(user.description)
		        image.append(user.profile_image_url)
		        user_name.append(user.name)
		        followers.append(user.public_metrics['followers_count'])
		        followings.append(user.public_metrics['following_count'])
		        total_no_tweets.append(user.public_metrics['tweet_count'])
		#         tweets = client.get_users_tweets(user_id, max_results=100, exclude='retweets')
		#         tweets_joined = ''
		#         if tweets.data:
		#             for tweet in tweets.data:
		#                 tweets_joined = tweets_joined + ";" + tweet.text
		#         else:
		#             tweets_joined = 'NA'
		#         latest_tweets.append(tweets_joined)
		    df = pd.DataFrame()
		    df['twitter_user_id'] = twitter_user_id
		    df['handle'] = handle
		    df['location'] = location
		    df['twitter_joined'] = twitter_joined
		    df['description'] = description
		    df['image'] = image
		    df['user_name'] = user_name
		    df['followers'] = followers
		    df['followings'] = followings
		    df['total_no_tweets'] = total_no_tweets
		    #df['latest_tweets'] = latest_tweets
		    return df

		df = pd.DataFrame()
		df_len=d.index[-1]
		for i in range(math.ceil(len(d)/100)):
		    start = i*100
		    end = start+100
		    if end>df_len:
		        end=df_len+1
		    print(start,end)
		    df1 = parse_twitter(list(d['Final Twitter Handles'][start:end]))
		    df = pd.concat([df,df1], ignore_index=True)
		    sleep(20)




		def namecleaning(x):
		    x= re.sub('[^\w\s,-]', '', x)
		    x= prep.clean(x)
		    return x
		df = df.fillna("")
		#df['user_name'] = df['user_name'].astype(str)
		df['Full Name cleaned'] = df['user_name'].apply(namecleaning)
		lst = ['AAH','AAN','AANPCP','AARCF','AAS','AAT','ACLS','ACNP','ACNPAG','ACNPBC','ACNPCAG','ACNS','ACNSBC','ACP','ACRN','ADCN','ADN','AEMSN','AGACCRNPBC','AGACNP','AGACNPBC','AGCNPBC','AGCNSBC','AGNP','AGNPBC','AGNPC','AGPCNP','AGPCNPBC','AGPCNPC','AGPCPNPBC','ALNC','AMSC','ANP','ANPBC','ANPBCCCRN','ANPC','AOCN','AOCNP','AOCNS','APC','APMC','APN','APNBC','APNC','APNCNP','APNCNS','APNFNP','APNP','APRN','APRNBC','APRNC','APRNCNP','APRNCNS','APRNNP','ARNP','ARNPBC','ARNPC','ARRT','ART','AS','ASCP','ASN','ASNC','ASPO','ASSOC','ATC','BA','BAO','BC','BCBA','BCH','BCLC','BCLS','BCOP','BCPCM','BCPS','BDS','BDSC','BHS','BHSCNSG','BHYG','BM','BMS','BN','BPHARM','BPHN','BS','BSED','BSEH','BSM','BSN','BSPH','BSW','BVMS','CANP','CAPA','CARN','CATNI','CATNP','CB','CBE','CBI','CCCA','CCCN','CCCSLP','CCE','CCES','CCM','CCNS','CCP','CCRN','CCSP','CCST','CCTC','CCTN','CCTRN','CD','CDA','CDDN','CDMS','CDN','LTC','CDONA','CEN','CETN','CFCN','CFN','CFNP','CFRN','CGN','CGRN','CGT','CH','CHB','CHD','CHES','CHN','CHPN','CHRN','CHUC','CIC','CLA','CLNC','CLPNI','CLS','CLT','CM','CMA','CMAA','CMAC','CMCN','CMDSC','CMSRN','CNA','CNAA','CNDLTC','CNE','CNI','CNLCP','CNM','CNMT','CNN','CNNP','CNO','CNP','CNRN','CNS','CNSN','CO','COCN','COHNCM','COHNS','COHNSCM','COMA','CORLN','CORN','COTA','CP','CPAN','CPDN','CPFT','CPHQ','CPN','CPNA','CPNL','CPNP','CPNPAC','CPON','CPSN','CRN','CRNA','CRNFA','CRNH','CRNI','CRNL','CRNO','CRNP','CRRN','CRRNA','CRT','CRTT','CS','CSN','CSPI','CST','CTN','CTRN','CUA','CUCNS','CUNP','CURN','CVN','CWCN','CWOCN','DA','DC','DCH','DCP','DDR','DDS','DDSC','DM','DMD','DME','DMSC','DMT','DMV','DN','DNC','DNE','DNP','DNS','DNSC','DO','DON','DOS','DP','DPH','DPHIL','DPHN','DR','DRNP','DRPH','DS','DSC','DSW','DVM','DVMS','DVR','DVS','EDD','EMTB','EMTD','EMTP','EN','ENP','ENPC','ET','FAAAI','FAAFP','FAAN','FAAO','FAAOS','FAAP','FAAPM','FACAAI','FACC','FACCP','FACD','FACE','FACEP','FACG','FACOG','FACOOG','FACP','FACPM','FACR','FACS','FACSM','FAEN','FAGD','FAMA','FAOTA','FAPA','FAPHA','FCAP','FCCM','FCPS','FDS','FFA','FFARCS','FICA','FICC','FICS','FNP','FNPBC','FNPC','FRCP','FRCPATH','GNP','GNPBC','IPN','JD','LAC','LAT','LATC','LC','LCCE','LCPC','LCSW','LDN','LDO','LICSW','LLC','LM','LMSW','LNC','LNCC','LPC','LRCP','LRCS','LRN','LSN','LVN','MA','MAS','MB','MBA','MBBCH','MBBS','MBCHB','MC','MCH','MD','MDFACP','MDPHD','MDS','ME','MED','MEMERGNSG','MFT','MHE','MHN','MHS','MICN','MLS','MLT','MMS','MN','MO','MPA','MPAS','MPH','MPP','MPT','MRAD','MRCP','MRCS','MRL','MS','MSC','MSCE','MSCI','MSCR','MSD','MSEE','MSEH','MSHA','MSLS','MSN','MSNC','MSNE','MSPAS','MSPH','MSSW','MSURG','MSW','MT','MTA','MV','NCSN','NCT','ND','NEABC','NEBC','NIC','NMT','NNP','NP','NPC','NPP','NPRN','NREMTP','OCN','OCNP','OCS','OD','OGNP','OHNCS','OLMO','ONC','ONP','OTA','OTC','OTL','OTR','OTRL','PA','PAC','PACMPAC','PALS','PAS','PCCN','PCNS','PD','PH','PHARMD','PHARMG','PHD','PHN','PHRN','PMHCNS','PMHNP','PMHNPBC','PMN','PNP','PNPAC','PTA','RA','RD','RDA','RDCS','RDH','RDMS','REEGT','REPT','RHIA','RHIT','RIPRN','RMA','RN','RNC','RNA','RNBC','RNC','RNCS','RNFA','RNP','RNPC','ROUB','RPA','RPAC','RPH','RPN','RPT','RRA','RRT','RT','RTR','RVT','SANEA','SANEP','SBB','SC','SCD','SCT','SH','SHN','SLS','SM','SN','SPN','ST','SV','SVN','TM','TNCCI','TNCCP','TNP','TNS','VMD','VT','WCC','WHNP','WOCN']
		    
		fullname=[]
		main_title=[]
		for j in range(len(df)):
		    name=[]
		    title=[]
		    for i in df['Full Name cleaned'][j].split():
		        i = re.sub('[^\w\s]', '', i)
		        if i.upper() in lst:
		            title.append(i)
		        else:
		            name.append(i)
		    fullname.append(" ".join(name))
		    main_title.append(" ".join(title))
		    
		df['newfullname'] = fullname
		df['title'] = main_title
		df['description'] = df['description'].str.replace(","," ")
		df["first"] = df["newfullname"].apply(lambda x: HumanName(x).first)
		df["middle"] = df["newfullname"].apply(lambda x: HumanName(x).middle)
		df["last"] = df["newfullname"].apply(lambda x: HumanName(x).last)
		df["Title2"] = df["newfullname"].apply(lambda x: HumanName(x).suffix)
		df.rename(columns={'description':'Desc'}, inplace=True)
		df['State code'] = df['location'].str.split(',').str[-1]
		df['State code'] = df['State code'].str.strip(" ")
		df['City'] = df['location'].str.split(',').str[0]
		df['Gender'] = df['first'].apply(lambda x: g.get_gender(x))
		df['Gender'].replace(['mostly_female','female','male','mostly_male','andy'],['F','F','M','M','Unknown'], inplace=True)
		df.twitter_joined = df.twitter_joined.dt.date

		return df


def calcscore(df, q1, q2):
	df = pd.DataFrame(df)
	df.columns = ['Value']
	df = df.astype(float)
	cut1 = df.quantile(q1).round(decimals=0).values[0]
	cut2 = df.quantile(q2).round(decimals=0).values[0]
	print(cut1, cut2)
	df.loc[df['Value']>cut1, 'Score'] = 0.5
	df.loc[(df['Value']<=cut1) & (df['Value']>cut2), 'Score'] = 0.3
	df.loc[df['Value']<=cut2, 'Score'] = 0.2
	df.loc[df['Value'].isin([0,'']), 'Score'] = 0
	return df['Score']
def calcscore_noroundoff(df, q1, q2):
	df = pd.DataFrame(df)
	df.columns = ['Value']
	df = df.astype(float)
	cut1 = df.quantile(q1).values[0]
	cut2 = df.quantile(q2).values[0]
	print(cut1,cut2)
	df.loc[df['Value']>cut1, 'Score'] = 0.5
	df.loc[(df['Value']<=cut1) & (df['Value']>cut2), 'Score'] = 0.3
	df.loc[df['Value']<=cut2, 'Score'] = 0.2
	df.loc[df['Value'].isin([0,'']), 'Score'] = 0
	return df['Score']

Oncology_keywords = ['Oncology', 'oncologist', 'onco ', 'onc ', 'nuclear medicine','cancer', 'lymphoma', 'chemo', 'leukaemia', 'leukemia', 'tumor', 'tumour', 'sarcoma', 'myeloma',  'cell therapy']

cardio_keywords = ['Cardiology','cardiologist', 'cardiothoracic', 'cardiac', 'heart-doctor','cardiothorac sur','cardiothoracic surgery','hypertension specialist']

Gastro_keywords = ['Gastroenterology','gastro ','bariatric']

internal_medicine = ['internal medicine','general medicine','junior resident', 'jr resident', 'dept of medicine','department of medicine','medical specialist', 'general practitioner','physician','internist',"physio","gen practice","specialty physician","pallmed","general practice","primary care"]

Hemato_keywords = ['Hematology','hem/onc ', 'hemato', "haemato"]  #Haematologist

Hepato_keywords = ['Hepatology','hepatologist','hepatic doctor','transplant hepatology']

Nephrology_keywords = ['Nephrology','Nephrologist','nephro ','kidney doctors',"Renal Medicine","Renal","Nephrology","Colorectal Surgery","ColorectalSur"]

Neurology_keywords = ['Neurology','neurologist','neurosurgeon','neuro ','brain-doctor',"Neuro ","Neurology Consultant","Neurology","Neurosur",
                      "Epileptology","Paediatric Neurology","Psychiatry","Elderly Psychiat",
                      "Parkinson's Disease","Migraine"]

Nurse_keywords = ['Nurse ']

Pediatrics_keywords = ['Pediatrics','pediatrician','pediatrist','pedology','baby doctor',"neonatology"]

Psychiatry_keywords = ['psychiatry','psychologist','psychotherapist','psychotherapy', "adult psychiatry","psychology","psychology","forensic psychia","c&apsych", 'mental health']

pharma_keywords = ['Pharmacy','Pharma ']

healthcare_keywords = ['Healthcare Administration','healthcare','health care']

Hospital_Lab_keywords = ['Hospital Laboratory','hospital lab','medical lab']

Patho_keywords = ['Pathology', 'pathologist']

immuno_keywords = ['Immunology', 'immunologist', 'immuno ']

Dermo_keywords = ['Dermatology','derma ', 'skin doctor', 'skin specialist']

Surgery_keywords = ['Surgery', 'surgeon']

Geriatrics_keywords = ['Geriatrics', 'geriatrics', 'geriatricist', 'elderly care', 'care of elderly']

Radiology_keywords = ['Radiology', 'radiologist']

Genetics_keywords = ['Genetics', 'geneticist']

Gynaecology_keywords = ['Gynaecology','gyna ','gyno ','gynarchy','obgyn','ob-gyn']

Nephrology_keywords = ['Nephrology','nephro','kidney doctors', "renal"]

Anaesthisia_keywords = ['Anaesthisia','anaesthisia', 'anasthesia', 'pain management', 'pain control']

Neurology_keywords = ['Neurology','neurology','neurologist','neurosurgeon','brain-doctor',"neurosur", "epileptology", "parkinson","migraine"]

Dietetics_keywords = ['Dietetics','dietetics', 'dietician']

Allergy_keywords = ['Allergy','allergy', 'allergist']

Infect_keywords = ['Infectious Diseases','infection']

Endocrinology_keywords = ['Endocrinology', 'endocrinologist', 'diabetes', 'diebetics', 'diab & endo', 'diab and endo']

Urology_keywords = ['Urology', 'Urologist','urological surgery']

Intensivecare_Keywords = ['Intensive Care', 'emergency med']

ENT_keywords = ['Otolaryngologist', 'otolaryngology']

Pulmonology_keywords = ['Pulmonology','pulmonologist','pulmonary',"respiratory med","pneumology", "respiratory lung transplant","respiratory consultant"]

#specs = [Nurse_keywords,Pediatrics_keywords,Psychiatry_keywords,pharma_keywords,healthcare_keywords,Hospital_Lab_keywords,Patho_keywords,immuno_keywords,Dermo_keywords,Surgery_keywords,Geriatrics_keywords,Radiology_keywords,Genetics_keywords,Gynaecology_keywords,Nephrology_keywords,Anaesthisia_keywords,Neurology_keywords,Dietetics_keywords,Allergy_keywords,Infect_keywords,Endocrinology_keywords ,Urology_keywords,Intensivecare_Keywords,ENT_keywords,Pulmonology_keywords]
specs = [Oncology_keywords,cardio_keywords,Gastro_keywords,internal_medicine,Hemato_keywords,Hepato_keywords,Neurology_keywords,Nurse_keywords,Pediatrics_keywords,Psychiatry_keywords,pharma_keywords,healthcare_keywords,Hospital_Lab_keywords,Patho_keywords,immuno_keywords,Dermo_keywords,Surgery_keywords,Geriatrics_keywords,Radiology_keywords,Genetics_keywords,Gynaecology_keywords,Nephrology_keywords,Anaesthisia_keywords,Neurology_keywords,Dietetics_keywords,Allergy_keywords,Infect_keywords,Endocrinology_keywords,Urology_keywords,Intensivecare_Keywords,ENT_keywords,Pulmonology_keywords]


def search_tag(x, specs):
    result = []
    x = x.lower()
    for tags in specs:
        for each in tags:
            if each.lower() in x:
                result.append(tags[0])
                #print(each)
                break
        if len(result)>0:
            break
    return result

def get_twitter_scores(scoring,twittertweetsdata,twitteruserdetails,weightagefile):
	#User Details
	scoring = pd.merge(scoring, twitteruserdetails[['External ID','twitter url','twitterhandle','total_no_tweets',
	                                               'tweets_per_day','followers','following']], 
	                  left_on='External ID',right_on='External ID', how='left')
	scoring['total_no_tweets'] = scoring['total_no_tweets'].replace([' '],[0])
	scoring = scoring.fillna(0)

	#Presence
	scoring['Twitter Presence'] = np.where(scoring['twitter url']==0,0,1)

	#Tweets Details
	#days_since_tweeted
	twittertweetsdatasort = twittertweetsdata.sort_values(['External ID','days_since_tweeted']).reset_index(drop=True)
	scoring = pd.merge(scoring, twittertweetsdatasort[['External ID','days_since_tweeted']], 
	                  left_on='External ID',
	                  right_on='External ID', how='left')
	scoring = scoring.drop_duplicates(['External ID'], keep="first")

	#tweet_favourtied_times
	scoring2 = pd.merge(scoring['External ID'],twittertweetsdatasort[['External ID','tweet_favourtied_times']], on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).mean()
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')

	#relevancy
	twittertweetsdatasort = twittertweetsdata.sort_values(['relevancy'], ascending=False).reset_index(drop=True)
	scoring = pd.merge(scoring, twittertweetsdatasort[['External ID','relevancy']], 
	                  on='External ID', how='left')
	scoring = scoring.drop_duplicates(['External ID'], keep="first")
	scoring['relevancy'] = scoring['relevancy'].fillna('')
	scoring['relevancy_twitter'] = scoring['relevancy'].replace([1,0,''],['Relevant','Non Relevant','None'])
	scoring.loc[scoring['total_no_tweets']==0, 'relevancy_twitter'] = 'None'
	scoring[['days_since_tweeted','tweet_favourtied_times','relevancy','relevancy_twitter']] = scoring[['days_since_tweeted','tweet_favourtied_times','relevancy','relevancy_twitter']].fillna(0)

	#Scores
	scoring.loc[scoring['Twitter Presence']==1, 'Scores - presence_twitter'] = 1
	scoring.loc[scoring['Twitter Presence'].isin([0,'']), 'Scores - presence_twitter'] = 0
	print("no. of Tweets")
	scoring['Score - no. of Tweets_twitter'] = calcscore(scoring[scoring['Twitter Presence']==1]['total_no_tweets'], 0.7, 0.4)
	print("Tweets per day")
	scoring['Scores - tweets_per_day'] = calcscore_noroundoff(scoring[scoring['Twitter Presence']==1]['tweets_per_day'], 0.7, 0.4)
	print("No. of followers")
	scoring['Scores - no. of Followers_twitter'] = calcscore(scoring[scoring['Twitter Presence']==1]['followers'], 0.7, 0.4)
	print("No. of followings")
	scoring['Scores - no. of Following_twitter'] = calcscore(scoring[scoring['Twitter Presence']==1]['following'], 0.7, 0.4)
	print("Favourited times")
	#scoring['Scores - Favourtied times per Tweet_twitter'] = calcscore(scoring[scoring['Twitter Presence']==1]['tweet_favourtied_times'], 0.7, 0.4)
	scoring['Scores - Favourtied times per Tweet_twitter'] = calcscore(scoring[scoring['total_no_tweets']>0]['tweet_favourtied_times'], 0.7, 0.4)
	scoring['days_since_tweeted'] = scoring['days_since_tweeted'].replace([''],[0])
	scoring.loc[scoring['days_since_tweeted']>60,  'Scores - Recency of post_twitter'] = 0.2
	scoring.loc[(scoring['days_since_tweeted']<=60) &
	           (scoring['days_since_tweeted']>30),  'Scores - Recency of post_twitter'] = 0.3
	scoring.loc[(scoring['days_since_tweeted']<=30) & 
	           (scoring['days_since_tweeted']>0),  'Scores - Recency of post_twitter'] = 0.5
	scoring.loc[scoring['days_since_tweeted']==0,  'Scores - Recency of post_twitter'] = 0
	scoring.loc[scoring['relevancy_twitter']=='Relevant', 'Scores - Relevancy_twitter'] = 0.5
	scoring.loc[(scoring['relevancy_twitter']=='Non Relevant') & (scoring['total_no_tweets']>0) , 'Scores - Relevancy_twitter'] = 0.3
	scoring.loc[(scoring['relevancy_twitter']=='None') | (scoring['total_no_tweets']==0), 'Scores - Relevancy_twitter'] = 0

	#weightages
	weightages = pd.read_excel(weightagefile, sheet_name='Twitter')
	twitterweightage = weightages.loc[weightages['Parameters'] == 'Total Score_twitter', 'Weightage'].iloc[0]
	#print(weightages.loc[weightages['Parameters'] == 'Scores - presence_twitter', 'Weightage'].iloc[0])
	#final score

	scoring = scoring.fillna(0)    
	scoring['Total Score_twitter'] = scoring.apply(lambda row: row['Scores - presence_twitter']*weightages.loc[weightages['Parameters'] == 'Scores - presence_twitter', 'Weightage'].iloc[0] + 
	                                  row['Score - no. of Tweets_twitter']*weightages.loc[weightages['Parameters'] == 'Score - no. of Tweets_twitter', 'Weightage'].iloc[0] +
	                                  row['Scores - tweets_per_day']*weightages.loc[weightages['Parameters'] == 'Scores - tweets_per_day', 'Weightage'].iloc[0] +
	                                  row['Scores - no. of Followers_twitter']*weightages.loc[weightages['Parameters'] == 'Scores - no. of Followers_twitter', 'Weightage'].iloc[0] +
	                                  row['Scores - no. of Following_twitter']*weightages.loc[weightages['Parameters'] == 'Scores - no. of Following_twitter', 'Weightage'].iloc[0] +
	                                  row['Scores - Favourtied times per Tweet_twitter']*weightages.loc[weightages['Parameters'] == 'Scores - Favourtied times per Tweet_twitter', 'Weightage'].iloc[0] +
	                                  row['Scores - Recency of post_twitter']*weightages.loc[weightages['Parameters'] == 'Scores - Recency of post_twitter', 'Weightage'].iloc[0] +
	                                  row['Scores - Relevancy_twitter']*weightages.loc[weightages['Parameters'] == 'Scores - Relevancy_twitter', 'Weightage'].iloc[0], axis=1)
	#print(scoring[scoring['Scores - presence_twitter']==1])
	return scoring,twitterweightage

def get_linkedin_scores(scoring, lnprofile, lnactivity, lnposts,weightagefile):
	lnposts['post_data'] = lnposts['post_data'].fillna('')
	lnposts['Relevancy_linkedin'] = lnposts['post_data'].apply(lambda x: search_tag(x, specs)).apply(lambda x: 1 if len(x)>0 else 0)
	scoring = pd.merge(scoring, lnprofile[['External ID','Linkedin_Url','LinkedIn Name','Followers']],
	                  on='External ID', how='left')
	scoring[['Linkedin_Url','LinkedIn Name']] = scoring[['Linkedin_Url','LinkedIn Name']].fillna('')
	scoring[['Followers']] = scoring[['Followers']].fillna(0)

	scoring2 = pd.merge(scoring['External ID'],lnposts[['External ID','Name']], on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).count()
	favfinal = favfinal.rename(columns={'Name':'No. of Posts'})
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')

	#scoring2 = pd.merge(scoring['External ID'],lnposts[['External ID','Relevancy_linkedin']], on='External ID', how='left')
	#groupbylist = ['External ID']
	#favfinal = scoring2.groupby(groupbylist).count()
	#scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	#scoring = scoring.rename(columns={'Count of Relevancy_linkedin':'Relevancy_linkedin'})
	#scoring.loc[scoring['Relevancy_linkedin']>=1,'Relevant_linkedin'] = 'Relevant'
	#scoring.loc[scoring['Relevancy_linkedin']==0,'Relevant_linkedin'] = 'Non Relevant'

	lnpostssort = lnposts.sort_values(['Relevancy_linkedin'], ascending=False).reset_index(drop=True)
	scoring = pd.merge(scoring, lnpostssort[['External ID','Relevancy_linkedin']], 
	                  on='External ID', how='left')
	scoring = scoring.drop_duplicates(['External ID'], keep="first")
	scoring['Relevancy_linkedin'] = scoring['Relevancy_linkedin'].fillna('')
	scoring['Relevancy_linkedin'] = scoring['Relevancy_linkedin'].replace([1,0,''],['Relevant','Non Relevant','None'])


	lnpostdatasort = lnposts.sort_values(['Recent Date'], ascending=False).reset_index(drop=True)
	scoring = pd.merge(scoring, lnposts[['External ID','Recent Date','Total Number of Days']], on='External ID', how='left')
	scoring = scoring.rename(columns={'Recent Date':'Date of the recent post', 'Total Number of Days':'No. of days from recent post'})
	scoring = scoring.drop_duplicates(['External ID'], keep="first")
	scoring['Date of the recent post'] = scoring['Date of the recent post'].astype(str)
	scoring['Date of the recent post'] = scoring['Date of the recent post'].replace(['NaT'],[''])

	lnactivity['reactions_count'] = lnactivity['reactions_count'].apply(lambda x:str(x).replace(',',''))
	lnactivity['no_of_comments'] = lnactivity['no_of_comments'].apply(lambda x:str(x).replace(',',''))
	lnactivity['no_of_comments'] = lnactivity['no_of_comments'].apply(lambda x:str(x).replace(' comments',''))
	lnactivity['no_of_comments'] = lnactivity['no_of_comments'].apply(lambda x:str(x).replace(' comment',''))
	lnactivity['no_of_comments'] = lnactivity['no_of_comments'].fillna(0)
	lnactivity['reactions_count'] = lnactivity['reactions_count'].fillna(0)
	lnactivity['no_of_comments'] = lnactivity['no_of_comments'].replace(['NA','nan'],[0,0])
	lnactivity['reactions_count'] = lnactivity['reactions_count'].replace(['NA','nan'],[0,0])
	lnactivity['no_of_comments'] = lnactivity['no_of_comments'].astype(int)
	lnactivity['reactions_count'] = lnactivity['reactions_count'].astype(int)


	scoring2 = pd.merge(scoring['External ID'],lnposts[['External ID','post_likes']], on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).mean()
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	scoring = scoring.rename(columns={'post_likes':'Average No. Reactions per post'})

	scoring2 = pd.merge(scoring['External ID'],lnposts[['External ID','post_comments']], on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).mean()
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	scoring = scoring.rename(columns={'post_comments':'Average Comments per post'})

	lnactdatasort = lnactivity.sort_values(['Recent Post Date'], ascending=False).reset_index(drop=True)
	scoring = pd.merge(scoring, lnactdatasort[['External ID','Recent Post Date','Total Days']], on='External ID', how='left')
	scoring = scoring.rename(columns={'Recent Post Date':'Last comment date','Total Days':'Days since last comment posted'})
	scoring = scoring.drop_duplicates(['External ID'], keep="first")

	scoring = scoring.fillna('')
	scoring[['No. of days from recent post','Last comment date']] = scoring[['No. of days from recent post','Last comment date']].astype(str)
	scoring[['No. of days from recent post','Last comment date']] = scoring[['No. of days from recent post','Last comment date']].replace(['NaT'],[''])
	#Score - presence
	scoring.loc[scoring['Linkedin_Url']=="", 'Scores - presence_linkedin'] = 0
	scoring.loc[scoring['Linkedin_Url']!="", 'Scores - presence_linkedin'] = 1
	#Score - no. of posts
	print("Number of posts")
	scoring['Scoring- no. of Posts_linkedin'] = calcscore(scoring[scoring['Scores - presence_linkedin']==1]['No. of Posts'], 0.9, 0.8)
	#Scores - no. of Followers
	print("no. of Followers")
	scoring['Followers'] = scoring['Followers'].replace([''],[0])
	scoring['Scores - no. of Followers_linkedin'] = calcscore(scoring[scoring['Scores - presence_linkedin']==1]['Followers'], 0.7, 0.4)
	#Average reactions per post
	print("Average Reactions per post")
	scoring['Average No. Reactions per post'] = scoring['Average No. Reactions per post'].replace([''],[0])
	#scoring['Average No. Reactions per post_linkedin'] = calcscore(scoring[scoring['Scores - presence_linkedin']==1]['Average No. Reactions per post'], 0.9, 0.8)
	scoring['Average No. Reactions per post_linkedin'] = calcscore(scoring[scoring['No. of Posts']>0]['Average No. Reactions per post'], 0.7, 0.4)
	#Average comments per post
	print("Average comments per post")
	scoring['Average Comments per post'] = scoring['Average Comments per post'].replace([''],[0])
	#scoring['Average comments per post_linkedin'] = calcscore(scoring[scoring['Scores - presence_linkedin']==1]['Average Comments per post'], 0.9, 0.8)
	scoring['Average comments per post_linkedin'] = calcscore(scoring[scoring['No. of Posts']>0]['Average Comments per post'], 0.9, 0.8)
	#Scores - Recency of comments
	scoring['Days since last comment posted'] = scoring['Days since last comment posted'].replace([''],[0])
	scoring.loc[(scoring['Days since last comment posted']>=365) & (scoring['Scores - presence_linkedin']==1),'Scores - Recency of comments_linkedin'] = 0.2
	scoring.loc[(scoring['Days since last comment posted']>30) & (scoring['Days since last comment posted']<365) & (scoring['Scores - presence_linkedin']==1),'Scores - Recency of comments_linkedin'] = 0.3
	scoring.loc[(scoring['Days since last comment posted']<=30) & (scoring['Days since last comment posted']>0) & (scoring['Scores - presence_linkedin']==1),'Scores - Recency of comments_linkedin'] = 0.5
	scoring.loc[(scoring['Days since last comment posted']==0) | (scoring['Scores - presence_linkedin']==0),'Scores - Recency of comments_linkedin'] = 0
	scoring['No. of days from recent post'] = scoring['No. of days from recent post'].apply(lambda x: x.split(" ")[0])
	scoring['No. of days from recent post'] = scoring['No. of days from recent post'].replace([''],[0])
	scoring['No. of days from recent post'] = scoring['No. of days from recent post'].astype(float)
	#Scores - Recency of post
	scoring['No. of days from recent post'] = scoring['No. of days from recent post'].replace([''],[0])
	scoring.loc[(scoring['No. of days from recent post']>60) & (scoring['Scores - presence_linkedin']==1),'Scores - Recency of post_linkedin'] = 0.2
	scoring.loc[(scoring['No. of days from recent post']>30) & (scoring['No. of days from recent post']<=60) & (scoring['Scores - presence_linkedin']==1),'Scores - Recency of post_linkedin'] = 0.3
	scoring.loc[(scoring['No. of days from recent post']<=30) & (scoring['No. of days from recent post']>0) & (scoring['Scores - presence_linkedin']==1),'Scores - Recency of post_linkedin'] = 0.5
	scoring.loc[(scoring['No. of days from recent post']==0) | (scoring['Scores - presence_linkedin']==0),'Scores - Recency of post_linkedin'] = 0
	#Scores - Relevancy
	scoring.loc[scoring['Relevancy_linkedin']=='Relevant', 'Scores - Relevancy_linkedin'] = 0.5
	scoring.loc[(scoring['Relevancy_linkedin']=='Non Relevant') & (scoring['No. of Posts']>0), 'Scores - Relevancy_linkedin'] = 0.3
	scoring.loc[(scoring['Relevancy_linkedin']=='None') | (scoring['No. of Posts']==0), 'Scores - Relevancy_linkedin'] = 0
	weightages = pd.read_excel(weightagefile, sheet_name='LinkedIn')
	linkedinweightage = weightages.loc[weightages['Parameters'] == 'Total Score_linkedin', 'Weightage'].iloc[0]
	scoring = scoring.fillna(0)
	scoring['Total Score_linkedin'] = scoring.apply(lambda row: row['Scores - presence_linkedin']*weightages.loc[weightages['Parameters'] == 'Scores - presence_linkedin', 'Weightage'].iloc[0] + 
	                                      row['Scoring- no. of Posts_linkedin']*weightages.loc[weightages['Parameters'] == 'Scoring- no. of Posts_linkedin', 'Weightage'].iloc[0] +
	                                      row['Average comments per post_linkedin']*weightages.loc[weightages['Parameters'] == 'Average comments per post_linkedin', 'Weightage'].iloc[0] +
	                                      row['Scores - no. of Followers_linkedin']*weightages.loc[weightages['Parameters'] == 'Scores - no. of Followers_linkedin', 'Weightage'].iloc[0] +
	                                      row['Average No. Reactions per post_linkedin']*weightages.loc[weightages['Parameters'] == 'Scores - Average reactions per post_linkedin', 'Weightage'].iloc[0] +
	                                      row['Scores - Recency of comments_linkedin']*weightages.loc[weightages['Parameters'] == 'Scores - Recency of comments_linkedin', 'Weightage'].iloc[0] +
	                                      row['Scores - Recency of post_linkedin']*weightages.loc[weightages['Parameters'] == 'Scores - Recency of post_linkedin', 'Weightage'].iloc[0] +
	                                      row['Scores - Relevancy_linkedin']*weightages.loc[weightages['Parameters'] == 'Scores - Relevancy_linkedin', 'Weightage'].iloc[0], axis=1)
	return scoring,linkedinweightage

def get_instagram_scores(scoring,insta,weightagefile):
	#User Details
	#st.dataframe(scoring)
	scoring = pd.merge(scoring, insta[['External ID','Instagram ID',
	      'Instagram_ Posts', 'Instagram_Followers', 'Instagram_Following']],
	                  on='External ID', how='left')
	scoring.rename(columns={'Instagram_ Posts':'Instagram Posts', 'Instagram_Followers':'Instagram Followers', 'Instagram_Following':'Instagram Following'}, inplace=True)

	scoring['Instagram ID'] = scoring['Instagram ID'].fillna('')

	scoring[['Instagram Posts', 'Instagram Followers', 'Instagram Following']] = scoring[['Instagram Posts', 'Instagram Followers', 'Instagram Following']].fillna(0)
	#scoring[scoring['Instagram ID']!='']
	#Score - presence
	scoring.loc[scoring['Instagram ID']!="",'Score -Instagram Presence']=1
	scoring.loc[scoring['Instagram ID']=="",'Score -Instagram Presence']=0
	#Score - no. of posts
	print("Number of posts")
	scoring['Scores - Instagram Posts'] = calcscore(scoring[scoring['Score -Instagram Presence']==1]['Instagram Posts'],0.7,0.4)
	#Scores - no. of Followers
	print("no. of Followers")
	scoring['Scores - no. of Instagram Followers'] = calcscore(scoring[scoring['Score -Instagram Presence']==1]['Instagram Followers'],0.7,0.4)
	#Scores - no. of Following
	print("no. of Following")
	scoring['Scores - no. of Instagram Following'] = calcscore(scoring[scoring['Score -Instagram Presence']==1]['Instagram Following'],0.7,0.4)
	#weightages
	weightages = pd.read_excel(weightagefile, sheet_name='Instagram')
	instagramweightage = weightages.loc[weightages['Parameters'] == 'Total Score_instagram', 'Weightage'].iloc[0]
	scoring = scoring.fillna(0)
	#total score calculation
	#final score
	scoring['Total Score_instagram'] = scoring.apply(lambda row: row['Score -Instagram Presence']*weightages.loc[weightages['Parameters'] == 'Score -Instagram Presence', 'Weightage'].iloc[0] + 
	                                      row['Scores - Instagram Posts']*weightages.loc[weightages['Parameters'] == 'Scores - Instagram Posts', 'Weightage'].iloc[0] +
	                                      row['Scores - no. of Instagram Followers']*weightages.loc[weightages['Parameters'] == 'Scores - no. of Instagram Followers', 'Weightage'].iloc[0] +
	                                      row['Scores - no. of Instagram Following']*weightages.loc[weightages['Parameters'] == 'Scores - no. of Instagram Following', 'Weightage'].iloc[0], axis=1)
	scoring['Instagram Presence'] = scoring['Score -Instagram Presence']
	#print(scoring[scoring['Score - Instagram Presence']==1])
	return scoring,instagramweightage

def get_facebook_scores(scoring,fb,weightagefile):
	# user details
	scoring = pd.merge(scoring, fb[['Facebook URL','External ID','No_of_posts_fb','days since last posted_fb', 'Friends2_fb']],
	                      on='External ID', how='left')
	scoring['Facebook URL'] = scoring['Facebook URL'].fillna('')
	scoring[['No_of_posts_fb', 'days since last posted_fb',"Friends2_fb"]] =scoring[['No_of_posts_fb', 'days since last posted_fb',"Friends2_fb"]].fillna(0)
	#scoring[scoring['Instagram ID']!='']
	#Score - presence
	scoring.loc[scoring['Facebook URL']!="",'Score -Facebook Presence']=1
	scoring.loc[scoring['Facebook URL']=="",'Score -Facebook Presence']=0
	#Score - no. of posts
	print("Number of posts")
	scoring['Scores -  Facebook Posts'] = calcscore(scoring[scoring['Score -Facebook Presence']==1]['No_of_posts_fb'],0.7,0.4)
	# score-friends
	print('number of friends')
	scoring['Scores -Friends_Facebook'] = calcscore(scoring[scoring['Score -Facebook Presence']==1]['Friends2_fb'],0.8,0.6)
	#score Recency
	print('recency of post')
	scoring.loc[scoring['days since last posted_fb']>60,  'Scores - Recency of post_Facebook'] = 0.2
	scoring.loc[(scoring['days since last posted_fb']<=60) &
	               (scoring['days since last posted_fb']>30),  'Scores - Recency of post_Facebook'] = 0.3
	scoring.loc[(scoring['days since last posted_fb']<=30) & 
	               (scoring['days since last posted_fb']>0),  'Scores - Recency of post_Facebook'] = 0.5
	scoring.loc[scoring['days since last posted_fb']==0,  'Scores - Recency of post_Facebook'] = 0
	#Score Relevancy
	#print('relvancy of post')
	#scoring.loc[scoring['Relevancy_fb']=='Relevant', 'Scores - Relevancy_Facebook'] = 0.5
	#scoring.loc[(scoring['No_of_posts_fb']>0) & (scoring['Relevancy_fb']!='Relevant'), 'Scores - Relevancy_Facebook'] = 0.3
	#scoring.loc[(scoring['No_of_posts_fb']==0) & (scoring['Relevancy_fb']!='Relevant'), 'Scores - Relevancy_Facebook'] = 0

	#weightages
	weightages = pd.read_excel(weightagefile, sheet_name='Facebook')
	Facebookweightage = weightages.loc[weightages['Parameters'] == 'Total Score_Facebook', 'Weightage'].iloc[0]
	scoring = scoring.fillna(0)
	#total score calculation
	#final score
	scoring['Total Score_Facebook'] = scoring.apply(lambda row: row['Score -Facebook Presence']*weightages.loc[weightages['Parameters'] == 'Score -Facebook Presence', 'Weightage'].iloc[0] + 
	                                          row['Scores -  Facebook Posts']*weightages.loc[weightages['Parameters'] == 'Scores -  Facebook Posts', 'Weightage'].iloc[0] +
	                                          row['Scores -Friends_Facebook']*weightages.loc[weightages['Parameters'] == 'Scores -Friends', 'Weightage'].iloc[0] +
	                                          row['Scores - Recency of post_Facebook']*weightages.loc[weightages['Parameters'] == 'Scores - Recency of post_Facebook', 'Weightage'].iloc[0], axis=1)
	return scoring,Facebookweightage

def get_Conference_scores(scoring,conferencemaster,weightagefile):
	#user details
	#scoring = pd.merge(scoring,conferencemaster[['External ID','Conference/Digital Event Name',
	#                                                   'Presentation Title','Conference Date',
	#                                                   'Presentor Role','Conference Type']], on='External ID', how='left')
	#scoring = scoring.fillna(0)
	#number of presentations calculations
	scoring2 = pd.merge(scoring['External ID'],conferencemaster[['External ID','Presentation Title']], 
	                   on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).count()
	favfinal.rename(columns={"Presentation Title":"Number_of_presentations_Conference"},inplace=True)
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	# type rank
	conferencemaster.loc[conferencemaster['Conference Type']=='Offline','Type_Rank']=1
	conferencemaster.loc[conferencemaster['Conference Type']=='Online','Type_Rank']=2
	#scoring.loc[scoring['Conference Type']=='','Type_Rank']=0
	# role rank
	conferencemaster.loc[conferencemaster['Presentor Role']=='Presentor','Role_Rank']=2
	conferencemaster.loc[conferencemaster['Presentor Role']=='Co-Presentor','Role_Rank']=1
	conferencemaster=conferencemaster.fillna(0)
	scoring = scoring.fillna(0)

	# date
	confdetsort = conferencemaster.sort_values(['Conference Year'], ascending=False).reset_index(drop=True)
	#scoring = scoring.rename(columns={'Conference Date':'Last_Conference_Date'})
	confdetsort = confdetsort.drop_duplicates(['External ID'], keep="first")
	#scoring = scoring.fillna(0)
	#scoring1 = scoring[scoring['Last_Conference_Date']!=0]
	scoring = pd.merge(scoring, confdetsort[['External ID','Conference Year',]], on='External ID', how='left')
	#scoring = scoring.drop(columns=['Last_Conference_Date_x'], axis=1)
	scoring = scoring.rename(columns={'Conference Year':'Last_Conference_Date'})
	scoring.loc[scoring['Last_Conference_Date']==2022,'Participation in last 1 year (Y=2. N=1)']=2
	scoring.loc[scoring['Last_Conference_Date']==2023,'Participation in last 1 year (Y=2. N=1)']=2
	scoring.loc[(scoring['Last_Conference_Date']<2022) & (scoring['Last_Conference_Date']>0), 'Participation in last 1 year (Y=2. N=1)']=1

	#scores of  presentations
	scoring.loc[scoring['Number_of_presentations_Conference']>0,'Conference Presence Score']=1
	scoring.loc[scoring['Number_of_presentations_Conference']==0,'Conference Presence Score']=0
	scoring['Scoring - No. of Presentations'] = calcscore(scoring[scoring['Number_of_presentations_Conference']>0]['Number_of_presentations_Conference'], 0.9, 0.6)
	scoring = scoring.fillna(0)
	#recency score of conference
	scoring.loc[scoring['Participation in last 1 year (Y=2. N=1)']==2,'Recency Score Conference']=0.5
	scoring.loc[scoring['Participation in last 1 year (Y=2. N=1)']==1,'Recency Score Conference']=0.3
	scoring.loc[scoring['Participation in last 1 year (Y=2. N=1)']==0,'Recency Score Conference']=0
	# score of type Rank
	confdetsort = conferencemaster.sort_values(['Type_Rank'], ascending=False).reset_index(drop=True)
	scoring = pd.merge(scoring, confdetsort[['External ID','Type_Rank']], on='External ID', how='left')
	scoring = scoring.rename(columns={'Type_Rank':'Confernce Type'})
	scoring = scoring.drop_duplicates(['External ID'], keep="first")
	scoring = scoring.fillna(0)
	scoring.loc[scoring['Confernce Type']==2, 'Confernce Type']=0.5
	scoring.loc[scoring['Confernce Type']==1, 'Confernce Type']=0.3

	#scoring of Role Rank
	confdetsort = conferencemaster.sort_values(['Role_Rank'], ascending=False).reset_index(drop=True)
	scoring = pd.merge(scoring, confdetsort[['External ID','Role_Rank']], on='External ID', how='left')
	scoring = scoring.rename(columns={'Role_Rank':'Score - Role_Conference'})
	scoring = scoring.drop_duplicates(['External ID'], keep="first")
	scoring = scoring.fillna(0)
	scoring.loc[scoring['Score - Role_Conference']==2, 'Score - Role_Conference']=0.5
	scoring.loc[scoring['Score - Role_Conference']==1, 'Score - Role_Conference']=0.3
	#weightages
	weightages = pd.read_excel(weightagefile, sheet_name='Conferences')
	conferenceweightage = weightages.loc[weightages['Parameters'] == 'Total Score_conferences', 'Weightage'].iloc[0]
	scoring['Total Score_conferences'] = scoring.apply(lambda row: row['Conference Presence Score']*weightages.loc[weightages['Parameters'] == 'Conference Presence Score', 'Weightage'].iloc[0] + 
	                                  row['Scoring - No. of Presentations']*weightages.loc[weightages['Parameters'] == 'Scoring - No. of Presentations', 'Weightage'].iloc[0] +
	                                  row['Recency Score Conference']*weightages.loc[weightages['Parameters'] == 'Recency Score Conference', 'Weightage'].iloc[0] +
	                                  row['Confernce Type']*weightages.loc[weightages['Parameters'] == 'Confernce Type', 'Weightage'].iloc[0] +
	                                  row['Score - Role_Conference']*weightages.loc[weightages['Parameters'] == 'Score - Role_Conference', 'Weightage'].iloc[0], axis=1)
	#     scoring.drop(columns=['Last_Conference_Date'], axis=1, inplace=True)
	#     scoring.loc[scoring['Scoring - No. of Presentations']==0.5, 'number of presentation category_conferences'] = 'High'
	#     scoring.loc[scoring['Scoring - No. of Presentations']==0.3, 'number of presentation category_conferences'] = 'Medium'
	#     scoring.loc[scoring['Scoring - No. of Presentations']==0.2, 'number of presentation category_conferences'] = 'Low'
	#     scoring.loc[scoring['Scoring - No. of Presentations']==0, 'number of presentation category_conferences'] = 'Nil'
	return scoring,conferenceweightage

def get_Digital_scores(scoring,digitalmaster,weightagefile):
	#user details
	#     scoring = pd.merge(scoring, digitalmaster[['External ID','Digital Event Name','Presentation Title',
	#                                                'Digital Event Year','Presentor Role']], 
	#                        left_on='External ID', right_on='External ID', how='left' )

	#number of presentations calculations
	scoring2 = pd.merge(scoring['External ID'],digitalmaster[['External ID','Digital Event Name']], on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).count()
	favfinal.rename(columns={"Digital Event Name":"Number_of_presentations_Digital"},inplace=True)
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')

	#role Rank
	digitalmaster.loc[digitalmaster['Presentor Role']=='Presentor','Role_Rank']=2
	digitalmaster.loc[digitalmaster['Presentor Role']=='Co-Presentor','Role_Rank']=1
	digitalmaster = digitalmaster.fillna(0)
	scoring = scoring.fillna(0)


	# score of digital Presence
	scoring.loc[scoring['Number_of_presentations_Digital']>0, 'Digital Events Participation'] = 1
	scoring.loc[scoring['Number_of_presentations_Digital']<=0, 'Digital Events Participation'] = 0


	#year
	digitaldetailssort = digitalmaster.sort_values(['Digital Event Year'], ascending=False).reset_index(drop=True)
	#scoring = scoring.rename(columns={'Year':'Participation in last 1 year (Y=2. N=1)'})
	digitaldetailssort = digitaldetailssort.drop_duplicates(['External ID'], keep="first")
	scoring = pd.merge(scoring, digitaldetailssort[['External ID','Digital Event Year']], on='External ID', how='left')
	scoring = scoring.fillna(0)
	#scoring = scoring.drop(columns=['Digital Event Year_x'], axis=1)
	#scoring = scoring.rename(columns={'Digital Event Year_y':'Digital Event Year'})
	scoring.loc[scoring['Digital Event Year']==2022, 'Participation in last 1 year (Y=2. N=1)_digital']=2
	scoring.loc[(scoring['Digital Event Year']<2022) & (scoring['Digital Event Year']>0), 'Participation in last 1 year (Y=2. N=1)_digital']=1  
	#score of events
	scoring['Score - No. of events'] = calcscore(scoring[scoring['Digital Events Participation']==1]['Number_of_presentations_Digital'],0.9,0.6)
	scoring = scoring.fillna(0)

	#Score of Digital recency
	scoring['Recency Score_Digital']=scoring['Participation in last 1 year (Y=2. N=1)_digital'].replace([2,1,0],[0.5,0.3,0])

	#score of role rank
	digitaldetailssort = digitalmaster.sort_values(['Role_Rank'], ascending=False).reset_index(drop=True)
	scoring = pd.merge(scoring, digitaldetailssort[['External ID','Role_Rank']], on='External ID', how='left')
	scoring = scoring.rename(columns={'Role_Rank':'Score - Role_Digital'})
	scoring = scoring.drop_duplicates(['External ID'], keep="first")
	scoring = scoring.fillna(0)
	scoring.loc[scoring['Score - Role_Digital']==2, 'Score - Role_Digital']=0.5
	scoring.loc[scoring['Score - Role_Digital']==1, 'Score - Role_Digital']=0.3

	#weightages
	weightages = pd.read_excel(weightagefile, sheet_name='Digital Events')
	digitaleventweightage = weightages.loc[weightages['Parameters'] == 'Total Score_digital', 'Weightage'].iloc[0]
	scoring['Total Score_digital'] = scoring.apply(lambda row: row['Digital Events Participation']*weightages.loc[weightages['Parameters'] == 'Digital Events Participation', 'Weightage'].iloc[0] + 
	                                  row['Score - No. of events']*weightages.loc[weightages['Parameters'] == 'Score - No. of events', 'Weightage'].iloc[0] +
	                                  row['Recency Score_Digital']*weightages.loc[weightages['Parameters'] == 'Recency Score_Digital', 'Weightage'].iloc[0] +
	                                  row['Score - Role_Digital']*weightages.loc[weightages['Parameters'] == 'Score - Role', 'Weightage'].iloc[0], axis=1)
	#     scoring.loc[scoring['Scoring - No. of Presentations']==0.5, 'number of presentation category_digital'] = 'High'
	#     scoring.loc[scoring['Scoring - No. of Presentations']==0.3, 'number of presentation category_digital'] = 'Medium'
	#     scoring.loc[scoring['Scoring - No. of Presentations']==0.2, 'number of presentation category_digital'] = 'Low'
	#     scoring.loc[scoring['Scoring - No. of Presentations']==0, 'number of presentation category_digital'] = 'Nil'
	return scoring,digitaleventweightage

def get_youtube_scores(scoring,youtube,weightagefile):
	#user details
	#you tube account id
	scoring2 = pd.merge(scoring['External ID'],youtube[['External ID','video_name']], on='External ID', how='left')
	groupbylist = ['External ID']
	scoring2 = scoring2.fillna('')
	favfinal = scoring2.groupby(groupbylist).agg({'video_name': ';'.join})
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	scoring = scoring.rename(columns={'video_name':'Youtube account ID'})
	#Relavancy Calculation
	Relevancy=[]
	try:
	   for i in youtube['video_name']:
	       post_text = str(i)
	       specialties_tagged = search_tag(post_text, specs)
	       if len(specialties_tagged)>0:
	           Relevancy.append("Relevant")
	       else:
	           Relevancy.append("Non Relevant")
	except:
	   Relevancy.append("None")
	youtube['Relevancy_Youtube']=Relevancy
	#you tubr links 
	scoring2 = pd.merge(scoring['External ID'],youtube[['External ID','page_url']], on='External ID', how='left')
	groupbylist = ['External ID']
	scoring2 = scoring2.fillna('')
	favfinal = scoring2.groupby(groupbylist).agg({'page_url': ';'.join})
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	scoring = scoring.rename(columns={'page_url':'Youtube Links'})
	#cal you tube presence
	scoring.loc[scoring['Youtube account ID']!='','Scores - presence_youtube'] = 1
	scoring.loc[scoring['Youtube account ID']=='','Scores - presence_youtube'] = 0
	#No. of Youtube Videos
	scoring2 = pd.merge(scoring['External ID'], youtube[['External ID','page_url']], on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).count()
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	scoring = scoring.rename(columns={'page_url':'No. of Youtube Videos'})
	#total likes
	scoring2 = pd.merge(scoring['External ID'], youtube[['External ID','likes']], on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).sum()
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	scoring = scoring.rename(columns={'likes':'Total Likes'})
	#total views
	scoring2 = pd.merge(scoring['External ID'], youtube[['External ID','views']], on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).sum()
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	scoring = scoring.rename(columns={'views':'Total Views'})
	#total comments
	scoring2 = pd.merge(scoring['External ID'], youtube[['External ID','no_of_comments']], on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).sum()
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	scoring = scoring.rename(columns={'no_of_comments':'Total Comments'})
	#total subcription
	scoring2 = pd.merge(scoring['External ID'], youtube[['External ID','subscriber count ']], on='External ID', how='left')
	groupbylist = ['External ID']
	favfinal = scoring2.groupby(groupbylist).sum()
	scoring = pd.merge(scoring, favfinal, on='External ID', how='left')
	scoring = scoring.rename(columns={'subscriber count ':'Total Subscriptions'})
	#score relavancy
	youtubesort = youtube.sort_values(['Relevancy_Youtube'], ascending=False).reset_index(drop=True)
	scoring = pd.merge(scoring, youtubesort[['External ID','Relevancy_Youtube']], on='External ID', how='left')
	scoring = scoring.rename(columns={'Relevancy_Youtube':'Presence of Relevant Videos'})
	scoring['Presence of Relevant Videos'] = scoring['Presence of Relevant Videos'].replace([1.0,0.0],['Relevant','Non Relevant'])
	scoring = scoring.drop_duplicates(['External ID'], keep="first")
	print('Scoring- no. of videos')
	scoring['Scoring- no. of videos_youtube'] = calcscore(scoring[scoring['Scores - presence_youtube']==1]['No. of Youtube Videos'], 0.9, 0.7)
	print('Scores - no. of views')
	scoring['Scores - no. of views_youtube'] = calcscore(scoring[scoring['Scores - presence_youtube']==1]['Total Views'], 0.7, 0.4)
	print('Scores - no. of likes')
	scoring['Scores - no. of likes_youtube'] = calcscore(scoring[scoring['Scores - presence_youtube']==1]['Total Likes'], 0.7, 0.4)
	print('Scores  - Total Subscriptions')
	scoring['Scores  - Total Subscriptions_youtube'] = calcscore(scoring[scoring['Scores - presence_youtube']==1]['Total Subscriptions'], 0.7, 0.4)
	#dates sorting
	youtubesort = youtube.sort_values(['Days since last video posted'], ascending=True).reset_index(drop=True)
	scoring = pd.merge(scoring, youtubesort[['External ID','Days since last video posted']], on='External ID', how='left')
	#scoring = scoring.rename(columns={'date_posted':'Recent Video posted date'})
	scoring = scoring.drop_duplicates(['External ID'], keep="first")
	#Scores - Recency
	scoring.loc[(scoring['Days since last video posted']>=365) & (scoring['Scores - presence_youtube']==1),'Scores - Recency_youtube'] = 0.2
	scoring.loc[(scoring['Days since last video posted']>30) & (scoring['Days since last video posted']<365) & (scoring['Scores - presence_youtube']==1),'Scores - Recency_youtube'] = 0.3
	scoring.loc[(scoring['Days since last video posted']<=30) & (scoring['Days since last video posted']>0) & (scoring['Scores - presence_youtube']==1),'Scores - Recency_youtube'] = 0.5
	scoring.loc[(scoring['Days since last video posted']==0) | (scoring['Scores - presence_youtube']==0),'Scores - Recency_youtube'] = 0
	#Scores - Relevancy
	scoring.loc[scoring['Presence of Relevant Videos']=='Relevant', 'Scores - Relevancy_youtube'] = 0.5
	scoring.loc[(scoring['Scores - presence_youtube']==1) & (scoring['Presence of Relevant Videos']!='Relevant'), 'Scores - Relevancy_youtube'] = 0.3
	scoring.loc[(scoring['Scores - presence_youtube']==0) & (scoring['Presence of Relevant Videos']!='Relevant'), 'Scores - Relevancy_youtube'] = 0
	#weightages
	weightages = pd.read_excel(weightagefile, sheet_name='YouTube')
	youtubeweightage = weightages.loc[weightages['Parameters'] == 'Total Score_youtube', 'Weightage'].iloc[0]
	#calculation
	scoring['Total Score_youtube'] = scoring.apply(lambda row: row['Scores - presence_youtube']*weightages.loc[weightages['Parameters'] == 'Scores - presence_youtube', 'Weightage'].iloc[0] + 
	                                  row['Scoring- no. of videos_youtube']*weightages.loc[weightages['Parameters'] == 'Scoring- no. of videos_youtube', 'Weightage'].iloc[0] +
	                                  row['Scores - no. of views_youtube']*weightages.loc[weightages['Parameters'] == 'Scores - no. of views_youtube', 'Weightage'].iloc[0] +
	                                  row['Scores - no. of likes_youtube']*weightages.loc[weightages['Parameters'] == 'Scores - no. of likes_youtube', 'Weightage'].iloc[0] +
	                                  row['Scores  - Total Subscriptions_youtube']*weightages.loc[weightages['Parameters'] == 'Scores  - Total Subscriptions_youtube', 'Weightage'].iloc[0] + 
	                                  row['Scores - Recency_youtube']*weightages.loc[weightages['Parameters'] == 'Scores - Recency_youtube', 'Weightage'].iloc[0] +
	                                  row['Scores - Relevancy_youtube']*weightages.loc[weightages['Parameters'] == 'Scores - Relevancy_youtube', 'Weightage'].iloc[0], axis=1)
	return scoring,youtubeweightage
    
def get_user_tweet_profile(d):
	client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAABYZgwEAAAAAb4tPJ1oHg5LIABwh6d9LWszHsho%3DUm7PFhkU4uuBrVPq3PT3jgUZxeUsJb52zaLlosv7pXLxv9ch93')




def func(choice):

	if choice == "Data Collection":
		st.subheader('Data Collection')
		result = pd.DataFrame(columns=['title', 'First_Name', 'Last_Name', 'Middle_Name', 'summary',
	       'location', 'city', 'region', 'country', 'connections', 'followers',
	       'about', 'current_company', 'designation', 'recent_posted', 'posts',
	       'linkedIn_url'])

		df = pd.DataFrame()	
		data = st.file_uploader("Upload HCP Dataset",type=["xlsx"])
		with st.spinner("Uploading and Reading the Data...."):
			if data is not None:
				flg = 'N'
				df = pd.read_excel(data)
				df = df.astype(str)
				df = df[["NPI","PersonFirstName","PersonMiddleName",'PersonLastName','PersonTitleCode','PersonGender','CityName','State']]
				st.dataframe(df.head())	
		if len(df)>0:
			with st.spinner("Searching through twitter and collecting handles, please wait..."):
				sleep(10)
			#sm_handles = profilesearch(df.head())
				sm_handles = pd.read_excel("sm_handles.xlsx")
			#st.write(len(sm_handles))
			#sm_handles.to_excel("sm_handles.xlsx", index=False, encoding='utf-8')

			st.markdown("""
			<style>
			div[data-testid="metric-container"] {
			   background-color: rgba(28, 131, 225, 0.1);
			   border: 1px solid rgba(28, 131, 225, 0.1);
			   padding: 5% 5% 5% 10%;
			   border-radius: 20px;
			   color: rgb(30, 103, 119);
			   overflow-wrap: break-word;
			}
			/* breakline for metric text         */
			div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
			   overflow-wrap: break-word;
			   white-space: break-spaces;
			   color: red;
			}
			</style>
			"""
			, unsafe_allow_html=True)


			st.markdown(
			    """
			<style>
			[data-testid="stMetricValue"] {
			    font-size: 40px;text-align:center;
			}
			</style>
			""",
			    unsafe_allow_html=True,
			)
			st.markdown(
			    """
			    <style>
			    [data-testid="stMetricDelta"] svg {
			        display: none;
			    }
			    [data-testid="stMetricDelta"]{
			    text-align:center
			    }
			    </style>
			    """,
			    unsafe_allow_html=True,
			)
			col3, col4 = st.columns(2)
			col3.metric("",len(df),"Number of HCPs searched for")
			col4.metric("",len(sm_handles),"Number of Handles Fetched")
			sm_handles['Platform'] = 'twitter'
			if len(sm_handles)>0:
				sm_handles = sm_handles[['handle_list','Platform']]


				with st.spinner("Collecting Full Profile information using Twitter API, Please wait...."):
					sleep(5)

					#df = twittwe_api(sm_handles)
					
					df = pd.read_excel("sm_full_profile.xlsx")

					#st.write(len(df))

					#df.to_excel("sm_full_profile.xlsx", index=False, encoding='utf-8')

					df = to_excel(df)
				st.download_button(label="Download Twitter Profiles information",data=df,file_name="Twitter Profiles information.xlsx",mime='xlsx')


	if choice =='Social Media Handle Mapping':
		st.subheader('Social Media Handle Identification')	
		threshold1 = st.sidebar.slider('Select Threshold (Minimum score for candidate to be considered as a match)',50, 100,value=85, step=1)
		threshold2 = st.sidebar.slider('Select Window Size (Window size for candidate to be considered for manual review)',1, 10,value=5, step=1)
		df=pd.DataFrame()
		df2=pd.DataFrame()
		data=pd.DataFrame()
		data2=pd.DataFrame()
		col1, col2 = st.columns(2)
		with col1:
			data = st.file_uploader("Upload HCP Dataset",type=["xlsx"])
		with col2:
			data2 = st.file_uploader("Upload Social Media Dataset",type=["xlsx"])
		if (data is not None) & (data2 is not None):
			df, df2 = upload1(data, data2)
			
			if (len(df)>0) & (len(df2)>0):
				with col1:
					dataframe_head(df)
					all_columns = df.columns.to_list()
				with col2:	
					dataframe_head(df2)
					all_columns1 = df2.columns.to_list()
				final_merge, final_merge_match,final_merge_manual,final_merge_non_match,potential_matches, candidates, df, df2 = getoutput(df, df2, threshold1, threshold2)
				final_merge_match,final_merge_manual,final_merge_non_match = view_data(df, df2, threshold1,threshold2,final_merge, final_merge_match,final_merge_manual,final_merge_non_match,potential_matches, candidates)

				if len(final_merge_manual)>0:
					manualfile = st.file_uploader("Upload Manually Reviewed Dataset",type=["csv"])
					if manualfile is not None:
						manualfile = pd.read_csv(manualfile)
						manualfile = manual_file_upload(manualfile)
						if manualfile is not None:
							manual_match = manualfile[manualfile['File']=='Match']
							manual_non_match = manualfile[manualfile['File']=='Non Match']
							manual_match = manual_match[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp','first_sm','middle_sm','last_sm','title_sm','Gender_sm','City_sm','State code_sm','Specialty_sm','handle','Score']]
							manual_non_match = manual_non_match[['NPI_hcp','SrNum1_hcp','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','PersonTitleCode_hcp','PersonGender_hcp','CityName_hcp','State_hcp','Specialty_hcp','first_sm','middle_sm','last_sm','title_sm','Gender_sm','City_sm','State code_sm','Specialty_sm','handle','Score']]
							final_merge_match = final_merge_match.append(manual_match)
							final_merge_non_match = final_merge_non_match.append(manual_non_match)
							view_data2(df, final_merge_match, final_merge_non_match, threshold1,threshold2)

				st.caption("Description:")
				st.caption("The model creates candidate pairs from Full Name in the HCP data and Socia Media data based on sortedneighbourhood indexing.")
				st.caption("All the features selected for name matching are then compared for each candidate pair. Each feature is compared with the corresponding feature using the exact match, bag of words or Levenstein method of string similarity. Based on the matching features, a probability score (%) is assigned to each candidate pair.")
				st.caption("All the candidate pairs with probability over a specified threshold is considered as a match and the rest are considered as non-matches. The threshold is 80% by default, however, the user has an option of modifying it on the UI.")
				
				#st.write(len(manual_match))
				#st.write(len(manual_non_match['SrNum1_hcp'].unique()))
				#st.write(final_merge_match.columns)
				#st.write(manual_match.columns)

				st.session_state['flag'] = final_merge_match

	if choice == "Insights":
		d = pd.DataFrame()
		select_data = st.sidebar.radio("Please choose an option",['Use valid profiles Data','Upload New File'])

		if select_data == "Use valid profiles Data":
			try:
				plchld = st.empty()
				click = plchld.checkbox("Please click here to perform the analysis")
				
				if click:
					plchld.empty()
					d = st.session_state.flag
					
			except:
				pass
				st.write("You have choosen an option to use the Valid profiles Data from previous session.")
				st.write("Please Navigate to the previous session to get the Valid profile Data or select an option to upload new file.")

		#col21,col22 =st.columns(2)
		#with col21:
		if select_data == "Upload New File":
				placeholder = st.empty()
				placeholder.write("Please upload the file for performing Analysis")
				upld = st.sidebar.file_uploader("Upload Valid profiles Data for Analysis",type=["csv"])
				if upld is not None:
					d = pd.read_csv(upld)
					placeholder.empty()
		if len(d)>0:
			#d = d.head(2)
			@st.cache(allow_output_mutation=True)
			def run_nlp_models(df):
				for i,v in enumerate(df['Insight'].astype(str)):
					df.loc[i,'Insight_word_cloud'] = prep.clean(v)

				with st.spinner("Running Sentiment Analysis, please wait..."):
					df = Sentiment_Analysis(df)

				with st.spinner("Running Aspect based Sentiment Analysis, please wait..."):				
					asbadf = pd.read_excel("absa_sentiment_v3 Medical Condition.xlsx")
					asbadf1 = asbadf

					symdf = pd.read_excel("absa_sentiment_v3 Symptom.xlsx")
					symdf1 = symdf

					meddf = pd.read_excel("absa_sentiment_v3 Medication.xlsx")
					meddf1 = meddf	

				with st.spinner("Running KIC Model, please wait..."):
					df = KIC_Predction_Pipeline(df)


				#df = pd.read_excel("HCP_KIC.xlsx")
				df['tweeted_time'] = pd.to_datetime(df['tweeted_time'])
				df['date']=pd.to_datetime(df['tweeted_time'])
				df['year']=pd.DatetimeIndex(df['date']).year
				df['year']=df['year'].astype('object')
				df['date'] = pd.to_datetime(df['date'])

				return df,asbadf1,symdf1,meddf1
			def f_word_cloud(column):
			    comment_words = ' '
			    stopwords = set(STOPWORDS)

			    stopwords = ["thanks","great","ln","thank","n","congratulations","good","better","best","now","agree","one","will","need","time","two","love","u","way","really","year","via","still","many","amp"] + list(STOPWORDS)

			    # iterate through the csv file 
			    for val in column: 

			        # typecaste each val to string 
			        val = str(val) 

			        # split the value 
			        tokens = val.split() 

			        # Converts each token into lowercase 
			        for i in range(len(tokens)): 
			            tokens[i] = tokens[i].lower() 

			        for words in tokens: 
			            comment_words = comment_words + words + ' '


			    wordcloud = WordCloud(width = 800, height = 800, 
			                    background_color ='white', 
			                    stopwords = stopwords, 
			                    min_font_size = 10).generate(comment_words) 
			    
			    return wordcloud
			def Sentiment_Analysis(df):

				# df["DISEASE"] = ""
				# df["CHEMICAL"] = ""
				# df["PERSON"] = ""
				# df["GPE"] = ""
				# df["ORG"] = ""

				# for i,row in df.iterrows():
				#     text = row["Insight"]
				#     # print(text)
				#     disease_ls, chemical_ls, person_ls, gpe_ls, org_ls = [], [], [], [], []
				#     disease_ls,chemical_ls = get_disease_chemical_ner(text)
				#     person_ls, gpe_ls, org_ls = get_person_gpe_org_ner(text)
				#     df.at[i,'DISEASE'] = disease_ls
				#     df.at[i,'CHEMICAL'] = chemical_ls
				#     df.at[i,'PERSON'] = person_ls
				#     df.at[i,'GPE'] = gpe_ls
				#     df.at[i,'ORG'] = org_ls
				# df['DISEASE'] = df['DISEASE'].apply(lambda x : ", ".join(x)) 
				# df['CHEMICAL'] = df['CHEMICAL'].apply(lambda x : ", ".join(x)) 
				# df['PERSON'] = df['PERSON'].apply(lambda x : ", ".join(x)) 
				# df['GPE'] = df['GPE'].apply(lambda x : ", ".join(x)) 
				# df['ORG'] = df['ORG'].apply(lambda x : ", ".join(x)) 
				# #st.dataframe(df)
				df["sentiment"] = ""

				#sid_obj = SentimentIntensityAnalyzer()
				
				for i,row in df.iterrows():
					text = df.loc[i,"Insight"]

					sentiment = sentiment_scores(text)
					df.loc[i,"sentiment"] = sentiment

				#st.dataframe(df.head(10))
				return df
			def sentiment_scores(sentence):
				sentiment = ""
				sid_obj = SentimentIntensityAnalyzer()
				sentiment_dict = sid_obj.polarity_scores(sentence)
				if sentiment_dict['compound'] >= 0.05 :
					sentiment="Positive"
				elif sentiment_dict['compound'] <= - 0.05 :
					sentiment = "Negative"
				else :
					sentiment = "Neutral"
				return sentiment	
			def get_disease_chemical_ner(text):
			    disease_ls = []
			    chemical_ls = []

			    doc = d_c_base_spacy_model(text)
			    for ent in doc.ents:
			        if ent.label_ == "DISEASE":
			            disease_ls.append(ent.text)
			        elif ent.label_ == "CHEMICAL":
			            chemical_ls.append(ent.text)

			    return disease_ls, chemical_ls
			def get_person_gpe_org_ner(text):
				person_ls = []
				gpe_ls = []
				org_ls = []
				doc = pgo_base_spacy_model(text)
				for ent in doc.ents:
				    if ent.label_ == "PERSON":
				        person_ls.append(ent.text)
				    elif ent.label_ == "GPE":
				        gpe_ls.append(ent.text)
				    elif ent.label_ == "ORG":
				        org_ls.append(ent.text)

				return person_ls, gpe_ls, org_ls
			def KIC_Predction_Pipeline(df):

				#df1=pd.read_excel('Test_Data (3).xlsx',index_col=[0]).reset_index(drop=True)
				# Stop words list from custom & nltk
				from nltk.corpus import stopwords
				df1 = df.astype(str)
				df1.reset_index(drop=True,inplace=True)
				nltk_st_words = stopwords.words('english')
				custom_st_words = ['also', 'sees','us']

				# Append custom list to ntlk list and remove duplicates
				nltk_st_words.extend(custom_st_words)
				stop_word = set(nltk_st_words)

				# Extract the clean data
				clean_data = []
				for i in range(df1.shape[0]):
					pun = re.sub(r"[^\w\s]", '', df1['Insight'][i].lstrip())
					stop_word_data = " ".join([w for w in pun.split() if w.lower() not in stop_word])
					clean_data.append(stop_word_data)

				df1['Insight_upd'] = clean_data

				df1['Insight_upd'] = df1['Insight_upd'].str.lower()
				num_labels = 11
				vocab_size = 829
				batch_size = 60
				num_epochs = 10
				data_imb=df1.copy()
				test_posts = data_imb['Insight_upd']
				# define Tokenizer with Vocab Size
				tokenizer = Tokenizer(num_words=vocab_size)
				tokenizer.fit_on_texts(test_posts)
				#x_train = tokenizer.texts_to_matrix(train_posts)
				x_test = tokenizer.texts_to_matrix(test_posts)

				keras_model_path = 'KIC1_save'	
				#model.save(keras_model_path)
				another_strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
				with another_strategy.scope():
				    restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
				    x_test1=restored_keras_model_ds(x_test)
				    #print(x_test1)
				yhat_probs=x_test1
				#print(yhat_probs) 
				classes_x=np.argmax(yhat_probs,axis=1)
				yhat_classes=classes_x

				x=pd.DataFrame(yhat_probs,columns=['ACCESS','COMPETITIVE LANDSCAPE','COSENTYX','DISEASE STATE/TREATMENT GOALS','EDUCATION/LEARNING','EFFICACY','HS', 'OTHER','PATIENT DYNAMICS','PSORIASIS','SAFETY']).reset_index()
				c = ['KIC-1_Pred']
				f1 = (x.set_index('index') .apply(lambda x: pd.Series(x.nlargest(1).index, index=c), axis=1).reset_index(drop=True)) 

				keras_model_path = 'KIC2_save'
				#model1.save(keras_model_path)
				another_strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
				with another_strategy.scope():
				    restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
				    x_test2=restored_keras_model_ds(x_test)
				    #print(x_test2)
				yhat_probs=x_test2
				#print(yhat_probs)
				#predict_x=model.predict(X_test) 
				classes_x=np.argmax(yhat_probs,axis=1)
				yhat_classes=classes_x

				x=pd.DataFrame(yhat_probs,columns=['ACCESS','COMPETITIVE LANDSCAPE','COSENTYX','DISEASE STATE/TREATMENT GOALS','EDUCATION/LEARNING','EFFICACY','HS', 'OTHER','PATIENT DYNAMICS','PSORIASIS','SAFETY']).reset_index()
				c = ['KIC-2_Pred']
				f2 = (x.set_index('index') .apply(lambda x: pd.Series(x.nlargest(1).index, index=c), axis=1).reset_index(drop=True))

				keras_model_path = 'KIC3_save'
				#model2.save(keras_model_path)
				another_strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
				with another_strategy.scope():
				    restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
				    x_test3=restored_keras_model_ds(x_test)
				    #print(x_test3)
				yhat_probs=x_test3
				classes_x=np.argmax(yhat_probs,axis=1)
				yhat_classes=classes_x

				x=pd.DataFrame(yhat_probs,columns=['ACCESS','COMPETITIVE LANDSCAPE','COSENTYX','DISEASE STATE/TREATMENT GOALS','EDUCATION/LEARNING','EFFICACY','HS', 'OTHER','PATIENT DYNAMICS','PSORIASIS','SAFETY']).reset_index()
				c = ['KIC-3_Pred']
				f3 = (x.set_index('index') .apply(lambda x: pd.Series(x.nlargest(1).index, index=c), axis=1).reset_index(drop=True))

				final=pd.concat([df1,f1,f2,f3], axis = 1)
				#final=final[['Insight_upd','KIC-1_Pred','KIC-2_Pred','KIC-3_Pred']]
				return final
			def descriptive_info(data_frame):
			    total_posts = data_frame.shape[0]
			    distinct_users = data_frame['handle'].nunique()
			    Avg_posts_per_user = round(total_posts/distinct_users)
			    time_period = data_frame['tweeted_time'].min().strftime("%b-%Y") + " to " + data_frame['tweeted_time'].max().strftime("%b-%Y")
			    return total_posts, distinct_users, Avg_posts_per_user, time_period
			def Message_distribution(data):
				plt.figure(figsize=[15,10])
				#data['month']=[ dt.strftime(i, '%Y-%m') for i in data['date']]
				data['month'] = data['date'].dt.strftime('%Y-%m')
				ans_2=data['month'].value_counts().sort_index()
				label=ans_2.index
				val=ans_2.values
				plt.plot(label,val)
				#plt.xticks(rotation = 45)
				#plt.title('Monthly Distribution of Messages', fontsize=25)
				plt.ylabel('No of messages', fontsize=15)
				plt.xlabel('Months', fontsize=15)
				plt.xticks(rotation=70, fontsize=15)
				plt.yticks(fontsize=15)
				for i,j in zip(label,val):
				    plt.text(x=i,y=j+0.20,s=str(j),fontsize=20)
				plt.tight_layout()	

				return plt
			def hastags(data):
				plt.figure(figsize=[15,11.5])
				top5_hashtag=data['Insight'].str.extract(r"#(\w+)")[0].str.lower().value_counts().sort_values(ascending=False)[0:5].sort_values()
				plt.barh(y='#'+top5_hashtag.index,width=top5_hashtag.values)
				#plt.title('Top 5 hashtags', fontsize=25)
				plt.ylabel('Hashtags', fontsize=15)
				plt.xlabel('Counts', fontsize=15)
				plt.xticks(fontsize=15)
				plt.yticks(fontsize=15)
				for i in range(len(top5_hashtag)):
				    plt.text(y=i,x=top5_hashtag.values[i]+0.1,s=str(top5_hashtag.values[i]),fontsize=15)
				return plt
			def wcloud(data):
				wordcloud = f_word_cloud(data['Insight_word_cloud'])
				plt.figure(figsize = [3, 3], facecolor = None)
				#plt.title('Word Cloud for Tweets', fontsize=5)
				plt.imshow(wordcloud) 
				plt.axis("off") 
				plt.tight_layout(pad = 0)
				return plt
			def absa_chart(df):
				df = df[~df['MEDICAL_CONDITION'].isnull()]
				dftest = df.groupby(['MEDICAL_CONDITION'])['MEDICAL_CONDITION'].count().reset_index(name='Count').sort_values(['Count'], ascending=False)[0:10]
				df = df[df['MEDICAL_CONDITION'].isin(list(dftest['MEDICAL_CONDITION'].unique()))]
				domain  = ['Positive', 'Negative', 'Neutral']
				range_ = ['green','red','orange']
				bars = alt.Chart(df).mark_bar().encode(
				    x=alt.X('count(MEDICAL_CONDITION):Q', stack='zero', axis=alt.Axis(title='Count',grid=False, format='.0f',tickMinStep=1), sort=alt.EncodingSortField(field='MEDICAL_CONDITION', op='count', order='descending')),
				    y=alt.Y('MEDICAL_CONDITION:N',axis=alt.Axis(grid=False)),
				    color=alt.Color('MEDICAL_CONDITION_Sentiment', scale=alt.Scale(domain=domain, range=range_),legend=alt.Legend(
						orient='bottom',
						#legendX=50, legendY=-40,
						direction='horizontal',
						titleAnchor='middle'))
				)
				text = alt.Chart(df).mark_text(dx=-15, dy=3, color='white', size=15).encode(
					x=alt.X('count(MEDICAL_CONDITION):Q', stack='zero'),
					y=alt.Y('MEDICAL_CONDITION:N'),
					detail='MEDICAL_CONDITION_Sentiment:N',
					text=alt.Text('count(MEDICAL_CONDITION):Q', format='.0f')
				)
				c = bars + text
				c.height=230
				#c.save('absachart.png')
				
				#return c
			def pie_chart(data):
				plt.figure(figsize=[20,20])
				var=data['sentiment'].value_counts()
				plt.pie(x=var.values,labels=var.index,autopct='%1.2f%%',pctdistance=0.85, colors=['grey', 'pink','orange'],textprops={'fontsize': 25})
				#plt.title('Distribution of Sentiments', fontsize=15)
				labels=var.index
				centre_circle = plt.Circle((0, 0), 0.70, fc='white')
				fig = plt.gcf()
				fig.gca().add_artist(centre_circle)
				#plt.legend(labels, loc="center right", title="Distribution of Sentiments",bbox_to_anchor=(1.7, 0.5))
				return plt
			def kic_graph(data):
				plt.figure(figsize=[15,8])
				#final_kic1 = final_kic['KIC-1_Pred'].drop(["OTHER","COSENTYX","HS"],axis=0)
				data['KIC-1_Pred'].value_counts().drop(['HS','COSENTYX','OTHER'],errors='ignore').plot(kind= 'barh')
				#plt.title('Distribution of Intents', fontsize=25)
				plt.xticks(fontsize=12) #rotation = 70,
				plt.yticks(fontsize=12)
				plt.ylabel('Counts', fontsize=15)
				plt.xlabel('KIC', fontsize=15)
				val1=data['KIC-1_Pred'].value_counts().drop(['HS','COSENTYX','OTHER'],errors='ignore')
				val=data['KIC-1_Pred'].value_counts(normalize=True).drop(['HS','COSENTYX','OTHER'],errors='ignore')
				for i in range(len(val1)):
				    plt.text(y=i-0.05,x=val1.values[i]+0.05,s=str(round(val.values[i]*100,2)),fontsize=15)
				return plt,val
			def symptom_graph(data):
				plt.figure(figsize=[15,8])
				if len(data)>0:
					#final_kic1 = final_kic['KIC-1_Pred'].drop(["OTHER","COSENTYX","HS"],axis=0)
					data['Symptom/Side Effect'].value_counts()[:5].plot(kind= 'bar')
					#plt.title('Distribution of Intents', fontsize=25)
					val=data['Symptom/Side Effect'].value_counts()[:5]
					for i in range(len(val)):
					    plt.text(x=i-0.05,y=val.values[i]+0.05,s=str(val.values[i]),fontsize=15)
				plt.xticks(rotation = 70,fontsize=15)
				plt.yticks(fontsize=15)
				plt.ylabel('Counts', fontsize=15)
				plt.xlabel('Symptoms', fontsize=15)
				return plt
			def medication_graph(data):
				plt.figure(figsize=[15,8])
				if len(data)>0:
					#final_kic1 = final_kic['KIC-1_Pred'].drop(["OTHER","COSENTYX","HS"],axis=0)
					data['MEDICATION'].value_counts()[:5].plot(kind= 'bar')
					#plt.title('Distribution of Intents', fontsize=25)

					val=data['MEDICATION'].value_counts()[:5]
					for i in range(len(val)):
					    plt.text(x=i-0.05,y=val.values[i]+0.05,s=str(val.values[i]),fontsize=15)
				plt.xticks(rotation = 70,fontsize=15)
				plt.yticks(fontsize=15)
				plt.ylabel('Counts', fontsize=15)
				plt.xlabel('Medication', fontsize=15)
				return plt				
			def country(data):
				plt.figure(figsize=[15,8])
				val = data.Country.value_counts()
				sns.countplot(data=data, x= data.Country, order=list(val.index))
				for i in range(len(val)):
				    plt.text(x=i-0.10,y=val.values[i]+0.05,s=str(val.values[i]), fontsize=15)
				return plt
			def specialty(data):
				plt.figure(figsize=[15,8])
				val = data.Profession.value_counts()
				sns.countplot(data=data, x= data.Profession, order=list(val.index))
				#val = data.Profession.value_counts()
				for i in range(len(val)):
				    plt.text(x=i-0.10,y=val.values[i]+0.05,s=str(val.values[i]), fontsize=15)	    		
				return plt
			def parse_twitter_tweets(usernames):
				user_name = []
				handle = []
				latest_tweets = []
				#twitter_user_id = []
				tweet_date = []
				users = client.get_users(usernames=usernames,user_fields=['id','name'])
				for user in users.data:
					#handle.append(user.username)
					#twitter_user_id.append(user.id)
					user_id = user.id
					name = user.name
					tweets = client.get_users_tweets(user_id, max_results=100,tweet_fields=['created_at'], exclude='retweets')
					if tweets.data:
					  for tweet in tweets.data:
					      latest_tweets.append(tweet.text)
					      handle.append(user.username)
					      user_name.append(user.name)
					      tweet_date.append(tweet.created_at)       
					else:
					  latest_tweets.append('NA')
					  handle.append(user.username)
					  user_name.append(user.name)
					  tweet_date.append('NA')
				df = pd.DataFrame()
				df['handle'] = handle
				df['Full_Name'] = user_name
				df['Insight'] = latest_tweets
				df['tweeted_time'] = tweet_date
				return df

			# d['Final Twitter Handles'] = d['handle']
			# with st.spinner("Collecting tweets Data from Valid profiles, please wait..."):
			# 	client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAABYZgwEAAAAAb4tPJ1oHg5LIABwh6d9LWszHsho%3DUm7PFhkU4uuBrVPq3PT3jgUZxeUsJb52zaLlosv7pXLxv9ch93')
			# 	d['Final Twitter Handles'] = d['Final Twitter Handles'].apply(lambda x:x.lstrip('@'))

			# 	df = pd.DataFrame()
			# 	df_len=d.index[-1]
			# 	for i in range(math.ceil(len(d)/100)):
			# 	    start = i*100
			# 	    end = start+100
			# 	    if end>df_len:
			# 	        end=df_len+1
			# 	    print(start,end)
			# 	    df1 = parse_twitter_tweets(list(d['Final Twitter Handles'][start:end]))
			# 	    df = pd.concat([df,df1], ignore_index=True)
			# 	    sleep(2)
			# 	df['tweeted_time'] = df['tweeted_time'].dt.date
			# 	df = df[df['tweeted_time']>dt.date.today()-timedelta(days=730)]
			# 	df.reset_index(drop=True, inplace=True)

			# data1 = pd.DataFrame()  
			# data = pd.DataFrame()

			# df['date']=pd.to_datetime(df['tweeted_time'])
			# df['year']=pd.DatetimeIndex(df['date']).year
			# df['year']=df['year'].astype('object')
			# for i,v in enumerate(df['Insight'].astype(str)):
			# 	df.loc[i,'Insight_word_cloud'] = prep.clean(v) #remove mentions, urls, hashtags, emoticons
			
			# def namecleaning(x):
			#     x= re.sub('[^\w\s,-]', '', x)
			#     x= prep.clean(x)
			#     return x
			# df['Full Name cleaned'] = df['Full_Name'].apply(namecleaning)
			# lst = ['AAH','AAN','AANPCP','AARCF','AAS','AAT','ACLS','ACNP','ACNPAG','ACNPBC','ACNPCAG','ACNS','ACNSBC','ACP','ACRN','ADCN','ADN','AEMSN','AGACCRNPBC','AGACNP','AGACNPBC','AGCNPBC','AGCNSBC','AGNP','AGNPBC','AGNPC','AGPCNP','AGPCNPBC','AGPCNPC','AGPCPNPBC','ALNC','AMSC','ANP','ANPBC','ANPBCCCRN','ANPC','AOCN','AOCNP','AOCNS','APC','APMC','APN','APNBC','APNC','APNCNP','APNCNS','APNFNP','APNP','APRN','APRNBC','APRNC','APRNCNP','APRNCNS','APRNNP','ARNP','ARNPBC','ARNPC','ARRT','ART','AS','ASCP','ASN','ASNC','ASPO','ASSOC','ATC','BA','BAO','BC','BCBA','BCH','BCLC','BCLS','BCOP','BCPCM','BCPS','BDS','BDSC','BHS','BHSCNSG','BHYG','BM','BMS','BN','BPHARM','BPHN','BS','BSED','BSEH','BSM','BSN','BSPH','BSW','BVMS','CANP','CAPA','CARN','CATNI','CATNP','CB','CBE','CBI','CCCA','CCCN','CCCSLP','CCE','CCES','CCM','CCNS','CCP','CCRN','CCSP','CCST','CCTC','CCTN','CCTRN','CD','CDA','CDDN','CDMS','CDN','LTC','CDONA','CEN','CETN','CFCN','CFN','CFNP','CFRN','CGN','CGRN','CGT','CH','CHB','CHD','CHES','CHN','CHPN','CHRN','CHUC','CIC','CLA','CLNC','CLPNI','CLS','CLT','CM','CMA','CMAA','CMAC','CMCN','CMDSC','CMSRN','CNA','CNAA','CNDLTC','CNE','CNI','CNLCP','CNM','CNMT','CNN','CNNP','CNO','CNP','CNRN','CNS','CNSN','CO','COCN','COHNCM','COHNS','COHNSCM','COMA','CORLN','CORN','COTA','CP','CPAN','CPDN','CPFT','CPHQ','CPN','CPNA','CPNL','CPNP','CPNPAC','CPON','CPSN','CRN','CRNA','CRNFA','CRNH','CRNI','CRNL','CRNO','CRNP','CRRN','CRRNA','CRT','CRTT','CS','CSN','CSPI','CST','CTN','CTRN','CUA','CUCNS','CUNP','CURN','CVN','CWCN','CWOCN','DA','DC','DCH','DCP','DDR','DDS','DDSC','DM','DMD','DME','DMSC','DMT','DMV','DN','DNC','DNE','DNP','DNS','DNSC','DO','DON','DOS','DP','DPH','DPHIL','DPHN','DR','DRNP','DRPH','DS','DSC','DSW','DVM','DVMS','DVR','DVS','EDD','EMTB','EMTD','EMTP','EN','ENP','ENPC','ET','FAAAI','FAAFP','FAAN','FAAO','FAAOS','FAAP','FAAPM','FACAAI','FACC','FACCP','FACD','FACE','FACEP','FACG','FACOG','FACOOG','FACP','FACPM','FACR','FACS','FACSM','FAEN','FAGD','FAMA','FAOTA','FAPA','FAPHA','FCAP','FCCM','FCPS','FDS','FFA','FFARCS','FICA','FICC','FICS','FNP','FNPBC','FNPC','FRCP','FRCPATH','GNP','GNPBC','IPN','JD','LAC','LAT','LATC','LC','LCCE','LCPC','LCSW','LDN','LDO','LICSW','LLC','LM','LMSW','LNC','LNCC','LPC','LRCP','LRCS','LRN','LSN','LVN','MA','MAS','MB','MBA','MBBCH','MBBS','MBCHB','MC','MCH','MD','MDFACP','MDPHD','MDS','ME','MED','MEMERGNSG','MFT','MHE','MHN','MHS','MICN','MLS','MLT','MMS','MN','MO','MPA','MPAS','MPH','MPP','MPT','MRAD','MRCP','MRCS','MRL','MS','MSC','MSCE','MSCI','MSCR','MSD','MSEE','MSEH','MSHA','MSLS','MSN','MSNC','MSNE','MSPAS','MSPH','MSSW','MSURG','MSW','MT','MTA','MV','NCSN','NCT','ND','NEABC','NEBC','NIC','NMT','NNP','NP','NPC','NPP','NPRN','NREMTP','OCN','OCNP','OCS','OD','OGNP','OHNCS','OLMO','ONC','ONP','OTA','OTC','OTL','OTR','OTRL','PA','PAC','PACMPAC','PALS','PAS','PCCN','PCNS','PD','PH','PHARMD','PHARMG','PHD','PHN','PHRN','PMHCNS','PMHNP','PMHNPBC','PMN','PNP','PNPAC','PTA','RA','RD','RDA','RDCS','RDH','RDMS','REEGT','REPT','RHIA','RHIT','RIPRN','RMA','RN','RNC','RNA','RNBC','RNC','RNCS','RNFA','RNP','RNPC','ROUB','RPA','RPAC','RPH','RPN','RPT','RRA','RRT','RT','RTR','RVT','SANEA','SANEP','SBB','SC','SCD','SCT','SH','SHN','SLS','SM','SN','SPN','ST','SV','SVN','TM','TNCCI','TNCCP','TNP','TNS','VMD','VT','WCC','WHNP','WOCN']
			    
			# fullname=[]
			# main_title=[]
			# for j in range(len(df)):
			#     name=[]
			#     title=[]
			#     for i in df['Full Name cleaned'][j].split():
			#         i = re.sub('[^\w\s]', '', i)
			#         if i.upper() in lst:
			#             title.append(i)
			#         else:
			#             name.append(i)
			#     fullname.append(" ".join(name))
			#     main_title.append(" ".join(title))
			    
			# df['HCP Name'] = fullname
			# df['title'] = main_title

			# df.to_excel("HCP_tweets.xlsx", index=False, encoding='utf-8')

			############################################
			# Getting the HCP Tweets from Twitter API #
			############################################
			with st.spinner("Analysing the Data and Initializing NLP models, please wait..."):
				import matplotlib.pyplot as plt
				import en_ner_bc5cdr_md
				import en_core_web_sm
				nltk.download('stopwords')
				d_c_base_spacy_model = en_ner_bc5cdr_md.load() #spacy.load("en_ner_bc5cdr_md")
				pgo_base_spacy_model = en_core_web_sm.load() #spacy.load("en_core_web_sm")
			df = pd.read_excel("HCP_Tweets.xlsx")
			df,asbadf1,symdf1,meddf1 = run_nlp_models(df)

			data1 = df
			data1['HCP Name'] = "Dr. "+ data1['Full_Name']
			if df is not None:
				#selection_list = list(data1['HCP Name'].unique()[:5])
				selection_list = list(data1['HCP Name'].value_counts()[(data1['HCP Name'].value_counts()>50) & (data1['HCP Name'].value_counts()<95)].index[:20])
				#selection_list = list(["Dr. Bryan Hambley","Dr. Alice Mims","Dr. Erel Joffe"])
				selection_list.insert(0,"ALL")
				selection_list.insert(0," ")
				#st.write(selection_list)
				#col21,col22,col23 = st.columns(3) 
				placeholder = st.empty()
				placeholder.write("Please choose an option to show the analysis")
				hcpname = st.sidebar.selectbox('Please select the Name of HCP',selection_list,key='sel1')
				if hcpname is not None and hcpname!=" ":
					placeholder.empty()
					if hcpname == "ALL":
						data = data1
						asbadf = asbadf1
						symdf = symdf1
						meddf = meddf1
					else:
						data = data1[data1['HCP Name']==hcpname]
						handlefilter = data['handle'].unique()[0]
						#st.write(handlefil)
						asbadf = asbadf1[asbadf1['Handle']==handlefilter]
						symdf = symdf1[symdf1['Handle']==handlefilter]						
						meddf = meddf1[meddf1['Handle']==handlefilter]

					with st.spinner("Ploting the Graphs"):
						#st.write(data.columns)
						#st.dataframe(data)
						total_posts, distinct_users, Avg_posts_per_user, time_period = descriptive_info(data)

						col7, col8, col9, col10, col11 = st.columns(5)

						st.markdown("""
						<style>
						div[data-testid="metric-container"] {
						   background-color: rgba(28, 131, 225, 0.1);
						   border: 1px solid rgba(28, 131, 225, 0.1);
						   padding: 5% 5% 5% 10%;
						   border-radius: 20px;
						   color: rgb(30, 103, 119);
						   overflow-wrap: break-word;
						}
						/* breakline for metric text         */
						div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
						   overflow-wrap: break-word;
						   white-space: break-spaces;
						   color: red;
						}
						</style>
						"""
						, unsafe_allow_html=True)


						st.markdown(
						    """
						<style>
						[data-testid="stMetricValue"] {
						    font-size: 40px;text-align:center;
						}
						</style>
						""",
						    unsafe_allow_html=True,
						)
						st.markdown(
						    """
						    <style>
						    [data-testid="stMetricDelta"] svg {
						        display: none;
						    }
						    [data-testid="stMetricDelta"]{
						    text-align:center
						    }
						    </style>
						    """,
						    unsafe_allow_html=True,
						)

						col7.metric("",total_posts,"Number of Tweets")
						col8.metric("", distinct_users, "Unique HCPs")
						col9.metric("", Avg_posts_per_user,"Avg. Tweets per HCP")
						col10.metric("",time_period.split("to")[0],"From")
						col11.metric("",time_period.split("to")[1],"Till")
						st.markdown("""---""")
						col1, col2 = st.columns(2)

						with col1:			
							#st.write("**How are the Tweets distributed over time:question:")
							st.markdown('**<p style="font-size:20px;border-radius:2%;text-align:center;">How are the Tweets distributed over time:question:</p>**',unsafe_allow_html=True)
							plt = Message_distribution(data)
							st.pyplot(plt)

						with col2:		
							#st.write("**What are the Top Hastags**:question:")
							st.markdown('**<p style="font-size:20px;border-radius:2%;text-align:center;">What are the Top Hastags:question:</p>**',unsafe_allow_html=True)
							plt = hastags(data)
							st.pyplot(plt)
						st.markdown("""---""")
						col3, col4 = st.columns(2)
						with col3:
							#st.write("**What are the most common Themes**:question:")
							st.markdown('**<p style="font-size:20px;border-radius:2%;text-align:center;">What are the most common Themes:question:</p>**',unsafe_allow_html=True)
							plt = wcloud(data)
							st.pyplot(plt)						

						with col4:
							#st.write("**How are the tweets sounding**:question:")
							with st.spinner("Running Sentiment Analysis Model..."):
								#data = Sentiment_Analysis(data)

								#data.to_excel("HCP_sentiment_Analysis.xlsx", index=False, encoding='utf-8')
								#data = pd.read_excel("HCP_sentiment_Analysis.xlsx")

								st.markdown('**<p style="font-size:20px;border-radius:2%;text-align:center;">How are the tweets sounding:question:</p>**',unsafe_allow_html=True)
								plt = pie_chart(data)
							st.pyplot(plt)
						st.markdown("""---""")
						#col5, 
						col6  = st.container()
						#with col5:
						#	st.markdown('**<p style="font-size:20px;border-radius:2%;text-align:center;">Whats the distribution of content:question:</p>**',unsafe_allow_html=True)
						#	with st.spinner("Running ASBA Model..."):
						#		c = absa_chart(asbadf)
						#	absaplot = Image.open('absachart.png')
						#	st.image(absaplot, clamp=False, channels="RGB", use_column_width=True)
							#st.dataframe(asbadf.head())
							
							#st.altair_chart(c, use_container_width=True)
						#final_kic = KIC_Predction_Pipeline(data)

						#final_kic.to_excel("HCP_KIC.xlsx", index=False, encoding='utf-8')

						#final_kic = pd.read_excel("HCP_KIC.xlsx")

						final_kic = data

						with col6:
							#st.write("**What are the key Topics discussed**:question:")
							st.markdown('**<p style="font-size:20px;border-radius:2%;text-align:center;">What are the key Topics discussed:question:</p>**',unsafe_allow_html=True)
							plt,val = kic_graph(data)
							st.pyplot(plt)
							#box_style = "background-color: #FFFF00; padding: 10px; border: 2px solid #FF0000; border-radius: 5px;"#<div style='{box_style}'>{hcpname}</div> 
							highlighted_text = f"<p style='color: black;''font-weight: bold;'><span style='color: orange;''font-weight: bold;'>{hcpname}</span> is discussing <span style='color: blue;'>{val.index[0]} </span>as a key area of focus with <span style='color: green;'>{round(val[0],2)}%</span> engagement rate.</p>"
							st.success(st.write(highlighted_text, unsafe_allow_html=True))
							

						#col7, col8 = st.columns(2)

						#with col7:
						#	st.markdown('**<p style="font-size:20px;border-radius:2%;text-align:center;">What are the Key Symptoms mentioned:question:</p>**',unsafe_allow_html=True)
						#	plt = symptom_graph(symdf)
						#	st.pyplot(plt)							


						#with col8:
						#	st.markdown('**<p style="font-size:20px;border-radius:2%;text-align:center;">What are the Key Medications:question:</p>**',unsafe_allow_html=True)
						#	plt = medication_graph(meddf)
						#	st.pyplot(plt)

						st.markdown("""---""")
	
						st.download_button(label="Download Output of Analysis",data=final_kic.to_csv(index=False).encode('utf-8'),file_name="Analysis Output.csv",mime='text/csv')

	if choice == "Affinity Calculation":
		#st.write("work in progress")
		plchld = st.empty()
		check = plchld.checkbox("Please check here to use the Match File from Name Matching Model")
		if check:
			#d = st.session_state.flag
			plchld.empty()
			with st.spinner("Reading the file, please wait..."):
				sleep(2)
				d=pd.read_excel("Matching HCPs.xlsx")

				if d is None:
					st.write("Please Navigate to Social Media Handle Mapping Tab to get the Matching HCP file")
				else:
					pass
			with st.spinner("scoring in progress, please wait..."):
				sleep(2)
				filename = 'Matching HCPs.xlsx'
				weightagefile = 'Weightages_UK_2.xlsx'
				scoring = pd.read_excel(filename, sheet_name='Scoring Sheet')
				scoring = scoring.fillna('')
				print("Twitter")
				twittertweetsdata = pd.read_excel(filename, sheet_name='Twitter Tweets Data')
				twitteruserdetails = pd.read_excel(filename, sheet_name='Twitter User Details')
				scoring,twitterweightage = get_twitter_scores(scoring,twittertweetsdata,twitteruserdetails,weightagefile)
				scoring['Overall Affinity Scores'] = scoring['Total Score_twitter']
			
			with st.spinner("Deciling and Segmentation of HCPs, please wait..."):
				sleep(2)
				deciledf = scoring[['External ID','Overall Affinity Scores']]
				deciledf = deciledf.sort_values(['Overall Affinity Scores'], ascending=False).reset_index(drop=True)
				oa_sum = deciledf['Overall Affinity Scores'].sum()
				cshare = 0
				share = []
				for i in deciledf['Overall Affinity Scores']:
				    cshare=cshare+i/oa_sum
				    share.append(cshare)
				deciledf['C.Share'] = share

				D=10
				deciledf['Decile']=0
				for i in range(len(deciledf)):
				    if deciledf['Overall Affinity Scores'].iloc[i] != 0:
				        if deciledf['C.Share'].iloc[i]> (11-D)*0.1:
				            deciledf.Decile.iloc[i] = D-1
				            D = deciledf.Decile.iloc[i]
				        else:
				            deciledf.Decile.iloc[i] = D 
				    else:
				        deciledf.Decile.iloc[i] = 0
				scoring = pd.merge(scoring,deciledf[['External ID','Decile']], on='External ID', how='left')

				scoring['Segmentation'] = ""
				scoring.loc[scoring['Decile']==0,'Segmentation']='No Digital Presence'
				scoring.loc[(scoring['Decile']>0) & (scoring['Decile']<4),'Segmentation']='Low'
				scoring.loc[(scoring['Decile']>=4) & (scoring['Decile']<8),'Segmentation']='Medium'
				scoring.loc[scoring['Decile']>=8,'Segmentation']='High'
				scoring['External ID'] = scoring['External ID'].astype(str)
				text_col = list(scoring.describe(include='O').columns)
				num_col = list(scoring.describe(include=np.number).columns)
				scoring[text_col] = scoring[text_col].astype(str)
				scoring[num_col] = scoring[num_col].astype(np.number)
			
			scoring_sample = scoring[['External ID','PersonFirstName_hcp','PersonMiddleName_hcp','PersonLastName_hcp','twitter url','Overall Affinity Scores','Decile','Segmentation']]

			st.write("Deciling And HCP Segmentation:")
			st.dataframe(scoring_sample.sample(10))
			st.write("Graphical Representation:")
			fig = px.histogram(scoring,x = 'Segmentation',text_auto=True)
			st.plotly_chart(fig)
			st.write("")
			scoring_xl = to_excel(scoring)
			st.download_button(label="Download Deciling and Segmentation Sheet",data=scoring_xl,file_name="Deciling and Segmentation Sheet.xlsx",mime='xlsx')



func(choice)	

		
