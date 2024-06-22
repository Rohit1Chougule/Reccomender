#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout='wide', page_title='Book Recommendation System')


# In[2]:


#### Loading data
books = pd.read_csv('Books.csv', encoding='latin-1', dtype={3: str}, low_memory=False)
users = pd.read_csv('Users.csv', encoding='latin-1')
ratings = pd.read_csv('Ratings.csv', encoding='latin-1')


# In[3]:


#### Data Preprocessing
ratings_with_name = ratings.merge(books, on='ISBN')
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
popularity_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popularity_df[popularity_df['num_ratings'] > 250].sort_values('avg_rating', ascending=False).head(50)
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'num_ratings', 'avg_rating']]


# In[4]:


#### Collaborative Filtering Based Recommender System
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
exp_users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(exp_users)]
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)


# In[5]:


user_similarity_score = cosine_similarity(pt.T)
user_similarity_df = pd.DataFrame(user_similarity_score, index=pt.columns, columns=pt.columns)


# In[6]:


def recommend_for_user(user_id):
    if user_id in user_similarity_df.columns:
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6].index
        similar_users_ratings = final_ratings[final_ratings['User-ID'].isin(similar_users)]
        
        recommendations = similar_users_ratings.groupby('Book-Title').agg({'Book-Rating': 'mean'}).reset_index()
        recommendations = recommendations.sort_values(by='Book-Rating', ascending=False)
        recommended_books = recommendations['Book-Title'].tolist()
        
        return recommended_books[:5]
    else:
        return []


# In[7]:


#### Streamlit App
st.sidebar.title('Book Recommendation System')
option = st.sidebar.selectbox('Select One', ['Top 50 Books', 'User-Based Recommendation'])

if option == 'Top 50 Books':
    st.title('Top 50 Books')
    st.dataframe(popular_df)
else:
    user_id_input = st.sidebar.text_input('Enter User ID')
    btn1 = st.sidebar.button('Search')
    st.title('Recommended Books for User')
    if btn1 and user_id_input:
        try:
            user_id = int(user_id_input)
            recommended_books = recommend_for_user(user_id)
            if recommended_books:
                for book in recommended_books:
                    st.write(book)
            else:
                st.write("No recommendations found for this user.")
        except ValueError:
            st.write("Please enter a valid User ID.")


# In[8]:


#Warning: to view this Streamlit app on a browser, run it with the following
  #command:

    #streamlit run C:\Users\Admin\anaconda3\lib\site-packages\ipykernel_launcher.py [ARGUMENTS]


# In[ ]:




